/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA && GOOGLE_TENSORRT

#include "tensorflow/compiler/tf2tensorrt/convert/convert_nodes.h"
#include "tensorflow/compiler/tf2tensorrt/convert/op_converter_registry.h"
#include "tensorflow/compiler/tf2tensorrt/convert/ops/layer_utils.h"

namespace tensorflow {
namespace tensorrt {
namespace convert {

class ConvertSelectV2 : public OpConverterBase<ConvertSelectV2> {
 public:
  explicit ConvertSelectV2(OpConverterParams* params)
      : OpConverterBase<ConvertSelectV2>(params) {}

  std::vector<DataType> AllowedDataTypes(OpConverterParams* params) {
    return {DataType::DT_FLOAT, DataType::DT_HALF, DataType::DT_INT32};
  }

  static constexpr std::array<InputArgSpec, 3> InputSpec() {
    return std::array<InputArgSpec, 3>{
        InputArgSpec::Create("cond", TrtInputArg::kBoth),
        InputArgSpec::Create("then", TrtInputArg::kBoth),
        InputArgSpec::Create("else", TrtInputArg::kBoth)};
  }

  Status Validate() {
    const auto& params = *this->params_;
    if (params.use_implicit_batch) {
      return errors::Unimplemented(
          "Conversion for SelectV2 is not supported in implicit batch mode.");
    }

    const auto& inputs = params.inputs;
    const TRT_TensorOrWeights& cond_input = inputs.at(0);
    const auto& node = params.node_def;
    TF_RETURN_IF_ERROR(
        check_type(cond_input.TrtDType(), nvinfer1::DataType::kBOOL, node));

    const auto type_then = inputs[1].TrtDType();
    const auto type_else = inputs[2].TrtDType();
    if (type_then != type_else && (type_then == nvinfer1::DataType::kINT32 ||
                                   type_else == nvinfer1::DataType::kINT32)) {
      // Both or none of (type_then, type_else) should be equal to kINT32.
      return errors::InvalidArgument(
          then_else_dtypes_error_msg(type_then, type_else, node));
    }

    all_tensors_ = true;
    for (int i = 0; i < 3; i++) {
      if (inputs.at(i).is_weights()) {
        if (!i) {
          return errors::InvalidArgument(bool_weight_error_msg(node));
        }
        all_tensors_ = false;
        break;
      }
    }

    nvinfer1::Dims broadcasted_dims[3];
    for (int i = 1; i < 3; i++) {
      TF_RETURN_IF_ERROR(GetTrtBroadcastShape(inputs.at(0), inputs.at(i), true,
                                              false, broadcasted_dims,
                                              broadcasted_dims + i));
    }

    for (int i = 0; i < tensor_.size(); i++) {
      // This will also convert constants to tensors.
      TF_RETURN_IF_ERROR(PrepareTensorForShape(
          params.converter, inputs.at(i), broadcasted_dims[i],
          params.validation_only, &tensor_[i], node, i));
    }

    return Status::OK();
  }

  Status Convert() {
    const auto& params = *this->params_;
    const auto& inputs = params.inputs;
    auto* converter = params.converter;

    nvinfer1::ISelectLayer* select_layer = converter->network()->addSelect(
        *tensor_[0]->trt_tensor(),  // cond_tensor
        *tensor_[1]->trt_tensor(),  // then_tensor
        *tensor_[2]->trt_tensor()   // else_tensor
        );

    converter->SetLayerName(select_layer, params.node_def.name(), "selectv2");
    AddOutput(TRT_TensorOrWeights(select_layer->getOutput(0)));
    return Status::OK();
  }

 private:
  bool all_tensors_;
  std::array<ITensorProxyPtr, 3> tensor_{nullptr, nullptr, nullptr};
};

std::string bool_weight_error_msg(const NodeDef& node_def) {
  return "Boolean parameter '" + node_def.input(0) + "' of " + node_def.op() +
         " operation in " + node_def.name() + " cannot be passed as a weight.";
}

std::string then_else_dtypes_error_msg(nvinfer1::DataType type_then,
                                       nvinfer1::DataType type_else,
                                       const NodeDef& node) {
  return "DataTypes (" + DebugString(type_then) + ", " +
         DebugString(type_else) + ") of parameters (" + node.input(1) + ", " +
         node.input(2) + ") of " + node.op() + " operation in " + node.name() +
         " are incompatible.";
}

REGISTER_DEFAULT_TRT_OP_CONVERTER(MakeConverterFunction<ConvertSelectV2>(),
                                  "SelectV2");
}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow
#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
