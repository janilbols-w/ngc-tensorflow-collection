/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_softmax_runner.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {
namespace {

struct CudnnSoftmaxParams {
  se::DeviceMemoryBase operand;
  se::DeviceMemoryBase output;
  se::dnn::BatchDescriptor operand_desc;
  bool log;
};

void AssignParams(const CudnnSoftmaxConfig &config,
                  CudnnSoftmaxParams* params,
                  const se::DeviceMemoryBase& operand,
                  se::DeviceMemoryBase& output) {
  const Shape& shape = config.output_shape;
  se::dnn::BatchDescriptor input_desc =
      MakeSoftmaxDescriptor(shape, config.feature_index);
  params->operand = operand;
  params->output = output;
  params->operand_desc = input_desc;
  params->log = config.log;
}

template <typename ElemType>
void RunCudnnSoftmaxImpl(
    CudnnSoftmaxParams* params, se::Stream* stream) {
  auto output_buf = se::DeviceMemory<ElemType>(params->output);
  stream->ThenSoftmax(
      se::DeviceMemory<ElemType>(params->operand),
      params->operand_desc,
      params->log,
      &output_buf);
}

}  // namespace

CudnnSoftmaxConfig GetCudnnSoftmaxConfig(const HloInstruction* instr,
                                         int64 feature_index, bool log) {
  CudnnSoftmaxConfig config;

  config.output_shape = instr->shape();
  config.output_type = config.output_shape.element_type();
  config.feature_index = feature_index;
  config.log = log;
  return config;
}

Status RunCudnnSoftmax(
    const CudnnSoftmaxConfig &config, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output, se::Stream* stream) {
  CudnnSoftmaxParams params;
  AssignParams(config, &params, operand, output);

  switch (config.output_type) {
    case F16:
      RunCudnnSoftmaxImpl<Eigen::half>(&params, stream);
      break;
    case F32:
      RunCudnnSoftmaxImpl<float>(&params, stream);
      break;
    default:
      return Unimplemented(
          "Primitive type %s not implemented for softmax",
          primitive_util::LowercasePrimitiveTypeName(config.output_type)
              .c_str());
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
