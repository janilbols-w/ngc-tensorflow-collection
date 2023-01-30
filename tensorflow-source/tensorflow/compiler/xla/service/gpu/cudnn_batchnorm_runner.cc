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

#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_runner.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

struct CudnnBatchNormParamsCommon {
  se::DeviceMemoryBase operand;
  se::dnn::BatchDescriptor operand_desc;
  se::dnn::BatchDescriptor scale_offset_desc;
  se::DeviceMemory<float> scale;
  float epsilon;
};

struct CudnnBatchNormForwardInferenceParams {
  CudnnBatchNormParamsCommon common;
  se::DeviceMemoryBase output;
  se::DeviceMemory<float> offset;
  se::DeviceMemory<float> mean;
  se::DeviceMemory<float> variance;
};

struct CudnnBatchNormForwardTrainingParams {
  CudnnBatchNormParamsCommon common;
  se::DeviceMemoryBase side_input;
  se::DeviceMemoryBase output_data;
  se::DeviceMemory<float> offset;
  se::DeviceMemory<float> output_mean;
  se::DeviceMemory<float> output_inv_stddev;
  se::dnn::ActivationMode activation_mode;
};

struct CudnnBatchNormBackwardParams {
  CudnnBatchNormParamsCommon common;
  se::DeviceMemoryBase output_grad_data;
  se::DeviceMemoryBase grad_output;
  se::DeviceMemory<float> output_grad_scale;
  se::DeviceMemory<float> output_grad_offset;
  se::DeviceMemory<float> mean;
  se::DeviceMemory<float> inv_stddev;
};

void AssignCommonParams(const CudnnBatchNormConfig& config,
                        CudnnBatchNormParamsCommon* params,
                        const se::DeviceMemoryBase& operand,
                        const se::DeviceMemory<float>& scale) {
  // The BatchNormTraining HLO outputs a tuple of three elements: output data,
  // batch mean, and batch variance.  We want to make our descriptors based on
  // the shape of the output data. Batchnorm backward call outputs a tuple of
  // three elements: grad data, grad offset, and grad scale.  We want to make
  // our descriptors based on the shape of the grad data.
  const Shape& shape = config.output_shape;
  DnnBatchDescriptors batch_descs =
      MakeBatchNormDescriptors(shape, config.feature_index);
  params->operand_desc = batch_descs.input_desc;
  params->scale_offset_desc = batch_descs.scale_offset_desc;
  params->operand = operand;
  params->scale = scale;
  params->epsilon = config.epsilon;
}

template <typename ElemType>
void RunCudnnBatchNormForwardInferenceImpl(
    CudnnBatchNormForwardInferenceParams* params, se::Stream* stream) {
  se::DeviceMemory<ElemType> null_device_ptr(nullptr);
  auto output_buf = se::DeviceMemory<ElemType>(params->output);
  stream->ThenBatchNormalizationForward(
      se::DeviceMemory<ElemType>(params->common.operand),
      params->common.scale,                                         //
      params->offset,                                               //
      params->mean,                                                 //
      params->variance,                                             //
      /*side_input=*/null_device_ptr, params->common.operand_desc,  //
      params->common.scale_offset_desc,                             //
      static_cast<double>(params->common.epsilon),                  //
      // TODO(b/137108598): Extend method to allow use of non-trivial
      // exponential averaging.
      /*exponential_average_factor=*/1.0,
      se::dnn::ActivationMode::kNone,       //
      &output_buf,                          //
      /*batch_mean=*/nullptr,               //
      /*batch_var=*/nullptr,                //
      /*saved_mean=*/nullptr,               //
      /*saved_inv_var=*/nullptr,            //
      /*is_training=*/false,                //
      /*reserve_space_allocator=*/nullptr,  //
      /*workspace_allocator=*/nullptr);
}

template <typename ElemType>
void RunCudnnBatchNormForwardTrainingImpl(
    CudnnBatchNormForwardTrainingParams* params,
    se::ScratchAllocator* reserve_space_allocator,
    se::ScratchAllocator* workspace_allocator, se::Stream* stream) {
  se::DeviceMemory<float> null_device_ptr(nullptr);
  se::DeviceMemory<ElemType> null_elem_device_ptr(nullptr);
  auto output_data = se::DeviceMemory<ElemType>(params->output_data);
  auto allocator =
      [&](se::ScratchAllocator* space_allocator) -> se::ScratchAllocator* {
    if (dynamic_cast<ScratchBufAllocator*>(space_allocator)->IsBufferNull()) {
      return nullptr;
    }
    return space_allocator;
  };
  reserve_space_allocator = allocator(reserve_space_allocator);
  workspace_allocator = allocator(workspace_allocator);
  stream->ThenBatchNormalizationForward(
      se::DeviceMemory<ElemType>(params->common.operand),
      params->common.scale,                            //
      params->offset,                                  //
      /*estimated_mean=*/null_device_ptr,              //
      /*estimated_variance=*/null_device_ptr,          //
      se::DeviceMemory<ElemType>(params->side_input),  //
      params->common.operand_desc,                     //
      params->common.scale_offset_desc,                //
      params->common.epsilon,                          //
      // TODO(b/137108598): Extend method to allow use of non-trivial
      // exponential averaging.
      /*exponential_average_factor=*/1.0,
      params->activation_mode,                       //
      &output_data,                                  //
      /*batch_mean=*/&null_device_ptr,               //
      /*batch_var=*/&null_device_ptr,                //
      /*saved_mean=*/&params->output_mean,           //
      /*saved_inv_var=*/&params->output_inv_stddev,  //
      /*is_training=*/true,                          //
      reserve_space_allocator,                       //
      workspace_allocator);
}

template <typename ElemType>
void RunCudnnBatchNormBackwardImpl(CudnnBatchNormBackwardParams* params,
                                   se::DeviceMemory<uint8> reserve_space,
                                   se::ScratchAllocator* workspace_allocator,
                                   se::Stream* stream) {
  se::DeviceMemory<float> null_device_ptr(nullptr);
  se::DeviceMemory<ElemType> null_elem_device_ptr(nullptr);
  if (dynamic_cast<ScratchBufAllocator*>(workspace_allocator)->IsBufferNull()) {
    workspace_allocator = nullptr;
  }
  auto output_grad_data = se::DeviceMemory<ElemType>(params->output_grad_data);
  stream->ThenBatchNormalizationBackward(
      se::DeviceMemory<ElemType>(params->grad_output),     //
      se::DeviceMemory<ElemType>(params->common.operand),  //
      params->common.scale,                                //
      /*offset=*/null_device_ptr,                          //
      params->mean,                                        //
      params->inv_stddev,                                  //
      /*y=*/null_elem_device_ptr,                          //
      params->common.operand_desc,                         //
      params->common.scale_offset_desc,                    //
      params->common.epsilon,                              //
      se::dnn::ActivationMode::kNone,                      //
      &output_grad_data,                                   //
      &params->output_grad_scale,                          //
      &params->output_grad_offset,                         //
      /*side_input_backprop=*/&null_elem_device_ptr,       //
      /*reserve_space_allocator=*/&reserve_space,          //
      /*workspace_allocator=*/workspace_allocator);
}

}  // namespace

CudnnBatchNormConfig GetCudnnBatchNormConfig(const HloInstruction* instr,
                                             float epsilon,
                                             int64_t feature_index) {
  CudnnBatchNormConfig config;

  config.output_shape = instr->shape().IsTuple()
                            ? instr->shape().tuple_shapes(0)
                            : instr->shape();
  config.output_type = config.output_shape.element_type();
  config.epsilon = epsilon;
  config.feature_index = feature_index;
  return config;
}

StatusOr<CudnnBatchNormForwardTrainingConfig>
GetCudnnBatchNormForwardTrainingConfig(const HloInstruction* instr,
                                       float epsilon, int64_t feature_index) {
  CudnnBatchNormForwardTrainingConfig config;
  config.batchnorm_config =
      GetCudnnBatchNormConfig(instr, epsilon, feature_index);
  TF_ASSIGN_OR_RETURN(CudnnBatchNormBackendConfig backend_config,
                      instr->backend_config<CudnnBatchNormBackendConfig>());
  if (!se::dnn::ActivationMode_IsValid(backend_config.activation_mode())) {
    return InternalError("Bad activation mode: %s",
                         backend_config.ShortDebugString());
  }
  config.activation_mode =
      static_cast<se::dnn::ActivationMode>(backend_config.activation_mode());

  return config;
}

Status RunCudnnBatchNormForwardInference(
    const CudnnBatchNormConfig& config, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, se::DeviceMemory<float> mean,
    se::DeviceMemory<float> variance, se::Stream* stream) {
  CudnnBatchNormForwardInferenceParams inference_params;
  AssignCommonParams(config, &inference_params.common, operand, scale);
  inference_params.offset = offset;
  inference_params.mean = mean;
  inference_params.variance = variance;
  inference_params.output = output;

  switch (config.output_type) {
    case F16:
      RunCudnnBatchNormForwardInferenceImpl<Eigen::half>(&inference_params,
                                                         stream);
      break;
    case F32:
      RunCudnnBatchNormForwardInferenceImpl<float>(&inference_params, stream);
      break;
    default:
      return Unimplemented(
          "Primitive type %s not implemented for batchnorm forward inference",
          primitive_util::LowercasePrimitiveTypeName(config.output_type)
              .c_str());
  }
  return Status::OK();
}

Status RunCudnnBatchNormForwardTraining(
    const CudnnBatchNormForwardTrainingConfig& config,
    se::DeviceMemoryBase operand, se::DeviceMemoryBase output_data,
    se::DeviceMemory<float> output_mean,
    se::DeviceMemory<float> output_inv_stddev, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> offset, se::DeviceMemoryBase side_input,
    se::DeviceMemoryBase reserve_space, se::DeviceMemoryBase workspace,
    se::Stream* stream) {
  CudnnBatchNormForwardTrainingParams forward_params;
  ScratchBufAllocator reserve_space_scratch_allocator(reserve_space);
  ScratchBufAllocator workspace_scratch_allocator(workspace);
  AssignCommonParams(config.batchnorm_config, &forward_params.common, operand,
                     scale);
  forward_params.offset = offset;
  forward_params.output_data = output_data;
  forward_params.output_mean = output_mean;
  forward_params.output_inv_stddev = output_inv_stddev;

  forward_params.side_input = side_input;
  forward_params.activation_mode = config.activation_mode;
  switch (config.batchnorm_config.output_type) {
    case F16:
      RunCudnnBatchNormForwardTrainingImpl<Eigen::half>(
          &forward_params, &reserve_space_scratch_allocator,
          &workspace_scratch_allocator, stream);
      break;
    case F32:
      RunCudnnBatchNormForwardTrainingImpl<float>(
          &forward_params, &reserve_space_scratch_allocator,
          &workspace_scratch_allocator, stream);
      break;
    default:
      return Unimplemented(
          "Primitive type %s not implemented for batchnorm forward training",
          primitive_util::LowercasePrimitiveTypeName(
              config.batchnorm_config.output_type)
              .c_str());
  }
  return Status::OK();
}

Status RunCudnnBatchNormBackward(
    const CudnnBatchNormConfig& config, se::DeviceMemoryBase operand,
    se::DeviceMemoryBase output_grad_data, se::DeviceMemoryBase grad_output,
    se::DeviceMemory<float> output_grad_scale,
    se::DeviceMemory<float> output_grad_offset, se::DeviceMemory<float> scale,
    se::DeviceMemory<float> mean, se::DeviceMemory<float> inv_stddev,
    se::DeviceMemory<uint8> reserve_space, se::DeviceMemoryBase workspace,
    se::Stream* stream) {
  CudnnBatchNormBackwardParams backward_params;
  ScratchBufAllocator workspace_scratch_allocator(workspace);
  AssignCommonParams(config, &backward_params.common, operand, scale);
  backward_params.output_grad_data = output_grad_data;
  backward_params.grad_output = grad_output;
  backward_params.output_grad_scale = output_grad_scale;
  backward_params.output_grad_offset = output_grad_offset;
  backward_params.mean = mean;
  backward_params.inv_stddev = inv_stddev;

  switch (config.output_type) {
    case F16:
      RunCudnnBatchNormBackwardImpl<Eigen::half>(
          &backward_params, reserve_space, &workspace_scratch_allocator,
          stream);
      break;
    case F32:
      RunCudnnBatchNormBackwardImpl<float>(&backward_params, reserve_space,
                                           &workspace_scratch_allocator,
                                           stream);
      break;
    default:
      return Unimplemented(
          "Primitive type %s not implemented for batchnorm backward",
          primitive_util::LowercasePrimitiveTypeName(config.output_type)
              .c_str());
  }
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
