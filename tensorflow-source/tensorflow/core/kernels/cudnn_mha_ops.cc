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
#define EIGEN_USE_THREADS

#include <stddef.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_set>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/use_cudnn.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/stream_executor_util.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

/*
 * This module implements ops that fuse a multi-head-attention model using the
 * underlying Cudnn library.
 *
 * Cudnn MHA library exposes an opaque parameter buffer with unknown layout and
 * format. And it is very likely that if saved, they cannot be used across
 * different GPUs. So users need to first query the size of the opaque
 * parameter buffer, and convert it to and from its canonical forms. But each
 * actual training step is carried out with the parameter buffer.
 *
 * Similar to many other ops, the forward op has two flavors: training and
 * inference. When training is specified, additional data in reserve_space will
 * be produced for the backward pass. So there is a performance penalty.
 *
 * In addition to the actual data and reserve_space, Cudnn also needs more
 * memory as temporary workspace. The memory management to and from
 * stream-executor is done through ScratchAllocator. In general,
 * stream-executor is responsible for creating the memory of proper size. And
 * TensorFlow is responsible for making sure the memory is alive long enough
 * and recycles afterwards.
 *
 */
namespace tensorflow {
using CPUDevice = Eigen::ThreadPoolDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

using GPUDevice = Eigen::GpuDevice;
using se::Stream;
using se::StreamExecutor;
using se::dnn::MhaDescriptor;

template <typename Device, typename T>
class CudnnMHAForwardOp;

template <typename Device, typename T>
class CudnnMHABackwardOp;

namespace {
using se::DeviceMemory;
using se::DeviceMemoryBase;
using se::ScratchAllocator;
using se::dnn::MhaSeqDataDescriptor;
using se::dnn::ProfileResult;
using se::dnn::ToDataType;
using se::port::StatusOr;

uint64_t HashList(const std::vector<int>& list) {
  if (list.empty()) {
    return 0;
  }
  uint64_t hash_code = list[0];
  for (int i = 1; i < list.size(); i++) {
    hash_code = Hash64Combine(hash_code, list[i]);
  }
  return hash_code;
}

inline se::port::Status ToExecutorStatus(const Status& s) {
  return s.ok() ? se::port::Status::OK()
                : se::port::Status(static_cast<se::port::error::Code>(
                                       static_cast<int>(s.code())),
                                   s.error_message());
}
template <typename T>
const DeviceMemory<T> AsDeviceMemory(const Tensor* tensor) {
  return DeviceMemory<T>::MakeFromByteSize(
      const_cast<T*>(tensor->template flat<T>().data()),
      tensor->template flat<T>().size() * sizeof(T));
}

template <typename T>
DeviceMemory<T> AsDeviceMemory(Tensor* tensor) {
  return DeviceMemory<T>::MakeFromByteSize(
      tensor->template flat<T>().data(),
      tensor->template flat<T>().size() * sizeof(T));
}

template <typename U, typename T>
DeviceMemory<U> CastDeviceMemory(Tensor* tensor) {
  return DeviceMemory<U>::MakeFromByteSize(
      tensor->template flat<T>().data(),
      tensor->template flat<T>().size() * sizeof(T));
}

// A helper to allocate temporary scratch memory for Cudnn Mha models. It
// takes the ownership of the underlying memory. The expectation is that the
// memory should be alive for the span of the Cudnn MHA itself.
template <typename T>
class CudnnMhaAllocatorInTemp : public ScratchAllocator {
 public:
  ~CudnnMhaAllocatorInTemp() override = default;

  explicit CudnnMhaAllocatorInTemp(OpKernelContext* context)
      : context_(context) {}
  int64_t GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64_t>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override {
    Tensor temporary_memory;
    const DataType tf_data_type = DataTypeToEnum<T>::value;
    int64_t allocate_count =
        Eigen::divup(byte_size, static_cast<int64_t>(sizeof(T)));
    Status allocation_status(context_->allocate_temp(
        tf_data_type, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    total_byte_size_ += byte_size;
    return DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory.template flat<T>().data(),
        temporary_memory.template flat<T>().size() * sizeof(T));
  }

  int64_t TotalByteSize() const { return total_byte_size_; }

  Tensor get_allocated_tensor(int index) const {
    return allocated_tensors_[index];
  }

 private:
  int64_t total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  std::vector<Tensor> allocated_tensors_;
};

// A helper to allocate memory for Cudnn MHA models as a kernel output. It is
// used by forward pass kernel to feed the output to the backward pass.
// The memory is expected to live long enough after the backward pass is
// finished.
template <typename T>
class CudnnMhaAllocatorInOutput : public ScratchAllocator {
 public:
  ~CudnnMhaAllocatorInOutput() override {}
  CudnnMhaAllocatorInOutput(OpKernelContext* context, int output_index)
      : context_(context), output_index_(output_index) {}
  int64_t GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64_t>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override {
    CHECK(total_byte_size_ == 0)
        << "Reserve space allocator can only be called once";
    int64_t allocate_count =
        Eigen::divup(byte_size, static_cast<int64_t>(sizeof(T)));

    Tensor* temporary_memory = nullptr;
    Status allocation_status(context_->allocate_output(
        output_index_, TensorShape({allocate_count}), &temporary_memory));
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    total_byte_size_ += byte_size;
    auto memory_uint8 = DeviceMemory<uint8>::MakeFromByteSize(
        temporary_memory->template flat<T>().data(),
        temporary_memory->template flat<T>().size() * sizeof(T));
    return StatusOr<DeviceMemory<uint8>>(memory_uint8);
  }
  int64_t TotalByteSize() const { return total_byte_size_; }

 private:
  int64_t total_byte_size_ = 0;
  OpKernelContext* context_;  // not owned
  int output_index_;
};

// A helper to allocate memory for Cudnn MHA models, which is
// expected to live between kernel invocations.
// This class is not thread-safe.
class CudnnMHASpaceAllocator : public ScratchAllocator {
 public:
  explicit CudnnMHASpaceAllocator(OpKernelContext* context)
      : context_(context) {}

  ~CudnnMHASpaceAllocator() override {}

  int64_t GetMemoryLimitInBytes() override {
    return std::numeric_limits<int64_t>::max();
  }

  StatusOr<DeviceMemory<uint8>> AllocateBytes(int64_t byte_size) override {
    if (total_byte_size_ != 0) {
      return Status(error::FAILED_PRECONDITION,
                    "Space allocator can only be called once");
    }

    Status allocation_status =
        context_->allocate_temp(DT_UINT8, TensorShape({byte_size}), &tensor_);
    if (!allocation_status.ok()) {
      return ToExecutorStatus(allocation_status);
    }
    total_byte_size_ += byte_size;
    return AsDeviceMemory<uint8>(&tensor_);
  }
  int64_t TotalByteSize() { return total_byte_size_; }

 private:
  int64_t total_byte_size_ = 0;
  Tensor tensor_;
  OpKernelContext* context_;  // not owned
};

// A helper class that collects the shapes to describe a MHA model.
struct CudnnMhaModelShapes {
  int num_heads;
  int batch_size;
  int q_size;
  int k_size;
  int v_size;
  int q_proj_size;
  int k_proj_size;
  int v_proj_size;
  int o_proj_size;
  int max_seq_len_qo;
  int max_seq_len_kv;
  bool proj_bias = true;
  bool is_training = false;
  // If you add new field to this structure, please take care of
  // updating IsCompatibleWith() below.
  TensorShape input_q_shape;
  TensorShape input_k_shape;
  TensorShape input_v_shape;
  TensorShape output_shape;

  // At present only fields related to cached MhaDescriptor are concerned.
  bool IsCompatibleWith(const CudnnMhaModelShapes& rhs) const {
    return num_heads == rhs.num_heads && batch_size == rhs.batch_size &&
           q_size == rhs.q_size && k_size == rhs.k_size &&
           v_size == rhs.v_size && q_proj_size == rhs.q_proj_size &&
           k_proj_size == rhs.k_proj_size && v_proj_size == rhs.v_proj_size &&
           o_proj_size == rhs.o_proj_size &&
           max_seq_len_qo == rhs.max_seq_len_qo &&
           max_seq_len_kv == rhs.max_seq_len_kv && proj_bias == rhs.proj_bias &&
           is_training == rhs.is_training;
  }
  string DebugString() const {
    return strings::Printf(
        "[num_heads, batch_size, q_size, k_size, v_size, "
        "q_proj_size, k_proj_size, v_proj_size, o_proj_size, max_seq_len_qo, "
        "max_seq_len_kv, proj_bias]: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
        "%d, %d] ",
        num_heads, batch_size, q_size, k_size, v_size, q_proj_size, k_proj_size,
        v_proj_size, o_proj_size, max_seq_len_qo, max_seq_len_kv, proj_bias);
  }
};

// Utility class for using CudnnMhaConfig as a hash table key.
struct CudnnMhaConfigHasher {
  uint64_t operator()(const CudnnMhaModelShapes& shapes) const {
    return HashList({shapes.num_heads, shapes.batch_size, shapes.q_size,
                     shapes.k_size, shapes.v_size, shapes.q_proj_size,
                     shapes.k_proj_size, shapes.v_proj_size, shapes.o_proj_size,
                     shapes.max_seq_len_qo, shapes.max_seq_len_kv,
                     shapes.proj_bias});
  }
};

struct CudnnMhaConfigComparator {
  bool operator()(const CudnnMhaModelShapes& lhs,
                  const CudnnMhaModelShapes& rhs) const {
    return lhs.IsCompatibleWith(rhs);
  }
};

// Pointers to mhA scratch space for a specific set of shape parameters (used as
// a hash table value in CudnnMHAForwardOp and CudnnMHABackwardOp).
struct MhaScratchSpace {
  std::unique_ptr<MhaDescriptor> mha_desc;
  std::unique_ptr<CudnnMHASpaceAllocator> dropout_state_allocator;
};

// Extract and checks the forward input tensors and shapes from the
// OpKernelContext.
Status ExtractForwardInput(OpKernelContext* context, const Tensor** input_q,
                           const Tensor** input_k, const Tensor** input_v,
                           const Tensor** params,
                           CudnnMhaModelShapes* model_shapes) {
  TF_RETURN_IF_ERROR(context->input("input_q", input_q));
  TF_RETURN_IF_ERROR(context->input("input_k", input_k));
  TF_RETURN_IF_ERROR(context->input("input_v", input_v));
  TF_RETURN_IF_ERROR(context->input("params", params));

  if ((*input_q)->dims() != 3) {
    return errors::InvalidArgument("MHA input must be a 3-D tensor.");
  }
  if ((*input_k)->dims() != 3) {
    return errors::InvalidArgument("MHA input must be a 3-D tensor.");
  }
  if ((*input_v)->dims() != 3) {
    return errors::InvalidArgument("MHA input must be a 3-D tensor.");
  }

  model_shapes->input_q_shape = (*input_q)->shape();
  model_shapes->input_k_shape = (*input_k)->shape();
  model_shapes->input_v_shape = (*input_v)->shape();

  model_shapes->batch_size = (*input_q)->dim_size(0);

  model_shapes->max_seq_len_qo = (*input_q)->dim_size(1);
  model_shapes->max_seq_len_kv = (*input_k)->dim_size(1);

  model_shapes->q_size = (*input_q)->dim_size(2);
  model_shapes->k_size = (*input_k)->dim_size(2);
  model_shapes->v_size = (*input_v)->dim_size(2);

  model_shapes->output_shape =
      TensorShape({model_shapes->batch_size, model_shapes->max_seq_len_qo,
                   model_shapes->o_proj_size});
  return Status::OK();
}

template <typename T>
Status CreateForwardAndBackwardIODescriptors(
    OpKernelContext* context, const CudnnMhaModelShapes& model_shapes,
    std::unique_ptr<MhaSeqDataDescriptor>* input_q_desc,
    std::unique_ptr<MhaSeqDataDescriptor>* input_k_desc,
    std::unique_ptr<MhaSeqDataDescriptor>* input_v_desc,
    std::unique_ptr<MhaSeqDataDescriptor>* output_desc,
    const absl::Span<const int> seq_lengths_qo,
    const absl::Span<const int> seq_lengths_kv) {
  StreamExecutor* executor = context->op_device_context()->stream()->parent();
  se::dnn::DataType data_type = ToDataType<T>::value;

  const TensorShape& input_q_shape = model_shapes.input_q_shape;
  const TensorShape& input_k_shape = model_shapes.input_k_shape;
  const TensorShape& input_v_shape = model_shapes.input_v_shape;
  const TensorShape& output_shape = model_shapes.output_shape;

  DCHECK_EQ(input_q_shape.dims(), 3);
  DCHECK_EQ(input_k_shape.dims(), 3);
  DCHECK_EQ(input_v_shape.dims(), 3);
  DCHECK_EQ(output_shape.dims(), 3);

  if (seq_lengths_qo.data() != nullptr) {
    // seq-q
    int q_seq_array_size = model_shapes.batch_size * 1;
    auto input_desc_s = executor->createMhaSeqDataDescriptor(
        model_shapes.batch_size, model_shapes.max_seq_len_qo,
        model_shapes.q_size, model_shapes.batch_size * 1, seq_lengths_qo,
        data_type);
    TF_RETURN_IF_ERROR(input_desc_s.status());
    *input_q_desc = input_desc_s.ConsumeValueOrDie();

    // seq-o
    int o_vect_size =
        model_shapes.o_proj_size > 0
            ? model_shapes.o_proj_size
            : (model_shapes.v_proj_size > 0 ? model_shapes.v_proj_size
                                            : model_shapes.v_size) *
                  model_shapes.num_heads;
    input_desc_s = executor->createMhaSeqDataDescriptor(
        model_shapes.batch_size, model_shapes.max_seq_len_qo, o_vect_size,
        q_seq_array_size, seq_lengths_qo, data_type);
    TF_RETURN_IF_ERROR(input_desc_s.status());
    *output_desc = input_desc_s.ConsumeValueOrDie();
  }

  if (seq_lengths_kv.data() != nullptr) {
    // seq-k
    int k_seq_array_size = model_shapes.batch_size;
    auto input_desc_s = executor->createMhaSeqDataDescriptor(
        model_shapes.batch_size, model_shapes.max_seq_len_kv,
        model_shapes.k_size, k_seq_array_size, seq_lengths_kv, data_type);
    TF_RETURN_IF_ERROR(input_desc_s.status());
    *input_k_desc = input_desc_s.ConsumeValueOrDie();

    // seq-v
    input_desc_s = executor->createMhaSeqDataDescriptor(
        model_shapes.batch_size, model_shapes.max_seq_len_kv,
        model_shapes.v_size, k_seq_array_size, seq_lengths_kv, data_type);
    TF_RETURN_IF_ERROR(input_desc_s.status());
    *input_v_desc = input_desc_s.ConsumeValueOrDie();
  }

  return Status::OK();
}

template <typename T>
Status DoForward(OpKernelContext* context, const MhaDescriptor& mha_desc,
                 const CudnnMhaModelShapes& model_shapes,
                 /* forward inputs */
                 const Tensor* input_q, const Tensor* input_k,
                 const Tensor* input_v, const Tensor* params,
                 const bool is_training, const bool residual_link,
                 /* forward outputs, outputs of the function */
                 Tensor* output, ScratchAllocator* reserve_space_allocator,
                 ScratchAllocator* workspace_allocator) {
  int q_seq_array_size = model_shapes.batch_size * 1;
  int k_seq_array_size = model_shapes.batch_size;

  std::unique_ptr<MhaSeqDataDescriptor> input_q_desc;
  std::unique_ptr<MhaSeqDataDescriptor> input_k_desc;
  std::unique_ptr<MhaSeqDataDescriptor> input_v_desc;
  std::unique_ptr<MhaSeqDataDescriptor> output_desc;

  std::vector<int> seq_len_qo_array(q_seq_array_size,
                                    model_shapes.max_seq_len_qo);
  std::vector<int> seq_len_kv_array(k_seq_array_size,
                                    model_shapes.max_seq_len_kv);
  auto seq_lengths_qo =
      absl::Span<const int>(seq_len_qo_array.data(), q_seq_array_size);
  auto seq_lengths_kv =
      absl::Span<const int>(seq_len_kv_array.data(), k_seq_array_size);

  TF_RETURN_IF_ERROR(CreateForwardAndBackwardIODescriptors<T>(
      context, model_shapes, &input_q_desc, &input_k_desc, &input_v_desc,
      &output_desc, seq_lengths_qo, seq_lengths_kv));

  auto input_q_data = AsDeviceMemory<T>(input_q);
  auto input_k_data = AsDeviceMemory<T>(input_k);
  auto input_v_data = AsDeviceMemory<T>(input_v);

  auto params_data = AsDeviceMemory<T>(params);
  auto output_data = AsDeviceMemory<T>(output);

  Stream* stream = context->op_device_context()->stream();

  Tensor seq_lengths_qo_tensor;
  DeviceMemory<int> seq_lengths_qo_ptr;

  if (q_seq_array_size > 0) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, {static_cast<long>(seq_lengths_qo.size())},
        &seq_lengths_qo_tensor));
    seq_lengths_qo_ptr = AsDeviceMemory<int>(&seq_lengths_qo_tensor);
    if (!stream
             ->ThenMemcpy(&seq_lengths_qo_ptr, seq_lengths_qo.data(),
                          seq_lengths_qo.size() * sizeof(int))
             .ok()) {
      return errors::InvalidArgument(
          "Failed to copy memory from host to "
          "device for sequence_lengths of Q/O in "
          "CudnnMHA");
    }
  }

  Tensor seq_lengths_kv_tensor;
  DeviceMemory<int> seq_lengths_kv_ptr;

  if (k_seq_array_size > 0) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, {static_cast<long>(seq_lengths_kv.size())},
        &seq_lengths_kv_tensor));
    seq_lengths_kv_ptr = AsDeviceMemory<int>(&seq_lengths_kv_tensor);
    if (!stream
             ->ThenMemcpy(&seq_lengths_kv_ptr, seq_lengths_kv.data(),
                          seq_lengths_kv.size() * sizeof(int))
             .ok()) {
      return errors::InvalidArgument(
          "Failed to copy memory from host to "
          "device for sequence_lengths of K/V in "
          "CudnnMHA");
    }
  }

  bool launch_success =
      stream
          ->ThenMhaForward(
              mha_desc, *input_q_desc, input_q_data, *input_k_desc,
              input_k_data, *input_v_desc, input_v_data, seq_lengths_qo_ptr,
              seq_lengths_kv_ptr, &params_data, *output_desc, &output_data,
              model_shapes.max_seq_len_qo, model_shapes.max_seq_len_kv,
              is_training, residual_link, reserve_space_allocator,
              workspace_allocator)
          .ok();
  return launch_success
             ? Status::OK()
             : errors::Internal(
                   "Failed to call ThenMhaForward with model config: ",
                   model_shapes.DebugString());
}

template <typename T>
Status DoBackward(OpKernelContext* context, const MhaDescriptor& mha_desc,
                  const CudnnMhaModelShapes& model_shapes,
                  /* forward inputs */
                  const Tensor* input_q, const Tensor* input_k,
                  const Tensor* input_v, const Tensor* params,
                  /* forward outputs */
                  const Tensor* output,
                  /* backprop inputs */
                  const Tensor* output_backprop, const Tensor* reserve_space,
                  /* backprop outputs, output of the function */
                  Tensor* input_q_backprop, Tensor* input_k_backprop,
                  Tensor* input_v_backprop, Tensor* params_backprop,
                  ScratchAllocator* workspace_allocator) {
  int q_seq_array_size = model_shapes.batch_size * 1;
  int k_seq_array_size = model_shapes.batch_size;

  std::unique_ptr<MhaSeqDataDescriptor> input_q_desc;
  std::unique_ptr<MhaSeqDataDescriptor> input_k_desc;
  std::unique_ptr<MhaSeqDataDescriptor> input_v_desc;
  std::unique_ptr<MhaSeqDataDescriptor> output_desc;

  std::vector<int> seq_len_qo_array(q_seq_array_size,
                                    model_shapes.max_seq_len_qo);
  std::vector<int> seq_len_kv_array(k_seq_array_size,
                                    model_shapes.max_seq_len_kv);
  auto seq_lengths_qo =
      absl::Span<const int>(seq_len_qo_array.data(), q_seq_array_size);
  auto seq_lengths_kv =
      absl::Span<const int>(seq_len_kv_array.data(), k_seq_array_size);

  TF_RETURN_IF_ERROR(CreateForwardAndBackwardIODescriptors<T>(
      context, model_shapes, &input_q_desc, &input_k_desc, &input_v_desc,
      &output_desc, seq_lengths_qo, seq_lengths_kv));

  auto input_q_data = AsDeviceMemory<T>(input_q);
  auto input_k_data = AsDeviceMemory<T>(input_k);
  auto input_v_data = AsDeviceMemory<T>(input_v);

  auto params_data = AsDeviceMemory<T>(params);
  auto output_data = AsDeviceMemory<T>(output);
  auto output_backprop_data = AsDeviceMemory<T>(output_backprop);

  auto input_q_backprop_data = AsDeviceMemory<T>(input_q_backprop);
  auto input_k_backprop_data = AsDeviceMemory<T>(input_k_backprop);
  auto input_v_backprop_data = AsDeviceMemory<T>(input_v_backprop);

  auto params_backprop_data = AsDeviceMemory<T>(params_backprop);
  auto reserve_space_uint8 =
      CastDeviceMemory<uint8, T>(const_cast<Tensor*>(reserve_space));

  // Creates a memory callback for the workspace. The memory lives to the end
  // of this kernel calls.
  Stream* stream = context->op_device_context()->stream();

  Tensor seq_lengths_qo_tensor;
  DeviceMemory<int> seq_lengths_qo_ptr;

  if (q_seq_array_size > 0) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, {static_cast<long>(seq_lengths_qo.size())},
        &seq_lengths_qo_tensor));
    seq_lengths_qo_ptr = AsDeviceMemory<int>(&seq_lengths_qo_tensor);
    if (!stream
             ->ThenMemcpy(&seq_lengths_qo_ptr, seq_lengths_qo.data(),
                          seq_lengths_qo.size() * sizeof(int))
             .ok()) {
      return errors::InvalidArgument(
          "Failed to copy memory from host to "
          "device for sequence_lengths of Q/O in "
          "CudnnMHA");
    }
  }

  Tensor seq_lengths_kv_tensor;
  DeviceMemory<int> seq_lengths_kv_ptr;
  if (k_seq_array_size > 0) {
    TF_RETURN_IF_ERROR(context->allocate_temp(
        DT_INT32, {static_cast<long>(seq_lengths_kv.size())},
        &seq_lengths_kv_tensor));
    seq_lengths_kv_ptr = AsDeviceMemory<int>(&seq_lengths_kv_tensor);
    if (!stream
             ->ThenMemcpy(&seq_lengths_kv_ptr, seq_lengths_kv.data(),
                          seq_lengths_kv.size() * sizeof(int))
             .ok()) {
      return errors::InvalidArgument(
          "Failed to copy memory from host to "
          "device for sequence_lengths of K/V in "
          "CudnnMHA");
    }
  }

  bool launch_success =
      stream
          ->ThenMhaBackward(mha_desc, *input_q_desc, input_q_data,
                            *input_k_desc, input_k_data, *input_v_desc,
                            input_v_data, &params_data, seq_lengths_qo_ptr,
                            seq_lengths_kv_ptr, *output_desc, output_data,
                            output_backprop_data, &input_q_backprop_data,
                            &input_k_backprop_data, &input_v_backprop_data,
                            &params_backprop_data, model_shapes.max_seq_len_qo,
                            model_shapes.max_seq_len_kv, &reserve_space_uint8,
                            workspace_allocator)
          .ok();
  return launch_success
             ? Status::OK()
             : errors::Internal(
                   "Failed to call ThenMhaBackward with model config: ",
                   model_shapes.DebugString());
}

}  // namespace

class CudnnMHAKernelCommon : public OpKernel {
 protected:
  explicit CudnnMHAKernelCommon(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dropout", &dropout_));
    OP_REQUIRES_OK(context, context->GetAttr("seed", &seed_));
    OP_REQUIRES_OK(context, context->GetAttr("use_bias", &is_proj_bias_));
  }
  uint64_t seed() { return (static_cast<uint64_t>(seed_) << 32); }

  float dropout() const { return dropout_; }
  bool is_proj_bias() const { return is_proj_bias_; }

  template <typename T, typename CompPrec>
  Status CreateMhaDescriptor(OpKernelContext* context,
                             const CudnnMhaModelShapes& model_shapes,
                             ScratchAllocator* dropout_state_allocator,
                             std::unique_ptr<MhaDescriptor>* mha_desc) {
    StreamExecutor* executor = context->op_device_context()->stream()->parent();
    se::dnn::DataType data_type = ToDataType<T>::value;
    se::dnn::DataType comp_prec = ToDataType<CompPrec>::value;

    auto mha_desc_s = executor->createMhaDescriptor(
        model_shapes.num_heads, model_shapes.batch_size, model_shapes.q_size,
        model_shapes.k_size, model_shapes.v_size, model_shapes.q_proj_size,
        model_shapes.k_proj_size, model_shapes.v_proj_size,
        model_shapes.o_proj_size, model_shapes.max_seq_len_qo,
        model_shapes.max_seq_len_kv, data_type, comp_prec, dropout(),
        is_proj_bias(), seed(), dropout_state_allocator);
    TF_RETURN_IF_ERROR(mha_desc_s.status());
    *mha_desc = mha_desc_s.ConsumeValueOrDie();

    return Status::OK();
  }

  using MhaStateCache =
      gtl::FlatMap<CudnnMhaModelShapes, MhaScratchSpace, CudnnMhaConfigHasher,
                   CudnnMhaConfigComparator>;
  // Returns a raw mha descriptor pointer. The cache owns the mha descriptor and
  // should outlive the returned pointer.
  template <typename T, typename CompPrec>
  Status GetCachedMhaDescriptor(OpKernelContext* context,
                                const CudnnMhaModelShapes& model_shapes,
                                MhaStateCache* cache,
                                MhaDescriptor** mha_desc) {
    MhaScratchSpace& mha_state = (*cache)[model_shapes];
    if (mha_state.mha_desc == nullptr) {
      CudnnMHASpaceAllocator* dropout_state_allocator =
          new CudnnMHASpaceAllocator(context);
      mha_state.dropout_state_allocator.reset(dropout_state_allocator);
      Status status = CreateMhaDescriptor<T, CompPrec>(
          context, model_shapes, dropout_state_allocator, &mha_state.mha_desc);

      TF_RETURN_IF_ERROR(status);
    }
    *mha_desc = mha_state.mha_desc.get();
    return Status::OK();
  }

 private:
  int seed_;
  float dropout_;
  bool is_proj_bias_;
  bool reset_rnd_gen_state_;
};

// Run the forward operation of the MHA model.
template <typename T>
class CudnnMHAForwardOp<GPUDevice, T> : public CudnnMHAKernelCommon {
 public:
  explicit CudnnMHAForwardOp(OpKernelConstruction* context)
      : CudnnMHAKernelCommon(context) {
    OP_REQUIRES_OK(context, context->GetAttr("is_training", &is_training_));
    OP_REQUIRES_OK(context, context->GetAttr("residual_link", &residual_link_));
    OP_REQUIRES_OK(context, context->GetAttr("num_heads", &num_heads_));
    OP_REQUIRES_OK(context, context->GetAttr("key_dim", &key_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("output_dim", &output_dim_));

    if (context->HasAttr("value_dim")) {
      OP_REQUIRES_OK(context, context->GetAttr("value_dim", &value_dim_));
    } else {
      value_dim_ = key_dim_;
    }

    // TODO(shuw) Read debug env variables.
  }

  void Compute(OpKernelContext* context) override {
    ComputeImpl(context, num_heads(), key_dim(), value_dim(), output_dim());
  }

 protected:
  virtual void ComputeImpl(OpKernelContext* context, int num_heads,
                           int k_proj_size, int v_proj_size, int o_proj_size) {
    const Tensor* input_q = nullptr;
    const Tensor* input_k = nullptr;
    const Tensor* input_v = nullptr;
    const Tensor* params = nullptr;

    CudnnMhaModelShapes model_shapes;
    model_shapes.num_heads = num_heads;
    // q and k proj_size have to be equal.
    model_shapes.q_proj_size = k_proj_size;
    model_shapes.k_proj_size = k_proj_size;
    model_shapes.v_proj_size = v_proj_size;
    model_shapes.o_proj_size = o_proj_size;

    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, &input_q, &input_k, &input_v,
                                       &params, &model_shapes));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, AllocateOutputs(context, model_shapes, &output));

    // Creates a memory callback for the reserve_space. The memory lives in the
    // output of this kernel. And it will be fed into the backward pass when
    // needed.
    CudnnMhaAllocatorInOutput<T> reserve_space_allocator(context, 1);
    // Creates a memory callback for the workspace. The memory lives to the end
    // of this kernel calls.
    CudnnMhaAllocatorInTemp<uint8> workspace_allocator(context);

    Status launch_status;
    {
      mutex_lock l(mu_);
      MhaDescriptor* mha_desc_ptr = nullptr;
      OP_REQUIRES_OK(context, GetCachedMhaDescriptor<T, T>(
                                  context, model_shapes, &mha_state_cache_,
                                  &mha_desc_ptr));

      launch_status =
          DoForward<T>(context, *mha_desc_ptr, model_shapes, input_q, input_k,
                       input_v, params, is_training_, residual_link_, output,
                       &reserve_space_allocator, &workspace_allocator);
    }
    OP_REQUIRES_OK(context, launch_status);

    Tensor* output_host_reserved = nullptr;
    // output_host_reserved stores opaque info used for backprop when running
    // in training mode. At present, it includes a num_heads, projection size
    // for each inputs and output.
    if (is_training()) {
      OP_REQUIRES_OK(context, context->allocate_output(2, TensorShape({4}),
                                                       &output_host_reserved));
      auto output_host_reserved_int = output_host_reserved->vec<int>();
      output_host_reserved_int(0) = num_heads;
      output_host_reserved_int(1) = k_proj_size;
      output_host_reserved_int(2) = v_proj_size;
      output_host_reserved_int(3) = o_proj_size;
    } else {
      OP_REQUIRES_OK(context,
                     context->allocate_output(2, {}, &output_host_reserved));
    }
  }

 protected:
  bool is_training() const { return is_training_; }
  bool residual_link() const { return residual_link_; }
  int num_heads() { return num_heads_; }
  int key_dim() { return key_dim_; }
  int value_dim() { return value_dim_; }
  int output_dim() { return output_dim_; }

 private:
  Status AllocateOutputs(OpKernelContext* context,
                         const CudnnMhaModelShapes& model_shapes,
                         Tensor** output) {
    const TensorShape& output_shape = model_shapes.output_shape;
    TF_RETURN_IF_ERROR(context->allocate_output(0, output_shape, output));

    if (!is_training_) {
      Tensor* dummy_reserve_space = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(1, {}, &dummy_reserve_space));
    }
    return Status::OK();
  }

  mutex mu_;
  bool is_training_;
  bool residual_link_;
  int num_heads_;
  int key_dim_;
  int value_dim_;
  int output_dim_;
  MhaStateCache mha_state_cache_ TF_GUARDED_BY(mu_);
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CudnnMHA")                 \
                              .Device(DEVICE_GPU)          \
                              .HostMemory("host_reserved") \
                              .TypeConstraint<T>("T"),     \
                          CudnnMHAForwardOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

// Run the backward operation of the MHA model.
template <typename T>
class CudnnMHABackwardOp<GPUDevice, T> : public CudnnMHAKernelCommon {
 public:
  explicit CudnnMHABackwardOp(OpKernelConstruction* context)
      : CudnnMHAKernelCommon(context) {}

  void Compute(OpKernelContext* context) override { ComputeImpl(context); }

 protected:
  virtual void ComputeImpl(OpKernelContext* context) {
    const Tensor* input_q = nullptr;
    const Tensor* input_k = nullptr;
    const Tensor* input_v = nullptr;
    const Tensor* params = nullptr;
    CudnnMhaModelShapes model_shapes;
    int num_heads(-1), k_proj_size(-1), v_proj_size(-1), o_proj_size(-1);
    OP_REQUIRES_OK(context,
                   GetMHAProjectionConfig(context, num_heads, k_proj_size,
                                          v_proj_size, o_proj_size));
    model_shapes.num_heads = num_heads;
    model_shapes.q_proj_size = k_proj_size;
    model_shapes.k_proj_size = k_proj_size;
    model_shapes.v_proj_size = v_proj_size;
    model_shapes.o_proj_size = o_proj_size;

    OP_REQUIRES_OK(context,
                   ExtractForwardInput(context, &input_q, &input_k, &input_v,
                                       &params, &model_shapes));

    const Tensor* output = nullptr;
    const Tensor* output_backprop = nullptr;
    const Tensor* reserve_space = nullptr;
    OP_REQUIRES_OK(context,
                   ExtractBackwardInputs(context, model_shapes, &output,
                                         &output_backprop, &reserve_space));

    Tensor* input_q_backprop = nullptr;
    Tensor* input_k_backprop = nullptr;
    Tensor* input_v_backprop = nullptr;
    Tensor* params_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   AllocateOutputs(context, model_shapes, params->shape(),
                                   &input_q_backprop, &input_k_backprop,
                                   &input_v_backprop, &params_backprop));

    // Creates a memory callback for the workspace. The memory lives to the end
    // of this kernel calls.
    CudnnMhaAllocatorInTemp<uint8> workspace_allocator(context);

    Status launch_status;
    {
      mutex_lock l(mu_);
      MhaDescriptor* mha_desc_ptr = nullptr;
      OP_REQUIRES_OK(context, GetCachedMhaDescriptor<T, T>(
                                  context, model_shapes, &mha_state_cache_,
                                  &mha_desc_ptr));
      launch_status =
          DoBackward<T>(context, *mha_desc_ptr, model_shapes, input_q, input_k,
                        input_v, params, output, output_backprop, reserve_space,
                        input_q_backprop, input_k_backprop, input_v_backprop,
                        params_backprop, &workspace_allocator);
    }
    OP_REQUIRES_OK(context, launch_status);
  }

 protected:
  Status GetMHAProjectionConfig(OpKernelContext* context, int& num_heads,
                                int& proj_k_size, int& proj_v_size,
                                int& proj_o_size) {
    const Tensor* host_reserved = nullptr;
    TF_RETURN_IF_ERROR(context->input("host_reserved", &host_reserved));

    auto host_reserved_int = host_reserved->vec<int>();
    num_heads = host_reserved_int(0);
    proj_k_size = host_reserved_int(1);
    proj_v_size = host_reserved_int(2);
    proj_o_size = host_reserved_int(3);
    return Status::OK();
  }

 private:
  mutex mu_;
  MhaStateCache mha_state_cache_ TF_GUARDED_BY(mu_);

  Status ExtractBackwardInputs(OpKernelContext* context,
                               const CudnnMhaModelShapes& model_shapes,
                               const Tensor** output,
                               const Tensor** output_backprop,
                               const Tensor** reserve_space) {
    TF_RETURN_IF_ERROR(context->input("output", output));
    TF_RETURN_IF_ERROR(context->input("output_backprop", output_backprop));
    TF_RETURN_IF_ERROR(context->input("reserve_space", reserve_space));

    const TensorShape& output_shape = model_shapes.output_shape;

    if (output_shape != (*output)->shape()) {
      return errors::InvalidArgument(
          "Invalid output shape: ", (*output)->shape().DebugString(), " vs ",
          output_shape.DebugString());
    }

    if (output_shape != (*output_backprop)->shape()) {
      return errors::InvalidArgument("Invalid output_backprop shape: ",
                                     (*output_backprop)->shape().DebugString(),
                                     " ", output_shape.DebugString());
    }
    return Status::OK();
  }

  Status AllocateOutputs(OpKernelContext* context,
                         const CudnnMhaModelShapes& model_shapes,
                         const TensorShape& params_shape,
                         Tensor** input_q_backprop, Tensor** input_k_backprop,
                         Tensor** input_v_backprop, Tensor** params_backprop) {
    const TensorShape& input_q_shape = model_shapes.input_q_shape;
    const TensorShape& input_k_shape = model_shapes.input_k_shape;
    const TensorShape& input_v_shape = model_shapes.input_v_shape;

    TF_RETURN_IF_ERROR(
        context->allocate_output(0, input_q_shape, input_q_backprop));
    TF_RETURN_IF_ERROR(
        context->allocate_output(1, input_k_shape, input_k_backprop));
    TF_RETURN_IF_ERROR(
        context->allocate_output(2, input_v_shape, input_v_backprop));
    TF_RETURN_IF_ERROR(
        context->allocate_output(3, params_shape, params_backprop));
    return Status::OK();
  }
};

#define REGISTER_GPU(T)                                    \
  REGISTER_KERNEL_BUILDER(Name("CudnnMHABackprop")         \
                              .Device(DEVICE_GPU)          \
                              .HostMemory("host_reserved") \
                              .TypeConstraint<T>("T"),     \
                          CudnnMHABackwardOp<GPUDevice, T>);

TF_CALL_half(REGISTER_GPU);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
