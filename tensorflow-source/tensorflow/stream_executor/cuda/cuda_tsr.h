/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TSR_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TSR_H_

#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/tsr.h"
#if CUDA_CUTENSOR
#include "third_party/gpus/cuda/include/cutensor.h"
#endif // CUDA_CUTENSOR

namespace stream_executor {
namespace gpu {

// Opaque and unique identifier for the cuTSR plugin.

extern const PluginId kCuTsrPlugin;

template <typename U>
struct CuTensorTypeTraits;

template <>
struct CuTensorTypeTraits<double> {
  static constexpr cudaDataType_t cudaType = CUDA_R_64F;
  static constexpr cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_64F;
  typedef double ScalarType;
};

template <>
struct CuTensorTypeTraits<float> {
  static constexpr cudaDataType_t cudaType = CUDA_R_32F;
  static constexpr cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_32F;
  typedef float ScalarType;
};

template <>
struct CuTensorTypeTraits<Eigen::half> {
  static constexpr cudaDataType_t cudaType = CUDA_R_16F;
  static constexpr cutensorComputeType_t cutensorType = CUTENSOR_R_MIN_16F;
  typedef float ScalarType;
};

template <>
struct CuTensorTypeTraits<std::complex<float>> {
  static constexpr cudaDataType_t cudaType = CUDA_C_32F;
  static constexpr cutensorComputeType_t cutensorType = CUTENSOR_C_MIN_32F;
  typedef std::complex<float> ScalarType;
};

template <>
struct CuTensorTypeTraits<std::complex<double>> {
  static constexpr cudaDataType_t cudaType = CUDA_C_64F;
  static constexpr cutensorComputeType_t cutensorType = CUTENSOR_C_MIN_64F;
  typedef std::complex<double> ScalarType;
};

constexpr int kMaxNumModes_ = 12;
static const std::vector<int> emptyVec;

// cuTensor-library based Tensor support. For details on overridden interface
// functions, see tsr.h.
class CUDATsr : public tsr::TsrSupport {
 public:
  explicit CUDATsr(GpuExecutor* parent) : parent_(parent) {}

  port::Status Init();

  ~CUDATsr() override {}

  TENSORFLOW_STREAM_EXECUTOR_GPU_TSR_SUPPORT_OVERRIDES
 private:
  GpuExecutor* parent_;  // Parent executor object. Not owned.

  // Provides access to the cuTENSOR handle.
  std::unique_ptr<class CuTensorAccess> cutensor_;
  bool isInitialized_ = false;

  tsr::DataType ComputeType_;
  uint32_t numModesA_;
  uint32_t numModesB_;
  uint32_t numModesC_;

  std::array<int, kMaxNumModes_> modesA_;
  std::array<int, kMaxNumModes_> modesB_;
  std::array<int, kMaxNumModes_> modesC_;
  std::array<int64_t, kMaxNumModes_> extentA_;
  std::array<int64_t, kMaxNumModes_> extentB_;
  std::array<int64_t, kMaxNumModes_> extentC_;

  // Encodes the strucutre of the contraction.
  cutensorContractionDescriptor_t descriptor_contraction_;
  // Encodes the execution plan.
  cutensorContractionPlan_t plan_;
  // Limits the search space (of viable candidates/implementations).
  cutensorContractionFind_t find_;

  cutensorTensorDescriptor_t descA_;
  cutensorTensorDescriptor_t descB_;
  cutensorTensorDescriptor_t descC_;

  size_t my_workspace_size = 0;

  bool InitializeModesInternal(Stream* stream, tsr::DataType type,
                               const std::string& equation,
                               const std::vector<int>& A_shape,
                               const std::vector<int>& B_shape = emptyVec);

  port::Status SetTensorContents(const std::string& equation,
                                 const std::vector<int>& A_shape,
                                 const std::vector<int>& B_shape = emptyVec);

  bool isInitialized() const { return isInitialized_; }

  std::vector<int64_t> getOutputShape() const {
    if (!isInitialized_) return {};
    std::vector<int64_t> extentC(numModesC_);
    for (int i = 0; i < numModesC_; ++i) {
      extentC[i] = extentC_.at(numModesC_ - i - 1);
    }
    return extentC;
  }

  tsr::DataType getComputeType() const { return ComputeType_; }

  template <typename T>
  bool DoCuTensorContractionInternal(Stream* stream, const void* A_raw,
                                     const void* B_raw, void* C_raw,
                                     void* work_raw);

  SE_DISALLOW_COPY_AND_ASSIGN(CUDATsr);
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_TSR_H_
