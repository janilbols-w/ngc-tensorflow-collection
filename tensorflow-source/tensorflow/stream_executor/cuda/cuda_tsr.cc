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

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/stream_executor/cuda/cuda_activation.h"
#include "tensorflow/stream_executor/cuda/cuda_gpu_executor.h"
#include "tensorflow/stream_executor/cuda/cuda_helpers.h"
#include "tensorflow/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/stream_executor/cuda/cuda_stream.h"
#include "tensorflow/stream_executor/cuda/cuda_tsr.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/lib/env.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"
#include "tensorflow/stream_executor/plugin_registry.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

#if CUDA_CUTENSOR
namespace stream_executor {
namespace gpu {

PLUGIN_REGISTRY_DEFINE_PLUGIN_ID(kCuTsrPlugin);

namespace {

#define CUTENSOR_VERSION \
  (CUTENSOR_MAJOR * 10000 + CUTENSOR_MINOR * 100 + CUTENSOR_PATCH)
static_assert(CUTENSOR_VERSION >= 10300,
              "cuTensor needs to be version 10300 or higher, otherwise reduction"
              " operations may suffer from slowing down.");

#define HANDLE_ERROR(x)                           \
  {                                               \
    const auto err = x;                           \
    if (err == CUTENSOR_STATUS_NOT_SUPPORTED) {   \
      return false;                               \
    }                                             \
    if (err != CUTENSOR_STATUS_SUCCESS) {         \
      printf(                                     \
          "cutensor_python: Error %s in  \
    line %d\n",                                   \
          cutensorGetErrorString(err), __LINE__); \
      return false;                               \
    }                                             \
  }

std::string ToString(cutensorStatus_t status) {
  switch (status) {
    case CUTENSOR_STATUS_SUCCESS:
      return "CUTENSOR_STATUS_SUCCESS";
    case CUTENSOR_STATUS_INVALID_VALUE:
      return "CUTENSOR_STATUS_INVALID_VALUE";
    case CUTENSOR_STATUS_IO_ERROR:
      return "CUTENSOR_STATUS_IO_ERROR";
    case CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
      return "CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    case CUTENSOR_STATUS_NOT_SUPPORTED:
      return "CUTENSOR_STATUS_NOT_SUPPORTED";
    default:
      return absl::StrCat(
          "<unknown cutensor status: ", static_cast<int>(status), ">");
  }
}

static inline void GetCuTsrComputeType(tsr::DataType type,
                                       cudaDataType_t* tensorType,
                                       cutensorComputeType_t* computeType) {
  switch (type) {
    case tsr::DataType::kFloat:
      *tensorType = CuTensorTypeTraits<float>::cudaType;
      *computeType = CuTensorTypeTraits<float>::cutensorType;
      break;
    case tsr::DataType::kDouble:
      *tensorType = CuTensorTypeTraits<double>::cudaType;
      *computeType = CuTensorTypeTraits<double>::cutensorType;
      break;
    case tsr::DataType::kHalf:
      *tensorType = CuTensorTypeTraits<Eigen::half>::cudaType;
      *computeType = CuTensorTypeTraits<Eigen::half>::cutensorType;
      break;
    case tsr::DataType::kComplexFloat:
      *tensorType = CuTensorTypeTraits<std::complex<float>>::cudaType;
      *computeType = CuTensorTypeTraits<std::complex<float>>::cutensorType;
      break;
    case tsr::DataType::kComplexDouble:
      *tensorType = CuTensorTypeTraits<std::complex<double>>::cudaType;
      *computeType = CuTensorTypeTraits<std::complex<double>>::cutensorType;
      break;
    default:
      LOG(FATAL) << "Non-supported types for CuTsrCompute.";
  }
}

class CuTensorHandle {
 public:
  // Takes ownership of the executor context and the lock to access cuTENSOR
  // using handle.
  CuTensorHandle(gpu::ScopedActivateExecutorContext context,
                 std::unique_ptr<absl::MutexLock> lock,
                 cutensorHandle_t* handle, void* cutensor_cacheline)
      : context_(std::move(context)),
        lock_(std::move(lock)),
        handle_(handle),
        cutensor_cachelines_(cutensor_cacheline) {}

  // Returns cuTENSOR handle. To be passed directly to cuTENSOR APIs, don't keep
  // a copy.
  cutensorHandle_t* handle() const { return handle_; }
  void* cacheline() const { return cutensor_cachelines_; }

 private:
  gpu::ScopedActivateExecutorContext context_;
  std::unique_ptr<absl::MutexLock> lock_;
  cutensorHandle_t* handle_;  // Not owned.
  void* cutensor_cachelines_ = nullptr;
};
}  // namespace

// Guards the enqueueing of cuTENSOR operations via the handle_ below.
static absl::Mutex mutex_;

class CuTensorAccess {
 public:
  // Takes ownership of the handle.
  explicit CuTensorAccess(std::unique_ptr<cutensorHandle_t> handle,
                          std::unique_ptr<char[]> cachelines)
      : handle_(std::move(handle)), cachelines_(std::move(cachelines)) {}

  ~CuTensorAccess() {
    absl::MutexLock lock(&mutex_);
    DetachCuTensorCacheline();
  }

  void DetachCuTensorCacheline() {
    if (cachelines_.get() != nullptr) {
      cutensorStatus_t err;
      const char* cacheFilename = getenv("TF_CUTENSOR_CACHEFILE");
      if (cacheFilename != nullptr) {
        err = cutensorHandleWriteCacheToFile(handle_.get(), cacheFilename);
        CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);
      }
      err = cutensorHandleDetachPlanCachelines(handle_.get());
      // free(cachelines_);
      CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);
    }
  }

  CuTensorHandle GetHandle(GpuExecutor* executor, Stream* stream) {
    auto lock = absl::make_unique<absl::MutexLock>(&mutex_);
    mutex_.AssertHeld();
    gpu::ScopedActivateExecutorContext context(executor);
    return CuTensorHandle(std::move(context), std::move(lock), handle_.get(),
                          (void*)cachelines_.get());
  }

 private:
  // cuTENSOR library handle.
  std::unique_ptr<cutensorHandle_t> handle_ TF_GUARDED_BY(mutex_);  // Owned.
  std::unique_ptr<char[]> cachelines_ TF_GUARDED_BY(mutex_);

  TF_DISALLOW_COPY_AND_ASSIGN(CuTensorAccess);
};

port::Status CUDATsr::Init() {
  ScopedActivateExecutorContext context(parent_);
  std::unique_ptr<cutensorHandle_t> cutensor_handle(new cutensorHandle_t());

  if (isInitialized()) {
    LOG(INFO) << "Try to repeatedly initialize cuTensor handle.";
    return port::InternalError("Try to repeatedly initialize cuTensor handle.");
  }

  const auto ret = cutensorInit(cutensor_handle.get());

  if (ret == CUTENSOR_STATUS_SUCCESS) {
    constexpr int32_t kNumCacheLines = 1024;
    size_t sizeCache = kNumCacheLines * sizeof(cutensorPlanCacheline_t);
    std::unique_ptr<char[]> cutensor_cachelines(new char[sizeCache]);
    CHECK_NE(cutensor_cachelines.get(), nullptr);

    cutensorStatus_t err = cutensorHandleAttachPlanCachelines(
        cutensor_handle.get(),
        (cutensorPlanCacheline_t*)cutensor_cachelines.get(), kNumCacheLines);
    CHECK_EQ(err, CUTENSOR_STATUS_SUCCESS) << cutensorGetErrorString(err);

    const char* cacheFilename = getenv("TF_CUTENSOR_CACHEFILE");
    if (cacheFilename != nullptr) {
      uint32_t numCachelinesRead = 0;
      cutensorStatus_t status = cutensorHandleReadCacheFromFile(
          cutensor_handle.get(), cacheFilename, &numCachelinesRead);
      if (status == CUTENSOR_STATUS_IO_ERROR) {
        printf("File (%s) doesn't seem to exist.\n", cacheFilename);
      } else if (status == CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE) {
        printf(
            "Cannot read cache: Please attach at least %d "
            "cachelines to the handle.\n",
            numCachelinesRead);
      }
    }

    cutensor_.reset(new CuTensorAccess(std::move(cutensor_handle),
                                       std::move(cutensor_cachelines)));
    LOG(INFO) << "Loaded cuTensor version " << cutensorGetVersion();

    return port::Status::OK();
  }

  return port::InternalError("Failed to create cuTensor handle");
}

port::Status CUDATsr::CutensorPreprocess(
    Stream* stream, std::vector<int64_t>* output_dims, tsr::DataType type,
    const std::string& equation, const std::vector<int>& A_shape,
    const std::vector<int>& B_shape = emptyVec) {
  absl::MutexLock lock(&mutex_);

  if (!InitializeModesInternal(stream, type, equation, A_shape, B_shape)) {
    return tensorflow::errors::InvalidArgument(
        "Invalid equation or tensor shapes.");
  }

  *output_dims = getOutputShape();
  return port::Status::OK();
}

template <typename DType>
bool CUDATsr::DoCuTensorContractionInternal(Stream* stream, const void* A_raw,
                                            const void* B_raw, void* C_raw,
                                            void* work_raw) {
  cutensorHandle_t* cutensor_handle =
      (cutensor_->GetHandle(parent_, stream)).handle();
  void* cacheline = (cutensor_->GetHandle(parent_, stream)).cacheline();

  typename CuTensorTypeTraits<DType>::ScalarType alpha = 1;
  typename CuTensorTypeTraits<DType>::ScalarType beta = 0;

  if (numModesB_ > 0) {
    // Dispatch to contraction.
    if (cacheline != nullptr) {
      HANDLE_ERROR(cutensorInitContractionPlan(cutensor_handle, &plan_,
                                               &descriptor_contraction_, &find_,
                                               my_workspace_size));
    }

    HANDLE_ERROR(cutensorContraction(
        cutensor_handle, &plan_, &alpha, A_raw, B_raw, &beta, C_raw, C_raw,
        work_raw, my_workspace_size, AsGpuStreamValue(stream)));
  } else {
    // Dispatch to reduction.
    cutensorComputeType_t computeType = CuTensorTypeTraits<DType>::cutensorType;

    HANDLE_ERROR(cutensorReduction(
        cutensor_handle, &alpha, A_raw, &descA_, modesA_.data(), &beta, A_raw,
        &descC_,
        modesC_.data(),  // beta == 0 => will not be used
        C_raw, &descC_, modesC_.data(), CUTENSOR_OP_ADD, computeType, work_raw,
        my_workspace_size, AsGpuStreamValue(stream)));
  }
  return true;
}

port::Status CUDATsr::SetTensorContents(const std::string& equation,
                                        const std::vector<int>& A_shape,
                                        const std::vector<int>& B_shape) {
  numModesA_ = A_shape.size();
  numModesB_ = B_shape.size();
  numModesC_ = 0;

  const auto arrow_pos = equation.find("->");
  const auto comma_pos = equation.find(",");
  const auto dots = equation.find("...");
  const bool isBroadcast = (dots != std::string::npos);
  const bool isImplicit = (arrow_pos == std::string::npos);

  if (isBroadcast) {
    // Broadcasting is not directly support by cuTENSOR.
    return port::InternalError("Broadcasting is not directly supported.");
  }
  const bool usesB = (comma_pos != std::string::npos);
  if (!usesB) {
    numModesB_ = 0;
  }

  size_t a_start = 0;
  size_t a_end =
      isImplicit
          ? ((comma_pos == std::string::npos) ? equation.size() : comma_pos)
          : ((comma_pos == std::string::npos) ? arrow_pos : comma_pos);
  size_t b_start = usesB ? comma_pos + 1 : 0;
  size_t b_end = usesB ? (isImplicit ? equation.size() : arrow_pos) : 0;
  size_t c_start = isImplicit ? equation.size() : arrow_pos + 2;
  size_t c_end = equation.size();

  char modeA[kMaxNumModes_ + 2];
  uint32_t numModesA = 0;
  for (int i = a_start; i < a_end && numModesA < kMaxNumModes_ + 2; ++i) {
    if (equation.at(i) != ' ') {
      // Skip spaces.
      modeA[numModesA++] = equation.at(i);
    }
  }

  char modeB[kMaxNumModes_ + 2];
  uint32_t numModesB = 0;
  for (int i = b_start; i < b_end && numModesB < kMaxNumModes_ + 2; ++i) {
    if (equation.at(i) != ' ') {
      modeB[numModesB++] = equation.at(i);
    }
  }

  char modeC[kMaxNumModes_ + 2];
  uint32_t numModesC = 0;
  for (int i = c_start; i < c_end && numModesC < kMaxNumModes_ + 2; ++i) {
    if (equation.at(i) != ' ') {
      modeC[numModesC++] = equation.at(i);
    }
  }

  // Substring size and shape don't match.
  if (numModesA != numModesA_) {
        return tensorflow::errors::InvalidArgument(
            "Substring size and shape don't match. Substrig size is ",
            numModesA, ", but shape of tensor A is ", numModesA_);
      }

  if (numModesB != numModesB_) {
    return tensorflow::errors::InvalidArgument(
        "Substring size and shape don't match. Substrig size is ", numModesB,
        ", but shape of tensor B is ", numModesB_);
  }

  // Too many modes.
  if (numModesA_ > kMaxNumModes_) {
    return tensorflow::errors::InvalidArgument("Too many modes. At most ",
                                               kMaxNumModes_, "modes, but ",
                                               numModesA_, " provided for A.");
  }

  if (numModesB_ > kMaxNumModes_) {
    return tensorflow::errors::InvalidArgument("Too many modes. At most ",
                                               kMaxNumModes_, "modes, but ",
                                               numModesB_, " provided for B.");
  }

  // Copy all modes from modeA to modeC if they don't appear in modeB.
  auto copyModesIf = [](const char* modeA, uint32_t numModesA,
                        const char* modeB, uint32_t numModesB, char* modeC,
                        uint32_t& numModesC) {
    for (uint32_t i = 0; i < numModesA; i++) {
      auto mode = modeA[i];
      bool found = false;
      for (uint32_t j = 0; j < numModesB; ++j) {
        if (mode == modeB[j]) {
          found = true;
          break;
        }
      }

      if (!found) {
        // Non-contracted mode.
        modeC[numModesC++] = mode;
        if (numModesC > kMaxNumModes_) {
          // Too many modes.
          return false;
        }
      }
    }
    return true;
  };

  std::array<char, kMaxNumModes_ + 1> implicitModeC;
  char* redirectModeC;
  if (isImplicit) {
    // Copy all non-contracted modes from A over to C.
    if (copyModesIf(modeA, numModesA_, modeB, numModesB_, implicitModeC.data(),
                    numModesC_) == false) {
      return tensorflow::errors::InvalidArgument("Too many modes.");
    }
    // Copy all non-contracted modes from B over to C.
    if (copyModesIf(modeB, numModesB_, modeA, numModesA_, implicitModeC.data(),
                    numModesC_) == false) {
      return tensorflow::errors::InvalidArgument("Too many modes.");
    }
    // Modes are sorted w.r.t. lexical order.
    std::sort(implicitModeC.begin(),
              std::next(implicitModeC.begin(), numModesC_));
    implicitModeC[numModesC_] = '\0';
    redirectModeC = implicitModeC.data();
  } else {
    redirectModeC = modeC;
    numModesC_ = numModesC;
  }

  for (uint32_t i = 0; i < numModesA_; i++) {
    modesA_[i] = modeA[numModesA_ - i - 1];
    extentA_[i] = A_shape[numModesA_ - i - 1];
  }

  for (uint32_t i = 0; i < numModesB_; i++) {
    modesB_[i] = modeB[numModesB_ - i - 1];
    extentB_[i] = B_shape[numModesB_ - i - 1];
  }

  for (uint32_t i = 0; i < numModesC_; i++) {
    const auto mode = redirectModeC[numModesC_ - i - 1];
    modesC_[i] = mode;
    bool found = false;
    for (uint32_t j = 0; j < numModesA_; ++j) {
      if (modesA_[j] == mode) {
        extentC_[i] = extentA_[j];
        found = true;
        break;
      }
    }
    for (uint32_t j = 0; !found && j < numModesB_; ++j) {
      if (modesB_[j] == mode) {
        extentC_[i] = extentB_[j];
        break;
      }
    }
  }

  isInitialized_ = true;
  return port::Status::OK();
}

bool CUDATsr::PrepareContraction(Stream* stream, size_t* workspace,
                                 const void* A_raw, const void* B_raw,
                                 void* C_raw) {
  cutensorHandle_t* cutensor_handle =
      (cutensor_->GetHandle(parent_, stream)).handle();
  void* cacheline = (cutensor_->GetHandle(parent_, stream)).cacheline();

  cudaDataType_t tensorType;
  cutensorComputeType_t computeType;

  GetCuTsrComputeType(getComputeType(), &tensorType, &computeType);

  HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle, &descA_,
                                            numModesA_, extentA_.data(), NULL,
                                            tensorType, CUTENSOR_OP_IDENTITY));

  HANDLE_ERROR(cutensorInitTensorDescriptor(cutensor_handle, &descC_,
                                            numModesC_, extentC_.data(), NULL,
                                            tensorType, CUTENSOR_OP_IDENTITY));

  // Retrieve the memory alignment for each tensor.
  uint32_t alignmentRequirementA;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle, A_raw, &descA_,
                                               &alignmentRequirementA));

  uint32_t alignmentRequirementC;
  HANDLE_ERROR(cutensorGetAlignmentRequirement(cutensor_handle, C_raw, &descC_,
                                               &alignmentRequirementC));

  if (numModesB_ > 0) {
    // Dispatch to contraction.
    HANDLE_ERROR(cutensorInitTensorDescriptor(
        cutensor_handle, &descB_, numModesB_, extentB_.data(), NULL, tensorType,
        CUTENSOR_OP_IDENTITY));

    uint32_t alignmentRequirementB;
    HANDLE_ERROR(cutensorGetAlignmentRequirement(
        cutensor_handle, B_raw, &descB_, &alignmentRequirementB));

    HANDLE_ERROR(cutensorInitContractionDescriptor(
        cutensor_handle, &descriptor_contraction_, &descA_, modesA_.data(),
        alignmentRequirementA, &descB_, modesB_.data(), alignmentRequirementB,
        &descC_, modesC_.data(), alignmentRequirementC, &descC_, modesC_.data(),
        alignmentRequirementC, computeType));

    cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    HANDLE_ERROR(cutensorInitContractionFind(cutensor_handle, &find_, algo));

    if (cacheline != nullptr) {
      const cutensorAutotuneMode_t autotuneMode = CUTENSOR_AUTOTUNE_INCREMENTAL;
      HANDLE_ERROR(cutensorContractionFindSetAttribute(
          cutensor_handle, &find_, CUTENSOR_CONTRACTION_FIND_AUTOTUNE_MODE,
          &autotuneMode, sizeof(cutensorAutotuneMode_t)));

      const uint32_t incCount = 5;
      HANDLE_ERROR(cutensorContractionFindSetAttribute(
          cutensor_handle, &find_, CUTENSOR_CONTRACTION_FIND_INCREMENTAL_COUNT,
          &incCount, sizeof(uint32_t)));
    }
#if CUTENSOR_VERSION < 10500
    HANDLE_ERROR(cutensorContractionGetWorkspace(
        cutensor_handle, &descriptor_contraction_, &find_,
        CUTENSOR_WORKSPACE_RECOMMENDED, &my_workspace_size));
#else
    HANDLE_ERROR(cutensorContractionGetWorkspaceSize(
        cutensor_handle, &descriptor_contraction_, &find_,
        CUTENSOR_WORKSPACE_RECOMMENDED, &my_workspace_size));
#endif

    if (cacheline == nullptr) {
      HANDLE_ERROR(cutensorInitContractionPlan(cutensor_handle, &plan_,
                                               &descriptor_contraction_, &find_,
                                               my_workspace_size));
    }
  }
  *workspace = my_workspace_size;
  return true;
}

bool CUDATsr::InitializeModesInternal(Stream* stream, tsr::DataType type,
                                      const std::string& equation,
                                      const std::vector<int>& A_shape,
                                      const std::vector<int>& B_shape) {
  ComputeType_ = type;
  port::Status status = SetTensorContents(equation, A_shape, B_shape);
  if (!status.ok()) {
    LOG(ERROR) << "Unable to set tensor contents: " << status.error_message();
    return true;
  }
  return true;
}

bool CUDATsr::DoTsrContraction(Stream* stream, const void* A_raw,
                               const void* B_raw, void* C_raw, void* work_raw) {
  if (getComputeType() == tsr::DataType::kDouble) {
    return DoCuTensorContractionInternal<double>(stream, A_raw, B_raw, C_raw,
                                                 work_raw);
  } else if (getComputeType() == tsr::DataType::kFloat) {
    return DoCuTensorContractionInternal<float>(stream, A_raw, B_raw, C_raw,
                                                work_raw);
  } else if (getComputeType() == tsr::DataType::kHalf) {
    return DoCuTensorContractionInternal<Eigen::half>(stream, A_raw, B_raw,
                                                      C_raw, work_raw);
  } else if (getComputeType() == tsr::DataType::kComplexFloat) {
    return DoCuTensorContractionInternal<std::complex<float>>(
        stream, A_raw, B_raw, C_raw, work_raw);
  } else if (getComputeType() == tsr::DataType::kComplexDouble) {
    return DoCuTensorContractionInternal<std::complex<double>>(
        stream, A_raw, B_raw, C_raw, work_raw);
  }
  return false;
}
}  // namespace gpu

void initialize_cutsr() {
  port::Status status =
      PluginRegistry::Instance()->RegisterFactory<PluginRegistry::TsrFactory>(
          cuda::kCudaPlatformId, gpu::kCuTsrPlugin, "cuTSR",
          [](internal::StreamExecutorInterface* parent) -> tsr::TsrSupport* {
            gpu::GpuExecutor* cuda_executor =
                dynamic_cast<gpu::GpuExecutor*>(parent);
            if (cuda_executor == nullptr) {
              LOG(ERROR) << "Attempting to initialize an instance of the cuTsr "
                         << "support library with a non-CUDA StreamExecutor";
              return nullptr;
            }

            gpu::CUDATsr* tsr = new gpu::CUDATsr(cuda_executor);
            if (!tsr->Init().ok()) {
              // Note: Init() will log a more specific error.
              delete tsr;
              return nullptr;
            }
            return tsr;
          });
  if (!status.ok()) {
    LOG(ERROR) << "Unable to register cuTSR factory: "
               << status.error_message();
  }

  PluginRegistry::Instance()->SetDefaultFactory(
      cuda::kCudaPlatformId, PluginKind::kTsr, gpu::kCuTsrPlugin);
}

}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(register_cutsr,
                            { stream_executor::initialize_cutsr(); });
#endif
