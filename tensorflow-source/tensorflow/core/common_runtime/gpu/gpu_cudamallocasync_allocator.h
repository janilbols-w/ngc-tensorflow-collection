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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_

#include <memory>

#ifdef GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#endif  // GOOGLE_CUDA

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// An allocator that wraps cudaMallocAsync. It has fewer fragmentation
// issues then the BFC memory allocator.  The compute-sanitizer tool
// helps to detect OOB memory errors when using cudaMallocAsync. Use
// the environment variable `TF_GPU_ALLOCATOR=cuda_malloc_async` to
// enable it.
//
// It needs CUDA 11.2+. When using a container, this only needs the
// container driver to be 11.2. It has a WAR again a driver bug in
// multi-GPU setup with CUDA 11.2. The WAR creates an extra context on
// GPU 0.
//
// We configure cudaMallocAsync to grow when more memory is needed
// instead of preallocating everything up front and to keep a local
// pool up to pool_size bytes that is never released to other processes.
// So no other process will "steal" the GPU memory already used by the
// current process. This is to speed up execution and prevent crashes
// of long-running jobs. Use `reserve_memory=true` if you want to
// preallocate the full pool_size. You can also use the environment
// variable `TF_CUDA_MALLOC_ASYNC_PREALLOC=nb_bytes` to preallocate
// that amount of memory. `TF_CUDA_MALLOC_ASYNC_PREALLOC=-1` is a
// special value that preallocate all what the BFC memory allocator
// would have allocated. This is useful when benchmarking as it doesn't
// change when driver allocations are done.
//
// Here, the pool_size isn't the absolute max as for [Gpu]BFCAllocator.
// The pool can grow above that up to the total GPU memory.  But the
// driver can return the excess memory to other processes.
class GpuCudaMallocAsyncAllocator : public Allocator {
 public:
  explicit GpuCudaMallocAsyncAllocator(PlatformGpuId platform_gpu_id,
                                       size_t pool_size,
                                       bool reserve_memory = false,
                                       bool compute_stats = false);
  ~GpuCudaMallocAsyncAllocator() override;
  string Name() override { return name_; }
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  bool TracksAllocationSizes() const override;

  size_t RequestedSize(const void* ptr) const override;

  size_t AllocatedSize(const void* ptr) const override;

  absl::optional<AllocatorStats> GetStats() override;

  void ClearStats() override;

 private:
  se::StreamExecutor* stream_exec_;  // Not owned.

  // cudaMallocAsync is stream aware. But TF StreamExecutor use only 1
  // compute stream and already synchronize with the h2d, d2h and d2d
  // stream. So we do not need to ask cudaMallocAsync to add extra
  // synchronization.
  cudaStream_t cuda_stream_;
  string name_;
  //Not owned. The default pool of the associated GPU.
  //If null, then the instanciation failed and the first allocation
  //will return an error.
#if CUDA_VERSION >= 11020
  cudaMemPool_t pool_;
  static std::vector<cudaMemPool_t*> all_pools_;
  static std::vector<PlatformGpuId> all_ids_;
#else
  char* pool_; // Dummy variable to allow to compile
#endif

  TF_DISALLOW_COPY_AND_ASSIGN(GpuCudaMallocAsyncAllocator);

  // Stats.
  // Structures mutable after construction
  mutable mutex lock_;
  AllocatorStats stats_ GUARDED_BY(lock_);
  absl::flat_hash_map<const void*, size_t> size_map_ GUARDED_BY(lock_);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GPU_GPU_CUDAMALLOCASYNC_ALLOCATOR_H_