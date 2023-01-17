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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/linalg/einsum_op.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/util/einsum_op_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if CUDA_CUTENSOR
#include "third_party/gpus/cuda/include/cutensor.h"
#define CUTENSOR_VERSION                                                      \
  (CUTENSOR_MAJOR * 10000 + CUTENSOR_MINOR * 100 + CUTENSOR_PATCH)
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/reduction_ops_common_gpu.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

static bool CuTensorReadyAndRequired() {
#if CUDA_CUTENSOR
  static bool load_cutensor_success = [] {
    if (!EnableCuTensorEinsum()) {
      return false;
    }
    return se::internal::DsoLoader::MaybeTryDlopenCUTENSORLibrary().ok();
  }();
  return load_cutensor_success;
#endif
   return false;
}

namespace functor {
#if CUDA_CUTENSOR
template <typename T>
struct EinsumCutensorGpuFunctor;
#endif
template <typename Device, typename T>
struct EinsumMaybeCutensorFunctor;

}  // namespace functor

struct EinsumHelper {
  // Each dimension is categorized into exactly one of five types based on
  // whether its corresponding label is present in the input and/or the output
  // subscripts.
  
  typedef struct {
    string equation;
    OperandLabels input_labels;
    Labels output_labels;
    std::vector<EinsumDimensionType> label_types;
    OperandLabelCounts input_label_counts;
    LabelCounts output_label_counts;
    gtl::InlinedVector<bool, 2> input_has_ellipsis;
    bool output_has_ellipsis = false;
    bool has_single_label = false;
    bool support_cutensor = true;
  } EinsumOpInputFeatures;

  // Parses and validates the equation and the input shapes. Single character
  // labels are integerized and we populate input and output label subscripts
  // and corresponding counts. Also create the mapping from (named) labels to
  // their EinsumDimensionType.
  static Status ParseEquationWithCuTensorSupport(
      const string& equation, OperandLabels* input_labels,
      Labels* output_labels, std::vector<EinsumDimensionType>* label_types,
      OperandLabelCounts* input_label_counts, LabelCounts* output_label_counts,
      gtl::InlinedVector<bool, 2>* input_has_ellipsis,
      bool* output_has_ellipsis, bool* has_single_labels,
      bool* support_cutensor) {
    gtl::InlinedVector<string, 2> input_str;
    string output_str;
    bool broadcast_required = false;
    TF_RETURN_IF_ERROR(ValidateEinsumEquation(equation, &input_str, &output_str));

    // Temporary map from single character labels to (consecutive) integer
    // labels.
    absl::flat_hash_map<char, int> label_mapping;
    int num_inputs = input_str.size();
    input_labels->resize(num_inputs);

    // Map from single characters to integer labels.
    for (int i = 0; i < num_inputs; ++i) {
      MapToLabels(input_str[i], &input_labels->at(i), &label_mapping);
    }
    MapToLabels(output_str, output_labels, &label_mapping);

    // Compute counts for input and output labels.
    int num_labels = label_mapping.size();
    input_label_counts->resize(num_inputs);
    input_has_ellipsis->resize(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      input_label_counts->at(i).resize(num_labels);
      input_has_ellipsis->at(i) = false;
      for (const int label : input_labels->at(i)) {
        if (label != kEllipsisLabel)
          input_label_counts->at(i)[label] += 1;
        else {
          input_has_ellipsis->at(i) = true;
          broadcast_required = true;
        }
      }
    }
    output_label_counts->resize(num_labels);
    *output_has_ellipsis = false;
    for (const int label : *output_labels) {
      if (label != kEllipsisLabel)
        output_label_counts->at(label) += 1;
      else {
        *output_has_ellipsis = true;
        if (!broadcast_required) broadcast_required = true;
      }
    }

    // Map each label to a unique DimensionType.
    label_types->resize(num_labels);
    for (int label = 0; label < num_labels; ++label) {
      if (label == kEllipsisLabel) continue;
      bool removed = (*output_label_counts)[label] == 0;
      bool unique = num_inputs == 1 || (*input_label_counts)[0][label] == 0 ||
                    (*input_label_counts)[1][label] == 0;
      (*label_types)[label] = GetDimensionType(removed, unique);
    }

    bool single_labels = false;
    bool repeated_labels = false;

    // Get repeated_labels.
    for (int i = 0; i < num_inputs; ++i) {
      if (!absl::c_all_of((*input_label_counts)[i],
                          [](int c) { return c <= 1; })) {
        repeated_labels = true;
        break;
      }
    }

    if (!repeated_labels) {
      repeated_labels =
          !absl::c_all_of(*output_label_counts, [](int c) { return c <= 1; });
    }

    // Single label is the label appears just once in one of the operand.
    if (!broadcast_required) {
      for (auto const& item : label_mapping) {
        int appear_counts = 0;
        auto label = item.second;
        for (int j = 0; j < num_inputs; ++j) {
          if (input_label_counts->at(j)[label]) appear_counts += 1;
        }
        if (appear_counts == 0) {
          single_labels = true;
        } else if (appear_counts < num_inputs) {
          single_labels = (*output_label_counts)[label] == 0;
        }

        if (single_labels) break;
      }
    }

    if (has_single_labels != nullptr) {
      *has_single_labels = single_labels;
    }

    bool trivial_reduction =
        equation.compare("->") == 0 || equation.compare(",->") == 0;
    *support_cutensor = !trivial_reduction &&
                        !repeated_labels && !broadcast_required;
    return Status::OK();
  }

  // Insert new (unnamed) broadcasting labels at the location of ellipsis.
  static void InsertBroadcastLabels(int num_bcast_dims, int num_named_labels,
                                    int ellipsis_axis, Labels* labels,
                                    LabelCounts* label_counts) {
    labels->erase(labels->begin() + ellipsis_axis);
    labels->insert(labels->begin() + ellipsis_axis, num_bcast_dims, 0);
    std::iota(labels->begin() + ellipsis_axis,
              labels->begin() + ellipsis_axis + num_bcast_dims,
              num_named_labels);
    // Increment label counts. Since these are new labels, the count is set
    // to 1.
    label_counts->resize(num_named_labels + num_bcast_dims, 1);
  }

  // Record and validate the label to dimension mapping. Must be a named
  // (non-broadcasting) label as broadcasting labels don't have a fixed
  // dimension.
  static Status RecordLabelToDimension(const int label, const int axis,
                                       const Tensor& input,
                                       LabelToDimSizes* label_to_dim_sizes) {
    const int64_t input_dim = input.dim_size(axis);
    // We know that label_to_dim_sizes has the size to accommodate named labels.
    if (label_to_dim_sizes->at(label) != 0 &&
        label_to_dim_sizes->at(label) != input_dim) {
      return errors::InvalidArgument(
          "Expected dimension ", label_to_dim_sizes->at(label), " at axis ",
          axis, " of the input shaped ", input.shape().DebugString(),
          " but got dimension ", input_dim);
    }
    (*label_to_dim_sizes)[label] = input_dim;
    return Status::OK();
  }

  // Validate input dimensions and populate unnamed labels and their label
  // counts.
  static Status ProcessDimensions(
      const OpInputList& inputs,
      const gtl::InlinedVector<bool, 2>& input_has_ellipsis,
      const bool output_has_ellipsis, OperandLabels* input_labels,
      Labels* output_labels, std::vector<EinsumDimensionType>* label_types,
      OperandLabelCounts* input_label_counts, LabelCounts* output_label_counts,
      LabelToDimSizes* label_to_dim_sizes) {
    if (inputs.size() != input_labels->size()) {
      return errors::InvalidArgument("Expected ", input_labels->size(),
                                     " inputs but got: ", inputs.size());
    }
    const int num_inputs = inputs.size();

    // We infer the number of broadcasting dimensions by taking the maximum rank
    // among the broadcasting subshapes of the input.
    int max_bcast_dims = 0;
    const int num_named_labels = label_types->size();
    label_to_dim_sizes->resize(num_named_labels);
    for (int i = 0; i < num_inputs; ++i) {
      Labels* labels = &(*input_labels)[i];

      if (!input_has_ellipsis[i]) {
        if (inputs[i].dims() != labels->size()) {
          return errors::InvalidArgument("Expected input ", i, " to have rank ",
                                         labels->size(),
                                         " but got: ", inputs[i].dims());
        }
        for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
          const int label = (*labels)[label_idx];
          TF_RETURN_IF_ERROR(RecordLabelToDimension(label, label_idx, inputs[i],
                                                    label_to_dim_sizes));
        }
        continue;
      }

      // Input has an ellipsis.
      if (inputs[i].dims() + 1 < labels->size()) {
        return errors::InvalidArgument(
            "Expected input ", i, " to have rank at least ", labels->size() - 1,
            " but got: ", inputs[i].dims());
      }
      int ellipsis_axis = -1;
      const int num_bcast_dims = inputs[i].dims() - labels->size() + 1;
      for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
        const int label = (*labels)[label_idx];
        if (label == kEllipsisLabel) {
          ellipsis_axis = label_idx;
          continue;
        }
        // Current label is not an ellipsis.
        const int axis =
            label_idx + (ellipsis_axis == -1 ? 0 : num_bcast_dims - 1);
        TF_RETURN_IF_ERROR(
            RecordLabelToDimension(label, axis, inputs[i], label_to_dim_sizes));
      }
      // Found an ellipsis. Replace 'kEllipsisLabel' with broadcasting
      // dimensions.
      if (ellipsis_axis != -1) {
        InsertBroadcastLabels(num_bcast_dims, num_named_labels, ellipsis_axis,
                              labels, &input_label_counts->at(i));
        max_bcast_dims = std::max(max_bcast_dims, num_bcast_dims);
      }
    }
    if (!absl::c_linear_search(input_has_ellipsis, true) &&
        !output_has_ellipsis) {
      return Status::OK();
    }
    // Insert broadcasting dimensions in the output labels.
    auto it =
        std::find(output_labels->begin(), output_labels->end(), kEllipsisLabel);
    if (it != output_labels->end()) {
      const int ellipsis_axis = it - output_labels->begin();
      InsertBroadcastLabels(max_bcast_dims, num_named_labels, ellipsis_axis,
                            output_labels, output_label_counts);
    } else if (max_bcast_dims > 0) {
      return errors::InvalidArgument(
          "Output contains ", max_bcast_dims,
          " broadcasting dimension(s) but no ellipsis "
          "(...) was found in the output subscripts.");
    }
    // Populate EinsumDimensionType for the new broadcasting labels.
    label_types->resize(num_named_labels + max_bcast_dims,
                        EinsumDimensionType::kBroadcasting);
    return Status::OK();
  }

  // Permutes the labels according to the given permutation.
  static void PermuteLabels(const std::vector<int>& permutation,
                            Labels* labels) {
    Labels permuted_labels(labels->size());
    for (int i = 0; i < labels->size(); ++i) {
      permuted_labels[i] = (*labels)[permutation[i]];
    }
    labels->swap(permuted_labels);
  }

  // Returns a reshaped input Tensor. The underlying buffer is not copied.
  static Status CopyFrom(const Tensor& input, const TensorShape& shape,
                         Tensor* output) {
    if (output->CopyFrom(input, shape)) return Status::OK();
    return errors::Internal(
        "Encountered error while reshaping a Tensor of shape ",
        input.shape().DebugString(), " to shape ", shape.DebugString());
  }

  // Returns whether transposing would be a no-op; whether input has rank < 2 or
  // the permutation is the identity permutation.
  static bool ShouldTranspose(const TensorShape& input_shape,
                              const std::vector<int>& permutation) {
    if (input_shape.dims() < 2) return false;
    for (int i = 0; i < permutation.size(); ++i) {
      if (permutation[i] != i) return true;
    }
    return false;
  }

  // Transpose the input given a permutation. Returns a reference to the input
  // if transposing is not necessary.
  template <typename Device, typename T>
  static Status TransposeOperand(OpKernelContext* ctx, const Tensor& input,
                                 const std::vector<int>& permutation,
                                 Tensor* output) {
    if (!ShouldTranspose(input.shape(), permutation)) {
      return CopyFrom(input, input.shape(), output);
    }
    TensorShape transposed_shape;
    for (int i = 0; i < input.dims(); ++i) {
      transposed_shape.AddDim(input.dim_size(permutation[i]));
    }
    // For empty Tensors, just change the shape. E.g. we may need to transpose
    // from shape [1, 0, 5] to [5, 1, 0].
    if (input.NumElements() == 0) {
      return CopyFrom(input, transposed_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, transposed_shape, output));
    const Device& device = ctx->eigen_device<Device>();
    TF_RETURN_IF_ERROR(DoTranspose(device, input, permutation, output));
    return Status::OK();
  }

  // If there are repeated labels in either the input or output, then this
  // strides the input (e.g. iii->i) or inflates it (e.g. i->iii), respectively.
  template <typename Device, typename T>
  static Status StrideOrInflate(OpKernelContext* ctx, const Tensor& input,
                                const Labels& labels,
                                const LabelCounts& label_counts,
                                const bool should_inflate, Tensor* output) {
    // Return early if there are no repeated indices.
    if (absl::c_all_of(label_counts, [](int c) { return c <= 1; })) {
      return CopyFrom(input, input.shape(), output);
    }
    // We reshape so that each repeated label is compressed to one dimension.
    // E.g. For iiij -> ij, The shape [3, 3, 3, 5] would be compressed to [27,
    // 5]. Striding appropriately (in this case with strides 14 (=1+3+9) and 1)
    // recovers the generalized diagonal of shape [3, 5].
    ShapeVec reshape;
    ShapeVec strides;
    // Strided and inflated shapes correspond to input and output shapes,
    // respectively, should_inflate is true (vice-versa if should_inflate is
    // false). E.g. they are [3, 5] and [3, 3, 3, 5] in the above example.
    ShapeVec strided_shape;
    ShapeVec inflated_shape;
    for (int label : labels) {
      const int count = label_counts[label];
      const int current_axis =
          should_inflate ? strided_shape.size() : inflated_shape.size();
      const int64_t dim = input.dim_size(current_axis);
      strided_shape.push_back(dim);
      inflated_shape.insert(inflated_shape.end(), count, dim);
      const int64_t reshape_dim = MathUtil::IPow(dim, count);
      reshape.push_back(reshape_dim);
      // While taking the d-diagonal in a rank k Tensor, we take d
      // equally-spaced elements including the first and last element. Then, (k
      // - 1) * stride = d^k - 1, or, stride = (d^k - 1)/(d - 1).
      const int64_t stride =
          (dim > 1 && count > 1) ? (reshape_dim - 1) / (dim - 1) : 1;
      strides.push_back(stride);
    }

    TensorShape output_shape =
        TensorShape(should_inflate ? inflated_shape : strided_shape);
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    const Device& device = ctx->eigen_device<Device>();
    switch (reshape.size()) {
#define NDIMS_CASE(N)                                                 \
  case N: {                                                           \
    if (should_inflate) {                                             \
      auto output_map = output->shaped<T, N>(reshape);                \
      auto input_map = input.shaped<T, N>(strided_shape);             \
      functor::InflateFunctor<Device, T, N>()(                        \
          device, input_map, TensorShape(strides).AsEigenDSizes<N>(), \
          output_map);                                                \
    } else {                                                          \
      auto input_map = input.shaped<T, N>(reshape);                   \
      auto output_map = output->shaped<T, N>(strided_shape);          \
      functor::StrideFunctor<Device, T, N>()(                         \
          device, input_map, TensorShape(strides).AsEigenDSizes<N>(), \
          output_map);                                                \
    }                                                                 \
  } break;
      NDIMS_CASE(1);
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      NDIMS_CASE(6);
      default:
        return errors::Unimplemented(
            "Unsupported rank: ", reshape.size(),
            " while handling repeated indices. Up to rank 6 is supported.");
#undef NDIMS_CASE
    }
    return Status::OK();
  }

  // Returns true if the input dimensions are already sorted in the order
  // [batch, contract, free, reduce]. Used to implement an optimization to avoid
  // an extra transpose and instead uses (adj_x and adj_y) in BatchMatMul.
  static bool ShouldSwapFreeAndContract(
      const Labels& labels,
      const std::vector<EinsumDimensionType>& label_types) {
    // Check that ordering is according to dimension type, with the role of
    // free and contract dimensions swapped.
    gtl::InlinedVector<int, 5> remap = {0, 1, 3, 2, 4};
    for (int i = 0; i + 1 < labels.size(); ++i) {
      const int dimtype_a = remap[label_types[labels[i]]];
      const int dimtype_b = remap[label_types[labels[i + 1]]];
      if (dimtype_a > dimtype_b ||
          (dimtype_a == dimtype_b && labels[i] > labels[i + 1])) {
        return false;
      }
    }
    return true;
  }


  template <typename Device, typename T>
  struct ReduceRank2TensorGenericImplFunctor {
    void operator()(OpKernelContext* ctx, const Tensor* input, Tensor* output,
                    const int64 output_size, const int64 reduce_size) {
      functor::ReduceFunctor<Device, Eigen::internal::SumReducer<T>>::Reduce(
          ctx, output->shaped<T, 1>({output_size}),
          const_cast<const Tensor&>(*input).shaped<T, 2>(
              {output_size, reduce_size}),
          Eigen::array<typename TTypes<T>::Tensor::Index, 1>({1}),
          Eigen::internal::SumReducer<T>());
    }
  };

  template <typename Device, typename T>
  struct ReduceRank2TensorMayBeCutensorFunctor {
    void operator()(OpKernelContext* ctx, const Tensor* input, Tensor* output,
                    const int64 output_size, const int64 reduce_size) {
      ReduceRank2TensorGenericImplFunctor<Device, T> reduce_rank2_generic;
      reduce_rank2_generic(
          ctx, input, output, output_size, reduce_size);
    }
  };

  template <typename T>
  struct ReduceRank2TensorMayBeCutensorFunctor<GPUDevice, T> {
    void operator()(OpKernelContext* ctx, const Tensor* input, Tensor* output,
                    const int64 output_size, const int64 reduce_size) {
#if CUDA_CUTENSOR
      if (CuTensorReadyAndRequired()) {
        functor::EinsumCutensorGpuFunctor<T>::ReduceRank2Tensor(
            ctx, input, output, output_size, reduce_size);
      } else {
#endif
      ReduceRank2TensorGenericImplFunctor<GPUDevice, T> reduce_rank2_generic;
      reduce_rank2_generic(
          ctx, input, output, output_size, reduce_size);
#if CUDA_CUTENSOR
      }
#endif
    }
  };

  template <typename Device, typename T>
  static Status ReduceOperand(
      OpKernelContext* ctx, const Tensor& input,
      const std::vector<EinsumDimensionType>& label_types,
      const LabelCounts& label_counts, const bool has_single_label,
      Labels* labels, Labels* free_labels,
      bool* swap_free_and_contract, Tensor* output) {
    // Find the permutation to transpose the input dimensions in the order of
    // EinsumDimensionType; i.e. batch, free, contract and reduce dimensions.
    // This makes it more convenient to invoke Reduce/Contract operations.
    std::vector<int> permutation(input.dims());
    absl::c_iota(permutation, 0);
    Tensor input_transposed;
    // Check if we can avoid the transpose. We need to flip the adj_x (or adj_y)
    // flag during BatchMatMul. This is an extra optimization not necessary for
    // correctness.
    if (ShouldSwapFreeAndContract(*labels, label_types)) {
      *swap_free_and_contract = true;
    } else {
      absl::c_sort(permutation, [&](int i, int j) {
        int label_i = (*labels)[i];
        int label_j = (*labels)[j];
        return std::tie(label_types[label_i], label_i) <
               std::tie(label_types[label_j], label_j);
      });
    }
    // Transpose the input so that EinsumDimensionTypes are in order.
    TF_RETURN_IF_ERROR(TransposeOperand<Device, T>(ctx, input, permutation,
                                                   &input_transposed));
    PermuteLabels(permutation, labels);

    // Take the generalized diagonal for dimensions with repeated axis labels.
    Tensor input_deduped;
    labels->erase(std::unique(labels->begin(), labels->end()), labels->end());
    TF_RETURN_IF_ERROR(
        StrideOrInflate<Device, T>(ctx, input_transposed, *labels, label_counts,
                                   false /* should_inflate */, &input_deduped));

    // Reshape denotes the rank-5 shape [broadcast, batch, free, contract,
    // reduce] where we've compacted the dimensions of each EinsumDimensionType.
    gtl::InlinedVector<int64_t, 5> reshape(5, 1);
    // The output shape is [batch shape] + [free size, contract size]
    // That is, the batch shape is preserved (for broadcasting while
    // contracting) while the free dims and contract dims are compressed to one
    // dimension each.
    TensorShape output_shape;
    for (int label_idx = 0; label_idx < labels->size(); ++label_idx) {
      const int label = labels->at(label_idx);
      int64_t dim = input_deduped.dim_size(label_idx);
      if (label_types[label] == EinsumDimensionType::kBroadcasting ||
          label_types[label] == EinsumDimensionType::kBatch) {
        output_shape.AddDim(dim);
      } else if (label_types[label] == EinsumDimensionType::kFree) {
        free_labels->push_back(label);
      }
      reshape[label_types[label]] *= dim;
    }
    if (*swap_free_and_contract)
      std::swap(reshape[EinsumDimensionType::kFree],
                reshape[EinsumDimensionType::kContract]);
    output_shape.AddDim(reshape[EinsumDimensionType::kFree]);
    output_shape.AddDim(reshape[EinsumDimensionType::kContract]);

    if (reshape[EinsumDimensionType::kReduce] ==
        1) {  // No need to actually reduce.
      return CopyFrom(input_deduped, output_shape, output);
    }
    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    using Reducer = Eigen::internal::SumReducer<T>;
    using Index = typename TTypes<T>::Tensor::Index;
    // Reduce along the last axis (i.e axis 1) of the rank-2 Tensor.
    const int64_t output_size = reshape[kBroadcasting] * reshape[kBatch] *
                              reshape[kFree] * reshape[kContract];
    
    if (!has_single_label) {
      ReduceRank2TensorMayBeCutensorFunctor<Device, T> rank2_tensor_reduction;
      rank2_tensor_reduction(ctx, &input_deduped, output, output_size,
                             reshape[kReduce]);
    } else {
      ReduceRank2TensorGenericImplFunctor<Device, T> rank2_tensor_reduction;
      rank2_tensor_reduction(ctx, &input_deduped, output, output_size,
                             reshape[kReduce]);
    }

    return Status::OK();
  }

  // Reshapes a Tensor of shape [b0,b1...bk,N,M] to [prod(b0,b1...bk),N,M].
  static Status ReshapeToRank3(const Tensor& input, int batch_size,
                               Tensor* output) {
    const int rank = input.dims();
    TensorShape output_shape = {batch_size, input.dim_size(rank - 2),
                                input.dim_size(rank - 1)};
    return CopyFrom(input, output_shape, output);
  }

  template <typename Device, typename T>
  struct ContractRank3TensorsFunctor {
    void operator()(OpKernelContext* ctx, const Tensor& input0,
                    const Tensor& input1, const bool trans_x,
                    const bool trans_y, const MatMulBCast& bcast,
                    Tensor* output) {
      Tensor lhs, rhs;
      OP_REQUIRES_OK(ctx, ReshapeToRank3(input0, bcast.x_batch_size(), &lhs));
      OP_REQUIRES_OK(ctx, ReshapeToRank3(input1, bcast.y_batch_size(), &rhs));

      LaunchBatchMatMul<Device, T>::Launch(ctx, lhs, rhs, /*adj_x=*/false,
                                           /*adj_y=*/false, trans_x, trans_y,
                                           bcast, output);
    }
  };

  template <typename T>
  struct ContractRank3TensorsFunctor<GPUDevice, T> {
    void operator()(OpKernelContext* ctx, const Tensor& input0,
                    const Tensor& input1, const bool trans_x,
                    const bool trans_y, const MatMulBCast& bcast,
                    Tensor* output) {
      // Strided batched gemm and no broadcasting required, dispatch to
      // cuTENSOR's ContractRank3TensorsFunctor
#if CUDA_CUTENSOR
      if (CuTensorReadyAndRequired()) {
        std::vector<int> input0_shape, input1_shape;
        for (int i = 0; i < input0.dims() - 2; ++i)
          input0_shape.push_back(input0.dim_size(i));
        for (int i = 0; i < input1.dims() - 2; ++i)
          input1_shape.push_back(input1.dim_size(i));

        int max_input_rank = std::max(input0_shape.size(), input1_shape.size());
        int min_input_rank = std::min(input0_shape.size(), input1_shape.size());
        for (int i = 0; i < max_input_rank - min_input_rank; ++i) {
          if (input0_shape.size() > input1_shape.size())
            input1_shape.insert(input1_shape.begin(), 1);
          else
            input0_shape.insert(input0_shape.begin(), 1);
        }

        string op0_equation, op1_equation, rhs_equation;
        int index_cnt = 0;
        constexpr char first_ascii_charactor = 'a';
        for (int i = 0; i < max_input_rank; i++) {
          if (input0_shape[i] == input1_shape[i] && input0_shape[i] == 1)
            continue;
          char ascii_charactor = first_ascii_charactor + index_cnt++;
          if (input0_shape[i] != input1_shape[i]) {
            input0_shape[i] > 1 ? op0_equation += ascii_charactor
                                : op1_equation += ascii_charactor;
          } else {
            op0_equation += ascii_charactor;
            op1_equation += ascii_charactor;
          }
          rhs_equation += ascii_charactor;
        }

        // 4 possible transposes in matmul.
        const std::vector<string> candidate_equations{"xy,yz->xz", "xy,zy->xz",
                                                      "yx,yz->xz", "yx,zy->xz"};

        string matmul_equation =
            candidate_equations.at(2 * int(trans_x) + int(trans_y));
        string equation = op0_equation + matmul_equation.substr(0, 2) + "," +
                          op1_equation + matmul_equation.substr(3, 2) + "->" +
                          rhs_equation + matmul_equation.substr(7, 2);

        std::vector<int> input0_shape_squeezed, input1_shape_squeezed;
        for (int i = 0; i < input0.dims() - 2; ++i) {
          if (input0.dim_size(i) > 1)
            input0_shape_squeezed.push_back(input0.dim_size(i));
        }
        for (int i = 0; i < input1.dims() - 2; ++i) {
          if (input1.dim_size(i) > 1)
            input1_shape_squeezed.push_back(input1.dim_size(i));
        }
        input0_shape_squeezed.push_back(input0.dim_size(input0.dims() - 2));
        input0_shape_squeezed.push_back(input0.dim_size(input0.dims() - 1));
        input1_shape_squeezed.push_back(input1.dim_size(input1.dims() - 2));
        input1_shape_squeezed.push_back(input1.dim_size(input1.dims() - 1));

        functor::EinsumCutensorGpuFunctor<T>::ContractRank3Tensors(
            ctx, equation, &input0, &input1, input0_shape_squeezed,
            input1_shape_squeezed, output);
      } else {
#endif
      if (CuTensorReadyAndRequired()) {
        VLOG(1) << "WARNING: " << "CuTENSOR support is requested but TF was "
                                  "not built with libcutensor, so fallbacks "
                                  "to generic GPU kernel.";
      }
      Tensor rhs, lhs;
      OP_REQUIRES_OK(ctx, ReshapeToRank3(input0, bcast.x_batch_size(), &lhs));
      OP_REQUIRES_OK(ctx, ReshapeToRank3(input1, bcast.y_batch_size(), &rhs));

      LaunchBatchMatMul<GPUDevice, T>::Launch(ctx, lhs, rhs, /*adj_x=*/false,
                                              /*adj_y=*/false, trans_x,
                                              trans_y, bcast, output);
#if CUDA_CUTENSOR
      }
#endif
     }
  };

  // Contracts the inputs along the last axis (or the second last if the
  // corresponding value of swap_free_and_contract is true). The batch
  // dimensions are broadcast to the output shape.
  // TODO(anudhyan): BatchMatMul might devolve into a component-wise
  // multiplication when the matrix shape is [1,1]; in this case BatchMatMul
  // functor would be very inefficient. The functor should detect if this is the
  // case and perform componentwise multiplication functor instead.
  template <typename Device, typename T>
  static Status ContractOperands(OpKernelContext* ctx,
                                 absl::Span<const Tensor> inputs,
                                 absl::Span<const bool> swap_free_and_contract,
                                 Tensor* output) {
    if (inputs.size() == 1)
      return CopyFrom(inputs[0], inputs[0].shape(), output);
    MatMulBCast bcast(inputs[0].shape().dim_sizes(),
                      inputs[1].shape().dim_sizes());
    if (!bcast.IsValid()) {
      return errors::InvalidArgument(
          "Invalid broadcasting dimensions: ", inputs[0].shape().DebugString(),
          " vs. ", inputs[1].shape().DebugString());
    }

    TensorShape output_shape = bcast.output_batch_shape();
    for (int i = 0; i < inputs.size(); ++i) {
      const int64_t free_axis =
          inputs[i].dims() - (swap_free_and_contract[i] ? 1 : 2);
      output_shape.AddDim(inputs[i].dim_size(free_axis));
    }
    bool trans_x = swap_free_and_contract[0];
    bool trans_y = !swap_free_and_contract[1];

    TF_RETURN_IF_ERROR(
        ctx->allocate_temp(DataTypeToEnum<T>::value, output_shape, output));
    if (inputs[0].NumElements() == 0 || inputs[1].NumElements() == 0) {
      functor::SetZeroFunctor<Device, T> set_zero;
      set_zero(ctx->eigen_device<Device>(), output->flat<T>());
      return Status::OK();
    }
    Tensor output_reshaped;
    TF_RETURN_IF_ERROR(
        ReshapeToRank3(*output, bcast.output_batch_size(), &output_reshaped));

    ContractRank3TensorsFunctor<Device, T> rank3_tensors_contraction_func;
    rank3_tensors_contraction_func(ctx, inputs[0], inputs[1], trans_x, trans_y,
                                   bcast, &output_reshaped);

    return Status::OK();
  }
};

namespace functor {

template <typename Device, typename T>
struct EinsumMaybeCutensorFunctor {
  void operator()(OpKernelContext* ctx,
                  const EinsumHelper::EinsumOpInputFeatures& features) {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));

    OperandLabels input_labels(features.input_labels);
    Labels output_labels(features.output_labels);
    gtl::InlinedVector<bool, 2> input_has_ellipsis(features.input_has_ellipsis);
    bool output_has_ellipsis(features.output_has_ellipsis);
    std::vector<EinsumDimensionType> label_types(features.label_types);
    OperandLabelCounts input_label_counts(features.input_label_counts);
    LabelCounts output_label_counts(features.output_label_counts);
    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(
        ctx, EinsumHelper::ProcessDimensions(
                 inputs, input_has_ellipsis, output_has_ellipsis, &input_labels,
                 &output_labels, &label_types, &input_label_counts,
                 &output_label_counts, &label_to_dim_sizes));

    // The reduction phase (a) sums across reduction dimensions, (b) takes
    // generalized diagonals, and (c) reshapes it into shape
    //   [(broadcasting) batch shape] + [F,C]
    // where F and C denote the total (compacted) size of free and contract
    // dimensions, respectively.
    const int num_inputs = inputs.size();
    OperandLabels free_labels(num_inputs);
    gtl::InlinedVector<Tensor, 2> inputs_reduced(num_inputs);
    gtl::InlinedVector<bool, 2> swap_free_and_contract(num_inputs);
    for (int i = 0; i < num_inputs; ++i) {
      OP_REQUIRES_OK(ctx,
                     EinsumHelper::ReduceOperand<Device, T>(
                         ctx, inputs[i], label_types, input_label_counts[i],
                         features.has_single_label,
                         &input_labels[i], &free_labels[i],
                         &swap_free_and_contract[i], &inputs_reduced[i]));
    }

    // After reduction, the inputs should be reshaped to Tensors suitable for
    // contraction. If num_inputs is 1, the reduced input is simply forwarded to
    // the output.
    Tensor contraction_output_reshaped;
    OP_REQUIRES_OK(ctx, EinsumHelper::ContractOperands<Device, T>(
                            ctx, inputs_reduced, swap_free_and_contract,
                            &contraction_output_reshaped));

    // Copy the batch labels from the contraction output. Recover the batch
    // shape, which may have been broadcasted.
    TensorShape result_shape = contraction_output_reshaped.shape();
    result_shape.RemoveLastDims(2);

    int num_labels = label_types.size();
    Labels result_labels;
    // All batch dimensions should be present in the contracted result. First
    // the broadcasting dimensions, then the named batch dimensions.
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBroadcasting)
        result_labels.push_back(label);
    }
    for (int label = 0; label < num_labels; ++label) {
      if (label_types[label] == EinsumDimensionType::kBatch)
        result_labels.push_back(label);
    }
    for (int i = 0; i < num_inputs; ++i) {
      for (int label : free_labels[i]) {
        result_labels.push_back(label);
        result_shape.AddDim(label_to_dim_sizes[label]);
      }
    }

    // Reshape the contraction (or reduction) result to its expanded shape:
    // [(broadcasted) batch shape] + [free shape 0] + [free shape 1].
    Tensor contraction_output;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::CopyFrom(contraction_output_reshaped, result_shape,
                                    &contraction_output));

    // Inflate the output if necessary. (E.g. for the equation 'i->iii' which
    // may arise while computing gradient of a regular Einsum).
    // TODO(anudhyan): It's possible that Eigen's contract and inflate can be
    // chained here to avoid materializing an intermediate.
    Tensor output_inflated;
    OP_REQUIRES_OK(
        ctx, EinsumHelper::StrideOrInflate<Device, T>(
                 ctx, contraction_output, result_labels, output_label_counts,
                 true /* should_inflate */, &output_inflated));
    if (output_inflated.dims() > contraction_output.dims()) {
      // We inflated the output. Modify result labels accordingly.
      Labels inflated_labels;
      for (int label : result_labels) {
        inflated_labels.insert(inflated_labels.end(),
                               output_label_counts[label], label);
      }
      result_labels.swap(inflated_labels);
    }
    // Find the permutation to map the result labels to the output labels. Note
    // that both the result and the final output may have the repeated labels,
    // in which case the permutation preserves the left-to-right ordering.
    // E.g. if result labels are [0, 0, 1] and output is [0, l, 0] then the
    // permutation should be [0, 2, 1]. We also use the fact that repeated
    // labels in the result are adjacent to each other.
    std::vector<int> output_permutation(output_labels.size());
    std::vector<int> label_to_position(num_labels, -1);
    for (int i = 0; i < result_labels.size(); ++i) {
      // Remember the position of only the leftmost result label.
      if (label_to_position[result_labels[i]] == -1) {
        label_to_position[result_labels[i]] = i;
      }
    }
    for (int i = 0; i < output_labels.size(); ++i) {
      output_permutation[i] = label_to_position[output_labels[i]];
      // We have found the leftmost occurrence. The next one would be adjacent.
      label_to_position[output_labels[i]] += 1;
    }
    Tensor output;
    OP_REQUIRES_OK(ctx, EinsumHelper::TransposeOperand<Device, T>(
                            ctx, output_inflated, output_permutation, &output));
    ctx->set_output(0, output);
  }
};

#if CUDA_CUTENSOR
template <typename T>
struct EinsumCutensorGpuFunctor {
  static void ContractRank3Tensors(OpKernelContext* ctx, const string& equation,
                                   const Tensor* input0, const Tensor* input1,
                                   const std::vector<int> input0_shape,
                                   const std::vector<int> input1_shape,
                                   Tensor* output_tensor) {
    Compute(ctx, equation, {input0, input1}, {input0_shape, input1_shape},
            &output_tensor);
  }

  static void ReduceRank2Tensor(OpKernelContext* ctx, const Tensor* input,
                                Tensor* output, const int64 output_size,
                                const int64 reduce_size) {
    if (!output_size || !reduce_size) {
      functor::SetZeroFunctor<Eigen::GpuDevice, T> set_zero;
      set_zero(ctx->eigen_device<Eigen::GpuDevice>(), output->flat<T>());
      return;
    }
    gtl::InlinedVector<std::vector<int>, 2> input_shape;
    input_shape.push_back({output_size, reduce_size});
    Compute(ctx, "ab->a", {input}, input_shape, &output);
  }

  void operator()(OpKernelContext* ctx,
                  EinsumHelper::EinsumOpInputFeatures features) {
    OpInputList inputs;
    OP_REQUIRES_OK(ctx, ctx->input_list("inputs", &inputs));
    const int num_inputs = inputs.size();

    LabelToDimSizes label_to_dim_sizes;

    OP_REQUIRES_OK(
        ctx,
        EinsumHelper::ProcessDimensions(
            inputs, features.input_has_ellipsis, features.output_has_ellipsis,
            &(features.input_labels), &(features.output_labels),
            &(features.label_types), &(features.input_label_counts),
            &(features.output_label_counts), &label_to_dim_sizes));

    gtl::InlinedVector<const Tensor*, 2> input_tensors;
    gtl::InlinedVector<std::vector<int>, 2> input_tensor_shapes;
    for (int i = 0; i < num_inputs; ++i) {
      input_tensors.push_back(&inputs[i]);
      std::vector<int> input_shape;
      for (int j = 0; j < inputs[i].dims(); ++j) {
        input_shape.push_back(inputs[i].dim_size(j));
      }
      input_tensor_shapes.push_back(input_shape);
    }
    Compute(ctx, features.equation, input_tensors, input_tensor_shapes);
  }

 private:
  static void Compute(
      OpKernelContext* ctx, const string equation,
      const gtl::InlinedVector<const Tensor*, 2>& input_tensors,
      const gtl::InlinedVector<std::vector<int>, 2>& input_tensor_shapes,
      Tensor** output_tensor_init = nullptr) {
    const int num_inputs = input_tensors.size();
    const Tensor* input_0_tensor = input_tensors[0];
    const Tensor* input_1_tensor;

    if (num_inputs == 2) {
      input_1_tensor = input_tensors[1];
    }

    std::vector<int64> output_dims;
    T alpha = (T)1.0f;
    T beta = (T)0.0f;
    size_t worksize = 0;

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES_OK(ctx, stream->parent()->AsTsr()->CutensorPreprocess(
                            stream, &output_dims, se::tsr::ToDataType<T>::value,
                            equation, input_tensor_shapes[0],
                            num_inputs == 2 ? input_tensor_shapes[1]
                                            : std::vector<int>()));

    Tensor* output_tensor(nullptr);
    TensorShape output_shape = TensorShape(output_dims);

    if (output_tensor_init == nullptr) {
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, output_shape, &output_tensor));
    } else {
      output_tensor = *output_tensor_init;
    }

    bool input_has_empty_dim =
        !absl::c_all_of(input_tensor_shapes[0], [](int c) { return c > 0; }) ||
        (num_inputs == 2 &&
         !absl::c_all_of(input_tensor_shapes[1], [](int c) { return c > 0; }));

    if (input_has_empty_dim ||
        !absl::c_all_of(output_dims, [](int c) { return c > 0; })) {
      functor::SetZeroFunctor<Eigen::GpuDevice, T> set_zero;
      set_zero(ctx->eigen_device<Eigen::GpuDevice>(), output_tensor->flat<T>());
      return;
    }

    OP_REQUIRES(
        ctx,
        stream->parent()->AsTsr()->PrepareContraction(
            stream, &worksize, input_0_tensor->flat<T>().data(),
            num_inputs == 1 ? nullptr : input_1_tensor->flat<T>().data(),
            output_tensor->flat<T>().data()),
        errors::Internal("Preparation for contraction failed!"));

    Tensor work_tensor;
    int64 work_tensor_size = worksize / sizeof(int8);
    TensorShape work_shape = {work_tensor_size};
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT8, work_shape, &work_tensor));

    OP_REQUIRES(
        ctx,
        stream
            ->ThenTsrContraction(
                input_0_tensor->flat<T>().data(),
                num_inputs == 1 ? nullptr : input_1_tensor->flat<T>().data(),
                output_tensor->flat<T>().data(),
                work_tensor.flat<int8>().data())
            .ok(),
        errors::Internal("Compute by CuTensor failed!"));
  }
};
#endif // CUDA_CUTENSOR
}  // namespace functor

template <typename Device, typename T>
class EinsumOp : public OpKernel {
 public:
  explicit EinsumOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("equation", &(InputFeatures.equation)));
    OP_REQUIRES_OK(
        c, ParseEinsumEquation(
               InputFeatures.equation, &(InputFeatures.input_labels),
               &(InputFeatures.output_labels), &(InputFeatures.label_types),
               &(InputFeatures.input_label_counts),
               &(InputFeatures.output_label_counts),
               &(InputFeatures.input_has_ellipsis),
               &(InputFeatures.output_has_ellipsis)));
  }

  void Compute(OpKernelContext* ctx) override {
    functor::EinsumMaybeCutensorFunctor<Device, T> einsum_generic_func;
    einsum_generic_func(ctx, InputFeatures);
  }

 private:
  EinsumHelper::EinsumOpInputFeatures InputFeatures;
};

template <typename T>
class EinsumOp<GPUDevice, T> : public OpKernel {
 public:
  explicit EinsumOp(OpKernelConstruction* c) : OpKernel(c) {
    OP_REQUIRES_OK(c, c->GetAttr("equation", &(InputFeatures.equation)));
    OP_REQUIRES_OK(
        c, EinsumHelper::ParseEquationWithCuTensorSupport(
           InputFeatures.equation, &(InputFeatures.input_labels),
          &(InputFeatures.output_labels), &(InputFeatures.label_types),
          &(InputFeatures.input_label_counts),
          &(InputFeatures.output_label_counts),
          &(InputFeatures.input_has_ellipsis),
          &(InputFeatures.output_has_ellipsis),
          &(InputFeatures.has_single_label),
          &(InputFeatures.support_cutensor)));
#if CUDA_CUTENSOR
    cutsr_ready_and_required = CuTensorReadyAndRequired();
#endif
  }

  void Compute(OpKernelContext* ctx) override {
#if CUDA_CUTENSOR
#if CUTENSOR_VERSION >= 10500
    // cuTENSOR only starts to support single label reduction/contraction after
    // version 1.5.0.
    bool use_direct_cutensor =
        cutsr_ready_and_required && InputFeatures.support_cutensor;
#else
    bool use_direct_cutensor = cutsr_ready_and_required &&
                               InputFeatures.support_cutensor &&
                               !InputFeatures.has_single_label;
#endif
    if (use_direct_cutensor) {
      functor::EinsumCutensorGpuFunctor<T> einsum_cutensor_func;
      einsum_cutensor_func(ctx, InputFeatures);
    } else {
#endif
    if (cutsr_ready_and_required) {
      VLOG(1) << "WARNING: " << "CuTENSOR support is requested but either TF "
                                "was not built with libcutensor or equation is "
                                "not currently supported so fallbacks to "
                                "generic GPU kernel.";
    }
    functor::EinsumMaybeCutensorFunctor<GPUDevice, T> einsum_generic_func;
    einsum_generic_func(ctx, InputFeatures);
#if CUDA_CUTENSOR
    }
#endif
  }

 private:
  EinsumHelper::EinsumOpInputFeatures InputFeatures;
  bool cutsr_ready_and_required = false;
};
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, N)                                      \
  template <>                                                       \
  void StrideFunctor<GPUDevice, T, N>::operator()(                  \
      const GPUDevice& d, typename TTypes<T, N>::ConstTensor input, \
      const Eigen::DSizes<Eigen::DenseIndex, N>& strides,           \
      typename TTypes<T, N>::Tensor output);                        \
  extern template struct StrideFunctor<GPUDevice, T, N>;            \
  template <>                                                       \
  void InflateFunctor<GPUDevice, T, N>::operator()(                 \
      const GPUDevice& d, typename TTypes<T, N>::ConstTensor input, \
      const Eigen::DSizes<Eigen::DenseIndex, N>& strides,           \
      typename TTypes<T, N>::Tensor output);                        \
  extern template struct InflateFunctor<GPUDevice, T, N>;
#if CUDA_CUTENSOR
#define DECLARE_GPU_SPEC_FOR_EINSUM(T) \
  template struct EinsumCutensorGpuFunctor<T>;
  DECLARE_GPU_SPEC_FOR_EINSUM(float);
  DECLARE_GPU_SPEC_FOR_EINSUM(double);
  DECLARE_GPU_SPEC_FOR_EINSUM(Eigen::half);
#undef DECLARE_GPU_SPEC_FOR_EINSUM
#endif

#define DECLARE_GPU_SPECS(T) \
  DECLARE_GPU_SPEC(T, 1);    \
  DECLARE_GPU_SPEC(T, 2);    \
  DECLARE_GPU_SPEC(T, 3);    \
  DECLARE_GPU_SPEC(T, 4);    \
  DECLARE_GPU_SPEC(T, 5);    \
  DECLARE_GPU_SPEC(T, 6);

DECLARE_GPU_SPECS(float);
DECLARE_GPU_SPECS(double);
DECLARE_GPU_SPECS(Eigen::half);

// TODO(rocm): Enable once complex types are supported.
#if GOOGLE_CUDA
DECLARE_GPU_SPECS(complex64);
DECLARE_GPU_SPECS(complex128);
#endif
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPECS
}  // namespace functor
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_EINSUM_OP_IMPL_H_

