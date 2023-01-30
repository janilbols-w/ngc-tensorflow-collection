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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace {}  // namespace

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("CudnnMHA")
    .Input("input_q: T")
    .Input("input_k: T")
    .Input("input_v: T")
    .Input("params: T")
    .SetIsStateful()
    .Output("output: T")
    .Output("reserve_space: T")
    .Output("host_reserved: int32")
    .Attr("T: {float16, float32, float64}")
    .Attr("num_heads: int")
    .Attr("key_dim: int")
    .Attr("output_dim: int")
    .Attr("value_dim: int = -1")
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("use_bias: bool = true")
    .Attr("is_training: bool = false")
    .Attr("residual_link: bool = false")
    .SetShapeFn([](InferenceContext* c) {
      auto input_q_shape = c->input(0);
      auto batch_size = c->Dim(input_q_shape, 0);
      auto max_seq_len_qo = c->Dim(input_q_shape, 1);
      int output_vect_size;
      TF_RETURN_IF_ERROR(c->GetAttr("output_dim", &output_vect_size));
      auto output_shape =
          c->MakeShape({batch_size, max_seq_len_qo, output_vect_size});
      c->set_output(0, output_shape);
      c->set_output(1, c->UnknownShape());
      c->set_output(2, c->UnknownShape());
      return Status::OK();
    });

REGISTER_OP("CudnnMHABackprop")
    .Input("input_q: T")
    .Input("input_k: T")
    .Input("input_v: T")
    .Input("params: T")
    .Input("output: T")
    .Input("output_backprop: T")
    .Input("reserve_space: T")
    .Input("host_reserved: int32")
    .SetIsStateful()
    .Output("input_q_backprop: T")
    .Output("input_k_backprop: T")
    .Output("input_v_backprop: T")
    .Output("params_backprop: T")
    .Attr("T: {float16, float32, float64}")
    .Attr("dropout: float = 0.0")
    .Attr("seed: int = 0")
    .Attr("use_bias: bool = true")
    .SetShapeFn([](InferenceContext* c) {
      auto input_q_shape = c->input(0);
      auto input_k_shape = c->input(1);
      auto input_v_shape = c->input(2);
      auto params_shape = c->input(3);
      c->set_output(0, input_q_shape);
      c->set_output(1, input_k_shape);
      c->set_output(2, input_v_shape);
      c->set_output(3, params_shape);
      return Status::OK();
    });
}  // namespace tensorflow
