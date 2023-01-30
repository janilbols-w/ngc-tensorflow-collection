# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradients for CuudnnRNN operators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_cudnn_mha_ops


@ops.RegisterGradient("CudnnMHA")
def _CudnnMHAGrad(op, *grads):
  """Gradients for the CudnnMHA op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use CudnnMHA in gradients, is_training must be set to True.")
  return gen_cudnn_mha_ops.cudnn_mha_backprop(
      input_q=op.inputs[0],
      input_k=op.inputs[1],
      input_v=op.inputs[2],
      params=op.inputs[3],
      output=op.outputs[0],
      output_backprop=grads[0],
      reserve_space=op.outputs[1],
      host_reserved=op.outputs[2],
      dropout=op.get_attr("dropout"),
      seed=op.get_attr("seed"))
