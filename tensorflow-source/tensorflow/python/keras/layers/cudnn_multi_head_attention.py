#Lint as : python3
#Copyright 2021 The TensorFlow Authors.All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http: // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == ==
"""Keras-based attention layer."""
#pylint : disable = g - classes - have - attributes

import collections
import string

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import einsum_dense
from tensorflow.python.keras.layers import multi_head_attention as MHA
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_cudnn_mha_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

_CHR_IDX = string.ascii_lowercase

@keras_export("keras.layers.CuDNNMHA")
class CuDNNMHA(MHA.MultiHeadAttention):
  """ Fast Multi-head-attention layer implementation backed by CuDNN.

  It doesn't support masking or attention scores output. Unlike its parent,
  MultiHeadAttention, CuDNNMHA only supports rank-3 input tensors as
  `(batch_size, <query/key/value dimensions>, key_dim)`. In addition to
  MultiHeadAttention, it enables the residual connection, see
  https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMultiHeadAttnForward
  In our implementation, the residual link connects to query data.

  Call arguments:
   query: Query `Tensor` of shape `(B, T, dim)`.
   value: Value `Tensor` of shape `(B, S, dim)`.
   key: Optional key `Tensor` of shape `(B, S, dim)`. If not given, will use
     `value` for both `key` and `value`, which is the most common case.
   training: Python boolean indicating whether the layer should behave in
     training mode (adding dropout) or in inference mode (no dropout).
     Defaults to either using the training mode of the parent layer/model,
     or False (inference) if there is no parent layer.
   residual_link: Python boolean indicating whether residual link should be
     enabled.
   attention_mask: Has and default to be None.
   return_attention_scores: Has and default to be None
  """
  def __init__(self,
               **kwargs):
    if "attention_axes" in kwargs:
      _attention_axes = kwargs.pop("attention_axes")
      if _attention_axes is not None and _attention_axes != (1,):
        logging.warning("CuDNNMHA only supports rank-3 tensors as input, "
                        "thus attention_axes has to be (1, ) or None")

    super(CuDNNMHA, self).__init__(**kwargs)
    self._support_attention_scores_output = False
    self._support_masking = False

  def get_config(self):
    base_config = super(  # pylint: disable=bad-super-call
        CuDNNMHA, self).get_config()
    base_config.pop("attention_axes")
    return dict(list(base_config.items()))

  def _build_from_signature(self, query, value, key=None):
    """Builds layers and variables.

      Once the method is called, self._built_from_signature will be set to True.

      Args:
        query: Query tensor or TensorShape.
        value: Value tensor or TensorShape.
        key: Key tensor or TensorShape.
      """
    self._built_from_signature = True
    if hasattr(query, "shape"):
      self._query_shape = tensor_shape.TensorShape(query.shape)
    else:
      self._query_shape = tensor_shape.TensorShape(query)
    if hasattr(value, "shape"):
      self._value_shape = tensor_shape.TensorShape(value.shape)
    else:
      self._value_shape = tensor_shape.TensorShape(value)
    if key is None:
      self._key_shape = self._value_shape
    elif hasattr(key, "shape"):
      self._key_shape = tensor_shape.TensorShape(key.shape)
    else:
      self._key_shape = tensor_shape.TensorShape(key)

    common_kwargs = dict(
        kernel_initializer=self._kernel_initializer,
        bias_initializer=self._bias_initializer,
        kernel_regularizer=self._kernel_regularizer,
        bias_regularizer=self._bias_regularizer,
        activity_regularizer=self._activity_regularizer,
        kernel_constraint=self._kernel_constraint,
        bias_constraint=self._bias_constraint)
#Any setup work performed only once should happen in an `init_scope`
#to avoid creating symbolic Tensors that will later pollute any eager
#operations.
    with tf_utils.maybe_init_scope(self):
      free_dims = self._query_shape.rank - 1

      einsum_equation_q, bias_axes_q, output_rank_q = MHA._build_proj_equation(
          free_dims, bound_dims=1, output_dims=2)
      einsum_equation_k, bias_axes_k, output_rank_k = MHA._build_proj_equation(
          self._key_shape.rank - 1, bound_dims=1, output_dims=2)
      einsum_equation_v, bias_axes_v, output_rank_v = MHA._build_proj_equation(
          self._value_shape.rank - 1, bound_dims=1, output_dims=2)

      kernel_shape_q, bias_shape_q, _ = einsum_dense._analyze_einsum_string(
          einsum_equation_q, bias_axes_q if self._use_bias else None,
          self._query_shape, MHA._get_output_shape(
              output_rank_q - 1, [self._num_heads, self._key_dim]))
      self._kernel_q = self.add_weight(
          "kernel_q",
          shape=kernel_shape_q,
          initializer=self._kernel_initializer,
          regularizer=self._kernel_regularizer,
          constraint=self._kernel_constraint,
          trainable=True)
      
      self._bias_q = self.add_weight(
          "bias_q",
          shape=bias_shape_q,
          initializer=self._bias_initializer,
          regularizer=self._bias_regularizer,
          constraint=self._bias_constraint,
          trainable=True) if self._use_bias else None

      kernel_shape_k, bias_shape_k, _ = einsum_dense._analyze_einsum_string(
          einsum_equation_k, bias_axes_k if self._use_bias else None,
          self._value_shape if key is None else self._key_shape,
          MHA._get_output_shape(output_rank_k - 1,
                                [self._num_heads, self._key_dim]))
      self._kernel_k = self.add_weight(
          "kernel_k",
          shape=kernel_shape_k,
          initializer=self._kernel_initializer,
          regularizer=self._kernel_regularizer,
          constraint=self._kernel_constraint,
          trainable=True)

      self._bias_k = self.add_weight(
          "bias_k",
          shape=bias_shape_k,
          initializer=self._bias_initializer,
          regularizer=self._bias_regularizer,
          constraint=self._bias_constraint,
          trainable=True) if self._use_bias else None

      kernel_shape_v, bias_shape_v, _ = einsum_dense._analyze_einsum_string(
          einsum_equation_v, bias_axes_v if self._use_bias else None,
          self._value_shape, MHA._get_output_shape(
              output_rank_v - 1, [self._num_heads, self._value_dim]))
      self._kernel_v = self.add_weight(
          "kernel_v",
          shape=kernel_shape_v,
          initializer=self._kernel_initializer,
          regularizer=self._kernel_regularizer,
          constraint=self._kernel_constraint,
          trainable=True)
      self._bias_v = self.add_weight(
          "bias_v",
          shape=bias_shape_v,
          initializer=self._bias_initializer,
          regularizer=self._bias_regularizer,
          constraint=self._bias_constraint,
          trainable=True) if self._use_bias else None

      self._kenel_o, self._bias_o = self._make_output_dense(
          free_dims, common_kwargs, "attention_output")

  def _make_output_dense(self, free_dims, common_kwargs, name=None):
    """Builds the output projection matrix.

    Args:
      free_dims: Number of free dimensions for einsum equation building.
      common_kwargs: Common keyword arguments for einsum layer.
      name: Name for the projection layer.

    Returns:
      kernel_o / bias_o
    """
    if self._output_shape:
      if not isinstance(self._output_shape, collections.abc.Sized):
        output_shape = [self._output_shape]
      else:
        output_shape = self._output_shape
    else:
      output_shape = [self._query_shape[-1]]

    einsum_equation, bias_axes, output_rank = MHA._build_proj_equation(
        free_dims, bound_dims=2, output_dims=len(output_shape))
    output_raw_shape = self._query_shape[:-1] + \
                       [self._num_heads, self._value_dim]

    kernel_shape_o, bias_shape_o, _ = einsum_dense._analyze_einsum_string(
        einsum_equation, bias_axes if self._use_bias else None,
        output_raw_shape, MHA._get_output_shape(output_rank - 1, output_shape))
    self._kernel_o = self.add_weight(
        "kernel_o",
        shape=kernel_shape_o,
        initializer=self._kernel_initializer,
        regularizer=self._kernel_regularizer,
        constraint=self._kernel_constraint,
        trainable=True)
    self._bias_o = self.add_weight(
        "bias_o",
        shape=bias_shape_o,
        initializer=self._bias_initializer,
        regularizer=self._bias_regularizer,
        constraint=self._bias_constraint,
        trainable=True) if self._use_bias else None
  
    return self._kernel_o, self._bias_o

  def call(self,
           query,
           value,
           key=None,
           training=True,
           residual_link=False,
           attention_mask=None,
           return_attention_scores=False):
    if isinstance(attention_mask, list):
      attention_mask = attention_mask[0]
    if attention_mask is not None:
      raise ValueError('Masking is not supported for CuDNN MHA.')

    if return_attention_scores:
      raise ValueError('Returning attention scores is not supported for '
                       'CuDNN MHA.')

    if not self._built_from_signature:
      self._build_from_signature(query=query, value=value, key=key)
    if key is None:
      key = value

    if self._output_shape:
      if not isinstance(self._output_shape, collections.abc.Sized):
        output_shape = [self._output_shape]
      else:
        output_shape = self._output_shape
    else:
      output_shape = [self._query_shape[-1]]
    
    def _to_params(shape, weights, biases=None):
      weights = [array_ops.reshape(x, shape) for x in weights]
      if biases is not None:
        biases = [array_ops.reshape(x, shape) for x in biases]
      return array_ops.concat(weights + (biases if biases is not None else []),
                              axis=0)

    params = _to_params(
      shape=constant_op.constant([-1]),
      weights=[self._kernel_q, self._kernel_k, self._kernel_v, self._kernel_o],
      biases=[self._bias_q, self._bias_k, self._bias_v, self._bias_o] if self._use_bias else None,
    )

    args = {
      'input_q': query,
      'input_k': key,
      'input_v': value,
      'params': params,
      'is_training': training,
      'residual_link': residual_link,
      'num_heads': self._num_heads,
      'key_dim': self._key_dim,
      'value_dim' : self._value_dim,
      'output_dim': output_shape[-1],
      'dropout' : self._dropout,
      'use_bias': self._use_bias,
    }
    output, _, _ = gen_cudnn_mha_ops.CudnnMHA(**args)

    return output
