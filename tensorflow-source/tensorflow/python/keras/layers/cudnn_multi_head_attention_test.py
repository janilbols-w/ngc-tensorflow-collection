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
"""Tests for the attention layer."""

import os
import tempfile

from absl.testing import parameterized

import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.layers import cudnn_multi_head_attention
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop
from tensorflow.python.platform import test

# This decorator runs the test in V1, V2-Eager, and V2-Functional mode. It
# guarantees forward compatibility of this code for the V2 switchover.
@keras_parameterized.run_all_keras_modes
class MultiHeadAttentionTest(keras_parameterized.TestCase):
  
  @parameterized.named_parameters(
      ("key_value_same_proj", None, None, [40, 80]),
      ("key_value_different_proj", 32, 60, [40, 60]),
  )
  @test_util.run_gpu_only
  def test_self_attention(self, value_dim, output_shape, output_dims):
    """Test with one input (self-attenntion) and no mask tensor."""
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
      num_heads=12,
      key_dim=64, 
      value_dim=value_dim,
      output_shape=output_shape)
    query = keras.Input(shape=(40, 80))
    value = keras.Input(shape=(20, 80))
    output = test_layer(query=query, value=value)
    self.assertEqual(output.shape.as_list(), [None] + output_dims)

  @parameterized.named_parameters(
      ("key_value_same_proj", None, None, [40, 80]),
      ("key_value_different_proj", 32, 60, [40, 60]),
  )
  @test_util.run_gpu_only
  def test_load_weights_and_compare_results(self, value_dim, output_shape, output_dims):
    batch_size = 2
    num_heads = 12
    key_dim = 64
    ref_layer = keras.layers.multi_head_attention.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        output_shape=output_shape)
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        output_shape=output_shape)
    query = np.random.random_sample((batch_size, 40, 80))
    value = np.random.random_sample((batch_size, 20, 80))
    ref_output = ref_layer(query=query, value=value)
    output = test_layer(query=query, value=value)
    self.assertNotAllClose(ref_output, output)

    test_layer.set_weights(ref_layer.get_weights())
    self.assertAllClose(
        ref_output, test_layer(query=query, value=value), atol=1e-2)

  @test_util.run_gpu_only
  def test_self_attention_keras_fit(self):
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=12, key_dim=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    batch_size = 3
    epochs = 5
    query = keras.Input(shape=(40, 80))
    value = keras.Input(shape=(40, 80))
    output = test_layer(query=query, value=value)

    # Create a model containing the test layer.
    model = keras.Model([query, value], output)

    # Generate data for the input (non-mask) tensors.
    from_data = np.random.random_sample((batch_size, 40, 80)).astype(np.float16)
    to_data = np.random.random_sample((batch_size, 40, 80)).astype(np.float16)
    model.compile(loss='mse',
                  optimizer=RMSprop(learning_rate=0.001))
    model.fit(x=[from_data, to_data],
              y=np.random.random_sample((batch_size, 40, 80)).astype(np.float16),
              epochs=epochs, batch_size=batch_size)
    # Invoke the data with a random set of mask data. This should mask at least
    # one element.
    out = model.predict([from_data, to_data])
    self.assertEqual(list(out.shape), [batch_size] + [40, 80])
  
  @test_util.run_in_graph_and_eager_modes
  @test_util.run_gpu_only
  def testExceptionThrowing(self):
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=2, key_dim=2, attention_axes=(2,) )
    batch_size = 32
    query = keras.backend.ones(shape=(batch_size, 4, 8))
    value = keras.backend.ones(shape=(batch_size, 2, 8))
    mask_data = np.random.randint(2, size=(batch_size, 4, 2))

    with self.assertRaisesRegex(
        ValueError, 'Masking is not supported for CuDNN MHA.'):
      output = test_layer(query, value, attention_mask=mask_data)
   
    with self.assertRaisesRegex(
        ValueError,
        'Returning attention scores is not supported for CuDNN MHA.'):
      output = test_layer(query, value, return_attention_scores=True)

  @parameterized.named_parameters(("with_bias", True), ("no_bias", False))
  def test_attention_with_bias(self, use_bias):
    """Test with a mask tensor."""
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=2, key_dim=2, use_bias=use_bias)
    # Create a 3-dimensional input (the first dimension is implicit).
    batch_size = 3
    query = keras.Input(shape=(4, 8))
    value = keras.Input(shape=(2, 8))
    output = test_layer(query=query, value=value)
    model = keras.Model([query, value], output)

    if use_bias:
      self.assertLen(test_layer.trainable_variables, 8)
    else:
      self.assertLen(test_layer.trainable_variables, 4)
  
  @test_util.run_gpu_only
  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=12,
        key_dim=64,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02))
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])
    
  @parameterized.named_parameters(
      ("4d_inputs", [3, 4], [3, 2]),
      ("5D_inputs", [5, 3, 4], [5, 3, 2]))
  @test_util.run_gpu_only
  def test_high_dim_attention(self, q_dims, v_dims):
    """Test with a mask tensor."""
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=2, key_dim=2)
    batch_size, hidden_size = 3, 8
    # Generate data for the input (non-mask) tensors.
    query_shape = [batch_size] + q_dims + [hidden_size]
    value_shape = [batch_size] + v_dims + [hidden_size]
    query = np.random.random_sample(query_shape)
    value = np.random.random_sample(value_shape)

    value_tensor = keras.Input(value_shape[1:], name="value")
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "MHA input must be a 3-D tensor"):
      output = test_layer(query=query, value=value)

  @test_util.run_gpu_only
  def test_dropout(self):
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=2, key_dim=2, dropout=0.5)

    # Generate data for the input (non-mask) tensors.
    from_data = keras.backend.ones(shape=(32, 4, 8)) #, dtype=dtypes.float64)
    to_data = keras.backend.ones(shape=(32, 2, 8)) #, dtype=dtypes.float64)
    train_out = test_layer(from_data, to_data, None, True)
    test_out = test_layer(from_data, to_data, None, False)

    # Output should be close when not in training mode,
    # and should not be close when enabling dropout in training mode.
    self.assertNotAllClose(
        keras.backend.eval(train_out),
        keras.backend.eval(test_out))

  @test_util.run_gpu_only
  def test_residual_link(self):
    test_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=2, key_dim=2)

    # Generate data for the input (non-mask) tensors.
    from_data = keras.backend.ones(shape=(32, 4, 8))
    to_data = keras.backend.ones(shape=(32, 2, 8))
    test_out_no_reslink = test_layer(from_data, to_data, None, True, False)
    test_out_with_reslink = test_layer(from_data, to_data, None, True, True)

    # Output should be close when not in training mode,
    # and should not be close when enabling dropout in training mode.
    self.assertNotAllClose(
        keras.backend.eval(test_out_no_reslink),
        keras.backend.eval(test_out_with_reslink))


class SubclassAttention(cudnn_multi_head_attention.CuDNNMHA):

  def _build_attention(self, qkv_rank):
    pass

  def _compute_attention(self,
                         query_tensor,
                         key_tensor,
                         value_tensor,
                         attention_mask=None,
                         training=None):
    return value_tensor, None


@keras_parameterized.run_all_keras_modes
class AttentionSubclassTest(keras_parameterized.TestCase):
  @test_util.run_gpu_only
  def test_initializer(self):
    """Test with a specified initializer."""
    test_layer = SubclassAttention(num_heads=12, key_dim=64)
    # Create a 3-dimensional input (the first dimension is implicit).
    query = keras.Input(shape=(40, 80))
    output = test_layer(query, query)
    self.assertEqual(output.shape.as_list(), [None, 40, 80])


class TestModel(keras.Model):

  def __init__(self):
    super(TestModel, self).__init__()
    self.attention = keras.layers.CuDNNMHA(
        num_heads=3,
        key_dim=4,
        value_dim=4,
        use_bias=True,
        dropout=0.0,
        output_shape=[12])

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def get_config(self):
    return {}

  def call(self, x, training=False):
    return self.attention(x, x, training=training)


@keras_parameterized.run_all_keras_modes(always_skip_v1=True)
class KerasModelSavingTest(keras_parameterized.TestCase):

  @test_util.run_gpu_only
  def test_keras_saving_subclass(self):
    model = TestModel()
    query = keras.Input(shape=(40, 80))
    _ = model(query)
    model_path = self.get_temp_dir() + "/tmp_model"
    keras.models.save_model(model, model_path, save_format="tf")
    reloaded_model = keras.models.load_model(model_path)
    self.assertEqual(
        len(model.trainable_variables), len(reloaded_model.trainable_variables))
    for src_v, loaded_v in zip(model.trainable_variables,
                               reloaded_model.trainable_variables):
      self.assertAllEqual(src_v, loaded_v)

  @parameterized.parameters("h5", "tf")
  @test_util.run_gpu_only
  def test_keras_saving_functional(self, save_format):
    model = TestModel()
    query = keras.Input(shape=(40, 80))
    output = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=3,
        key_dim=4,
        value_dim=4,
        use_bias=True,
        dropout=0.0)(query, query)
    model = keras.Model(inputs=query, outputs=output)
    model_path = self.get_temp_dir() + "/tmp_model"
    keras.models.save_model(model, model_path, save_format=save_format)
    reloaded_model = keras.models.load_model(
        model_path,
        custom_objects={"CuDNNMHA":cudnn_multi_head_attention.CuDNNMHA})
    self.assertEqual(
        len(model.trainable_variables), len(reloaded_model.trainable_variables))
    for src_v, loaded_v in zip(model.trainable_variables,
                               reloaded_model.trainable_variables):
      self.assertAllEqual(src_v, loaded_v)

  @test_util.run_gpu_only
  def test_create_without_build(self):
    not_intialized_layer = cudnn_multi_head_attention.CuDNNMHA(
        num_heads=3, key_dim=4, value_dim=4)
    cudnn_multi_head_attention.CuDNNMHA.from_config(
        not_intialized_layer.get_config())


if __name__ == "__main__":
  test.main()
