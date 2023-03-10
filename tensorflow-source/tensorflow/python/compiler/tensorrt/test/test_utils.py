# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities to test TF-TensorRT integration."""

import os

from contextlib import contextmanager
from functools import wraps

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test


def disable_tf32_testing(_class):
  """Force Tensorflow to use FP32 instead of TensorFloat-32 computation.

  This helper decorator helps to avoid some false negative tests due to
  TF32 requiring sometimes slightly higher tolerances to pass."""

  if not issubclass(_class, trt_test.TfTrtIntegrationTestBase):
    raise ValueError("Can only decorate a `TfTrtIntegrationTestBase` test.")

  @wraps(_class.ShouldAllowTF32Computation)
  def _should_allow_tf32_testing(self, *args ,**kwargs):
    return False

  _class.ShouldAllowTF32Computation = _should_allow_tf32_testing
  return _class


@contextmanager
def experimental_feature_scope(feature_name):
  """Creates a context manager to enable the given experimental feature.

  This helper function creates a context manager setting up an experimental
  feature temporarily.

  Example:

  ```python
  with self._experimental_feature_scope("feature_1"):
    do_smthg()
  ```

  Args:
    feature_name: Name of the feature being tested for activation.
  """

  env_varname = "TF_TRT_EXPERIMENTAL_FEATURES"
  env_value_bckp = os.environ.get(env_varname, default="")

  exp_features = env_value_bckp.split(",")
  os.environ[env_varname] = ",".join(list(set(exp_features + [feature_name])))

  try:
    yield
  finally:
    os.environ[env_varname] = env_value_bckp
