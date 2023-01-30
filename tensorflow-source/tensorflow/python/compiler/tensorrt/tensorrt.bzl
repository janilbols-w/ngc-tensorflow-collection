"""Tags used for TF-TRT python module and tests."""

# buildifier: disable=same-origin-load
load("//tensorflow:tensorflow.bzl", "cuda_py_test")

TFTRT_PY_TEST_TAGS = [
    "no_cuda_on_cpu_tap",
    "no_rocm",
    "no_windows",
    "nomac",
]


TFTRT_PY_TEST_DEPS = [
    "//tensorflow/python:client_testlib",
    "//tensorflow/python:framework_test_lib",
    "//tensorflow/python/compiler/tensorrt:tf_trt_integration_test_base",
]


def tftrt_py_test(*args, **kwargs):
    """Helper function providing a common base for Python TF-TRT Unittests."""

    kwargs.setdefault("python_version", "PY3")
    kwargs.setdefault("xla_enable_strict_auto_jit", False)
    kwargs.setdefault("tags", TFTRT_PY_TEST_TAGS)
    kwargs.setdefault("deps", TFTRT_PY_TEST_DEPS)

    kwargs["deps"] = list(kwargs["deps"]) + kwargs.pop("extra_deps", [])
    kwargs["tags"] = list(kwargs["tags"]) + kwargs.pop("extra_tags", [])

    cuda_py_test(*args, **kwargs)
