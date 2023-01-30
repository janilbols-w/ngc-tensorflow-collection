#!/bin/bash
# Build the components of tensorflow that require Bazel


# Inputs: 
# OUTPUT_DIRS - String of space-delimited directories to store outputs, in order of:
#     1)kernel test list
#     2)xla test list
#     3)tensorflow whl
# TESTLIST - Determines whether the test lists are built (1 to build, 0 to skip)
# TRITON - Determines whether the TRITON-specific libary is built (1 to build, 0 to skip)
# NOCLEAN - Determines whether bazel clean is run and the tensorflow whl is
#     removed after the build and install (0 to clean, 1 to skip)
# PYVER - The version of python
# BUILD_OPTS - File containing desired bazel flags for building tensorflow
# BAZEL_CACHE_FLAG - flag to add to BUILD_OPTS to enable bazel cache
# LIBCUDA_FOUND - Determines whether a libcuda stub was created and needs to be cleaned (0 to clean, 1 to skip)
# IN_CONTAINER - Flag for whether Tensorflow is being built within a container (1 for yes, 0 for bare-metal)
# TF_API - TensorFlow API version: 1 => v1.x, 2 => 2.x

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"


read -ra OUTPUT_LIST <<<"$OUTPUT_DIRS"
KERNEL_OUT=${OUTPUT_LIST[0]}
XLA_OUT=${OUTPUT_LIST[1]}
WHL_OUT=${OUTPUT_LIST[2]}
LIBS_OUT=${OUTPUT_LIST[3]}

for d in ${OUTPUT_LIST[@]}
do
  mkdir -p ${d}
done

KERNEL_TEST_RETURN=0
BAZEL_BUILD_RETURN=0
if [[ "$TF_API" == "2" ]]; then
  BAZEL_OPTS="--config=v2 $(cat $BUILD_OPTS) $BAZEL_CACHE_FLAG"
else
  BAZEL_OPTS="--config=v1 $(cat $BUILD_OPTS) $BAZEL_CACHE_FLAG"
fi
echo "BAZEL_OPTS: $BAZEL_OPTS"

TRITON_TARGET=
if [[ $TRITON -eq 1 ]]; then
  TRITON_TARGET="//tensorflow:libtensorflow_triton.so"
fi

if [[ $IN_CONTAINER -eq 1 ]]; then

  bazel build $BAZEL_OPTS \
      tensorflow/tools/pip_package:build_pip_package \
      //tensorflow:libtensorflow_cc.so $TRITON_TARGET
  BAZEL_BUILD_RETURN=$?
  cp bazel-bin/tensorflow/libtensorflow_cc.so.? ${LIBS_OUT}

else
  bazel build $BAZEL_OPTS \
      tensorflow/tools/pip_package:build_pip_package ${TRITON_TARGET}
  BAZEL_BUILD_RETURN=$?
fi

if [[ $TRITON -eq 1 ]]; then
  cp bazel-bin/tensorflow/libtensorflow_triton.so.? ${LIBS_OUT}
fi

if [ ${BAZEL_BUILD_RETURN} -gt 0 ]
then
  exit ${BAZEL_BUILD_RETURN}
fi
# Build the test lists for L1 kernel and xla tests
if [[ $TESTLIST -eq 1 ]]; then
  rm -f "${KERNEL_OUT}/tests.list" \
        "${XLA_OUT}/tests.list"

  bazel test --verbose_failures --local_test_jobs=1 \
             --run_under="$THIS_DIR/test_grabber.sh $KERNEL_OUT" \
             --build_tests_only --test_tag_filters=-no_gpu,-benchmark-test \
             --cache_test_results=no $BAZEL_OPTS -- \
             //tensorflow/python/kernel_tests/... \
             `# The following tests are skipped because they depend on additional binaries.` \
             -//tensorflow/python/kernel_tests:ackermann_test \
             -//tensorflow/python/kernel_tests:duplicate_op_test \
             -//tensorflow/python/kernel_tests:invalid_op_test \
             -//tensorflow/python/kernel_tests/proto:encode_proto_op_test \
             -//tensorflow/python/kernel_tests/proto:decode_proto_op_test \
             -//tensorflow/python/kernel_tests/proto:descriptor_source_test \
             `# Enable the following when nvbug 3098132 is fixed.` \
             -//tensorflow/python/kernel_tests:conv_ops_3d_test \
             -//tensorflow/python/kernel_tests:conv_ops_3d_test_gpu \
             `# The following test is a CPU only test.` \
             -//tensorflow/python/kernel_tests/random:random_binomial_test
      KERNEL_TEST_RETURN=$?  
fi

bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHL_OUT --gpu --project_name tensorflow
PIP_PACKAGE_RETURN=$?
if [ ${PIP_PACKAGE_RETURN} -gt 0 ]; then
  echo "Assembly of TF pip package failed."
  exit ${PIP_PACKAGE_RETURN}
fi

pip$PYVER install --no-cache-dir --upgrade $WHL_OUT/tensorflow-*.whl
PIP_INSTALL_RETURN=$?
if [ ${PIP_INSTALL_RETURN} -gt 0 ]; then
  echo "Installation of TF pip package failed."
  exit ${PIP_INSTALL_RETURN}
fi

if [[ $NOCLEAN -eq 0 ]]; then
  rm -f $WHL_OUT/tensorflow-*.whl
  bazel clean --expunge
  rm .tf_configure.bazelrc
  rm -rf ${HOME}/.cache/bazel /tmp/*
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
fi

