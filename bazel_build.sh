#!/bin/bash
# Build the components of tensorflow that require Bazel


# Inputs:
# OUTPUT_DIRS - String of space-delimited directories to store outputs, in order of:
#     1)tensorflow whl
#     2)other lib.so outputs
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
WHL_OUT=${OUTPUT_LIST[0]}
LIBS_OUT=${OUTPUT_LIST[1]}

for d in ${OUTPUT_LIST[@]}
do
  mkdir -p ${d}
done

echo "TARGETARCH: ${TARGETARCH}"

BAZEL_BUILD_RETURN=0
if [[ "$TF_API" == "2" ]]; then
  BAZEL_OPTS="--config=v2 $(cat $BUILD_OPTS) $BAZEL_CACHE_FLAG"
else
  BAZEL_OPTS="--config=v1 $(cat $BUILD_OPTS) $BAZEL_CACHE_FLAG"
fi

echo "BAZEL_OPTS: $BAZEL_OPTS"

SCRIPT_DIR=$(pwd)

if [[ $MANYLINUX_BUILD_STAGE -eq 1 ]]; then
  # SKIP BUILD (assumes prebuilt wheel from manylinux stage)
  if [[ $SKIPBUILD -eq 1 ]]; then
    cd ${SCRIPT_DIR}  # move to dl/tf/tf directory
    pip$PYVER install --no-cache-dir --no-deps $WHL_OUT/tensorflow-*.whl
    PIP_INSTALL_RETURN=$?
    if [ ${PIP_INSTALL_RETURN} -gt 0 ]; then
      echo "Installation of TF pip package failed."
      exit ${PIP_INSTALL_RETURN}
    fi

    pip$PYVER check
    if [[ $? -gt 0 ]]; then
      echo "Dependency check failed."
      exit 1
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
    exit
  fi
fi
# DO BUILD

# install `psutil` & `pytz` if needed & start a timer before bazel build action
pip$PYVER install --no-cache-dir psutil pytz

cd ${THIS_DIR}    # move to dl/dgx/tf directory
export BUILD_START_TIMESTAMP=$(
    python -c "import bazel_build_data_collector as script; \
    print(script._get_datetime_now(as_string=True))"
)
echo "BUILD_START_TIMESTAMP: ${BUILD_START_TIMESTAMP}"
echo "BAZEL OPTS: ${BAZEL_OPTS}"
echo "GCC VERSION: $(gcc --version)"

cd ${SCRIPT_DIR}  # move to dl/tf/tf directory

if [[ $IN_CONTAINER -eq 1 ]]; then
  bazel build $BAZEL_OPTS \
      tensorflow/tools/pip_package:build_pip_package \
      //tensorflow:libtensorflow_cc.so
  BAZEL_BUILD_RETURN=$?
  cp bazel-bin/tensorflow/libtensorflow_cc.so.? ${LIBS_OUT}

else
  bazel build $BAZEL_OPTS \
      tensorflow/tools/pip_package:build_pip_package
  BAZEL_BUILD_RETURN=$?
fi

# Push data to the log collection server
cd ${THIS_DIR}    # move to dl/dgx/tf directory
python$PYVER bazel_build_data_collector.py || true
cd ${SCRIPT_DIR}  # move to dl/tf/tf directory


if [ ${BAZEL_BUILD_RETURN} -gt 0 ]
then
  exit ${BAZEL_BUILD_RETURN}
fi

bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHL_OUT --gpu --project_name tensorflow
PIP_PACKAGE_RETURN=$?
if [ ${PIP_PACKAGE_RETURN} -gt 0 ]; then
  echo "Assembly of TF pip package failed."
  exit ${PIP_PACKAGE_RETURN}
fi

if [[ $MANYLINUX_BUILD_STAGE -eq 1 ]]; then
  bazel-bin/tensorflow/tools/pip_package/build_pip_package $WHL_OUT \
      --gpu --project_name nvidia_tensorflow --build_number $CI_PIPELINE_ID
  PIP_PACKAGE_RETURN=$?
  if [ ${PIP_PACKAGE_RETURN} -gt 0 ]; then
    echo "Assembly of standalone TF pip package failed."
    exit ${PIP_PACKAGE_RETURN}
  fi
  bazel clean --expunge
  rm .tf_configure.bazelrc
  rm -rf ${HOME}/.cache/bazel
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
  exit 0
fi

pip$PYVER install --no-cache-dir --no-deps $WHL_OUT/tensorflow-*.whl
PIP_INSTALL_RETURN=$?
if [ ${PIP_INSTALL_RETURN} -gt 0 ]; then
  echo "Installation of TF pip package failed."
  exit ${PIP_INSTALL_RETURN}
fi

pip$PYVER check
if [[ $? -gt 0 ]]; then
  echo "Dependency check failed."
  exit 1
fi

if [[ $POSTCLEAN -eq 1 ]]; then
  rm -f $WHL_OUT/tensorflow-*.whl
  bazel clean --expunge
  rm .tf_configure.bazelrc
  rm -rf ${HOME}/.cache/bazel /tmp/*
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
    rm /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
fi
