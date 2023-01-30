#!/bin/bash
#
# Configure, build, and install Tensorflow
#

# Exit at error
set -e

Usage() {
  echo "Configure, build, and install Tensorflow."
  echo ""
  echo "  Usage: $0 [OPTIONS]"
  echo ""
  echo "    OPTIONS                        DESCRIPTION"
  echo "    --python2.7                    Build python2.7 package (default)"
  echo "    --python3.6                    Build python3.6 package"
  echo "    --configonly                   Run configure step only"
  echo "    --noconfig                     Skip configure step"
  echo "    --noclean                      Retain intermediate build files"
  echo "    --testlist                     Build list of python kernel_tests"
  echo "    --triton                       Build TRITON-specific library"
  echo "    --v1                           Build TensorFlow v1 API"
  echo "    --v2                           Build TensorFlow v2 API"
  echo "    --bazel-cache                  Use Bazel build cache"
  echo "    --bazel-cache-download-only    Use Bazel build cache in download mode only. No cache upload"
}

PYVER=3.6
CONFIGONLY=0
NOCONFIG=0
NOCLEAN=0
TESTLIST=0
TF_API=2
BAZEL_CACHE=0
BAZEL_CACHE_NOUPLOAD=0

while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")  Usage; exit 1 ;;
    "--python2.7")  PYVER=2.7 ;;
    "--python3.6")  PYVER=3.6 ;;
    "--configonly") CONFIGONLY=1 ;;
    "--noconfig")   NOCONFIG=1 ;;
    "--noclean")    NOCLEAN=1 ;;
    "--testlist")   TESTLIST=1 ;;
    "--triton")     TRITON=1 ;;
    "--v1")         TF_API=1 ;;
    "--v2")         TF_API=2 ;;
    "--bazel-cache") BAZEL_CACHE=1 ;;
    "--bazel-cache-download-only") BAZEL_CACHE_NOUPLOAD=1 ;;
    *)
      echo UNKNOWN OPTION $1
      echo Run $0 -h for help
      exit 1
  esac
  shift 1
done

export TF_NEED_CUDA=1
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDA_VERSION=$(echo "${CUDA_VERSION}" | cut -d . -f 1-2)
export TF_CUBLAS_VERSION=$(echo "${CUBLAS_VERSION}" | cut -d . -f 1)
export TF_CUDNN_VERSION=$(echo "${CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)
export TF_TENSORRT_VERSION=$(echo "${TRT_VERSION}" | cut -d . -f 1)
export TF_CUDA_COMPUTE_CAPABILITIES="5.2,6.0,6.1,7.0,7.5,8.0"
export TF_ENABLE_XLA=1
export TF_NEED_HDFS=0
export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell -Wno-sign-compare"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${THIS_DIR}/tensorflow-source"
export PYTHON_BIN_PATH=/usr/bin/python$PYVER
LIBCUDA_FOUND=$(ldconfig -p | awk '{print $1}' | grep libcuda.so | wc -l)
if [[ $NOCONFIG -eq 0 ]]; then
  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
      ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
  yes "" | ./configure
fi

if [[ $CONFIGONLY -eq 1 ]]; then
  exit 0
fi

unset BAZEL_CACHE_FLAG
if [[ $BAZEL_CACHE -eq 1 ]]; then
  export BAZEL_CACHE_FLAG="$(cat ${THIS_DIR}/nvbazelcache)"
fi
if [[ $BAZEL_CACHE_NOUPLOAD -eq 1 ]]; then
  export BAZEL_CACHE_FLAG="$(cat ${THIS_DIR}/nvbazelcache) --remote_upload_local_results=false"

fi

if [[ ! -z "$BAZEL_CACHE_FLAG" ]]; then
  BAZEL_CACHE_ADDR="$(cut -d '=' -f2 <<< $(cat ${THIS_DIR}/nvbazelcache))/cas/0000000000000000000000000000000000000000000000000000000000000000"
  BAZEL_CACHE_RESPONSE=$(curl --silent --max-time 10 --connect-timeout 10 ${BAZEL_CACHE_ADDR})
  if [[ "Not found" != "${BAZEL_CACHE_RESPONSE}" ]]; then
    export BAZEL_CACHE_FLAG=""
  fi;
fi;
echo "Bazel Cache Flag: $BAZEL_CACHE_FLAG"

export OUTPUT_DIRS="tensorflow/python/kernel_tests tensorflow/compiler/tests /tmp/pip"
export BUILD_OPTS="${THIS_DIR}/nvbuildopts"
export IN_CONTAINER="1"
export TESTLIST
export TRITON
export NOCLEAN
export PYVER
export LIBCUDA_FOUND
export TF_API
bash ${THIS_DIR}/bazel_build.sh
