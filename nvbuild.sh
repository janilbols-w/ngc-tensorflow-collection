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
  echo "    --python3.6                    Build python3.6 package"
  echo "    --python3.8                    Build python3.8 package (default)"
  echo "    --configonly                   Run configure step only"
  echo "    --noconfig                     Skip configure step"
  echo "    --noclean                      Retain intermediate build files"
  echo "    --triton                       Build TRITON-specific library"
  echo "    --v1                           Build TensorFlow v1 API"
  echo "    --v2                           Build TensorFlow v2 API"
  echo "    --bazel-cache                  Use Bazel build cache"
  echo "    --bazel-cache-download-only    Use Bazel build cache in download mode only. No cache upload"
  echo "    --ccache                       Use ccache"
}

PYVER=3.8
CONFIGONLY=0
NOCONFIG=0
NOCLEAN=0
TF_API=2
BAZEL_CACHE=0
BAZEL_CACHE_NOUPLOAD=0
TF_USE_CCACHE=0

while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")  Usage; exit 1 ;;
    "--python3.8")  PYVER=3.8 ;;
    "--python3.6")  PYVER=3.6 ;;
    "--configonly") CONFIGONLY=1 ;;
    "--noconfig")   NOCONFIG=1 ;;
    "--noclean")    NOCLEAN=1 ;;
    "--triton")     TRITON=1 ;;
    "--v1")         TF_API=1 ;;
    "--v2")         TF_API=2 ;;
    "--bazel-cache") BAZEL_CACHE=1 ;;
    "--bazel-cache-download-only") BAZEL_CACHE_NOUPLOAD=1 ;;
    "--ccache")     TF_USE_CCACHE=1 ;;
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
export TF_CUDA_COMPUTE_CAPABILITIES="5.2,6.0,6.1,7.0,7.5,8.0,8.6"
export TF_ENABLE_XLA=1
export TF_NEED_HDFS=0
export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell"
export TF_USE_CCACHE

# This is the flags that Google use internally with clang.
# Adding them help make the compilation error more readable even if g++ doesn't do exactly as clang.
# Google use -Werror too. But we can't use it as the error from g++ isn't exactly the same and will trigger useless compilation error.
export CC_OPT_FLAGS="$CC_OPT_FLAGS -Wno-address-of-packed-member -Wno-defaulted-function-deleted -Wno-enum-compare-switch -Wno-expansion-to-defined -Wno-ignored-attributes -Wno-ignored-qualifiers -Wno-inconsistent-missing-override -Wno-int-in-bool-context -Wno-misleading-indentation -Wno-potentially-evaluated-expression -Wno-psabi -Wno-range-loop-analysis -Wno-return-std-move -Wno-sizeof-pointer-div -Wno-sizeof-array-div -Wno-string-concatenation -Wno-tautological-constant-compare -Wno-tautological-type-limit-compare -Wno-tautological-undefined-compare -Wno-tautological-unsigned-zero-compare -Wno-tautological-unsigned-enum-zero-compare -Wno-undefined-func-template -Wno-unused-lambda-capture -Wno-unused-local-typedef -Wno-void-pointer-to-int-cast -Wno-uninitialized-const-reference -Wno-compound-token-split -Wno-ambiguous-member-template -Wno-char-subscripts -Wno-error=deprecated-declarations -Wno-extern-c-compat -Wno-gnu-alignof-expression -Wno-gnu-variable-sized-type-not-at-end -Wno-implicit-int-float-conversion -Wno-invalid-source-encoding -Wno-mismatched-tags -Wno-pointer-sign -Wno-private-header -Wno-sign-compare -Wno-signed-unsigned-wchar -Wno-strict-overflow -Wno-trigraphs -Wno-unknown-pragmas -Wno-unused-const-variable -Wno-unused-function -Wno-unused-private-field -Wno-user-defined-warnings -Wvla -Wno-reserved-user-defined-literal -Wno-return-type-c-linkage -Wno-self-assign-overloaded -Woverloaded-virtual -Wnon-virtual-dtor -Wno-deprecated -Wno-invalid-offsetof -Wimplicit-fallthrough -Wno-final-dtor-non-final-class -Wno-c++20-designator -Wno-register -Wno-dynamic-exception-spec"

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
if [[ $BAZEL_CACHE -eq 1 || $BAZEL_CACHE_NOUPLOAD -eq 1 ]]; then
  export BAZEL_CACHE_FLAG="$(${THIS_DIR}/nvbazelcache)"
fi
if [[ $BAZEL_CACHE_NOUPLOAD -eq 1 && ! -z "$BAZEL_CACHE_FLAG" ]]; then
  export BAZEL_CACHE_FLAG="$BAZEL_CACHE_FLAG --remote_upload_local_results=false"
fi
echo "Bazel Cache Flag: $BAZEL_CACHE_FLAG"

export OUTPUT_DIRS="/tmp/pip /usr/local/lib/tensorflow"
export BUILD_OPTS="${THIS_DIR}/nvbuildopts"
export IN_CONTAINER="1"
export TRITON
export NOCLEAN
export PYVER
export LIBCUDA_FOUND
export TF_API
bash ${THIS_DIR}/bazel_build.sh
