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
  echo "    --clean                        Delete local configuration and bazel cache"
  echo "    --post_clean                   Delete intermediate build files"
  echo "    --v1                           Build TensorFlow v1 API"
  echo "    --v2                           Build TensorFlow v2 API"
  echo "    --bazel-cache                  Use Bazel build cache"
  echo "    --bazel-cache-download-only    Use Bazel build cache in download mode only. No cache upload"
  echo "    --sm SM1,SM2,...               The SM to use to compile TF"
  echo "    --sm local                     Query the SM of available GPUs. This is the default behavior."
  echo "    --ccache                       Use ccache"
  echo "    --manylinux_build              Build .so and whl in manylinux"
  echo "    --copy_from_manylinux          Copy over prebuilt .so and whl from manylinux, set env vars, and install whl"
}

PYVER=3.8
CONFIGONLY=0
CLEAN=0
POSTCLEAN=0
TF_API=2
BAZEL_CACHE=0
BAZEL_CACHE_NOUPLOAD=0
TF_USE_CCACHE=0
SKIPBUILD=0
MANYLINUX=0
MAX_BUILD_JOBS=-1

if [[ -z "${TARGETARCH}" ]]; then
  ARCH=$(uname -m)
  if [[ "$ARCH" == "x86_64" ]]; then
    TARGETARCH="amd64"
  elif [[ "$ARCH" == "aarch64" ]]; then
    TARGETARCH="arm64"
    # Avoid ABORT failures on SBSA builders due to insufficeint memory.
    N_CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
    [[ $N_CPU_CORES -gt 40 ]] && MAX_BUILD_JOBS=40 || MAX_BUILD_JOBS=$N_CPU_CORES
  else
    echo Unknown arch $ARCH
    exit 1
  fi
fi

export TF_CUDA_COMPUTE_CAPABILITIES="local"

while [[ $# -gt 0 ]]; do
  case $1 in
    "--help"|"-h")  Usage; exit 1 ;;
    "--python3.8")  PYVER=3.8 ;;
    "--python3.7")   PYVER=3.7 ;;
    "--python3.6")  PYVER=3.6 ;;
    "--configonly") CONFIGONLY=1 ;;
    "--clean")      CLEAN=1 ;;
    "--post_clean") POSTCLEAN=1 ;;
    "--manylinux_build")  MANYLINUX=1 ;;
    "--copy_from_manylinux")  SKIPBUILD=1 ;;
    "--v1")         TF_API=1 ;;
    "--v2")         TF_API=2 ;;
    "--bazel-cache") BAZEL_CACHE=1 ;;
    "--bazel-cache-download-only") BAZEL_CACHE_NOUPLOAD=1 ;;
    "--ccache")     TF_USE_CCACHE=1 ;;
    "--sm")         shift 1;
                    TF_CUDA_COMPUTE_CAPABILITIES=$1
                    ;;
    *)
      echo UNKNOWN OPTION $1
      echo Run $0 -h for help
      exit 1
  esac
  shift 1
done

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${THIS_DIR}/tensorflow-source"

if [[ "$TF_CUDA_COMPUTE_CAPABILITIES" == "all" ]]; then
  TF_CUDA_COMPUTE_CAPABILITIES="$(${THIS_DIR}/nvarch.sh ${TARGETARCH})"
  if [[ $? -ne 0 ]]; then exit 1; fi
elif [ "$TF_CUDA_COMPUTE_CAPABILITIES" == "local" ]; then
echo DISCOVERING LOCAL COMPUTE CAPABILITIES
set +e # Allow errors so that a.out can be cleaned up
TF_CUDA_COMPUTE_CAPABILITIES=$( \
cat <<EOF | nvcc -x c++ --run -
#include <stdio.h>
#include <string>
#include <set>
#include <cuda_runtime.h>
#define CK(cmd) do {                    \
  cudaError_t r = (cmd);                \
  if (r != cudaSuccess) {               \
    fprintf(stderr,                     \
            "CUDA Runtime error: %s\n", \
            cudaGetErrorString(r));     \
    exit(EXIT_FAILURE);                 \
  }                                     \
 } while (false)
using namespace std;
int main() {
  int device_count;
  CK(cudaGetDeviceCount(&device_count));
  set<string> set;
  for(int i=0; i<device_count; i++) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, i));
    set.insert(to_string(prop.major)+"."+to_string(prop.minor));
  }
  int nb_printed = 0;
  for(string sm: set) {
    if (nb_printed > 0) printf(",");
    printf("%s", sm.data());
    ++nb_printed;
  }
  printf("\n");
}
EOF
)
R=$?
rm a.out
if [[ "$R" -ne 0 ]]; then
  exit 1
fi
set -e
fi

echo "CUDA COMPUTE: ${TF_CUDA_COMPUTE_CAPABILITIES}"

if [[ $CLEAN -eq 1 ]]; then
    echo "Clearing build artifacts and cache... Building from scratch"
  	rm -rf /root/.cache/bazel/*
  	rm -f .tf_configure.bazelrc
  	rm -f bazel-*
fi


export TF_NEED_CUDA=1
export TF_NEED_CUTENSOR=1
export TF_NEED_TENSORRT=1
export TF_CUDA_PATHS=/usr,/usr/local/cuda
export TF_CUDA_VERSION=$(echo "${CUDA_VERSION}" | cut -d . -f 1-2)
export TF_CUBLAS_VERSION=$(echo "${CUBLAS_VERSION}" | cut -d . -f 1)
export TF_CUDNN_VERSION=$(echo "${CUDNN_VERSION}" | cut -d . -f 1)
export TF_NCCL_VERSION=$(echo "${NCCL_VERSION}" | cut -d . -f 1)
export TF_TENSORRT_VERSION=$(echo "${TRT_VERSION}" | cut -d . -f 1)
export TF_ENABLE_XLA=1
export TF_NEED_HDFS=0
if [ "${TARGETARCH}" = "amd64" ] ; then export CC_OPT_FLAGS="-march=sandybridge -mtune=broadwell" ; fi
if [ "${TARGETARCH}" = "arm64" ] ; then export CC_OPT_FLAGS="-march=armv8-a" ; fi
export TF_USE_CCACHE
export MAX_BUILD_JOBS

# This is the flags that Google use internally with clang.
# Adding them help make the compilation error more readable even if g++ doesn't do exactly as clang.
# Google use -Werror too. But we can't use it as the error from g++ isn't exactly the same and will trigger useless compilation error.
export CC_OPT_FLAGS="$CC_OPT_FLAGS -Wno-address-of-packed-member -Wno-defaulted-function-deleted -Wno-enum-compare-switch -Wno-expansion-to-defined -Wno-ignored-attributes -Wno-ignored-qualifiers -Wno-inconsistent-missing-override -Wno-int-in-bool-context -Wno-misleading-indentation -Wno-potentially-evaluated-expression -Wno-psabi -Wno-range-loop-analysis -Wno-return-std-move -Wno-sizeof-pointer-div -Wno-sizeof-array-div -Wno-string-concatenation -Wno-tautological-constant-compare -Wno-tautological-type-limit-compare -Wno-tautological-undefined-compare -Wno-tautological-unsigned-zero-compare -Wno-tautological-unsigned-enum-zero-compare -Wno-undefined-func-template -Wno-unused-lambda-capture -Wno-unused-local-typedef -Wno-void-pointer-to-int-cast -Wno-uninitialized-const-reference -Wno-compound-token-split -Wno-ambiguous-member-template -Wno-char-subscripts -Wno-error=deprecated-declarations -Wno-extern-c-compat -Wno-gnu-alignof-expression -Wno-gnu-variable-sized-type-not-at-end -Wno-implicit-int-float-conversion -Wno-invalid-source-encoding -Wno-mismatched-tags -Wno-pointer-sign -Wno-private-header -Wno-sign-compare -Wno-signed-unsigned-wchar -Wno-strict-overflow -Wno-trigraphs -Wno-unknown-pragmas -Wno-unused-const-variable -Wno-unused-function -Wno-unused-private-field -Wno-user-defined-warnings -Wvla -Wno-reserved-user-defined-literal -Wno-return-type-c-linkage -Wno-self-assign-overloaded -Woverloaded-virtual -Wnon-virtual-dtor -Wno-deprecated -Wno-invalid-offsetof -Wimplicit-fallthrough -Wno-final-dtor-non-final-class -Wno-c++20-designator -Wno-register -Wno-dynamic-exception-spec"

# Compare installed and expected bazel versions and re-install bazel as needed.
BAZEL_INSTALLED=$(bazel --version 2>/dev/null | cut -d' ' -f2)
BAZEL_EXPECTED=$(cat .bazelversion 2>/dev/null || true)

if [[ -n "$BAZEL_EXPECTED" && "$BAZEL_INSTALLED" != "$BAZEL_EXPECTED" ]]; then
  set +e # Temporariliy allow errors so that we can cleanup the bazel tmp dir.
  echo Re-installing bazel $BAZEL_EXPECTED.
  mkdir bazel-setup.tmp
  if [[ $? -ne 0 ]]; then
    exit 1
  fi
  cd bazel-setup.tmp
  bazel_cleanup() {
    cd ..
    rm -rf bazel-setup.tmp
  }
  if [[ "$TARGETARCH" == "amd64" ]]; then
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_EXPECTED/bazel-$BAZEL_EXPECTED-installer-linux-x86_64.sh
    curl -fSsL -o LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE
    bash ./bazel-$BAZEL_EXPECTED-installer-linux-x86_64.sh
    if [[ $? -ne 0 ]]; then
      echo "Failed to setup bazel."
      bazel_cleanup
      exit 1
    fi
  elif [[ "$TARGETARCH" == "arm64" ]]; then
    curl -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_EXPECTED/bazel-$BAZEL_EXPECTED-linux-arm64
    mv bazel-$BAZEL_EXPECTED-linux-arm64 bazel
    chmod 0755 bazel
    if [[ "$(./bazel --version | cut -d' ' -f2)" != "$BAZEL_EXPECTED" ]]; then
      echo Failed to download appropriate bazel binary.
      bazel_cleanup
      exit 1
    fi
    install bazel /usr/local/bin/
    if [[ $? -ne 0 ]]; then
      echo "Failed to install bazel."
      bazel_cleanup
      exit 1
    fi
  else
    echo "Unexpected TARGETARCH: $TARGETARCH"
    bazel_cleanup
    exit 1
  fi
  bazel_cleanup
  echo "Successfully installed bazel $(bazel --version | cut -d' ' -f2)"
  set -e
else
  echo Using bazel $BAZEL_INSTALLED
fi

export PYTHON_BIN_PATH=$(which python$PYVER)
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  echo Failed to locate python$PYVER
  exit 1
fi

LIBCUDA_FOUND=$(ldconfig -p | awk '{print $1}' | grep libcuda.so | wc -l)

NEED_CONFIGURE=0
if [[ ! -f ".tf_configure.bazelrc" ]]; then
  NEED_CONFIGURE=1
else
  PYTHON_IN_CONFIG=$(grep PYTHON_BIN_PATH .tf_configure.bazelrc | grep -o '".*"' | tr -d '"')
  if [ "$PYTHON_IN_CONFIG" != "$PYTHON_BIN_PATH" ]; then
    echo Reconfiguring for changed python version.
    NEED_CONFIGURE=1
  fi
  SMS_IN_CONFIG=$(grep TF_CUDA_COMPUTE_CAPABILITIES .tf_configure.bazelrc | grep -o '".*"' | tr -d '"')
  if [ "$SMS_IN_CONFIG" != "$TF_CUDA_COMPUTE_CAPABILITIES" ]; then
    echo Reconfiguring for changed CUDA ARCH list.
    NEED_CONFIGURE=1
  fi
fi
if [[ "$NEED_CONFIGURE" -eq 1 ]]; then
  echo "Generating a fresh bazel configuration ..."

  if [[ "$LIBCUDA_FOUND" -eq 0 ]]; then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
      ln -fs /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
  fi
  yes "" | ./configure
fi

if ! grep -Fxq "startup --max_idle_secs=0" .tf_configure.bazelrc; then
  echo "$(printf '\n'; cat .tf_configure.bazelrc)" > .tf_configure.bazelrc
  echo "$(printf 'startup --max_idle_secs=0\n'; cat .tf_configure.bazelrc)" > .tf_configure.bazelrc
  echo "$(printf '# Prevent cache from being invalidated after 3 hours\n'; cat .tf_configure.bazelrc)" > .tf_configure.bazelrc
fi

if [[ $CONFIGONLY -eq 1 ]]; then
  exit 0
fi

# Uninstall any previous TF version
pip$PYVER uninstall -y tensorflow

export OUTPUT_DIRS="/tmp/pip /usr/local/lib/tensorflow"
export BUILD_OPTS="${THIS_DIR}/nvbuildopts"
export IN_CONTAINER="1"
export POSTCLEAN
export PYVER
export LIBCUDA_FOUND
export TF_API
export SKIPBUILD

unset BAZEL_CACHE_FLAG
if [[ $BAZEL_CACHE -eq 1 || $BAZEL_CACHE_NOUPLOAD -eq 1 ]]; then
  export BAZEL_CACHE_FLAG="$(${THIS_DIR}/nvbazelcache)"
fi
if [[ $BAZEL_CACHE_NOUPLOAD -eq 1 && ! -z "$BAZEL_CACHE_FLAG" ]]; then
  export BAZEL_CACHE_FLAG="$BAZEL_CACHE_FLAG --remote_upload_local_results=false"
fi
echo "Bazel Cache Flag: $BAZEL_CACHE_FLAG"

if [[ $SKIPBUILD -eq 1 || $MANYLINUX -eq 1 ]]; then
  export MANYLINUX_BUILD_STAGE=1
fi

bash ${THIS_DIR}/bazel_build.sh
