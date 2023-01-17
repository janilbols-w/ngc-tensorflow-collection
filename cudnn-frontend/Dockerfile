# https://hub.docker.com/r/nvidia/cuda
# https://gitlab.com/nvidia/container-images/cuda/blob/master/dist/11.2.1/ubuntu18.04-x86_64/devel/Dockerfile

ARG IMAGE_NAME="nvidia/cuda"
FROM ${IMAGE_NAME}:11.6.0-devel-ubuntu18.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# https://developer.nvidia.com/rdp/cudnn-download#
# https://developer.nvidia.com/compute/cudnn/secure/8.3.1/local_installers/11.5/cudnn-local-repo-ubuntu1804-8.3.1.22_1.0-1_amd64.deb

ENV CUDNN_VERSION 8.4.0.27

LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=$CUDNN_VERSION-1+cuda11.6 \
    libcudnn8-dev=$CUDNN_VERSION-1+cuda11.6 \
    && apt-mark hold libcudnn8 && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && \
    apt-get update && \ 
    apt-get install -y cmake

COPY . .

RUN mkdir build 
WORKDIR /build
RUN cmake ..
RUN cmake --build .

RUN pwd
