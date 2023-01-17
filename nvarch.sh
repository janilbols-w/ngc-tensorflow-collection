#!/bin/bash

USE_DOT=1
ARCH=""

while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--nodot" ]]; then
    USE_DOT=0
  elif [[ "$1" == "--dot" ]]; then
    USE_DOT=1
  elif [[ -z "$ARCH" ]]; then
    ARCH="$1"
  else
    echo Invalid argument $1
    exit 1
  fi
  shift
done

if [[ "${ARCH}" == "amd64" ]]; then
  DOT_LIST="5.2,6.0,6.1,7.0,7.5,8.0,8.6,9.0"
elif [[ "${ARCH}" == "arm64" ]]; then
  DOT_LIST="5.3,6.2,7.0,7.2,7.5,8.0,8.6,8.7,9.0"
else
  echo "Invalid arch '$ARCH' (expected 'amd64' or 'arm64')" 1>&2
  exit 1
fi

if [[ $USE_DOT -eq 0 ]]; then
  echo "${DOT_LIST//[.]/}"
else
  echo "${DOT_LIST}"
fi

