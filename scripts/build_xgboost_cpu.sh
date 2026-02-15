#!/usr/bin/env bash
set -euo pipefail

XGB_VER="v3.2.0"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP_DIR="${ROOT_DIR}/third_party/xgboost"
SRC_DIR="${TP_DIR}/src"
BUILD_DIR="${TP_DIR}/build_cpu"
INSTALL_DIR="${TP_DIR}/install_cpu"

mkdir -p "${TP_DIR}"
if [ ! -d "${SRC_DIR}" ]; then
  git clone --recursive https://github.com/dmlc/xgboost.git "${SRC_DIR}"
fi

cd "${SRC_DIR}"
git fetch --tags
git checkout "${XGB_VER}"
git submodule update --init --recursive

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=OFF \
  -DUSE_OPENMP=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  ..

ninja
ninja install
