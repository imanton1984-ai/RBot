#!/usr/bin/env bash
set -euo pipefail

XGB_VER="v2.1.1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP_DIR="${ROOT_DIR}/third_party/xgboost"
SRC_DIR="${TP_DIR}/src"
BUILD_DIR="${TP_DIR}/build"
INSTALL_DIR="${TP_DIR}/install"

echo "[xgb] root=${ROOT_DIR}"
echo "[xgb] installing deps (ubuntu/debian expected)..."
sudo apt-get update
sudo apt-get install -y git cmake ninja-build build-essential libomp-dev

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

# IMPORTANT:
# - USE_CUDA=ON включает GPU predictor/inference
# - BUILD_SHARED_LIBS=ON чтобы получить libxgboost.so для Rust
# - CMAKE_CUDA_ARCHITECTURES можно поставить "native" или явно (например 75)
cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_OPENMP=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  ..

ninja
ninja install

echo
echo "[xgb] installed to: ${INSTALL_DIR}"
echo "[xgb] export for current shell:"
echo "  export XGBOOST_LIB_DIR=${INSTALL_DIR}/lib"
echo "  export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"