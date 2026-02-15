#!/usr/bin/env bash
set -euo pipefail

XGB_VER="v3.2.0"
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
# - CMAKE_CUDA_ARCHITECTURES определяем автоматически или через переменную
ARCH="${XGB_CUDA_ARCH:-}"
if [ -z "$ARCH" ]; then
  # пробуем достать compute capability, например "8.6" -> "86"
  CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.' | tr -d ' ')"
  if [ -n "$CC" ]; then
    ARCH="$CC"
  else
    ARCH="86"  # безопасный дефолт под Ampere; при желании меняешь на 75/89/90
  fi
fi

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_CUDA=ON \
  -DUSE_OPENMP=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
  ..

ninja
ninja install

echo
echo "[xgb] installed to: ${INSTALL_DIR}"
echo "[xgb] export for current shell:"
echo "  export XGBOOST_LIB_DIR=${INSTALL_DIR}/lib"
echo "  export LD_LIBRARY_PATH=${INSTALL_DIR}/lib:\$LD_LIBRARY_PATH"