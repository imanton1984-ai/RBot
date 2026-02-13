#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
section "INSTALL CPU DEPS (XGBoost)"

XGBOOST_LIB_DIR="$ROOT_DIR/third_party/xgboost/install_cpu/lib"
XGBOOST_BIN="$ROOT_DIR/third_party/xgboost/install_cpu/lib/libxgboost.so"

# Check if XGBoost is already built
if [ -f "$XGBOOST_BIN" ]; then
    ok "XGBoost already built. Skipping build."
else
    # Create a CPU build script on the fly if it doesn't exist
    # based on the CUDA one but with USE_CUDA=OFF
    CPU_BUILD_SCRIPT="$ROOT_DIR/scripts/build_xgboost_cpu.sh"

    if [ ! -f "$CPU_BUILD_SCRIPT" ]; then
        log "Creating $CPU_BUILD_SCRIPT from template..."
        cat > "$CPU_BUILD_SCRIPT" << 'EOF'
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
EOF
        chmod +x "$CPU_BUILD_SCRIPT"
    fi

    log "Building XGBoost (CPU ONLY)..."
    "$CPU_BUILD_SCRIPT"
fi

export XGBOOST_LIB_DIR="$XGBOOST_LIB_DIR"
export LD_LIBRARY_PATH="$XGBOOST_LIB_DIR:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$XGBOOST_LIB_DIR:${LIBRARY_PATH:-}"

ok "XGBoost CPU lib path set: $XGBOOST_LIB_DIR"