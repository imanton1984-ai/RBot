#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
section "INSTALL GPU DEPS (XGBoost)"

XGBOOST_LIB_DIR="$ROOT_DIR/third_party/xgboost/install/lib"
XGBOOST_BIN="$ROOT_DIR/third_party/xgboost/install/lib/libxgboost.so"

# Check if XGBoost is already built
if [ -f "$XGBOOST_BIN" ]; then
    ok "XGBoost already built. Skipping build."
else
    log "XGBoost not found. Building with CUDA support..."
    
    # Logic to pick the right script based on CUDA version check (simplified)
    # Assuming newer CUDA for simplicity, otherwise fallback logic from original script can be added here
    XGBOOST_SCRIPT="$ROOT_DIR/scripts/build_xgboost_cuda.sh"

    if [ -f "$XGBOOST_SCRIPT" ]; then
        chmod +x "$XGBOOST_SCRIPT"
        "$XGBOOST_SCRIPT"
        
        ok "XGBoost GPU build completed."
    else
        die "XGBoost build script not found: $XGBOOST_SCRIPT"
    fi
fi

export XGBOOST_LIB_DIR="$XGBOOST_LIB_DIR"
export LD_LIBRARY_PATH="$XGBOOST_LIB_DIR:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$XGBOOST_LIB_DIR:${LIBRARY_PATH:-}"

ok "XGBoost GPU lib path set: $XGBOOST_LIB_DIR"