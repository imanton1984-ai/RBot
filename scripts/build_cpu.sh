#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

section "BUILD (CPU MODE)"

# Ensure XGBoost CPU libs are available
export XGBOOST_LIB_DIR="$ROOT_DIR/third_party/xgboost/install_cpu/lib"
export LD_LIBRARY_PATH="$XGBOOST_LIB_DIR:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$XGBOOST_LIB_DIR:${LIBRARY_PATH:-}"

TARGET_DIR="$ROOT_DIR/target/release"

# Check if binaries already exist and are up-to-date
BINARIES=("compute_history" "compute_realtime" "connections" "ingestor")

all_built=true
for bin in "${BINARIES[@]}"; do
    if [[ ! -x "$TARGET_DIR/$bin" ]]; then
        all_built=false
        break
    fi
done

if [[ "$all_built" == true ]]; then
    ok "All binaries already built. Skipping build."
else
    log "Building without CUDA features"
    export CARGO_TERM_COLOR="always"

    # Build everything with default features (no cuda)
    # Using release for performance, change to debug if laptop is too slow to compile release
    PROFILE="release" # Change to debug if laptop is too slow to compile release

    run_with_pty "cargo build --$PROFILE -p compute --bin compute_history"
    run_with_pty "cargo build --$PROFILE -p compute --bin compute_realtime"
    run_with_pty "cargo build --$PROFILE -p connections"
    run_with_pty "cargo build --$PROFILE -p ingestor"
    run_with_pty "cargo build --$PROFILE -p order_manager"

    ok "CPU Build finished."
fi