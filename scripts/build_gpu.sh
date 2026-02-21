#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

section "BUILD (GPU MODE)"

# Ensure XGBoost libs are available
export XGBOOST_LIB_DIR="$ROOT_DIR/third_party/xgboost/install/lib"
export LD_LIBRARY_PATH="$XGBOOST_LIB_DIR:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$XGBOOST_LIB_DIR:${LIBRARY_PATH:-}"

TARGET_DIR="$ROOT_DIR/target/release"

# Check if binaries already exist and are up-to-date
BINARIES=("compute_history" "compute_realtime" "connections" "ingestor" "order_manager")

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
    log "Building with features: cuda"
    # Progress bar settings
    export CARGO_TERM_COLOR="always"

    # Build history (GPU enabled)
    run_with_pty "cargo build --release -p compute --bin compute_history --features cuda"

    # Build realtime (GPU enabled - optional, usually CPU, but if requested)
    run_with_pty "cargo build --release -p compute --bin compute_realtime --features cuda"

    # Build others
    run_with_pty "cargo build --release -p connections"
    run_with_pty "cargo build --release -p ingestor"
    run_with_pty "cargo build --release -p order_manager"

    ok "GPU Build finished."
fi