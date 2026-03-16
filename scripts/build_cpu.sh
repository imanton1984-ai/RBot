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

# Always run cargo build — it handles incremental compilation internally.
# Previous "skip if binaries exist" logic was WRONG: it never recompiled
# when source files changed, causing stale binaries without new features (e.g. cooldown).
log "Building without CUDA features (cargo handles incremental compilation)"
export CARGO_TERM_COLOR="always"

PROFILE="release"

run_with_pty "cargo build --$PROFILE -p compute --bin compute_history"
run_with_pty "cargo build --$PROFILE -p compute --bin compute_realtime"
run_with_pty "cargo build --$PROFILE -p connections"
run_with_pty "cargo build --$PROFILE -p ingestor"
run_with_pty "cargo build --$PROFILE -p order_manager"

ok "CPU Build finished."