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

# Always run cargo build — it handles incremental compilation internally.
# Previous "skip if binaries exist" logic was WRONG: it never recompiled
# when source files changed, causing stale binaries without new features (e.g. cooldown).
log "Building with features: cuda (cargo handles incremental compilation)"
export CARGO_TERM_COLOR="always"

# Build history (GPU enabled)
run_with_pty "cargo build --release -p compute --bin compute_history --features cuda"

# Build realtime (GPU enabled)
run_with_pty "cargo build --release -p compute --bin compute_realtime --features cuda"

# Build others
run_with_pty "cargo build --release -p connections"
run_with_pty "cargo build --release -p ingestor"
run_with_pty "cargo build --release -p order_manager"

ok "GPU Build finished."