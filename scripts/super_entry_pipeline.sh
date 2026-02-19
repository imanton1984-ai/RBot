#!/usr/bin/env bash
set -euo pipefail

# Super Entry Strategy — Production Pipeline Script
#
# This script manages the Super Entry production lifecycle:
#   1. Check models exist
#   2. Run service for real-time signal generation
#
# NOTE: This does NOT run backtest or training.
#       For backtesting, use: ./scripts/super_entry_backtester.sh
#
# USAGE:
#   ./scripts/super_entry_pipeline.sh              # Run service (real-time signals)
#   ./scripts/super_entry_pipeline.sh --backfill   # Force backfill historical data
#   ./scripts/super_entry_pipeline.sh --check      # Only check if models exist
#
# PREREQUISITES:
#   1. Database must have market.candles_* and market.indicators_wide data
#   2. Models must be trained: ./scripts/super_entry_backtester.sh --train

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODELS_DIR="${ROOT_DIR}/models"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "$LOG_DIR"

# Parse arguments
RUN_BACKFILL=false
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backfill)
            RUN_BACKFILL=true
            shift
            ;;
        --check)
            CHECK_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--backfill] [--check]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "[super_entry] Super Entry Strategy Pipeline"
echo "[super_entry] Root: ${ROOT_DIR}"
echo "=============================================="

# Load env
if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5433/timescaledb_binance}"
export MODELS_DIR="${MODELS_DIR}"
export RUST_LOG="${RUST_LOG:-info}"

# ============================================
# 1. Check Models Exist
# ============================================
echo ""
echo "=============================================="
echo "[super_entry] Stage 1: Checking Models"
echo "=============================================="

MISSING_MODELS=false
for tf in 1 5 15 60 240; do
    SUPER_MODEL="${MODELS_DIR}/super_entry_v1_tf${tf}.ubj"
    DIR_MODEL="${MODELS_DIR}/super_dir_v1_tf${tf}.ubj"

    if [ ! -f "${SUPER_MODEL}" ]; then
        echo "[super_entry] MISSING: ${SUPER_MODEL}"
        MISSING_MODELS=true
    else
        echo "[super_entry] FOUND:  ${SUPER_MODEL}"
    fi

    if [ ! -f "${DIR_MODEL}" ]; then
        echo "[super_entry] MISSING: ${DIR_MODEL}"
        MISSING_MODELS=true
    else
        echo "[super_entry] FOUND:  ${DIR_MODEL}"
    fi
done

if [ "$MISSING_MODELS" = true ]; then
    echo ""
    echo "[super_entry] ❌ Models missing! Train first:"
    echo "[super_entry]   ./scripts/super_entry_backtester.sh --dataset"
    echo "[super_entry]   ./scripts/super_entry_backtester.sh --train"
    exit 1
fi

echo "[super_entry] ✅ All models found"

if [ "$CHECK_ONLY" = true ]; then
    echo "[super_entry] Check complete. Exiting."
    exit 0
fi

# ============================================
# 2. Build Service Binary
# ============================================
echo ""
echo "=============================================="
echo "[super_entry] Stage 2: Building Service"
echo "=============================================="

echo "[super_entry] Building super_entry_service..."
cargo build --release -p ml_entry_strategy --bin super_entry_service 2>&1 | tail -5

if [ ! -x "${ROOT_DIR}/target/release/super_entry_service" ]; then
    echo "[super_entry] ❌ Failed to build super_entry_service!"
    exit 1
fi

echo "[super_entry] ✅ Binary ready: target/release/super_entry_service"

# ============================================
# 3. Run Service
# ============================================
echo ""
echo "=============================================="
echo "[super_entry] Stage 3: Starting Service"
echo "=============================================="

if [ "$RUN_BACKFILL" = true ]; then
    export SUPER_ENTRY_FORCE_BACKFILL=true
    echo "[super_entry] Force backfill enabled"
fi

echo "[super_entry] Starting super_entry_service..."
echo "[super_entry] Logs: logs/super_entry_service.out"

nohup "${ROOT_DIR}/target/release/super_entry_service" >> "${LOG_DIR}/super_entry_service.out" 2>&1 &
PID=$!

echo $PID > "${ROOT_DIR}/run/super_entry_service.pid"

echo "[super_entry] ✅ Service started (PID: $PID)"
echo ""
echo "=============================================="
echo "[super_entry] Pipeline complete!"
echo "=============================================="
echo ""
echo "[super_entry] Service running in background"
echo "[super_entry] PID: ${PID}"
echo "[super_entry] Log: logs/super_entry_service.out"
echo "[super_entry] Stop: kill $PID  # or ./scripts/stop.sh"
echo ""
