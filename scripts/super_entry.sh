#!/usr/bin/env bash
set -euo pipefail

# Super Entry Strategy — Full Pipeline Script
#
# This script manages the complete Super Entry lifecycle:
#   1. Build dataset from DB (Rust)
#   2. Train XGBoost models (Python)
#   3. Run backtest (Rust)
#
# USAGE:
#   ./scripts/super_entry.sh              # Full pipeline: dataset → train → backtest
#   ./scripts/super_entry.sh --dataset    # Only build dataset
#   ./scripts/super_entry.sh --train      # Only train models
#   ./scripts/super_entry.sh --backtest   # Only run backtest
#   ./scripts/super_entry.sh --gpu        # Use GPU for training + inference
#
# PREREQUISITES:
#   1. Database must have market.candles_* and market.indicators_wide data
#   2. Run compute_history first: ./scripts/run.sh
#   3. Python environment with xgboost, scikit-learn, pandas, numpy

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ROOT_DIR}/trainer/.venv"
MODELS_DIR="${ROOT_DIR}/models"

# Parse arguments
RUN_DATASET=false
RUN_TRAIN=false
RUN_BACKTEST=false
USE_GPU=""
ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            RUN_DATASET=true
            ALL=false
            shift
            ;;
        --train|--train-only)
            RUN_TRAIN=true
            ALL=false
            shift
            ;;
        --backtest)
            RUN_BACKTEST=true
            ALL=false
            shift
            ;;
        --gpu)
            USE_GPU="--gpu"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--dataset] [--train] [--backtest] [--gpu]"
            exit 1
            ;;
    esac
done

# Default: run all stages
if [ "$ALL" = true ]; then
    RUN_DATASET=true
    RUN_TRAIN=true
    RUN_BACKTEST=true
fi

echo "=============================================="
echo "[super_entry] Super Entry Strategy Pipeline"
echo "[super_entry] Root: ${ROOT_DIR}"
echo "[super_entry] GPU:  ${USE_GPU:-disabled}"
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
# 1. Build Dataset
# ============================================
if [ "$RUN_DATASET" = true ]; then
    echo ""
    echo "=============================================="
    echo "[super_entry] Stage 1: Building Dataset"
    echo "=============================================="

    echo "[super_entry] Building Rust binary..."
    cargo build --release -p ml_entry_strategy --bin super_entry_dataset 2>&1 | tail -5

    echo "[super_entry] Running dataset builder..."
    export SUPER_ENTRY_DATASET_OUTPUT="super_entry_dataset.csv"
    ./target/release/super_entry_dataset 2>&1 | tee logs/super_entry_dataset.out

    if [ -f "super_entry_dataset.csv" ]; then
        ROWS=$(wc -l < super_entry_dataset.csv)
        echo "[super_entry] ✅ Dataset built: super_entry_dataset.csv ($ROWS rows)"
    else
        echo "[super_entry] ❌ Dataset build failed!"
        exit 1
    fi
fi

# ============================================
# 2. Train Models
# ============================================
if [ "$RUN_TRAIN" = true ]; then
    echo ""
    echo "=============================================="
    echo "[super_entry] Stage 2: Training Models"
    echo "=============================================="

    DATASET_CSV="${ROOT_DIR}/super_entry_dataset.csv"

    if [ ! -f "${DATASET_CSV}" ]; then
        echo "[super_entry] WARNING: ${DATASET_CSV} not found!"
        echo "[super_entry] Run dataset stage first: $0 --dataset"
        exit 1
    fi

    # Setup Python venv
    if [ ! -d "${VENV_DIR}" ]; then
        echo "[super_entry] Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip wheel setuptools > /dev/null 2>&1
    python -m pip install numpy pandas scikit-learn xgboost > /dev/null 2>&1

    echo "[super_entry] Training super entry models..."
    python "${ROOT_DIR}/trainer/src/train_super_entry.py" \
        --csv "${DATASET_CSV}" \
        --output-dir "${MODELS_DIR}" \
        ${USE_GPU} 2>&1 | tee logs/super_entry_train.out

    echo "[super_entry] ✅ Training complete"
    echo "[super_entry] Models:"
    ls -la "${MODELS_DIR}"/super_entry_*.ubj "${MODELS_DIR}"/super_dir_*.ubj 2>/dev/null || echo "  No models found"
fi

# ============================================
# 3. Backtest
# ============================================
if [ "$RUN_BACKTEST" = true ]; then
    echo ""
    echo "=============================================="
    echo "[super_entry] Stage 3: Backtesting"
    echo "=============================================="

    echo "[super_entry] Building backtest binary..."
    cargo build --release -p ml_entry_strategy --bin super_entry_backtest 2>&1 | tail -5

    if [ -n "$USE_GPU" ]; then
        export SUPER_ENTRY_USE_GPU=true
    fi

    export SUPER_ENTRY_BACKTEST_CSV="super_entry_backtest_results.csv"

    echo "[super_entry] Running backtest..."
    echo "[super_entry] Logs: logs/super_entry_backtest.out"
    ./target/release/super_entry_backtest 2>&1 | tee logs/super_entry_backtest.out

    echo ""
    echo "[super_entry] Full log saved to: logs/super_entry_backtest.out"

    if [ -f "super_entry_backtest_results.csv" ]; then
        echo "[super_entry] ✅ Backtest results: super_entry_backtest_results.csv"
    fi
fi

echo ""
echo "=============================================="
echo "[super_entry] Pipeline complete!"
echo "=============================================="
echo ""
echo "[super_entry] Summary:"
[ -f "super_entry_dataset.csv" ] && echo "  📊 Dataset:  super_entry_dataset.csv"
ls "${MODELS_DIR}"/super_entry_*.ubj 2>/dev/null && echo "  🤖 Models:   ${MODELS_DIR}/super_entry_*.ubj"
ls "${MODELS_DIR}"/super_dir_*.ubj 2>/dev/null && echo "              ${MODELS_DIR}/super_dir_*.ubj"
[ -f "super_entry_backtest_results.csv" ] && echo "  📈 Backtest: super_entry_backtest_results.csv"
echo ""
