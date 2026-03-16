#!/usr/bin/env bash
set -euo pipefail

# Super Level Strategy — Full Pipeline Script
#
# Управляет полным жизненным циклом бектеста Super Level Strategy:
#   1. Build dataset from DB (Rust) — генерирует CSV с уровнями по касаниям
#   2. Train 5 XGBoost models (Python) — level, entry, direction, bounce_break, evaluator
#   3. Run backtest (Rust) — полная статистика с funnel, bounce/break, direction
#
# USAGE:
#   ./scripts/super_level_backtest.sh                # Full pipeline: dataset → train → backtest
#   ./scripts/super_level_backtest.sh --dataset      # Only build dataset
#   ./scripts/super_level_backtest.sh --train        # Only train 5 models
#   ./scripts/super_level_backtest.sh --backtest     # Only run backtest
#   ./scripts/super_level_backtest.sh --gpu          # Use GPU for training + inference
#
# PREREQUISITES:
#   1. Database with market.candles_* and market.indicators_wide
#   2. Python venv with xgboost, scikit-learn, pandas, numpy

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ROOT_DIR}/trainer/.venv"
MODELS_DIR="${ROOT_DIR}/models"
LOGS_DIR="${ROOT_DIR}/logs"

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

# Auto-detect GPU
if [ -z "$USE_GPU" ] && command -v nvidia-smi &>/dev/null; then
    echo "[super_level] GPU auto-detected (nvidia-smi found), enabling GPU"
    USE_GPU="--gpu"
fi

echo "=============================================="
echo "[super_level] Super Level Strategy Pipeline"
echo "[super_level] Root:   ${ROOT_DIR}"
echo "[super_level] GPU:    ${USE_GPU:-disabled}"
echo "[super_level] Models: 5 (level, entry, direction, bounce_break, evaluator)"
echo "=============================================="

# Ensure directories
mkdir -p "${LOGS_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${ROOT_DIR}/dataset"

# Load .env
if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5433/timescaledb_binance}"
export MODELS_DIR="${MODELS_DIR}"
export RUST_LOG="${RUST_LOG:-info}"

# Set XGBoost library path for GPU/CPU
if [ -n "$USE_GPU" ]; then
    export XGBOOST_LIB_DIR="${ROOT_DIR}/third_party/xgboost/install/lib"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${XGBOOST_LIB_DIR}:${LD_LIBRARY_PATH:-}"
else
    export XGBOOST_LIB_DIR="${ROOT_DIR}/third_party/xgboost/install_cpu/lib"
    export LD_LIBRARY_PATH="${XGBOOST_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# ============================================
# 1. Build Dataset
# ============================================
if [ "$RUN_DATASET" = true ]; then
    echo ""
    echo "=============================================="
    echo "[super_level] Stage 1: Building Dataset"
    echo "=============================================="

    echo "[super_level] Building Rust binary..."
    cargo build --release -p super_level_strategy --bin super_level_dataset 2>&1 | tail -5

    echo "[super_level] Running dataset builder..."
    export SUPER_LEVEL_DATASET_OUTPUT="${ROOT_DIR}/dataset/super_level_dataset.csv"
    ./target/release/super_level_dataset 2>&1 | tee "${LOGS_DIR}/super_level_dataset.out"

    if [ -f "${ROOT_DIR}/dataset/super_level_dataset.csv" ]; then
        ROWS=$(wc -l < "${ROOT_DIR}/dataset/super_level_dataset.csv")
        echo "[super_level] ✅ Dataset built: dataset/super_level_dataset.csv ($ROWS rows)"
    else
        echo "[super_level] ❌ Dataset build failed!"
        exit 1
    fi
fi

# ============================================
# 2. Train 5 Models
# ============================================
if [ "$RUN_TRAIN" = true ]; then
    echo ""
    echo "=============================================="
    echo "[super_level] Stage 2: Training 5 Models"
    echo "=============================================="

    DATASET_CSV="${ROOT_DIR}/dataset/super_level_dataset.csv"

    if [ ! -f "${DATASET_CSV}" ]; then
        echo "[super_level] WARNING: ${DATASET_CSV} not found!"
        echo "[super_level] Run dataset stage first: $0 --dataset"
        exit 1
    fi

    # Setup Python venv
    if [ ! -d "${VENV_DIR}" ]; then
        echo "[super_level] Creating virtual environment..."
        python3 -m venv "${VENV_DIR}"
    fi

    source "${VENV_DIR}/bin/activate"
    python -m pip install --upgrade pip wheel setuptools > /dev/null 2>&1
    python -m pip install numpy pandas scikit-learn xgboost > /dev/null 2>&1

    echo "[super_level] Training 5 models (level, entry, direction, bounce_break, evaluator)..."
    python "${ROOT_DIR}/trainer/src/train_super_level.py" \
        --csv "${DATASET_CSV}" \
        --output-dir "${MODELS_DIR}" \
        ${USE_GPU} 2>&1 | tee "${LOGS_DIR}/super_level_train.out"

    echo "[super_level] ✅ Training complete"
    echo "[super_level] Models:"
    ls -la "${MODELS_DIR}"/slvl_*.ubj 2>/dev/null || echo "  No models found"
fi

# ============================================
# 3. Backtest
# ============================================
if [ "$RUN_BACKTEST" = true ]; then
    echo ""
    echo "=============================================="
    echo "[super_level] Stage 3: Backtesting"
    echo "=============================================="

    echo "[super_level] Building backtest binary..."
    cargo build --release -p super_level_strategy --bin super_level_backtest 2>&1 | tail -5

    if [ -n "$USE_GPU" ]; then
        export SUPER_LEVEL_USE_GPU=true
    fi

    export SUPER_LEVEL_BACKTEST_CSV="${ROOT_DIR}/dataset/super_level_backtest_results.csv"

    echo "[super_level] Running backtest..."
    echo "[super_level] Logs: ${LOGS_DIR}/super_level_backtest.out"
    ./target/release/super_level_backtest 2>&1 | tee "${LOGS_DIR}/super_level_backtest.out"

    echo ""
    echo "[super_level] Full log saved to: ${LOGS_DIR}/super_level_backtest.out"

    if [ -f "${ROOT_DIR}/dataset/super_level_backtest_results.csv" ]; then
        echo "[super_level] ✅ Backtest results: dataset/super_level_backtest_results.csv"
    fi
fi

echo ""
echo "=============================================="
echo "[super_level] Pipeline complete!"
echo "=============================================="
echo ""
echo "[super_level] Summary:"
[ -f "${ROOT_DIR}/dataset/super_level_dataset.csv" ] && echo "  📊 Dataset:  dataset/super_level_dataset.csv"
ls "${MODELS_DIR}"/slvl_level_*.ubj 2>/dev/null | head -1 && echo "  🤖 Models:   ${MODELS_DIR}/slvl_*_v1_tf*.ubj (5 models × N TFs)"
[ -f "${ROOT_DIR}/dataset/super_level_backtest_results.csv" ] && echo "  📈 Backtest: dataset/super_level_backtest_results.csv"
echo ""
echo "[super_level] Logs:"
[ -f "${LOGS_DIR}/super_level_dataset.out" ] && echo "  📝 Dataset:  ${LOGS_DIR}/super_level_dataset.out"
[ -f "${LOGS_DIR}/super_level_train.out" ] && echo "  📝 Training: ${LOGS_DIR}/super_level_train.out"
[ -f "${LOGS_DIR}/super_level_backtest.out" ] && echo "  📝 Backtest: ${LOGS_DIR}/super_level_backtest.out"
echo ""
