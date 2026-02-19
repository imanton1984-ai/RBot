#!/usr/bin/env bash
set -euo pipefail

# Teacher Script — Train ALL ML models for Rust trading bot
#
# This script trains 4 model types:
#
#   1. Price Models (train_from_db.py)
#      - Input: market.indicators_wide + market.candles_* (from DB)
#      - Output: models/price_v1_tf{1,5,15,60,240,1440}.ubj
#      - Task: Regression (predict price return after HORIZON bars)
#
#   2. Levels Models (train_from_db.py)
#      - Input: market.indicators_wide + market.candles_* (from DB)
#      - Output: models/levels_v1_tf{1,5,15,60,240,1440}.ubj
#      - Task: Binary classification (predict direction up/down)
#
#   3. Signal Quality Model (train_final_scorer.py)
#      - Input: backtest_results.csv (from backtester)
#      - Output: models/signal_quality_v1.ubj
#      - Task: Binary classification (predict if signal reaches TP1)
#
#   4. Entry Policy Models (train_entry_policy.py)
#      - Input: entry_policy_dataset.csv (from backtester)
#      - Output: models/entry_enter_v1_tf{1,5,15,60,240}.ubj
#                models/entry_cancel_v1_tf{1,5,15,60,240}.ubj
#      - Task: Binary classification (ENTER/WAIT/CANCEL decision)
#
# USAGE:
#   ./scripts/teacher.sh [--gpu] [--skip-db] [--skip-quality] [--skip-entry]
#
# OPTIONS:
#   --gpu          Use GPU for training (requires CUDA)
#   --skip-db      Skip price/levels model training (from DB)
#   --skip-quality Skip signal quality model training
#   --skip-entry   Skip entry policy models training
#
# PREREQUISITES:
#   1. Database must have market.indicators_wide and market.candles_* data
#   2. For signal quality: Run backtester first: ./scripts/backtester.sh
#   3. For entry policy: Run backtester first: ./scripts/backtester.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/trainer/.venv"
MODELS_DIR="${ROOT_DIR}/models"

# Parse arguments
USE_GPU=""
SKIP_DB=false
SKIP_QUALITY=false
SKIP_ENTRY=false
SKIP_SUPER_ENTRY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU="--gpu"
            shift
            ;;
        --skip-db)
            SKIP_DB=true
            shift
            ;;
        --skip-quality)
            SKIP_QUALITY=true
            shift
            ;;
        --skip-entry)
            SKIP_ENTRY=true
            shift
            ;;
        --skip-super-entry)
            SKIP_SUPER_ENTRY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu] [--skip-db] [--skip-quality] [--skip-entry] [--skip-super-entry]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "[teacher] Root directory: ${ROOT_DIR}"
echo "[teacher] Models directory: ${MODELS_DIR}"
echo "[teacher] GPU training: ${USE_GPU:-no}"
echo "=============================================="

# Create virtual environment if it doesn't exist
if [ ! -d "${VENV_DIR}" ]; then
    echo "[teacher] Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtual environment
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Upgrade pip and install build tools
python -m pip install --upgrade pip wheel setuptools

# Install training dependencies
echo "[teacher] Installing dependencies..."
python -m pip install \
    numpy pandas sqlalchemy psycopg2-binary scikit-learn \
    xgboost

# Set environment variables
export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5433/timescaledb_binance}"
export MODELS_DIR="${MODELS_DIR}"

# Ensure models directory exists
mkdir -p "${MODELS_DIR}"

echo ""
echo "=============================================="
echo "[teacher] Starting training pipeline..."
echo "=============================================="
echo ""

# Track success
DB_SUCCESS=false
QUALITY_SUCCESS=false
ENTRY_SUCCESS=false

# ============================================
# 1. Train Price & Levels Models (from DB)
# ============================================
if [ "$SKIP_DB" = false ]; then
    echo "=============================================="
    echo "[teacher] === Price & Levels Models (from DB) ==="
    echo "=============================================="
    echo "[teacher] This trains 2 models per timeframe:"
    echo "[teacher]   - price_v1_tf*.ubj (regression)"
    echo "[teacher]   - levels_v1_tf*.ubj (binary direction)"
    echo ""
    
    echo "[teacher] Training from database..."
    echo "[teacher] Output: ${MODELS_DIR}/"
    echo ""
    
    if python "${ROOT_DIR}/trainer/src/train_from_db.py"; then
        echo ""
        echo "[teacher] ✅ Price & Levels models training completed!"
        DB_SUCCESS=true
    else
        echo ""
        echo "[teacher] ❌ Price & Levels models training failed!"
    fi
else
    echo "[teacher] Skipping price/levels models training (--skip-db)"
fi

echo ""
echo "----------------------------------------------"
echo ""

# ============================================
# 2. Train Signal Quality Model
# ============================================
if [ "$SKIP_QUALITY" = false ]; then
    echo "=============================================="
    echo "[teacher] === Signal Quality Model ==="
    echo "=============================================="
    
    QUALITY_CSV="${ROOT_DIR}/backtest_results.csv"
    
    if [ ! -f "${QUALITY_CSV}" ]; then
        echo "[teacher] WARNING: ${QUALITY_CSV} not found!"
        echo "[teacher] Run backtester first: ./scripts/backtester.sh"
        echo "[teacher] Skipping signal quality training..."
    else
        echo "[teacher] Training signal quality model..."
        echo "[teacher] Input: ${QUALITY_CSV}"
        echo "[teacher] Output: ${MODELS_DIR}/signal_quality_v1.ubj"
        echo ""
        
        if python "${ROOT_DIR}/trainer/src/train_final_scorer.py" \
            --csv "${QUALITY_CSV}" \
            --output-dir "${MODELS_DIR}" \
            ${USE_GPU}; then
            echo ""
            echo "[teacher] ✅ Signal quality model training completed!"
            QUALITY_SUCCESS=true
        else
            echo ""
            echo "[teacher] ❌ Signal quality model training failed!"
        fi
    fi
else
    echo "[teacher] Skipping signal quality model training (--skip-quality)"
fi

echo ""
echo "----------------------------------------------"
echo ""

# ============================================
# 3. Train Entry Policy Models
# ============================================
if [ "$SKIP_ENTRY" = false ]; then
    echo "=============================================="
    echo "[teacher] === Entry Policy Models ==="
    echo "=============================================="
    
    ENTRY_CSV="${ROOT_DIR}/entry_policy_dataset.csv"
    
    if [ ! -f "${ENTRY_CSV}" ]; then
        echo "[teacher] WARNING: ${ENTRY_CSV} not found!"
        echo "[teacher] Run backtester first: ./scripts/backtester.sh"
        echo "[teacher] Skipping entry policy training..."
    else
        echo "[teacher] Training entry policy models..."
        echo "[teacher] Input: ${ENTRY_CSV}"
        echo "[teacher] Output: ${MODELS_DIR}/entry_enter_v1_tf*.ubj"
        echo "[teacher]        ${MODELS_DIR}/entry_cancel_v1_tf*.ubj"
        echo ""
        
        if python "${ROOT_DIR}/trainer/src/train_entry_policy.py" \
            --csv "${ENTRY_CSV}" \
            --output-dir "${MODELS_DIR}" \
            ${USE_GPU}; then
            echo ""
            echo "[teacher] ✅ Entry policy models training completed!"
            ENTRY_SUCCESS=true
        else
            echo ""
            echo "[teacher] ❌ Entry policy models training failed!"
        fi
    fi
else
    echo "[teacher] Skipping entry policy models training (--skip-entry)"
fi

echo ""
echo "----------------------------------------------"
echo ""

# ============================================
# 4. Train Super Entry Models
# ============================================
SUPER_ENTRY_SUCCESS=false
if [ "$SKIP_SUPER_ENTRY" = false ]; then
    echo "=============================================="
    echo "[teacher] === Super Entry Models ==="
    echo "=============================================="
    
    SUPER_CSV="${ROOT_DIR}/super_entry_dataset.csv"
    
    if [ ! -f "${SUPER_CSV}" ]; then
        echo "[teacher] WARNING: ${SUPER_CSV} not found!"
        echo "[teacher] Building dataset first..."
        
        # Build dataset from DB
        if cargo build --release -p ml_entry_strategy --bin super_entry_dataset 2>&1 | tail -3; then
            if ./target/release/super_entry_dataset 2>&1 | tail -10; then
                echo "[teacher] Dataset built successfully"
            else
                echo "[teacher] Dataset build failed, skipping super entry training"
            fi
        fi
    fi
    
    if [ -f "${SUPER_CSV}" ]; then
        echo "[teacher] Training super entry models..."
        echo "[teacher] Input: ${SUPER_CSV}"
        echo "[teacher] Output: ${MODELS_DIR}/super_entry_v1_tf*.ubj"
        echo "[teacher]        ${MODELS_DIR}/super_dir_v1_tf*.ubj"
        echo ""
        
        if python "${ROOT_DIR}/trainer/src/train_super_entry.py" \
            --csv "${SUPER_CSV}" \
            --output-dir "${MODELS_DIR}" \
            ${USE_GPU}; then
            echo ""
            echo "[teacher] ✅ Super entry models training completed!"
            SUPER_ENTRY_SUCCESS=true
        else
            echo ""
            echo "[teacher] ❌ Super entry models training failed!"
        fi
    else
        echo "[teacher] Skipping super entry training (no dataset)"
    fi
else
    echo "[teacher] Skipping super entry models training (--skip-super-entry)"
fi

echo ""
echo "=============================================="
echo "[teacher] Training pipeline finished!"
echo "=============================================="
echo ""

# Summary
echo "[teacher] Summary:"
if [ "$SKIP_DB" = false ]; then
    if [ "$DB_SUCCESS" = true ]; then
        echo "  ✅ Price/Levels: models/price_v1_tf*.ubj"
        echo "                   models/levels_v1_tf*.ubj"
    else
        echo "  ❌ Price/Levels: FAILED or SKIPPED"
    fi
fi

if [ "$SKIP_QUALITY" = false ]; then
    if [ "$QUALITY_SUCCESS" = true ]; then
        echo "  ✅ Signal Quality: models/signal_quality_v1.ubj"
    else
        echo "  ❌ Signal Quality: FAILED or SKIPPED"
    fi
fi

if [ "$SKIP_ENTRY" = false ]; then
    if [ "$ENTRY_SUCCESS" = true ]; then
        echo "  ✅ Entry Policy: models/entry_enter_v1_tf*.ubj"
        echo "                   models/entry_cancel_v1_tf*.ubj"
    else
        echo "  ❌ Entry Policy: FAILED or SKIPPED"
    fi
fi

if [ "$SKIP_SUPER_ENTRY" = false ]; then
    if [ "$SUPER_ENTRY_SUCCESS" = true ]; then
        echo "  ✅ Super Entry: models/super_entry_v1_tf*.ubj"
        echo "                  models/super_dir_v1_tf*.ubj"
    else
        echo "  ❌ Super Entry: FAILED or SKIPPED"
    fi
fi

echo ""
echo "[teacher] Models saved to: ${MODELS_DIR}/"
echo ""

# List generated models
if [ -d "${MODELS_DIR}" ]; then
    echo "[teacher] Generated models:"
    ls -la "${MODELS_DIR}"/*.ubj 2>/dev/null || echo "[teacher] No .ubj models found"
fi

echo ""
echo "=============================================="
echo "[teacher] All done!"
echo "=============================================="
