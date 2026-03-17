#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Walk-Forward Optimization Training — Super Entry Strategy
# ═══════════════════════════════════════════════════════════════════════
#
# Usage:
#   ./scripts/train_wfo.sh                    # Full WFO (all TFs, CPU)
#   ./scripts/train_wfo.sh --gpu              # Full WFO (GPU)
#   ./scripts/train_wfo.sh --gpu --tf 15,60   # Specific TFs only
#   ./scripts/train_wfo.sh --evaluate-only    # OOS metrics without final model
#
# Prerequisite:
#   Dataset must exist: dataset/super_entry_dataset.csv
#   Generate with: cargo run --release -p ml_entry_strategy --bin super_entry_dataset

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

CSV="dataset/super_entry_dataset.csv"
OUTPUT_DIR="models"
GPU_FLAG=""
TF_FLAG=""
EVAL_FLAG=""
EXTRA_FLAGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            GPU_FLAG="--gpu"
            shift
            ;;
        --tf|--timeframes)
            TF_FLAG="--timeframes $2"
            shift 2
            ;;
        --evaluate-only)
            EVAL_FLAG="--evaluate-only"
            shift
            ;;
        --csv)
            CSV="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-fold-models)
            EXTRA_FLAGS="$EXTRA_FLAGS --no-fold-models"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu] [--tf 15,60,240] [--evaluate-only] [--csv path] [--no-fold-models]"
            exit 1
            ;;
    esac
done

# Check dataset exists
if [ ! -f "$CSV" ]; then
    echo "❌ Dataset not found: $CSV"
    echo ""
    echo "Generate dataset first:"
    echo "  cargo run --release -p ml_entry_strategy --bin super_entry_dataset"
    exit 1
fi

# Show config
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  WFO Training — Super Entry Strategy                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Dataset:    $CSV ($(wc -l < "$CSV") lines)"
echo "  Output:     $OUTPUT_DIR/"
echo "  GPU:        ${GPU_FLAG:-CPU}"
echo "  Timeframes: ${TF_FLAG:-all}"
echo "  Mode:       ${EVAL_FLAG:-full training}"
echo ""

# Check Python deps
python3 -c "import xgboost, sklearn, pandas, numpy" 2>/dev/null || {
    echo "❌ Missing Python packages. Install:"
    echo "  pip install xgboost scikit-learn pandas numpy"
    exit 1
}

# Create output dir
mkdir -p "$OUTPUT_DIR"

# Run WFO training
python3 trainer/src/train_super_entry_wfo.py \
    --csv "$CSV" \
    --output-dir "$OUTPUT_DIR" \
    $GPU_FLAG \
    $TF_FLAG \
    $EVAL_FLAG \
    $EXTRA_FLAGS \
    2>&1 | tee "logs/wfo_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "✅ WFO training complete. Check models/ for results."
