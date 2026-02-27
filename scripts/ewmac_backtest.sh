#!/usr/bin/env bash
set -euo pipefail

# EWMAC Strategy — Backtest Script
#
# Runs the EWMAC (Exponentially Weighted Moving Average Crossover) backtest
# and saves results to logs/ewmac_backtest.out
#
# EWMAC is a rule-based strategy — no models or training required.
# It only needs OHLCV candle data in the database.
#
# USAGE:
#   ./scripts/ewmac_backtest.sh                    # Run with defaults
#   ./scripts/ewmac_backtest.sh --csv              # Also export CSV
#   ./scripts/ewmac_backtest.sh --min-forecast 8   # Custom min forecast
#   ./scripts/ewmac_backtest.sh --max-hold 50      # Custom max hold bars
#   ./scripts/ewmac_backtest.sh --fdm 1.5          # Custom FDM
#
# PREREQUISITES:
#   1. Database must have market.candles_* data
#   2. Run compute_history or ingestor first to populate candles
#
# OUTPUT:
#   - Console: summary tables (WinRate, Sharpe, PnL, Direction, Forecast buckets)
#   - File:    logs/ewmac_backtest.out (full log)
#   - CSV:     dataset/ewmac_backtest_results.csv (if --csv flag used)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/ewmac_backtest.out"

# Parse arguments
EXPORT_CSV=false
MIN_FORECAST=""
MAX_FORECAST=""
MAX_HOLD=""
COOLDOWN=""
MIN_PAIRS=""
FDM=""
WARMUP=""
MIN_ATR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --csv)
            EXPORT_CSV=true
            shift
            ;;
        --min-forecast)
            MIN_FORECAST="$2"
            shift 2
            ;;
        --max-forecast)
            MAX_FORECAST="$2"
            shift 2
            ;;
        --max-hold)
            MAX_HOLD="$2"
            shift 2
            ;;
        --cooldown)
            COOLDOWN="$2"
            shift 2
            ;;
        --min-pairs)
            MIN_PAIRS="$2"
            shift 2
            ;;
        --fdm)
            FDM="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --min-atr)
            MIN_ATR="$2"
            shift 2
            ;;
        -h|--help)
            echo "EWMAC Strategy Backtester"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --csv                Export results to CSV"
            echo "  --min-forecast N     Minimum |forecast| for signal (default: 5)"
            echo "  --max-forecast N     Maximum forecast cap (default: 20)"
            echo "  --max-hold N         Max bars to hold a trade (default: 30)"
            echo "  --fdm N              Forecast diversification multiplier (default: 1.2)"
            echo "  --warmup N           Warmup bars (default: 300)"
            echo "  --min-atr N          Minimum ATR% filter (default: 0.05)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# Load .env
if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
fi

export DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgres@localhost:5433/timescaledb_binance}"
export RUST_LOG="${RUST_LOG:-info}"

# Apply custom parameters via env vars
[ -n "$MIN_FORECAST" ] && export EWMAC_MIN_FORECAST="$MIN_FORECAST"
[ -n "$MAX_FORECAST" ] && export EWMAC_MAX_FORECAST="$MAX_FORECAST"
[ -n "$MAX_HOLD" ]     && export EWMAC_MAX_HOLD_BARS="$MAX_HOLD"
[ -n "$COOLDOWN" ]     && export EWMAC_COOLDOWN_BARS="$COOLDOWN"
[ -n "$MIN_PAIRS" ]    && export EWMAC_MIN_PAIRS_AGREE="$MIN_PAIRS"
[ -n "$FDM" ]          && export EWMAC_FDM="$FDM"
[ -n "$WARMUP" ]       && export EWMAC_WARMUP_BARS="$WARMUP"
[ -n "$MIN_ATR" ]      && export EWMAC_MIN_ATR_PCT="$MIN_ATR"

if [ "$EXPORT_CSV" = true ]; then
    mkdir -p "${ROOT_DIR}/dataset"
    export EWMAC_BACKTEST_CSV="${ROOT_DIR}/dataset/ewmac_backtest_results.csv"
fi

# Ensure log directory exists
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "[ewmac] EWMAC Strategy Backtester"
echo "[ewmac] Root: ${ROOT_DIR}"
echo "[ewmac] Log:  ${LOG_FILE}"
echo "=============================================="
echo ""

# Build
echo "[ewmac] Building backtest binary..."
cargo build --release -p ewmac_strategy --bin ewmac_backtest 2>&1 | tail -5

echo "[ewmac] Running backtest..."
echo "[ewmac] Output → ${LOG_FILE}"
echo ""

# Run backtest, output to both console and log file
./target/release/ewmac_backtest 2>&1 | tee "$LOG_FILE"

echo ""
echo "=============================================="
echo "[ewmac] Backtest complete!"
echo "[ewmac] Full log: ${LOG_FILE}"

if [ "$EXPORT_CSV" = true ] && [ -f "${ROOT_DIR}/dataset/ewmac_backtest_results.csv" ]; then
    ROWS=$(wc -l < "${ROOT_DIR}/dataset/ewmac_backtest_results.csv")
    echo "[ewmac] CSV export: dataset/ewmac_backtest_results.csv ($ROWS rows)"
fi

echo "=============================================="
