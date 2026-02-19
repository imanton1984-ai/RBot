#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Signal Quality Backtester ==="

# Build
echo "Building backtester..."
cargo build --release -p backtester 2>&1 | tail -5

# Config (override via env vars)
export BACKTEST_MIN_SCORE="${BACKTEST_MIN_SCORE:-0.55}"
export BACKTEST_MAX_SIGNALS="${BACKTEST_MAX_SIGNALS:-500000}"
export BACKTEST_TIMEOUT_BARS="${BACKTEST_TIMEOUT_BARS:-100}"
export BACKTEST_CSV_OUTPUT="${BACKTEST_CSV_OUTPUT:-dataset/backtest_results.csv}"
export RUST_LOG="${RUST_LOG:-info}"

echo "Config:"
echo "  MIN_SCORE:     $BACKTEST_MIN_SCORE"
echo "  MAX_SIGNALS:   $BACKTEST_MAX_SIGNALS"
echo "  TIMEOUT_BARS:  $BACKTEST_TIMEOUT_BARS"
echo "  CSV_OUTPUT:    $BACKTEST_CSV_OUTPUT"
echo ""

# Run
echo "Running backtester..."
./target/release/backtester 2>&1 | tee logs/backtester.out

echo ""
echo "Done. CSV output: $BACKTEST_CSV_OUTPUT"
echo "DB results: SELECT * FROM trade.backtest_results ORDER BY signal_time DESC LIMIT 20;"
