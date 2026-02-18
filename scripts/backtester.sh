#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "=== Signal Quality Backtester ==="

# Build
echo "Building backtester..."
cargo build --release -p backtester 2>&1 | tail -5

# Ensure database schema is up-to-date (apply strategy support migration)
echo "Ensuring database schema is up-to-date..."
STRATEGY_MIGRATION="$ROOT_DIR/database/ddl/090_strategy_support.sql"
if [[ -f "$STRATEGY_MIGRATION" ]]; then
  PGPASSWORD=postgres psql -h 127.0.0.1 -p 5433 -U postgres -d timescaledb_binance \
    -v ON_ERROR_STOP=1 \
    -f "$STRATEGY_MIGRATION" >/dev/null 2>&1 || true
  echo "Database schema check complete."
else
  echo "Warning: Strategy migration file not found: $STRATEGY_MIGRATION"
fi

# Config (override via env vars)
export BACKTEST_MIN_SCORE="${BACKTEST_MIN_SCORE:-0.55}"
export BACKTEST_MAX_SIGNALS="${BACKTEST_MAX_SIGNALS:-500000}"
export BACKTEST_TIMEOUT_BARS="${BACKTEST_TIMEOUT_BARS:-100}"
export BACKTEST_CSV_OUTPUT="${BACKTEST_CSV_OUTPUT:-backtest_results.csv}"
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
