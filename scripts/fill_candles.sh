#!/usr/bin/env bash
#
# fill_candles.sh — Independent candle ingestion (no pipeline dependency)
#
# Запускает ТОЛЬКО ингестор (backfill REST + WS) для наполнения таблиц свечей.
# Не запускает compute, order_manager или другие сервисы.
#
# Usage:
#   ./scripts/fill_candles.sh              # Default limits from runtime.toml
#   ./scripts/fill_candles.sh --deep       # Force 12000 candles for all TFs
#   BACKFILL_CANDLES=5000 ./scripts/fill_candles.sh  # Custom global override
#
# Per-TF limits are configured in config/runtime.toml under [runtime.backfill_candles_per_tf]:
#   "15m" = 12000
#   "1h"  = 12000
#   "4h"  = 12000
#   "1d"  = 3700
#
# After filling candles, run compute_history to generate indicators:
#   cargo run --release -p compute --bin compute_history
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ─── Parse args ────────────────────────────────────────────────────
if [[ "${1:-}" == "--deep" ]]; then
    echo "🔥 Deep backfill mode: overriding BACKFILL_CANDLES=12000 for all TFs"
    export BACKFILL_CANDLES=12000
fi

# ─── Load DATABASE_URL from .env or config ─────────────────────────
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

if [ -z "${DATABASE_URL:-}" ]; then
    echo "⚠️  DATABASE_URL not set. Attempting to construct from config/database.toml..."
    DB_HOST=$(grep -oP 'host\s*=\s*"\K[^"]+' config/database.toml 2>/dev/null || echo "localhost")
    DB_PORT=$(grep -oP 'port\s*=\s*\K\d+' config/database.toml 2>/dev/null || echo "5433")
    DB_USER=$(grep -oP 'user\s*=\s*"\K[^"]+' config/database.toml 2>/dev/null || echo "postgres")
    DB_PASS=$(grep -oP 'password\s*=\s*"\K[^"]+' config/database.toml 2>/dev/null || echo "postgres")
    DB_NAME=$(grep -oP 'name\s*=\s*"\K[^"]+' config/database.toml 2>/dev/null || echo "rust_trader")
    export DATABASE_URL="postgres://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}?sslmode=disable"
    echo "   DATABASE_URL=$DATABASE_URL"
fi

# ─── Check prerequisites ──────────────────────────────────────────
echo "📊 Checking database connectivity..."
if ! psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM market.pairs WHERE is_active = true" 2>/dev/null; then
    echo "❌ Cannot connect to database or market.pairs not ready."
    echo "   Make sure TimescaleDB is running and schema is initialized."
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  🕯️  Independent Candle Ingestion"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "  Per-TF limits from config/runtime.toml:"
echo "    1m  = ${BACKFILL_CANDLES:-1000 (default)}"
echo "    5m  = ${BACKFILL_CANDLES:-1000 (default)}"
echo "    15m = ${BACKFILL_CANDLES:-12000 (per-TF config)}"
echo "    1h  = ${BACKFILL_CANDLES:-12000 (per-TF config)}"
echo "    4h  = ${BACKFILL_CANDLES:-12000 (per-TF config)}"
echo "    1d  = ${BACKFILL_CANDLES:-3700 (per-TF config)}"
echo ""

# ─── Build ingestor ───────────────────────────────────────────────
echo "🔨 Building ingestor (release)..."
cargo build --release -p ingestor 2>&1 | tail -5

# ─── Run ingestor ─────────────────────────────────────────────────
echo ""
echo "🚀 Starting candle ingestion..."
echo "   This may take 10-30 minutes depending on the number of pairs and depth."
echo "   Press Ctrl+C to stop after backfill (WS will be interrupted)."
echo ""

# Set max backfill loops high enough for deep history
export INGEST_BACKFILL_MAX_LOOPS="${INGEST_BACKFILL_MAX_LOOPS:-50000}"

# Run the ingestor
./target/release/ingestor

echo ""
echo "✅ Candle ingestion complete."
echo ""
echo "Next steps:"
echo "  1. Run compute_history to generate indicators for the new candles:"
echo "     cargo run --release -p compute --bin compute_history"
echo "  2. Rebuild the super_entry dataset:"
echo "     ./scripts/super_entry.sh --dataset"
echo "  3. Retrain models:"
echo "     ./scripts/super_entry.sh --train"
echo "  4. Run backtest to validate:"
echo "     ./scripts/super_entry.sh --backtest"
