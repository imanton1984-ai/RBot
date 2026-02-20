#!/bin/bash
# runweb.sh - Build and run WebUI only
# Usage: ./scripts/runweb.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEBUI_DIR="$PROJECT_ROOT/webui"

echo "========================================"
echo "  WebUI Build & Run Script"
echo "========================================"
echo ""

# Check if DATABASE_URL is set
if [ -z "$DATABASE_URL" ]; then
    export DATABASE_URL="postgres://postgres:postgres@localhost:5433/timescaledb_binance"
    echo "[INFO] Using default DATABASE_URL"
fi

# Build frontend
echo "[1/3] Building frontend..."
cd "$WEBUI_DIR"
npm run build

# Build Rust backend
echo ""
echo "[2/3] Building Rust backend..."
cd "$PROJECT_ROOT"
cargo build --release --package webui

# Run server
echo ""
echo "[3/3] Starting WebUI server..."
echo "========================================"
echo "  WebUI will be available at:"
echo "  http://localhost:3000"
echo "========================================"
echo ""

exec "$PROJECT_ROOT/target/release/webui_server"
