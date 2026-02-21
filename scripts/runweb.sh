#!/bin/bash
# runweb.sh - Build and run WebUI, auto-open Chrome
# Usage: ./scripts/runweb.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WEBUI_DIR="$PROJECT_ROOT/webui"
WEBUI_PORT=3000
WEBUI_URL="http://localhost:${WEBUI_PORT}"

echo "══════════════════════════════════════════"
echo "  🚀 WebUI Build & Run Script"
echo "══════════════════════════════════════════"
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

# Open Chrome in background after server starts
echo ""
echo "[3/3] Starting WebUI server + opening Chrome..."
echo "══════════════════════════════════════════"
echo "  🌐 WebUI: ${WEBUI_URL}"
echo "  Press Ctrl+C to stop"
echo "══════════════════════════════════════════"
echo ""

# Launch browser with 3s delay (in background)
(
    sleep 3
    if command -v google-chrome &>/dev/null; then
        google-chrome "${WEBUI_URL}" &>/dev/null &
    elif command -v chromium-browser &>/dev/null; then
        chromium-browser "${WEBUI_URL}" &>/dev/null &
    elif command -v chromium &>/dev/null; then
        chromium "${WEBUI_URL}" &>/dev/null &
    elif command -v xdg-open &>/dev/null; then
        xdg-open "${WEBUI_URL}" &>/dev/null &
    else
        echo "⚠️  No browser found. Open ${WEBUI_URL} manually."
    fi
) &

exec "$PROJECT_ROOT/target/release/webui_server"
