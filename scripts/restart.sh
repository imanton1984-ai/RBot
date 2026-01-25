#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="./run"

stop_pidfile() {
  local name="$1"
  local pidfile="$PID_DIR/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "Stopping $name ($pid)..."
      kill "$pid" || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

main() {
  echo "Restarting services..."
  mkdir -p "$PID_DIR"

  # Останавливаем в обратном порядке зависимости
  stop_pidfile "live"
  stop_pidfile "backfill"
  stop_pidfile "writer"
  stop_pidfile "ingest"
  stop_pidfile "health"
  
  # Жесткая зачистка по именам бинарников (на случай потери pid файлов)
  pkill -f "svc_live" || true
  pkill -f "svc_backfill" || true
  pkill -f "svc_writer" || true
  pkill -f "svc_market_ingest" || true
  pkill -f "svc_health" || true

  echo "All stopped. Starting..."
  ./scripts/start.sh
}

main "$@"
