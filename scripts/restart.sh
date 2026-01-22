#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="${PID_DIR:-./run}"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yml}"

log(){ echo "[$(date +'%H:%M:%S')] $*"; }

stop_pidfile() {
  local name="$1"
  local pidfile="$PID_DIR/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    local pid; pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      log "Stopping $name (pid=$pid)..."
      kill "$pid" || true
      # wait a bit
      for _ in {1..20}; do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.2
      done
      kill -9 "$pid" 2>/dev/null || true
      log "$name stopped"
    fi
    rm -f "$pidfile"
  fi
}

main() {
  log "Restarting services..."
  mkdir -p "$PID_DIR"

  stop_pidfile "db_writer"
  # stop_pidfile "market_collector"
  # stop_pidfile "indicator_engine"
  # stop_pidfile "signal_scorer"
  # stop_pidfile "order_engine"

  log "Optional: restarting docker infra (kept running by default)"
  # docker compose -f "$COMPOSE_FILE" restart

  ./scripts/start.sh
}

main "$@"
