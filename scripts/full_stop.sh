#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="./run"
LOG_DIR="./logs"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yaml}"

DO_DOCKER_DOWN=1
DO_DOCKER_VOL=1
DO_CARGO_CLEAN=0
DO_LOGS_CLEAN=0

log(){ echo "[$(date +'%H:%M:%S')] $*"; }

usage() {
  cat <<EOF
Usage: scripts/full_stop.sh [options]

Options:
  --no-docker        do not touch docker compose
  --keep-volumes     docker down without -v
  --cargo-clean      run cargo clean (workspace)
  --clean-logs       remove ./logs/*.log
  -h, --help         show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-docker) DO_DOCKER_DOWN=0; shift ;;
    --keep-volumes) DO_DOCKER_VOL=0; shift ;;
    --cargo-clean) DO_CARGO_CLEAN=1; shift ;;
    --clean-logs) DO_LOGS_CLEAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) log "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

mkdir -p "$PID_DIR" "$LOG_DIR"

log "🔪 FULL STOP: killing services, cleaning pidfiles, stopping docker..."

# 1) Try graceful stop for any pidfiles we have
if compgen -G "$PID_DIR/*.pid" > /dev/null; then
  for f in "$PID_DIR"/*.pid; do
    name="$(basename "$f" .pid)"
    pid="$(cat "$f" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      log "🛑 SIGTERM $name pid=$pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  sleep 1
fi

# 2) Hard kill known binaries no matter what (this is what you asked for)
#    (паттерны сделаны так, чтобы прибить и ./target/release/..., и отладочные, и "svc_*" процессы)
PATTERNS=(
  "svc_market_ingest"
  "svc_health"
  "svc_writer"
  "svc_live"
  "svc_backfill"
  "svc_db_init"
  "svc_compute"
  "svc_orders"
  "svc_position_tracker"
  "svc_api_gateway"
)

for p in "${PATTERNS[@]}"; do
  if pgrep -f "$p" >/dev/null 2>&1; then
    log "💀 pkill -9 -f $p"
    pkill -9 -f "$p" || true
  fi
done

# 3) Also kill anything still holding our known ports (just in case)
for port in 9001 9005 19092 9644 8080 5433; do
  if sudo ss -ltnp 2>/dev/null | grep -q ":$port "; then
    log "🔌 Port $port still in use. PIDs:"
    sudo ss -ltnp | grep ":$port " || true
  fi
done

# 4) Remove pidfiles (they're now garbage)
rm -f "$PID_DIR"/*.pid 2>/dev/null || true

# 5) Docker down
if (( DO_DOCKER_DOWN == 1 )); then
  if [[ -f "$COMPOSE_FILE" ]]; then
    if (( DO_DOCKER_VOL == 1 )); then
      log "🐳 docker compose down -v ($COMPOSE_FILE)"
      docker compose -f "$COMPOSE_FILE" down -v --remove-orphans || true
    else
      log "🐳 docker compose down ($COMPOSE_FILE)"
      docker compose -f "$COMPOSE_FILE" down --remove-orphans || true
    fi
  else
    log "⚠️ compose file not found: $COMPOSE_FILE (skip docker down)"
  fi
fi

# 6) Optional: clean logs
if (( DO_LOGS_CLEAN == 1 )); then
  log "🧹 cleaning logs..."
  rm -f "$LOG_DIR"/*.log 2>/dev/null || true
fi

# 7) Optional: cargo clean
if (( DO_CARGO_CLEAN == 1 )); then
  log "🧽 cargo clean..."
  cargo clean || true
fi

log "✅ FULL STOP DONE"
