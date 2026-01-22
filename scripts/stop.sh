#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="${PID_DIR:-./run}"
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yml}"

GRACE_SECONDS="${GRACE_SECONDS:-4}"
HARD_KILL_AFTER_SECONDS="${HARD_KILL_AFTER_SECONDS:-2}"

DOCKER_ACTION="none"   # none|stop|down
STOP_ALL=0

log(){ echo "[$(date +'%H:%M:%S')] $*"; }
die(){ echo "❌ $*" >&2; exit 1; }

usage() {
  cat <<EOF
Usage: scripts/stop.sh [options]

Options:
  --all              Stop all known services (default list inside script)
  --service NAME     Stop only one service by name (can be repeated)
  --docker-stop      docker compose stop
  --docker-down      docker compose down (removes containers)
  --pid-dir PATH     pid dir (default: ./run)
  -h, --help         show help

Env:
  GRACE_SECONDS              soft stop wait (default: 4)
  HARD_KILL_AFTER_SECONDS    extra wait before -9 (default: 2)
  COMPOSE_FILE               docker compose file (default: docker-compose.yml)
EOF
}

SERVICES_DEFAULT=(
  "db_writer"
  "market_collector"
  "indicator_engine"
  "signal_scorer"
  "order_engine"
  "position_tracker"
)

SERVICES_TO_STOP=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      STOP_ALL=1
      shift
      ;;
    --service)
      shift
      [[ $# -gt 0 ]] || die "--service requires a NAME"
      SERVICES_TO_STOP+=("$1")
      shift
      ;;
    --docker-stop)
      DOCKER_ACTION="stop"
      shift
      ;;
    --docker-down)
      DOCKER_ACTION="down"
      shift
      ;;
    --pid-dir)
      shift
      [[ $# -gt 0 ]] || die "--pid-dir requires a PATH"
      PID_DIR="$1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown arg: $1 (use --help)"
      ;;
  esac
done

mkdir -p "$PID_DIR"

stop_pid() {
  local name="$1"
  local pidfile="$PID_DIR/${name}.pid"

  if [[ ! -f "$pidfile" ]]; then
    log "ℹ️  $name: pidfile not found, skip"
    return 0
  fi

  local pid
  pid="$(cat "$pidfile" || true)"
  if [[ -z "$pid" ]]; then
    log "⚠️  $name: empty pidfile, removing"
    rm -f "$pidfile"
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    log "ℹ️  $name: pid $pid not running, removing pidfile"
    rm -f "$pidfile"
    return 0
  fi

  log "🛑 Stopping $name (pid=$pid) ..."
  kill "$pid" 2>/dev/null || true

  # graceful wait
  local t=0
  while kill -0 "$pid" 2>/dev/null; do
    if (( t >= GRACE_SECONDS * 10 )); then
      break
    fi
    sleep 0.1
    t=$((t+1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    log "⚠️  $name: still alive after ${GRACE_SECONDS}s, sending SIGKILL..."
    kill -9 "$pid" 2>/dev/null || true

    # short extra wait
    local t2=0
    while kill -0 "$pid" 2>/dev/null; do
      if (( t2 >= HARD_KILL_AFTER_SECONDS * 10 )); then
        break
      fi
      sleep 0.1
      t2=$((t2+1))
    done
  fi

  if kill -0 "$pid" 2>/dev/null; then
    log "❌ $name: failed to stop pid=$pid"
    return 1
  fi

  rm -f "$pidfile"
  log "✅ $name stopped"
  return 0
}

main() {
  local list=()

  if (( STOP_ALL == 1 )) || (( ${#SERVICES_TO_STOP[@]} == 0 )); then
    list=("${SERVICES_DEFAULT[@]}")
  else
    list=("${SERVICES_TO_STOP[@]}")
  fi

  log "Stopping services (pid_dir=$PID_DIR)..."
  local failed=0
  for s in "${list[@]}"; do
    stop_pid "$s" || failed=1
  done

  case "$DOCKER_ACTION" in
    none)
      ;;
    stop)
      log "🐳 docker compose stop..."
      docker compose -f "$COMPOSE_FILE" stop || true
      ;;
    down)
      log "🐳 docker compose down..."
      docker compose -f "$COMPOSE_FILE" down || true
      ;;
  esac

  if (( failed == 1 )); then
    die "Some services failed to stop"
  fi

  log "✅ STOP OK"
}

main
