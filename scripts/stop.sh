#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PID_DIR="${PID_DIR:-$ROOT_DIR/run}"
mkdir -p "$LOG_DIR" "$PID_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
STOP_LOG="$LOG_DIR/stop_$TS.log"
exec > >(tee -a "$STOP_LOG") 2>&1

echo "==========================================="
echo "STOP @ $(date)"
echo "ROOT: $ROOT_DIR"
echo "LOG : $STOP_LOG"
echo "==========================================="

# env (опционально, чтобы stop знал KAFKA_BROKERS/и т.д.)
load_env_file() {
  local f="$1"
  echo "Loading env: $f"
  set -a
  # shellcheck disable=SC1090
  source "$f"
  set +a
}

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  load_env_file "$ENV_FILE"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  load_env_file "$ROOT_DIR/.env"
elif [[ -f "$ROOT_DIR/connections/.env" ]]; then
  load_env_file "$ROOT_DIR/connections/.env"
fi

# args
WITH_INFRA_DOWN=0
usage() {
  cat <<EOF
Usage: ./scripts/stop.sh [options]
  --down     also stop docker compose infra (compose down)
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --down) WITH_INFRA_DOWN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

compose() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    return 127
  fi
}

COMPOSE_FILE=""
if [[ -f "$ROOT_DIR/infra/docker-compose.yaml" ]]; then
  COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yaml"
elif [[ -f "$ROOT_DIR/infra/docker-compose.yml" ]]; then
  COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"
fi

# stop by pidfiles first (надежно, без pgrep -f по всему миру)
stop_pidfile() {
  local pidfile="$1"
  local name; name="$(basename "$pidfile" .pid)"
  local pid; pid="$(cat "$pidfile" 2>/dev/null || true)"

  if [[ -z "$pid" ]]; then
    rm -f "$pidfile" || true
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Not running: $name (stale pidfile $pidfile)"
    rm -f "$pidfile" || true
    return 0
  fi

  echo "Stopping $name pid=$pid (TERM)..."
  kill "$pid" 2>/dev/null || true

  # wait up to 10s
  for _ in {1..10}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Stopped: $name"
      rm -f "$pidfile" || true
      return 0
    fi
    sleep 1
  done

  echo "Force killing $name pid=$pid (KILL)..."
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$pidfile" || true
}

shopt -s nullglob
pidfiles=("$PID_DIR"/*.pid)
if [[ ${#pidfiles[@]} -eq 0 ]]; then
  echo "No pidfiles in $PID_DIR. Nothing to stop via pidfiles."
else
  for pf in "${pidfiles[@]}"; do
    stop_pidfile "$pf"
  done
fi
shopt -u nullglob

# убираем артефакт старого health-loop (в твоем текущем варианте он есть) :contentReference[oaicite:5]{index=5}
rm -f /tmp/health_monitor_running 2>/dev/null || true

# optionally: infra down
if [[ "$WITH_INFRA_DOWN" -eq 1 ]]; then
  if [[ -n "$COMPOSE_FILE" ]]; then
    echo "Stopping infra via compose down: $COMPOSE_FILE"
    compose -f "$COMPOSE_FILE" down
  else
    echo "infra/docker-compose.(yml|yaml) not found -> skipping down"
  fi
else
  echo "Leaving infra running (use --down to stop docker compose)."
fi

# stop healthcheck grafana services
echo "Stopping healthcheck Grafana services..."
if command -v docker >/dev/null 2>&1 && (command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1); then
  cd "$ROOT_DIR/healthcheck/grafana"
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose down
  else
    docker compose down
  fi
else
  echo "docker-compose not found. Skipping Grafana shutdown."
fi

# Stop sccache server if available
if command -v sccache >/dev/null 2>&1; then
  echo "Stopping sccache server..."
  sccache --stop-server || true
  echo "sccache server stopped"
fi

echo "==========================================="
echo "STOP DONE @ $(date)"
echo "Logs in : $LOG_DIR"
echo "==========================================="
