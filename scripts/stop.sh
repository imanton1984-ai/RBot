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

# ------------------ helpers ------------------
compose() {
  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose "$@"
  else
    return 127
  fi
}

load_env_file() {
  local f="$1"
  echo "Loading env: $f"
  set -a
  # shellcheck disable=SC1090
  source "$f"
  set +a
}

kill_pid_force() {
  local pid="$1" name="${2:-proc}"
  if [[ -z "$pid" ]]; then return 0; fi
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Already dead: $name pid=$pid"
    return 0
  fi

  echo "TERM: $name pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true

  for _ in {1..6}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Stopped: $name pid=$pid"
      return 0
    fi
    sleep 1
  done

  echo "KILL: $name pid=$pid"
  kill -KILL "$pid" 2>/dev/null || true
}

# env (опционально, чтобы stop знал KAFKA_BROKERS/и т.д.)
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  load_env_file "$ENV_FILE"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  load_env_file "$ROOT_DIR/.env"
elif [[ -f "$ROOT_DIR/connections/.env" ]]; then
  load_env_file "$ROOT_DIR/connections/.env"
fi

# args
WITH_INFRA_DOWN=0
WITH_DOCKER=1
NUKE_DOCKER_ALL=0
REMOVE_VOLUMES=1

usage() {
  cat <<EOF
Usage: ./scripts/stop.sh [options]
  --down           also stop docker compose infra (compose down)
  --no-docker      do not touch docker at all
  --nuke-docker    DANGEROUS: stop ALL running docker containers on the machine
  --keep-volumes   do NOT remove compose volumes (no -v)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --down) WITH_INFRA_DOWN=1; shift ;;
    --no-docker) WITH_DOCKER=0; shift ;;
    --nuke-docker) NUKE_DOCKER_ALL=1; shift ;;
    --keep-volumes) REMOVE_VOLUMES=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# stop connections
if [[ -f "./run/connections.pid" ]]; then
  PID="$(cat ./run/connections.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping connections (pid $PID)..."
    kill "$PID" || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f ./run/connections.pid
fi

# stop compute service
if [[ -f "./run/compute.pid" ]]; then
  PID="$(cat ./run/compute.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping compute service (pid $PID)..."
    kill "$PID" || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f ./run/compute.pid
fi

# stop ingestor service
if [[ -f "./run/ingestor.pid" ]]; then
  PID="$(cat ./run/ingestor.pid)"
  if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping ingestor service (pid $PID)..."
    kill "$PID" || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
  fi
  rm -f ./run/ingestor.pid
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

# ------------------ 1) Stop by pidfiles ------------------
echo "--- Stopping by pidfiles in $PID_DIR ---"
shopt -s nullglob
pidfiles=("$PID_DIR"/*.pid)
if [[ ${#pidfiles[@]} -eq 0 ]]; then
  echo "No pidfiles found."
  # Try to find and stop Rust compute process that includes predictor module
  echo "Looking for Rust compute processes with predictor module..."
  while read -r pid cmd; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    echo "TERM compute process: pid=$pid cmd=$cmd"
    kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f "compute.*--module.*predictors?" -l || true)
  
  sleep 2
  
  # Force kill if still running
  while read -r pid cmd; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "KILL compute process: pid=$pid cmd=$cmd"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done < <(pgrep -f "compute.*--module.*predictors?" -l || true)
else
  for pf in "${pidfiles[@]}"; do
    stop_pidfile "$pf"
  done
fi
shopt -u nullglob

# ------------------ 2) Kill leftover ONLY from this repo ------------------
# ВАЖНО: больше не убиваем процессы "по словам" (writer/backfill и т.д.) по всей системе.
# Убиваем только то, что в cmdline содержит ROOT_DIR (т.е. реально процессы этого репо).
echo "--- Killing leftover processes that belong to this repo only (cmdline contains ROOT_DIR) ---"
while read -r pid cmd; do
  [[ -z "$pid" ]] && continue
  # защита от пустых/битых строк
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "TERM leftover: pid=$pid cmd=$cmd"
  kill -TERM "$pid" 2>/dev/null || true
done < <(ps -eo pid=,cmd= | grep -F "$ROOT_DIR" | grep -v grep || true)

sleep 2

while read -r pid cmd; do
  [[ -z "$pid" ]] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "KILL leftover: pid=$pid cmd=$cmd"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done < <(ps -eo pid=,cmd= | grep -F "$ROOT_DIR" | grep -v grep || true)

# Also look for any remaining compute processes with predictor module
echo "Looking for any remaining compute processes with predictor module..."
while read -r pid cmd; do
  [[ -z "$pid" ]] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "TERM compute process: pid=$pid cmd=$cmd"
  kill -TERM "$pid" 2>/dev/null || true
done < <(pgrep -f "compute.*predictor" -l || true)

# Also stop any super_entry strategy processes (dataset builder, backtester)
echo "Looking for super_entry strategy processes..."
while read -r pid cmd; do
  [[ -z "$pid" ]] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  echo "TERM super_entry: pid=$pid cmd=$cmd"
  kill -TERM "$pid" 2>/dev/null || true
done < <(pgrep -f "super_entry" -l || true)

sleep 2

# Force kill if still running
while read -r pid cmd; do
  [[ -z "$pid" ]] && continue
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  if kill -0 "$pid" 2>/dev/null; then
    echo "KILL compute process: pid=$pid cmd=$cmd"
    kill -KILL "$pid" 2>/dev/null || true
  fi
done < <(pgrep -f "compute.*predictor" -l || true)

# убираем артефакт старого health-loop (в твоем текущем варианте он есть) :contentReference[oaicite:0]{index=0}
rm -f /tmp/health_monitor_running 2>/dev/null || true

# ------------------ 3) Docker: stop infra containers ------------------
if [[ "$WITH_DOCKER" -eq 1 ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker not found -> skipping docker stop"
  else
    if [[ "$NUKE_DOCKER_ALL" -eq 1 ]]; then
      echo "!!! NUKE DOCKER: stopping ALL running containers !!!"
      ids="$(docker ps -q || true)"
      if [[ -n "$ids" ]]; then
        docker rm -f $ids || true
      fi
      echo "All containers stopped."
    else
      COMPOSE_FILE=""
      if [[ -f "$ROOT_DIR/infra/docker-compose.yaml" ]]; then
        COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yaml"
      elif [[ -f "$ROOT_DIR/infra/docker-compose.yml" ]]; then
        COMPOSE_FILE="$ROOT_DIR/infra/docker-compose.yml"
      fi

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
        
        # Still try to bring down compose if file exists (but don't force)
        if [[ -n "$COMPOSE_FILE" ]] && compose -f "$COMPOSE_FILE" ps >/dev/null 2>&1; then
          echo "--- docker compose down (force) ---"
          if [[ "$REMOVE_VOLUMES" -eq 1 ]]; then
            compose -f "$COMPOSE_FILE" down -v --remove-orphans || true
          else
            compose -f "$COMPOSE_FILE" down --remove-orphans || true
          fi
        else
          echo "compose file not found or compose not available -> fallback to container-name stop"
        fi
      fi

      # Fallback by known container names
      CONTAINERS="${BOT_DOCKER_CONTAINERS:-redpanda redpanda_console redpanda-console timescaledb}"
      echo "--- docker rm -f known bot containers: $CONTAINERS ---"
      for c in $CONTAINERS; do
        if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
          echo "docker rm -f $c"
          docker rm -f "$c" || true
        fi
      done

      # Optional: remove bot networks if exist
      NETS="${BOT_DOCKER_NETWORKS:-infra_botnet botnet}"
      echo "--- removing known bot networks (if exist): $NETS ---"
      for n in $NETS; do
        if docker network ls --format '{{.Name}}' | grep -qx "$n"; then
          echo "docker network rm $n"
          docker network rm "$n" || true
        fi
      done
    fi
  fi
else
  echo "Docker untouched (--no-docker)."
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