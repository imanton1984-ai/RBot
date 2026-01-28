#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PID_DIR="${PID_DIR:-$ROOT_DIR/run}"
mkdir -p "$LOG_DIR" "$PID_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/full_stop_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

echo "==========================================="
echo "FULL STOP @ $(date)"
echo "ROOT: $ROOT_DIR"
echo "LOG : $LOG"
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

# ------------------ load env (optional) ------------------
if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  load_env_file "$ENV_FILE"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  load_env_file "$ROOT_DIR/.env"
elif [[ -f "$ROOT_DIR/connections/.env" ]]; then
  load_env_file "$ROOT_DIR/connections/.env"
fi

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

# ------------------ options ------------------
WITH_DOCKER=1
NUKE_DOCKER_ALL=0
REMOVE_VOLUMES=1

usage() {
  cat <<EOF
Usage: ./scripts/full_stop.sh [options]
  --no-docker        do not touch docker at all
  --nuke-docker      DANGEROUS: stop ALL running docker containers on the machine
  --keep-volumes     do NOT remove compose volumes (no -v)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-docker) WITH_DOCKER=0; shift ;;
    --nuke-docker) NUKE_DOCKER_ALL=1; shift ;;
    --keep-volumes) REMOVE_VOLUMES=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

# ------------------ 1) Stop by pidfiles ------------------
echo "--- Stopping by pidfiles in $PID_DIR ---"
shopt -s nullglob
pidfiles=("$PID_DIR"/*.pid)
if [[ ${#pidfiles[@]} -eq 0 ]]; then
  echo "No pidfiles found."
else
  for pf in "${pidfiles[@]}"; do
    name="$(basename "$pf" .pid)"
    pid="$(cat "$pf" 2>/dev/null || true)"
    if [[ -n "$pid" ]]; then
      kill_pid_force "$pid" "$name"
    fi
    rm -f "$pf" 2>/dev/null || true
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

# старые артефакты (если были)
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

echo "==========================================="
echo "FULL STOP DONE @ $(date)"
echo "==========================================="
