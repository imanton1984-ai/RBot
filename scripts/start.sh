#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-.env}"
DB_WAIT_SECONDS="${DB_WAIT_SECONDS:-5}"

# release|debug
BUILD_MODE="${BUILD_MODE:-release}"

# build flags
CARGO_FLAGS=()
if [[ "$BUILD_MODE" == "release" ]]; then
  CARGO_FLAGS+=(--release)
elif [[ "$BUILD_MODE" != "debug" ]]; then
  echo "❌ BUILD_MODE must be 'release' or 'debug' (got: $BUILD_MODE)" >&2
  exit 1
fi

BIN_PATH="./target/$BUILD_MODE"
DB_INIT_BIN="$BIN_PATH/db_init"
DB_WRITER_BIN="$BIN_PATH/db_writer"
INGEST_BIN="$BIN_PATH/ingest"
TELEMETRY_BIN="$BIN_PATH/telemetry"
LIVE_BIN="$BIN_PATH/svc_live"
BACKFILL_BIN="$BIN_PATH/svc_backfill"

PID_DIR="./run"
LOG_DIR="./logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

log() { echo "[$(date +'%H:%M:%S')] $*"; }
die() { echo "❌ $*" >&2; exit 1; }

load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    log "Loaded env: $ENV_FILE"
  else
    die ".env file not found at $ENV_FILE"
  fi

  : "${DATABASE_URL:?DATABASE_URL is required in .env}"

  # дефолты, чтобы set -u не ронял скрипт
  : "${DB_USER:=postgres}"
  : "${DB_PASSWORD:=postgres}"
  : "${DB_NAME:=timescaledb_binance}"
  : "${DB_PORT:=5433}"
}

stop_by_pidfile() {
  local name="$1"
  local pidfile="$PID_DIR/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      log "Stopping $name (pid $pid)..."
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
}

ensure_built() {
  local package="$1"
  local bin="$2"
  if [[ ! -x "$bin" ]]; then
    log "Binary $bin not found. Building $package..."
    cargo build "${CARGO_FLAGS[@]}" -p "$package"
  fi
}

docker_up() {
  log "Starting docker infrastructure (if not running)..."
  docker compose -f "$COMPOSE_FILE" up -d
}

wait_postgres() {
  log "Waiting for PostgreSQL ($DB_WAIT_SECONDS s max)..."
  local start_ts
  start_ts=$(date +%s)

  while true; do
    if nc -z 127.0.0.1 "$DB_PORT" >/dev/null 2>&1; then
      log "PostgreSQL port $DB_PORT is open"

      # сервис в compose называется timescaledb :contentReference[oaicite:2]{index=2}
      if docker compose -f "$COMPOSE_FILE" exec -T timescaledb \
        pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
        log "PostgreSQL is fully ready"
        return 0
      fi
    fi

    local now
    now=$(date +%s)
    if (( now - start_ts > DB_WAIT_SECONDS )); then
      log "DB not ready. Container status:"
      docker compose -f "$COMPOSE_FILE" ps
      die "PostgreSQL not ready at localhost:$DB_PORT after ${DB_WAIT_SECONDS}s"
    fi
    sleep 2
  done
}

run_db_init() {
  log "Running db_init migrations..."
  "$DB_INIT_BIN" --dir database/ddl
}

start_service() {
  local name="$1"
  local bin="$2"
  local pidfile="$PID_DIR/${name}.pid"
  local logfile="$LOG_DIR/${name}.log"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    log "Service $name already running (pid $(cat "$pidfile"))"
    return 0
  fi

  log "Starting $name..."
  nohup "$bin" >> "$logfile" 2>&1 &
  echo $! > "$pidfile"

  sleep 1
  if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    log "$name started. PID: $(cat "$pidfile"), Log: $logfile"
  else
    die "$name failed to start! Check $logfile"
  fi
}

main() {
  load_env

  log "[*] Cleaning up old processes..."
  stop_by_pidfile "ingest"
  stop_by_pidfile "db_writer"
  stop_by_pidfile "telemetry"
  pkill -f "/ingest" 2>/dev/null || true
  pkill -f "/db_writer" 2>/dev/null || true
  pkill -f "/telemetry" 2>/dev/null || true
  sleep 2

  log "Checking/Building binaries..."
  ensure_built "db_init" "$DB_INIT_BIN"
  ensure_built "db_writer" "$DB_WRITER_BIN"
  ensure_built "ingest" "$INGEST_BIN"
  ensure_built "telemetry" "$TELEMETRY_BIN"
  ensure_built "svc_live" "$LIVE_BIN"
  ensure_built "svc_backfill" "$BACKFILL_BIN"

  docker_up
  wait_postgres
  # ADDED: Wait for Redpanda explicitly
  log "Waiting for Redpanda..."
  ./scripts/wait_redpanda.sh || die "Redpanda failed to start"
  run_db_init

  if [[ -x "./scripts/db_health.sh" ]]; then
    ./scripts/db_health.sh || die "DB health check failed"
  fi

  # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
  
  # 1. Запускаем ingest (он начнет с загрузки пар, но не начнет backfill, пока не пройдет Loading Pairs)
  start_service "ingest" "$INGEST_BIN"

  # 2. Ждем только готовности списка пар (это быстро)
  log "Waiting for stage PAIRS_READY..."
  ./scripts/wait_stage.sh "http://localhost:8081/stagez" "PAIRS_READY" 120

  # 3. Запускаем backfill service для загрузки исторических данных
  log "Starting backfill service to load historical data..."
  start_service "backfill" "$BACKFILL_BIN"

  # 4. СРАЗУ запускаем db_writer, чтобы он был готов принимать поток данных от backfill
  log "Starting db_writer to consume backfill stream..."
  start_service "db_writer" "$DB_WRITER_BIN"

  # 5. Start the telemetry service to monitor all connections
  log "Starting telemetry service to monitor connections..."
  start_service "telemetry" "$TELEMETRY_BIN"

  # 6. Now wait for backfill to complete
  log "Waiting for stage BACKFILL_CANDLES_READY..."
  ./scripts/wait_stage.sh "http://localhost:8081/stagez" "BACKFILL_CANDLES_READY" 3600

  # 7. Finally start the live service to begin real-time WebSocket data processing
  log "Starting live service for real-time WebSocket data processing..."
  start_service "live" "$LIVE_BIN"

  log "✅ ALL SYSTEMS GO (Realtime mode active)"
}

main "$@"


