#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-.env}"
DB_WAIT_SECONDS="${DB_WAIT_SECONDS:-10}"

# release|debug
BUILD_MODE="${BUILD_MODE:-release}"
CARGO_FLAGS=()
if [[ "$BUILD_MODE" == "release" ]]; then
  CARGO_FLAGS+=(--release)
fi

TARGET_DIR="./target/$BUILD_MODE"
PID_DIR="./run"
LOG_DIR="./logs"
mkdir -p "$PID_DIR" "$LOG_DIR"

# --- ОПРЕДЕЛЕНИЕ БИНАРНИКОВ (Новые имена из apps) ---
BIN_DB_INIT="$TARGET_DIR/svc_db_init"
BIN_INGEST="$TARGET_DIR/svc_market_ingest"
BIN_WRITER="$TARGET_DIR/svc_writer"
BIN_LIVE="$TARGET_DIR/svc_live"
BIN_BACKFILL="$TARGET_DIR/svc_backfill"
BIN_COMPUTE="$TARGET_DIR/svc_compute"
BIN_ORDERS="$TARGET_DIR/svc_orders"
BIN_TRACKER="$TARGET_DIR/svc_position_tracker"
BIN_API="$TARGET_DIR/svc_api_gateway"
BIN_HEALTH="$TARGET_DIR/svc_health"

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
}

ensure_built() {
  local bin_name="$1" # Имя бинарника в apps/src/bin/ (без .rs)
  local bin_path="$2"
  
  if [[ ! -f "$bin_path" ]]; then
    log "Binary $bin_name not found. Building..."
    cargo build "${CARGO_FLAGS[@]}" -p apps --bin "$bin_name"
  fi
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
  if [[ ! -x "$bin" ]]; then die "Binary $bin not found or not executable"; fi

  # Очистка старых процессов
  pkill -f "$(basename "$bin")" 2>/dev/null || true
  
  nohup "$bin" >> "$logfile" 2>&1 &
  echo $! > "$pidfile"
  
  sleep 1
  if kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    log "$name started. PID: $(cat "$pidfile")"
  else
    die "$name failed to start! Check $logfile"
  fi
}

main() {
  load_env
  
  # 1. Сборка всех необходимых сервисов
  log "Checking binaries..."
  ensure_built "svc_db_init"         "$BIN_DB_INIT"
  ensure_built "svc_market_ingest"   "$BIN_INGEST"
  ensure_built "svc_writer"          "$BIN_WRITER"
  ensure_built "svc_live"            "$BIN_LIVE"
  ensure_built "svc_backfill"        "$BIN_BACKFILL"
  ensure_built "svc_health"          "$BIN_HEALTH"
  # ensure_built "svc_compute"       "$BIN_COMPUTE" # Раскомментируй, когда будут готовы
  
  # 2. Инфраструктура
  log "Starting Docker infrastructure..."
  docker compose -f "$COMPOSE_FILE" up -d
  
  # Ожидание БД
  ./scripts/wait_pg.sh || die "DB wait failed"
  # Ожидание Redpanda
  ./scripts/wait_redpanda.sh || die "Redpanda wait failed"

  # 3. Миграции
  log "Running DB Init..."
  "$BIN_DB_INIT" --dir database/ddl

  # 4. Проверка здоровья БД
  if [[ -x "./scripts/db_health.sh" ]]; then
    ./scripts/db_health.sh || die "DB health check failed"
  fi

  # 5. Запуск сервисов
  
  # Healthcheck сервис (мониторит остальные)
  start_service "health" "$BIN_HEALTH"

  # Ingest (загружает пары)
  start_service "ingest" "$BIN_INGEST"
  log "Waiting for PAIRS_READY stage via health service..."
  # ingest публикует stagez сам (MARKET_INGEST_PORT из .env, по умолчанию 9001)
  ./scripts/wait_stage.sh "http://localhost:\${MARKET_INGEST_PORT:-9001}/stagez" "PAIRS_READY" 120
  start_service "writer" "$BIN_WRITER"

  # Backfill (исторические данные)
  log "Starting Backfill..."
  start_service "backfill" "$BIN_BACKFILL"
  
  # Ждем окончания загрузки истории перед лайвом (опционально, но надежнее)
  log "Waiting for BACKFILL_CANDLES_READY via ingest service..."
  ./scripts/wait_stage.sh "http://localhost:\${MARKET_INGEST_PORT:-9001}/stagez" "BACKFILL_CANDLES_READY" 3600

  # Live (Realtime WS)
  start_service "live" "$BIN_LIVE"

  log "✅ SYSTEM STARTED"
  log "Logs are in $LOG_DIR"
}

main "$@"


