#!/usr/bin/env bash
set -euo pipefail

# Определяем корневую директорию проекта
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# --- Конфигурация ---
COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yaml}"
ENV_FILE="${ENV_FILE:-.env}"
DB_WAIT_SECONDS="${DB_WAIT_SECONDS:-5}"
BUILD_MODE="${BUILD_MODE:-release}" # можно менять на debug для разработки

# Пути к бинарникам зависят от режима сборки
BIN_PATH="./target/$BUILD_MODE"
DB_INIT_BIN="$BIN_PATH/db_init"
DB_WRITER_BIN="$BIN_PATH/db_writer"

# Директории для логов и пидов
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
}

echo "[*] Cleaning up old processes..."
pkill -f "target/release/ingest" || true
pkill -f "target/release/db_writer" || true
# Даем портам 1-2 секунды освободиться
sleep 2

# Функция автоматической сборки
ensure_built() {
  local package=$1
  local bin=$2
  if [[ ! -x "$bin" ]]; then
    log "Binary $bin not found. Building $package..."
    cargo build --${BUILD_MODE} -p "$package"
  fi
}

wait_postgres() {
  log "Waiting for PostgreSQL ($DB_WAIT_SECONDS s max)..."
  local start_ts=$(date +%s)
  
  # Используем DB_PORT из .env (у вас там 5433)
  local port=${DB_PORT:-5432}

  while true; do
    # 1. Проверяем открыт ли порт локально
    if nc -z 127.0.0.1 "$port" >/dev/null 2>&1; then
      log "PostgreSQL port $port is open"
      
      # 2. Проверяем готовность самой БД внутри контейнера
      # ВАЖНО: имя сервиса тут 'timescaledb' (как в вашем выводе Docker)
      if docker compose -f "$COMPOSE_FILE" exec -T timescaledb pg_isready -U "$DB_USER" >/dev/null 2>&1; then
        log "PostgreSQL is fully ready"
        return 0
      fi
    fi

    local now=$(date +%s)
    if (( now - start_ts > DB_WAIT_SECONDS )); then
      log "Full DATABASE_URL check failed. Checking container status..."
      docker compose -f "$COMPOSE_FILE" ps
      die "PostgreSQL not ready at localhost:$port after ${DB_WAIT_SECONDS}s"
    fi
    sleep 2
  done
}

docker_up() {
  log "Starting docker infrastructure (if not running)..."
  docker compose -f "$COMPOSE_FILE" up -d
}

run_db_init() {
  log "Running db_init migrations..."
  # Передаем DATABASE_URL явно, если db_init его ожидает
  "$DB_INIT_BIN" --dir infra/database/init
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
  # Запуск в фоне. Используем env для проброса переменных
  nohup "$bin" >> "$logfile" 2>&1 &
  echo $! > "$pidfile"
  
  sleep 1
  if kill -0 $! 2>/dev/null; then
    log "$name started. PID: $(cat "$pidfile"), Log: $logfile"
  else
    die "$name failed to start! Check $logfile"
  fi
}

main() {
  load_env
  
  log "Checking/Building binaries..."
  ensure_built "db_init" "$DB_INIT_BIN"
  ensure_built "db_writer" "$DB_WRITER_BIN"

  docker_up
  wait_postgres

  run_db_init
  
  # Опционально: проверка здоровья через ваш скрипт
  if [[ -x "./scripts/db_health.sh" ]]; then
     ./scripts/db_health.sh || die "DB health check failed"
  fi
  
  echo "[...] Starting ingest..."
  ./target/release/ingest > ./logs/ingest.log 2>&1 &
  echo $! > ./run/ingest.pid

  echo "[...] Waiting for stage PAIRS_READY..."
  ./scripts/wait_stage.sh "http://localhost:8081/stagez" "PAIRS_READY" 120

  # Запуск сервисов
  start_service "db_writer" "$DB_WRITER_BIN"

  log "✅ ALL SYSTEMS GO"
}

main "$@"

