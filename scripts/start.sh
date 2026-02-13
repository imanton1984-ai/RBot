#!/usr/bin/env bash

# --- 1. ОПРЕДЕЛЕНИЕ ФУНКЦИЙ (Сначала это!) ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_RED='\033[31m'
C_GREEN='\033[32m'
C_YELLOW='\033[33m'

log() { echo -e "${C_BOLD}[$(date +%T)]${C_RESET} $*"; }
ok() { echo -e "${C_BOLD}[$(date +%T)]${C_RESET} ${C_GREEN}OK:${C_RESET} $*"; }
warn() { echo -e "${C_BOLD}[$(date +%T)]${C_RESET} ${C_YELLOW}WARN:${C_RESET} $*"; }
die() { echo -e "${C_BOLD}[$(date +%T)]${C_RESET} ${C_RED}ERROR:${C_RESET} $*"; exit 1; }
section() { echo -e "\n${C_BOLD}=== $* ===${C_RESET}"; }

# --- 2. НАСТРОЙКА CUDA 13.1 ---
# Мы используем универсальный путь /usr/local/cuda, который ссылается на 13.1
SELECTED_CUDA="/usr/local/cuda"

if [ ! -x "$SELECTED_CUDA/bin/nvcc" ]; then
    # Резервный поиск, если симлинк не настроен
    if [ -x "/usr/local/cuda-13.1/bin/nvcc" ]; then
        SELECTED_CUDA="/usr/local/cuda-13.1"
    else
        die "CUDA 13.1 binaries not found. Please run: sudo ln -s /usr/local/cuda-13.1 /usr/local/cuda"
    fi
fi

export CUDA_HOME="$SELECTED_CUDA"
export PATH="$CUDA_HOME/bin:$PATH"
# Добавляем стандартные пути библиотек и пути для CUPTI (нужно для профилирования)
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_ROOT="$CUDA_HOME"

section "CUDA ENVIRONMENT"
ok "Using CUDA from: $CUDA_HOME"
log "NVCC Path: $(which nvcc)"
log "NVCC Version: $(nvcc --version | grep release)"

# --- 3. НАСТРОЙКА XGBOOST PATHS ---
XGB_LIB_PATH="/home/anton/Desktop/Rust_trader/third_party/xgboost/install/lib"
export LIBRARY_PATH="${LIBRARY_PATH:-}:$XGB_LIB_PATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$XGB_LIB_PATH"

# Проверка для логов
log "Building with CUDA from: $(which nvcc)"
log "CUDA Version: $(nvcc --version | grep release)"

# ==============================================================================
# Automatic cuDNN Installation for GPU Acceleration
# ==============================================================================
# This section checks if cuDNN 9 is installed, which is required for XGBoost GPU support.
# If it's not found, it will attempt to install it. This requires sudo privileges.

# Check if we are on a debian-based system with apt-get
if command -v apt-get &> /dev/null; then
    # Check for libcudnn.so.9
    if ! ldconfig -p | grep -q libcudnn.so.9; then
        echo "WARNING: libcudnn.so.9 not found. Attempting to install NVIDIA cuDNN 9 for CUDA 12."
        echo "This is required for GPU acceleration and will require sudo privileges."

        # Check if running with sudo, if not, prompt
        if [ "$EUID" -ne 0 ]; then
            echo "Please enter your password for sudo to continue with the installation."
        fi

        set -e # Exit immediately if a command exits with a non-zero status.

        # Install wget if not present
        if ! command -v wget &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y wget
        fi

        # Install cuDNN 9 for CUDA 12 on Ubuntu 24.04
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb
        sudo apt-get update
        sudo apt-get install -y libcudnn9-cuda-12
        rm /tmp/cuda-keyring.deb

        echo "cuDNN 9 installation complete."
        set +e # Return to default error handling
    else
        echo "Found libcudnn.so.9. Skipping cuDNN installation."
    fi
else
    echo "WARNING: 'apt-get' not found. Cannot automatically check or install cuDNN. Please ensure cuDNN 9 for your CUDA version is installed manually."
fi

# ==============================================================================

# Check CUDA version compatibility for XGBoost
echo "Checking CUDA version compatibility for XGBoost..."
CUDA_VERSION=$(nvcc --version | grep "V[0-9]" | cut -d' ' -f6 | sed 's/V//')
REQUIRED_CUDA_VERSION="12.9"

echo "Detected CUDA version: $CUDA_VERSION"
echo "Required CUDA version for XGBoost GPU support: >= $REQUIRED_CUDA_VERSION"

# Compare versions using sort -V (version sort) to properly handle version comparison
if [ "$(printf '%s\n%s' "$REQUIRED_CUDA_VERSION" "$CUDA_VERSION" | sort -V | head -n1)" = "$REQUIRED_CUDA_VERSION" ]; then
    echo "CUDA version is sufficient for latest XGBoost."
    XGBOOST_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_xgboost_cuda.sh"
else
    echo "WARNING: CUDA version $CUDA_VERSION is less than required $REQUIRED_CUDA_VERSION"
    echo ""
    echo "********************************************************************************"
    echo "CUDA VERSION ISSUE:"
    echo "Your CUDA version ($CUDA_VERSION) is insufficient for the latest XGBoost."
    echo "Checking for newer CUDA installations..."
    
    # Check if newer CUDA versions are installed but not properly linked
    if [ -d "/usr/local/cuda-13.1" ] || [ -d "/usr/local/cuda-13.0" ] || [ -d "/usr/local/cuda-12.9" ]; then
        echo "Newer CUDA versions found but not properly linked!"
        echo "Current nvcc path: $(which nvcc)"
        echo "Available CUDA versions:"
        ls -la /usr/local/cuda* | grep -E "(cuda-13|cuda-12\\.[9-9])" || echo "No newer CUDA versions found"
        echo ""
        echo "To fix this, you may need to update your CUDA symlink:"
        echo "sudo ln -sf /usr/local/cuda-13.1 /usr/local/cuda"
        echo "Then restart your shell or run: source ~/.bashrc"
        echo ""
        echo "Would you like to continue with an older XGBoost version instead? (y/n): \c"
        read -p "" -n 1 -r REPLY
        echo
    else
        echo "No newer CUDA versions found on the system."
        echo ""
        echo "To upgrade CUDA:"
        echo "1. Visit https://developer.nvidia.com/cuda-downloads"
        echo "2. Download CUDA 12.9 or higher for your system"
        echo "3. Follow the installation instructions"
        echo "4. Restart your terminal/shell after installation"
        echo "5. Verify with: nvcc --version"
        echo ""
        read -p "Would you like to continue with an older XGBoost version instead? (y/n): " -n 1 -r REPLY
        echo
    fi
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Using compatible XGBoost version v2.1.1 for CUDA $CUDA_VERSION..."
        XGBOOST_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_xgboost_cuda_compat.sh"
        if [ ! -f "$XGBOOST_SCRIPT_PATH" ]; then
            echo "Creating compatible build script..."
            sed 's/XGB_VER="v3.2.0"/XGB_VER="v2.1.1"/' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_xgboost_cuda.sh" > "$XGBOOST_SCRIPT_PATH"
        fi
    else
        echo "Exiting. Please update your CUDA installation and try again."
        exit 1
    fi
fi

# Build XGBoost with CUDA support
echo "Building XGBoost with detected compatible script..."
if [ -f "$XGBOOST_SCRIPT_PATH" ]; then
    echo "Running XGBoost build script: $XGBOOST_SCRIPT_PATH"
    chmod +x "$XGBOOST_SCRIPT_PATH"
    "$XGBOOST_SCRIPT_PATH"

    # Set XGBoost library path
    export XGBOOST_LIB_DIR="/home/anton/Desktop/Rust_trader/third_party/xgboost/install/lib"
    export LD_LIBRARY_PATH="$XGBOOST_LIB_DIR:$LD_LIBRARY_PATH"
    export LIBRARY_PATH="$XGBOOST_LIB_DIR:$LIBRARY_PATH"

    echo "XGBoost library path set to: $XGBOOST_LIB_DIR"
else
    echo "WARNING: XGBoost build script not found at $XGBOOST_SCRIPT_PATH"
fi

# scripts/start.sh

# CUDA Environment Variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"

# ----------------- Продолжение основного скрипта -----------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"
PID_DIR="${PID_DIR:-$ROOT_DIR/run}"
mkdir -p "$LOG_DIR" "$PID_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
START_LOG="$LOG_DIR/start_${TS}.log"

# Логи как раньше: и в файл, и в консоль
exec > >(tee -a "$START_LOG") 2>&1

# ----------------- pretty output helpers -----------------

NO_COLOR="${NO_COLOR:-0}"
if [[ "$NO_COLOR" == "1" ]]; then
  C_RESET=""; C_DIM=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
else
  C_RESET=$'\033[0m'
  C_DIM=$'\033[2m'
  C_BOLD=$'\033[1m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_CYAN=$'\033[36m'
fi


ts() { date +"%H:%M:%S"; }

hr() {
  echo -e "${C_DIM}===========================================${C_RESET}"
}

section() {
  local title="$1"
  hr
  echo -e "${C_BOLD}${C_CYAN}${title}${C_RESET}"
  hr
}

log()  { echo -e "${C_DIM}[$(ts)]${C_RESET} $*"; }
ok()   { echo -e "${C_DIM}[$(ts)]${C_RESET} ${C_GREEN}✅${C_RESET} $*"; }
warn() { echo -e "${C_DIM}[$(ts)]${C_RESET} ${C_YELLOW}⚠️${C_RESET} $*"; }
err()  { echo -e "${C_DIM}[$(ts)]${C_RESET} ${C_RED}❌${C_RESET} $*"; }

die() { err "$*"; exit 1; }

# ----------------- core helpers -----------------

load_env_file() {
  local f="$1"
  log "Loading env: ${C_BOLD}$f${C_RESET}"
  set -a
  # shellcheck disable=SC1090
  source "$f"
  set +a
}

compose() {
  if command -v docker >/dev/null 2>&1; then
    docker compose "$@"
  else
    die "docker not found"
  fi
}

wait_for_tcp() {
  local host="$1" port="$2" timeout="$3"
  log "Waiting for TCP ${C_BOLD}${host}:${port}${C_RESET} (timeout ${timeout}s)..."
  local start now
  start="$(date +%s)"
  while true; do
    if (echo >/dev/tcp/"$host"/"$port") >/dev/null 2>&1; then
      ok "TCP ready: ${host}:${port}"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      die "Timeout waiting for ${host}:${port}"
    fi
    sleep 1
  done
}

# Запуск команды с pseudo-TTY, чтобы cargo рисовал progress bar даже при tee.
run_with_pty() {
  local cmd="$1"
  if command -v script >/dev/null 2>&1; then
    # -q quiet, -e return exit code of command (если доступно), /dev/null как typescript file
    # В разных дистрибутивах ключ -e есть; если нет — просто уберем.
    if script -q -e -c "true" /dev/null >/dev/null 2>&1; then
      script -q -e -c "$cmd" /dev/null
    else
      script -q -c "$cmd" /dev/null
    fi
  else
    warn "utility 'script' not found -> cargo progress bar may be disabled"
    bash -lc "$cmd"
  fi
}

# ----------------- banner -----------------

section "START"
log "ROOT: ${C_BOLD}$ROOT_DIR${C_RESET}"
log "LOG : ${C_BOLD}$START_LOG${C_RESET}"

# Start sccache server if available
if command -v sccache >/dev/null 2>&1; then
  if ! pgrep sccache > /dev/null; then
    log "Starting sccache server..."
    sccache --start-server
    ok "sccache server started"
  else
    ok "sccache server is already running."
  fi
else
  warn "sccache not found, skipping server start"
fi

# ----------------- env -----------------

if [[ -n "${ENV_FILE:-}" && -f "${ENV_FILE}" ]]; then
  load_env_file "$ENV_FILE"
elif [[ -f "$ROOT_DIR/.env" ]]; then
  load_env_file "$ROOT_DIR/.env"
else
  warn "No .env found (ENV_FILE/.env). Continuing..."
fi

# ----------------- args -----------------

WITH_INFRA=1
BUILD_MODE="auto"          # auto|never|always
COMPOSE_BUILD=0
TAIL_LOGS=0
SERVICES_OVERRIDE=""

usage() {
  cat <<EOF
Usage: ./scripts/start.sh [options]
  --no-infra           do not start docker compose infra
  --build              force cargo build
  --no-build           never build (fail if binaries missing)
  --compose-build      docker compose up -d --build
  --services "a b c"   start only selected services (space-separated)
  --tail               tail service logs after start
  -h, --help           show help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-infra) WITH_INFRA=0; shift ;;
    --build) BUILD_MODE="always"; shift ;;
    --no-build) BUILD_MODE="never"; shift ;;
    --compose-build) COMPOSE_BUILD=1; shift ;;
    --services) SERVICES_OVERRIDE="${2:-}"; shift 2 ;;
    --tail) TAIL_LOGS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown arg: $1" ;;
  esac
done

# ----------------- toolchain -----------------

section "TOOLCHAIN"
log "rustc: $(rustc -V 2>/dev/null || echo 'not found')"
log "cargo: $(cargo -V 2>/dev/null || echo 'not found')"

# ----------------- config -----------------

COMPOSE_FILE="${COMPOSE_FILE:-infra/docker-compose.yaml}"
BUILD_PROFILE="${BUILD_PROFILE:-release}" # release|debug
SERVICE_BINS="${SERVICE_BINS:-connections ingestor}"

if [[ -n "$SERVICES_OVERRIDE" ]]; then
  SERVICE_BINS="$SERVICES_OVERRIDE"
fi

if [[ "$BUILD_PROFILE" != "release" && "$BUILD_PROFILE" != "debug" ]]; then
  die "BUILD_PROFILE must be release|debug (got: $BUILD_PROFILE)"
fi

TARGET_DIR="$ROOT_DIR/target/$BUILD_PROFILE"
mkdir -p "$TARGET_DIR" 2>/dev/null || true

find_bin_path() {
  local name="$1"
  local p="$TARGET_DIR/$name"
  [[ -x "$p" ]] && { echo "$p"; return 0; }
  local p2="$ROOT_DIR/target/release/$name"
  [[ -x "$p2" ]] && { echo "$p2"; return 0; }
  return 1
}

# ----------------- 1) infra -----------------

if [[ "$WITH_INFRA" -eq 1 ]]; then
  section "INFRA"
  log "Compose file: ${C_BOLD}$COMPOSE_FILE${C_RESET}"

  STAMP_FILE="$ROOT_DIR/.compose_last_up"
  if [[ "$COMPOSE_BUILD" -eq 0 ]]; then
    if [[ ! -f "$STAMP_FILE" || "$COMPOSE_FILE" -nt "$STAMP_FILE" ]]; then
      warn "Compose file changed since last up -> enabling compose build automatically."
      COMPOSE_BUILD=1
    fi
  fi

  if [[ "$COMPOSE_BUILD" -eq 1 ]]; then
    log "docker compose up -d --build --remove-orphans"
    compose -f "$COMPOSE_FILE" up -d --build --remove-orphans
  else
    log "docker compose up -d --remove-orphans"
    compose -f "$COMPOSE_FILE" up -d --remove-orphans
  fi
  date > "$STAMP_FILE"

  DB_HOST="${DB_TCP_HOST:-127.0.0.1}"
  DB_PORT="${DB_TCP_PORT:-5433}"
  KAFKA_HOST="${KAFKA_TCP_HOST:-127.0.0.1}"
  KAFKA_PORT="${KAFKA_TCP_PORT:-19092}"

  wait_for_tcp "$DB_HOST" "$DB_PORT" 45
  wait_for_tcp "$KAFKA_HOST" "$KAFKA_PORT" 45
else
  section "INFRA"
  warn "Skipping infra (--no-infra)."
fi

# --- DB INITIALIZATION: check if DB is initialized and initialize if needed ---
check_db_initialized() {
  local db_url="${DATABASE_URL:-postgres://postgres:postgres@127.0.0.1:5433/timescaledb_binance}"

  # Check if market.pairs table exists
  if PGPASSWORD=postgres psql -h 127.0.0.1 -p 5433 -U postgres -d timescaledb_binance -tAc "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'market' AND table_name = 'pairs');" 2>/dev/null | grep -q 't'; then
    return 0  # Database is initialized
  else
    return 1  # Database is not initialized
  fi
}

initialize_database() {
  log "Database not initialized. Initializing database schema..."

  # Build the database initialization service if it doesn't exist
  local db_init_bin="$TARGET_DIR/svc_db_init"
  if [[ ! -x "$db_init_bin" ]]; then
    log "Building database initialization service..."
    if [[ "$BUILD_PROFILE" == "release" ]]; then
      run_with_pty "cargo build --release -p database --bin svc_db_init"
    else
      run_with_pty "cargo build -p database --bin svc_db_init"
    fi
  fi

  # Run the database initialization
  local db_init_bin_path
  db_init_bin_path="$(find_bin_path "svc_db_init" || true)"
  if [[ -n "$db_init_bin_path" ]]; then
    log "Running database initialization..."
    "$db_init_bin_path" 2>&1 | tee -a "$LOG_DIR/db_init.out"
    if [[ $? -eq 0 ]]; then
      ok "Database initialization completed successfully."
    else
      err "Database initialization failed."
      return 1
    fi
  else
    err "Database initialization binary not found: svc_db_init"
    return 1
  fi
}

export MIN_PAIRS="${MIN_PAIRS:-300}"
export RUST_LOG="${RUST_LOG:-info,ingestor=info,connections=info}"

# Check if database is initialized, and initialize if needed
section "DATABASE INITIALIZATION CHECK"
if check_db_initialized; then
  ok "Database is already initialized."
else
  warn "Database not initialized. Starting initialization..."
  initialize_database
  if [[ $? -ne 0 ]]; then
    die "Database initialization failed. Cannot proceed."
  fi
fi

# ----------------- 2) build -----------------

section "BUILD"

need_build=0

if [[ "$BUILD_MODE" == "always" ]]; then
  need_build=1
elif [[ "$BUILD_MODE" == "never" ]]; then
  need_build=0
else
  # auto: если нет бинарника — билд
  for s in $SERVICE_BINS; do
    if ! find_bin_path "$s" >/dev/null 2>&1; then
      need_build=1
      break
    fi
  done

  # toolchain stamp
  if [[ "$need_build" -eq 0 ]]; then
    RUSTC_STAMP_FILE="$TARGET_DIR/.rustc_version"
    CUR_RUSTC="$(rustc -V 2>/dev/null || true)"
    OLD_RUSTC="$(cat "$RUSTC_STAMP_FILE" 2>/dev/null || true)"
    if [[ -n "$CUR_RUSTC" && "$CUR_RUSTC" != "$OLD_RUSTC" ]]; then
      warn "rustc changed since last build:"
      log "old: ${OLD_RUSTC:-<none>}"
      log "new: $CUR_RUSTC"
      need_build=1
    fi
  fi

  # Cargo.lock / Cargo.toml newer than binaries
  if [[ "$need_build" -eq 0 ]]; then
    newest_src=0
    for f in "$ROOT_DIR/Cargo.lock" "$ROOT_DIR/Cargo.toml"; do
      [[ -f "$f" ]] || continue
      t="$(stat -c %Y "$f" 2>/dev/null || echo 0)"
      (( t > newest_src )) && newest_src="$t"
    done

    oldest_bin=9999999999
    for s in $SERVICE_BINS; do
      b="$(find_bin_path "$s" || true)"
      [[ -n "$b" ]] || continue
      t="$(stat -c %Y "$b" 2>/dev/null || echo 9999999999)"
      (( t < oldest_bin )) && oldest_bin="$t"
    done

    if (( newest_src > 0 && oldest_bin < 9999999999 && newest_src > oldest_bin )); then
      warn "Cargo.lock/Cargo.toml newer than binaries -> enabling build."
      need_build=1
    fi
  fi
fi

if [[ "$BUILD_MODE" == "never" ]]; then
  for s in $SERVICE_BINS; do
    if ! find_bin_path "$s" >/dev/null 2>&1; then
      die "--no-build was set but binary missing: $s"
    fi
  done
fi

if [[ "$need_build" -eq 1 ]]; then
  command -v cargo >/dev/null 2>&1 || die "cargo not found but build is required."

  # ВАЖНО: прогресс-бар cargo
  export CARGO_TERM_COLOR="${CARGO_TERM_COLOR:-always}"
  export CARGO_TERM_PROGRESS_WHEN="${CARGO_TERM_PROGRESS_WHEN:-always}"
  export CARGO_TERM_PROGRESS_WIDTH="${CARGO_TERM_PROGRESS_WIDTH:-80}"

  log "Profile: ${C_BOLD}$BUILD_PROFILE${C_RESET}"
  log "Targets : ${C_BOLD}$SERVICE_BINS${C_RESET}"

  if [[ "$BUILD_PROFILE" == "release" ]]; then
    # Build history compute with CUDA features
    cargo build --release -p compute --bin compute_history --features cuda
    # Build realtime compute without CUDA features (for laptop fallback)
    cargo build --release -p compute --bin compute_realtime
  else
    # Build history compute with CUDA features
    cargo build -p compute --bin compute_history --features cuda
    # Build realtime compute without CUDA features (for laptop fallback)
    cargo build -p compute --bin compute_realtime
  fi


  # Если хочешь ограничить сборку только нужными бинари (сильно ускоряет),
  # можно выставить: CARGO_BUILD_CMD="cargo build --release --bin svc_a --bin svc_b"
  log "+ ${C_BOLD}$CARGO_BUILD_CMD${C_RESET}"

  run_with_pty "$CARGO_BUILD_CMD"

  rustc -V 2>/dev/null > "$TARGET_DIR/.rustc_version" || true
  ok "Build finished."
else
  ok "Skipping build (auto-mode: binaries are up-to-date)."
fi





# Create necessary Kafka topics
section "KAFKA TOPICS SETUP"
log "Creating necessary Kafka topics..."
docker exec redpanda rpk topic create candles.close --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic candles.close already exists or error occurred"
docker exec redpanda rpk topic create candles.update --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic candles.update already exists or error occurred"
docker exec redpanda rpk topic create indicators.close --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic indicators.close already exists or error occurred"
docker exec redpanda rpk topic create ws_feed --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic ws_feed already exists or error occurred"
docker exec redpanda rpk topic create signals.raw --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic signals.raw already exists or error occurred"
docker exec redpanda rpk topic create signals.final --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic signals.final already exists or error occurred"
docker exec redpanda rpk topic create features.snapshot --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic features.snapshot already exists or error occurred"
docker exec redpanda rpk topic create orders.cmd --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic orders.cmd already exists or error occurred"
docker exec redpanda rpk topic create orders.events --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic orders.events already exists or error occurred"
docker exec redpanda rpk topic create positions.events --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || echo "  Topic positions.events already exists or error occurred"
ok "Kafka topics setup completed."

# Set recommended environment variables for history/realtime modes
export DB_PERSIST_MODE=history
export DB_PERSIST_CHUNK_SIZE_HISTORY=50000
export DB_PERSIST_MAX_BATCH=200000
export DB_PERSIST_FLUSH_MS=150
export DB_PERSIST_HISTORY_SKIP_JSON=1   # огромный выигрыш, если jsonb тяжёлый
export DB_PERSIST_HISTORY_UPSERT=0      # append

# For realtime (безопасно):
export DB_PERSIST_MODE_REALTIME=realtime
export DB_PERSIST_CHUNK_SIZE_REALTIME=2000
export DB_PERSIST_MAX_BATCH_REALTIME=20000
export DB_PERSIST_FLUSH_MS_REALTIME=50

# --- CANDLES: load historical candles and start real-time ingestion ---
section "ONESHOT: load_candles"
log "Loading historical candles and starting real-time ingestion..."
log "This may take a few minutes depending on the number of pairs..."

# Run candle loading (this will load historical and start real-time)
# Only run after pairs are confirmed ready
# run_oneshot "load_candles"
  warn "Skipping oneshot binaries (ingestor_pairs/load_candles) - not used in current architecture."



section "START SERVICES"

start_one() {
  local name="$1"
  local pidfile="$PID_DIR/$name.pid"
  local outfile="$LOG_DIR/${name}.out"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    ok "Already running: ${C_BOLD}$name${C_RESET} (pid=$(cat "$pidfile"))"
    return 0
  fi

  local bin_path
  bin_path="$(find_bin_path "$name" || true)"
  [[ -n "$bin_path" ]] || die "binary not found for service: $name"

  log "Starting: ${C_BOLD}$name${C_RESET}"
  log "  bin: $bin_path"
  log "  log: $outfile"

  nohup "$bin_path" >>"$outfile" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidfile"

  ok "Started: ${C_BOLD}$name${C_RESET} pid=$pid"
}

for b in $SERVICE_BINS; do
  start_one "$b"
done

# Start compute_history and compute_realtime separately
section "START COMPUTE SERVICES"

start_compute_history() {
  local name="compute_history"
  local pidfile="$PID_DIR/$name.pid"
  local outfile="$LOG_DIR/${name}.out"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    ok "Already running: ${C_BOLD}$name${C_RESET} (pid=$(cat "$pidfile"))"
    return 0
  fi

  local bin_path="$TARGET_DIR/compute_history"
  if [[ ! -x "$bin_path" ]]; then
    die "compute_history binary not found at $bin_path"
  fi

  # Set environment for history mode
  export DB_PERSIST_MODE=history
  export DB_PERSIST_CHUNK_SIZE_HISTORY=50000
  export DB_PERSIST_MAX_BATCH=200000
  export DB_PERSIST_FLUSH_MS=150
  export DB_PERSIST_HISTORY_SKIP_JSON=1
  export DB_PERSIST_HISTORY_UPSERT=0

  log "Starting: ${C_BOLD}$name${C_RESET}"
  log "  bin: $bin_path"
  log "  log: $outfile"

  nohup "$bin_path" >>"$outfile" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidfile"

  ok "Started: ${C_BOLD}$name${C_RESET} pid=$pid"
}

start_compute_realtime() {
  local name="compute_realtime"
  local pidfile="$PID_DIR/$name.pid"
  local outfile="$LOG_DIR/${name}.out"

  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    ok "Already running: ${C_BOLD}$name${C_RESET} (pid=$(cat "$pidfile"))"
    return 0
  fi

  local bin_path="$TARGET_DIR/compute_realtime"
  if [[ ! -x "$bin_path" ]]; then
    die "compute_realtime binary not found at $bin_path"
  fi

  # Set environment for realtime mode
  export DB_PERSIST_MODE=realtime
  export DB_PERSIST_CHUNK_SIZE_REALTIME=2000
  export DB_PERSIST_MAX_BATCH_REALTIME=20000
  export DB_PERSIST_FLUSH_MS_REALTIME=50

  log "Starting: ${C_BOLD}$name${C_RESET}"
  log "  bin: $bin_path"
  log "  log: $outfile"

  nohup "$bin_path" >>"$outfile" 2>&1 &
  local pid=$!
  echo "$pid" > "$pidfile"

  ok "Started: ${C_BOLD}$name${C_RESET} pid=$pid"
}

# Start both compute services
start_compute_history
start_compute_realtime

section "DONE"
ok "START DONE @ $(date)"
log "Services: $SERVICE_BINS"
log "PIDs in : $PID_DIR"
log "Logs in : $LOG_DIR"

if [[ "$TAIL_LOGS" -eq 1 ]]; then
  section "TAIL LOGS"
  log "Tailing logs (Ctrl+C to stop)..."
  tail -n 200 -F "$LOG_DIR"/*.out
fi

# ----------------- 4) start grafana for healthcheck -----------------

section "START GRAFANA HEALTHCHECK"

if command -v docker >/dev/null 2>&1 && (command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1); then
  log "Starting Grafana for healthcheck monitoring..."
  cd "$ROOT_DIR/healthcheck/grafana"
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose up -d
  else
    docker compose up -d
  fi
  ok "Grafana started. Access at http://localhost:3001 (admin/admin)"
else
  warn "Docker or docker-compose not found. Skipping Grafana startup."
fi



