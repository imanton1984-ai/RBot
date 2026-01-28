#!/usr/bin/env bash
# scripts/start.sh
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
SERVICE_BINS="${SERVICE_BINS:-connections}"

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
    CARGO_BUILD_CMD="${CARGO_BUILD_CMD:-cargo build --release -p connections -p ingestor}"
  else
    CARGO_BUILD_CMD="${CARGO_BUILD_CMD:-cargo build -p connections -p ingestor}"
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

# # ----------------- 2.5) one-shot steps -----------------

run_oneshot() {
  local name="$1"
  local outfile="$LOG_DIR/${name}.out"

  local bin_path
  bin_path="$(find_bin_path "$name" || true)"
  [[ -n "$bin_path" ]] || die "oneshot binary not found: $name"

  section "ONESHOT: $name"
  log "Running: ${C_BOLD}$name${C_RESET}"
  log "  bin: $bin_path"
  log "  log: $outfile"

  # Пишем лог, но НЕ в nohup — это шаг, который должен завершиться сейчас.
  "$bin_path" 2>&1 | tee -a "$outfile"
  ok "Oneshot finished: $name"
}

# --- PAIRS: refresh universe into DB ---
# по умолчанию ждём хотя бы 300 активных пар
export MIN_PAIRS="${MIN_PAIRS:-300}"
export RUST_LOG="${RUST_LOG:-info}"

run_oneshot "ingestor_pairs"

# ----------------- 3) start services -----------------



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



