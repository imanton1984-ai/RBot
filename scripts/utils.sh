#!/usr/bin/env bash

# --- Colors & Logging ---
NO_COLOR="${NO_COLOR:-0}"
if [[ "$NO_COLOR" == "1" ]]; then
  C_RESET=""; C_DIM=""; C_BOLD=""; C_RED=""; C_GREEN=""; C_YELLOW=""; C_BLUE=""; C_CYAN=""
else
  C_RESET='\033[0m'
  C_DIM='\033[2m'
  C_BOLD='\033[1m'
  C_RED='\033[31m'
  C_GREEN='\033[32m'
  C_YELLOW='\033[33m'
  C_BLUE='\033[34m'
  C_CYAN='\033[36m'
fi

ts() { date +"%H:%M:%S"; }
hr() { echo -e "${C_DIM}===========================================${C_RESET}"; }
section() { hr; echo -e "${C_BOLD}${C_CYAN}${1}${C_RESET}"; hr; }
log()  { echo -e "${C_DIM}[$(ts)]${C_RESET} $*"; }
ok()   { echo -e "${C_DIM}[$(ts)]${C_RESET} ${C_GREEN}✅${C_RESET} $*"; }
warn() { echo -e "${C_DIM}[$(ts)]${C_RESET} ${C_YELLOW}⚠️${C_RESET} $*"; }
err()  { echo -e "${C_DIM}[$(ts)]${C_RESET} ${C_RED}❌${C_RESET} $*"; }
die()  { err "$*"; exit 1; }

# --- Helper Functions ---

load_env_file() {
  local f="$1"
  if [[ -f "$f" ]]; then
    log "Loading env: ${C_BOLD}$f${C_RESET}"
    set -a
    source "$f"
    set +a
  fi
}

find_bin_path() {
  local name="$1"
  local target_dir="$2"
  local root_dir="$3"
  
  local p="$target_dir/$name"
  [[ -x "$p" ]] && { echo "$p"; return 0; }
  local p2="$root_dir/target/release/$name"
  [[ -x "$p2" ]] && { echo "$p2"; return 0; }
  return 1
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local timeout="$3"
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

run_with_pty() {
  local cmd="$1"
  if command -v script >/dev/null 2>&1; then
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