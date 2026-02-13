#!/usr/bin/env bash

# --- Color definitions ---
C_RESET='\033[0m'
C_BOLD='\033[1m'
C_GREEN='\033[32m'
C_CYAN='\033[36m'

# --- Logging functions ---
ts() { date +"%H:%M:%S"; }
hr() { echo -e "${C_BOLD}===========================================${C_RESET}"; }
section() { hr; echo -e "${C_BOLD}${C_CYAN}${1}${C_RESET}"; hr; }
log()  { echo -e "${C_BOLD}[$(ts)]${C_RESET} $*"; }
ok()   { echo -e "${C_BOLD}[$(ts)]${C_RESET} ${C_GREEN}✅${C_RESET} $*"; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Create timestamped log file like in original start.sh
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT_DIR/logs"
START_LOG="$LOG_DIR/start_${TS}.log"
mkdir -p "$LOG_DIR"

# Log to both file and console like original
exec > >(tee -a "$START_LOG") 2>&1

section "START (CPU MODE)"

log "ROOT: ${C_BOLD}$ROOT_DIR${C_RESET}"
log "LOG : ${C_BOLD}$START_LOG${C_RESET}"

# 1. System Check
./scripts/sys_check_cpu.sh

# 2. Install Deps (if needed)
./scripts/install_cpu.sh

# 3. Start Infra
./scripts/infra_start.sh

# 4. Init DB
./scripts/init_db.sh

# 5. Build Rust (CPU)
# Using release, change to debug inside scripts/build_cpu.sh if compilation is too slow
./scripts/build_cpu.sh 

# 6. Run Services
./scripts/run.sh "release" "cpu"

echo -e "\n\033[1;32mCPU BOT STARTED SUCCESSFULLY\033[0m"