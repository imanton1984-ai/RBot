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

# Parse strategy flags
STRATEGY="${STRATEGY:-default}"
RUN_SUPER_ENTRY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy=*)
            STRATEGY="${1#*=}"
            shift
            ;;
        --super-entry)
            RUN_SUPER_ENTRY=true
            STRATEGY="super_entry"  # Override strategy to super_entry
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Export strategy for downstream scripts
export ACTIVE_STRATEGY="$STRATEGY"
export SUPER_ENTRY_ENABLED="$RUN_SUPER_ENTRY"
# Configurable: which timeframes to compute indicators/signals for (super_entry only).
# Default: "15,60,240,1440" (15m, 1h, 4h, 1d). Set to e.g. "15,60,240" to drop 1d.
# Candle loading is NOT affected — all TFs still load candles.
export SUPER_ENTRY_TIMEFRAMES="${SUPER_ENTRY_TIMEFRAMES:-15,60,240,1440}"

log "ROOT:     ${C_BOLD}$ROOT_DIR${C_RESET}"
log "LOG :     ${C_BOLD}$START_LOG${C_RESET}"
log "STRATEGY: ${C_BOLD}$STRATEGY${C_RESET}"
if [ "$RUN_SUPER_ENTRY" = true ]; then
    log "SUPER ENTRY: ${C_GREEN}ENABLED${C_RESET}"
    log "SUPER_ENTRY_TIMEFRAMES: ${C_BOLD}$SUPER_ENTRY_TIMEFRAMES${C_RESET}"
fi

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

# 7. Super Entry Strategy info
if [ "$RUN_SUPER_ENTRY" = true ]; then
    section "SUPER ENTRY STRATEGY"
    log "Super Entry strategy is ENABLED"
    log "  History backfill: integrated into compute_history (batch mode after indicators)"
    log "  Realtime signals: integrated into compute_realtime (super_entry_stage)"
    log "  NOTE: For backtesting, run separately: ./scripts/super_entry_backtester.sh"
    ok "Super Entry integrated into compute pipeline"
fi

echo -e "\n\033[1;32mCPU BOT STARTED SUCCESSFULLY\033[0m"
echo -e "Active strategy: $STRATEGY"
[ "$RUN_SUPER_ENTRY" = true ] && echo -e "Super Entry: \033[1;32mENABLED\033[0m"