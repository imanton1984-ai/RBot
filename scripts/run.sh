#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Args
BUILD_PROFILE="${1:-release}"
MODE="${2:-cpu}" # cpu or gpu used for lib path setting

LOG_DIR="$ROOT_DIR/logs"
PID_DIR="$ROOT_DIR/run"
mkdir -p "$LOG_DIR" "$PID_DIR"

# Create timestamped log file like in original start.sh
TS="$(date +%Y%m%d_%H%M%S)"
START_LOG="$LOG_DIR/start_${TS}.log"

# Log to both file and console like original
exec > >(tee -a "$START_LOG") 2>&1

section "STARTING SERVICES"

# Set Library Paths based on mode
if [[ "$MODE" == "gpu" ]]; then
    export XGBOOST_LIB_DIR="$ROOT_DIR/third_party/xgboost/install/lib"
    # Also standard CUDA paths
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$XGBOOST_LIB_DIR:${LD_LIBRARY_PATH:-}"
else
    export XGBOOST_LIB_DIR="$ROOT_DIR/third_party/xgboost/install_cpu/lib"
    export LD_LIBRARY_PATH="$XGBOOST_LIB_DIR:${LD_LIBRARY_PATH:-}"
fi

TARGET_DIR="$ROOT_DIR/target/$BUILD_PROFILE"

# Env vars for services
export MIN_PAIRS="${MIN_PAIRS:-300}"

# RUST_LOG: when super_entry is active, enable info logging for compute_history
# so that ZERO-COPY pipeline messages and shutdown chain are visible.
ACTIVE_STRATEGY="${ACTIVE_STRATEGY:-default}"
if [ "$ACTIVE_STRATEGY" = "super_entry" ] || [ "${SUPER_ENTRY_ENABLED:-false}" = "true" ]; then
    export RUST_LOG="${RUST_LOG:-info,compute_history=info,trade_signal_stage=info,ingestor=info,connections=info,super_entry=info,super_entry_stage=info,ml_entry_strategy=info}"
else
    export RUST_LOG="${RUST_LOG:-warn,compute_history=warn,trade_signal_stage=info,ingestor=info,connections=info,super_entry=info,super_entry_stage=info,ml_entry_strategy=info}"
fi

# Start Helper
start_svc() {
    local name="$1"
    local bin_name="$2"
    local pidfile="$PID_DIR/$name.pid"
    local outfile="$LOG_DIR/${name}.out"
    
    if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
        ok "$name already running."
        return
    fi
    
    local bin="$TARGET_DIR/$bin_name"
    if [[ ! -x "$bin" ]]; then 
        err "Binary not found: $bin"
        return
    fi
    
    log "Starting $name..."
    log "  bin: $bin"
    log "  log: $outfile"
    
    nohup "$bin" >>"$outfile" 2>&1 &
    local pid=$!
    echo "$pid" > "$pidfile"
    ok "Started $name (pid $pid)"
}

# 1. Connections
start_svc "connections" "connections"

# 2. Ingestor
start_svc "ingestor" "ingestor"

# 3. Compute History
# Set ENV for history
export DB_PERSIST_MODE=history
export DB_PERSIST_CHUNK_SIZE_HISTORY=50000
export DB_PERSIST_MAX_BATCH=200000
export DB_PERSIST_FLUSH_MS=150
export DB_PERSIST_HISTORY_SKIP_JSON=1   # huge saving if jsonb is heavy
export DB_PERSIST_HISTORY_UPSERT=0      # append

log "Starting compute_history service..."
start_svc "compute_history" "compute_history"

# 4. Compute Realtime (Reset ENV first)
export DB_PERSIST_MODE=realtime
export DB_PERSIST_CHUNK_SIZE_REALTIME=2000
export DB_PERSIST_MAX_BATCH_REALTIME=20000
export DB_PERSIST_FLUSH_MS_REALTIME=50

log "Starting compute_realtime service..."
start_svc "compute_realtime" "compute_realtime"

# 5. Grafana (Optional)
if [ -d "$ROOT_DIR/healthcheck/grafana" ]; then
    section "START GRAFANA HEALTHCHECK"
    log "Starting Grafana for healthcheck monitoring..."
    cd "$ROOT_DIR/healthcheck/grafana"
    docker compose up -d >/dev/null 2>&1 || true
    ok "Grafana started. Access at http://localhost:3001 (admin/admin)"
fi

# 6. Strategy-specific Services
# Determine which strategy to run
ACTIVE_STRATEGY="${ACTIVE_STRATEGY:-default}"
export ACTIVE_STRATEGY  # Export for child processes (compute_history, compute_realtime)

log "Active strategy: ${ACTIVE_STRATEGY}"

if [ "$ACTIVE_STRATEGY" = "super_entry" ] || [ "${SUPER_ENTRY_ENABLED:-false}" = "true" ]; then
    # ═══════════════════════════════════════════
    # SUPER ENTRY STRATEGY MODE
    # History backfill: handled by compute_history (runs batch backfill after indicators)
    # Realtime signals: handled by compute_realtime (integrated super_entry_stage)
    # No separate super_entry_service needed!
    # ═══════════════════════════════════════════
    section "SUPER ENTRY STRATEGY"
    log "Super Entry strategy active"
    log "  History: ZERO-COPY pipeline in compute_history (GPU→indicators→XGBoost→DB)"
    log "  Realtime: SuperEntryStage in compute_realtime (single-candle inference)"
    log "  Predictors/trade_signals pipeline: SKIPPED"
    log "  NOTE: Must have models in models/super_entry_v1_tf{X}.ubj"

else
    # ═══════════════════════════════════════════
    # DEFAULT (LEVEL) STRATEGY MODE
    # Runs full pipeline: predictors → trade_signals
    # ═══════════════════════════════════════════
    section "LEVEL STRATEGY SERVICES"
    log "Starting default strategy (predictors + trade signals)"

    # The predictors and trade signals are computed within compute_history/compute_realtime
    # They use the same compute services but process data differently
    log "Predictors and trade signals will be computed by compute_history/compute_realtime"
fi

section "ALL SERVICES STARTED"
ok "START DONE @ $(date)"
log "Services started successfully"
log "PIDs in : $PID_DIR"
log "Logs in : $LOG_DIR"