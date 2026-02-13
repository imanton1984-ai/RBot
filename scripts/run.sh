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
export RUST_LOG="${RUST_LOG:-info,ingestor=info,connections=info}"

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

section "ALL SERVICES STARTED"
ok "START DONE @ $(date)"
log "Services started successfully"
log "PIDs in : $PID_DIR"
log "Logs in : $LOG_DIR"