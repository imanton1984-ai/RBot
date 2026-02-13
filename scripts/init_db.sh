#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

section "DATABASE INITIALIZATION"

# Check if DB needs init
check_db_initialized() {
  PGPASSWORD=postgres psql -h 127.0.0.1 -p 5433 -U postgres -d timescaledb_binance -tAc "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'market' AND table_name = 'pairs');" 2>/dev/null | grep -q 't'
}

if check_db_initialized; then
  ok "Database already initialized."
else
  warn "Database schema missing. Running initialization..."
  # Build tool if needed (using release profile for speed usually)
  cargo build -p database --bin svc_db_init
  
  # Run it
  "$ROOT_DIR/target/debug/svc_db_init" 2>&1 | tee -a "logs/db_init.out"
  
  if check_db_initialized; then
      ok "Database initialized successfully."
  else
      die "Database initialization failed."
  fi
fi