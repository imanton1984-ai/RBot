#!/usr/bin/env bash
# Script to stop healthcheck grafana services

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "Stopping healthcheck Grafana services..."

if command -v docker >/dev/null 2>&1 && (command -v docker-compose >/dev/null 2>&1 || docker compose version >/dev/null 2>&1); then
  if command -v docker-compose >/dev/null 2>&1; then
    docker-compose down
  else
    docker compose down
  fi
  log "Healthcheck Grafana services stopped."
else
  log "docker-compose not found. Skipping Grafana shutdown."
fi