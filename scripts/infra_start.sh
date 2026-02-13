#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/utils.sh"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load Env
if [[ -f "$ROOT_DIR/.env" ]]; then load_env_file "$ROOT_DIR/.env"; fi

section "INFRASTRUCTURE"

COMPOSE_FILE="infra/docker-compose.yaml"
# Check if compose file needs rebuild/up
log "Starting Docker Compose..."
if command -v docker >/dev/null 2>&1; then
    docker compose -f "$COMPOSE_FILE" up -d --remove-orphans
else
    die "Docker not found"
fi

DB_HOST="${DB_TCP_HOST:-127.0.0.1}"
DB_PORT="${DB_TCP_PORT:-5433}"
KAFKA_HOST="${KAFKA_TCP_HOST:-127.0.0.1}"
KAFKA_PORT="${KAFKA_TCP_PORT:-19092}"

wait_for_tcp "$DB_HOST" "$DB_PORT" 45
wait_for_tcp "$KAFKA_HOST" "$KAFKA_PORT" 45

section "KAFKA TOPICS"
# Create topics (idempotent usually)
TOPICS=(
    "candles.close" "candles.update" "indicators.close" "ws_feed"
    "signals.raw" "signals.final" "features.snapshot"
    "orders.cmd" "orders.events" "positions.events"
)

for t in "${TOPICS[@]}"; do
    docker exec redpanda rpk topic create "$t" --brokers=redpanda:29092 --partitions=3 --replicas=1 2>/dev/null || true
done
ok "Kafka topics ready."