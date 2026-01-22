#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for TimescaleDB to be ready..."
timeout=300
start_time=$(date +%s)

while true; do
    if docker compose -f infra/docker-compose.yml exec -T timescaledb pg_isready -U postgres >/dev/null 2>&1; then
        echo "✅ TimescaleDB is ready"
        exit 0
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -ge $timeout ]; then
        echo "❌ Timeout waiting for TimescaleDB"
        exit 1
    fi
    
    echo "⏳ TimescaleDB not ready yet, waiting..."
    sleep 5
done