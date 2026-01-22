#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for Redpanda to be ready..."
timeout=300
start_time=$(date +%s)

while true; do
    if docker compose -f infra/docker-compose.yml exec -T redpanda rpk cluster health >/dev/null 2>&1; then
        echo "✅ Redpanda is ready"
        exit 0
    fi
    
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    
    if [ $elapsed -ge $timeout ]; then
        echo "❌ Timeout waiting for Redpanda"
        exit 1
    fi
    
    echo "⏳ Redpanda not ready yet, waiting..."
    sleep 5
done