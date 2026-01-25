#!/usr/bin/env bash
set -euo pipefail

echo "Waiting for Redpanda to be ready..."
max_attempts=5
attempt=1

while [ $attempt -le $max_attempts ]; do
    # CHANGED: Added --api-urls to match docker-compose healthcheck
    if docker compose -f infra/docker-compose.yaml exec -T redpanda rpk cluster health --api-urls=http://127.0.0.1:9644 >/dev/null 2>&1; then
        echo "✅ Redpanda is ready"
        exit 0
    fi
    
    if [ $attempt -lt $max_attempts ]; then
        echo "⏳ Redpanda not ready yet, waiting... (attempt $attempt/$max_attempts)"
        sleep 5
    fi
    
    attempt=$((attempt + 1))
done

echo "❌ Failed to wait for Redpanda after $max_attempts attempts"
exit 1