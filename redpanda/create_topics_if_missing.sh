#!/usr/bin/env bash
set -euo pipefail

# Проверяем, запущен ли redpanda контейнер
if ! docker ps --format "table {{.Names}}" | grep -q "^redpanda$"; then
    echo "❌ redpanda container is not running. Please start it first."
    exit 1
fi

# Получаем список существующих топиков
existing_topics=$(docker exec redpanda rpk topic list --no-headers 2>/dev/null | tr '\n' ' ')

# Определяем необходимые топики
required_topics=("candles.close" "indicators.close")

echo "Checking for required topics..."

for topic in "${required_topics[@]}"; do
    if [[ $existing_topics =~ (^|[[:space:]])"$topic"($|[[:space:]]) ]]; then
        echo "✅ Topic '$topic' already exists"
    else
        echo "⏳ Creating topic '$topic'..."
        docker exec redpanda rpk topic create "$topic" -p 16
        echo "✅ Topic '$topic' created"
    fi
done

echo "All required topics are ready."