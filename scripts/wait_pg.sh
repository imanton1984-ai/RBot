#!/bin/bash
# scripts/wait_pg.sh
set -e

echo "Waiting for TimescaleDB to be ready..."
while ! PGPASSWORD=postgres psql -h localhost -p 5433 -U postgres -d timescaledb_binance -c "SELECT 1" > /dev/null 2>&1; do
  echo "⏳ TimescaleDB not ready yet, waiting..."
  sleep 2
done
echo "✅ TimescaleDB is ready!"