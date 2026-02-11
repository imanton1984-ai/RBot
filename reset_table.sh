#!/bin/bash
# Script to reset the indicators_wide table structure

echo "Dropping market.indicators_wide table..."
psql -h localhost -p 5433 -U postgres -d timescaledb_binance -c "DROP TABLE IF EXISTS market.indicators_wide CASCADE;"

echo "Table dropped. Restart your Rust bot to recreate it with the correct schema."
echo "Run: ./scripts/restart.sh"