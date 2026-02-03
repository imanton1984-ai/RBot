#!/bin/bash

echo "Cleaning market tables..."

# Load environment variables
source .env

# Set the PostgreSQL password from environment variable
export PGPASSWORD="${DB_PASSWORD:-postgres}"

# Connect to PostgreSQL and drop all market candle tables and pairs table
psql -h localhost -p 5433 -U postgres -d timescaledb_binance -c "
BEGIN;

-- Stop any active connections to prevent locking issues
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'timescaledb_binance'
AND pid <> pg_backend_pid();

-- Drop all candle tables
DROP TABLE IF EXISTS market.candles_1m CASCADE;
DROP TABLE IF EXISTS market.candles_5m CASCADE;
DROP TABLE IF EXISTS market.candles_15m CASCADE;
DROP TABLE IF EXISTS market.candles_1h CASCADE;
DROP TABLE IF EXISTS market.candles_4h CASCADE;
DROP TABLE IF EXISTS market.candles_1d CASCADE;
DROP TABLE IF EXISTS market.candles_live CASCADE;

-- Drop all indicator tables
DROP TABLE IF EXISTS market.indicators_1m CASCADE;
DROP TABLE IF EXISTS market.indicators_5m CASCADE;
DROP TABLE IF EXISTS market.indicators_15m CASCADE;
DROP TABLE IF EXISTS market.indicators_1h CASCADE;
DROP TABLE IF EXISTS market.indicators_4h CASCADE;
DROP TABLE IF EXISTS market.indicators_1d CASCADE;

-- Drop pairs table
DROP TABLE IF EXISTS market.pairs CASCADE;

-- Drop raw signals table
DROP TABLE IF EXISTS market.raw_signals CASCADE;

-- Recreate the market schema
CREATE SCHEMA IF NOT EXISTS market;

COMMIT;

SELECT 'Tables cleaned successfully' AS status;
"

echo "Market tables have been cleaned. You can now restart your services."