-- 056_raw_signals_pk_update.sql: Update primary key constraint for raw_signals table to support UPSERT operations
-- This allows updating existing signals instead of throwing duplicate key errors

-- Drop and recreate the primary key constraint to include time_ms in the correct order for UPSERT operations
-- Note: We keep 'time' in the constraint as it's required for TimescaleDB hypertable partitioning
ALTER TABLE market.raw_signals DROP CONSTRAINT IF EXISTS raw_signals_pkey;
ALTER TABLE market.raw_signals ADD CONSTRAINT raw_signals_pkey PRIMARY KEY (time_ms, symbol_id, tf_minutes, indicator_id, signal_kind, time);