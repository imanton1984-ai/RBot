-- 056_raw_signals_pk_update.sql: Fix PK to match Rust ON CONFLICT clause
-- We MUST include signal_sub_id, otherwise we cannot store multiple EMAs (20, 50, 200) for the same candle.

ALTER TABLE market.raw_signals DROP CONSTRAINT IF EXISTS raw_signals_pkey;

-- Primary Key must include 'time' for TimescaleDB partitioning
-- Primary Key must include 'signal_sub_id' for logic correctness (EMA_20 vs EMA_50)
ALTER TABLE market.raw_signals ADD CONSTRAINT raw_signals_pkey PRIMARY KEY (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id);

-- Create index for time_ms if needed for queries, but not strictly required for the PK constraint
CREATE INDEX IF NOT EXISTS idx_raw_signals_time_ms ON market.raw_signals(time_ms);