-- 110_performance_indexes.sql
-- Performance indexes to fix slow queries in WebUI and background tasks.
--
-- Root causes of slow queries:
--   1. candles_live: broadcaster polls by open_time_ms without symbol/timeframe — no matching index
--   2. candles_live: positions query does LATERAL JOIN ORDER BY open_time_ms DESC LIMIT 1
--      per symbol, hitting many rows in a large table
--   3. super_entry_signals: archiver filters by `time < now() - 24h` — needs efficient scan
--   4. candles_live grows unbounded — needs periodic cleanup

BEGIN;

-- ═══════════════════════════════════════════════════════════
-- 1. candles_live: index for broadcaster's time-range scan
--    Query: WHERE open_time_ms > (now_epoch_ms - 120000)
-- ═══════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS candles_live_open_time_desc_idx
    ON market.candles_live (open_time_ms DESC);

-- ═══════════════════════════════════════════════════════════
-- 2. candles_live: cleanup old rows to prevent table bloat
--    Keep only last 48 hours of data (the table is "hot" cache only).
-- ═══════════════════════════════════════════════════════════
-- This is done in application code (signal_archiver), but we add
-- a partial index to speed up the DELETE of old rows.
CREATE INDEX IF NOT EXISTS candles_live_cleanup_idx
    ON market.candles_live (open_time_ms)
    WHERE open_time_ms < (EXTRACT(EPOCH FROM now() - INTERVAL '48 hours') * 1000)::bigint;

-- ═══════════════════════════════════════════════════════════
-- 3. super_entry_signals: index for archiver's time filter
--    Query: WHERE time < now() - INTERVAL '24 hours'
--    The PK is (symbol_id, tf_minutes, time) — not efficient for
--    a pure time-range scan across all symbols.
-- ═══════════════════════════════════════════════════════════
-- Already exists: ix_super_entry_signals_time ON (time DESC)
-- But for the archiver's `time < X` scan, an ASC index on time is better.
CREATE INDEX IF NOT EXISTS ix_super_entry_signals_time_asc
    ON trade.super_entry_signals (time ASC);

-- ═══════════════════════════════════════════════════════════
-- 4. position_history: index for position_id dedup check
--    Query: NOT EXISTS (SELECT 1 FROM position_history WHERE position_id = p.id)
-- ═══════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS ix_position_history_position_id
    ON trade.position_history (position_id);

COMMIT;
