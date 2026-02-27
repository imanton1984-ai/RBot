-- 091_super_entry_outdated.sql
-- DDL for Super Entry Strategy outdated signals table
-- Signals older than 24 hours are moved here from trade.super_entry_signals
-- by an hourly job. Retention: 60,000 signals per timeframe.

CREATE TABLE IF NOT EXISTS trade.super_entry_outdated (
    time            TIMESTAMPTZ NOT NULL,
    time_ms         BIGINT NOT NULL,
    symbol          TEXT NOT NULL,
    symbol_id       BIGINT NOT NULL,
    tf_minutes      SMALLINT NOT NULL,
    side            SMALLINT NOT NULL,           -- 1=LONG, -1=SHORT
    entry_price     FLOAT8 NOT NULL,
    sl_price        FLOAT8 NOT NULL,
    tp_price        FLOAT8 NOT NULL,
    p_super         REAL NOT NULL,               -- P(super move) from model
    p_long          REAL NOT NULL,               -- P(direction=LONG) from model
    combined_score  REAL NOT NULL,               -- combined scorer output
    dir_confidence  REAL NOT NULL DEFAULT 0.0,   -- directional confidence |p_long - 0.5|
    strategy        TEXT NOT NULL DEFAULT 'super_entry_v1',
    reason          JSONB,                       -- full metadata (target_pct, sl_pct, etc.)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    moved_at        TIMESTAMPTZ NOT NULL DEFAULT now(),  -- when signal was moved from active table
    PRIMARY KEY (symbol_id, tf_minutes, time)
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS ix_super_entry_outdated_time
    ON trade.super_entry_outdated (time DESC);

CREATE INDEX IF NOT EXISTS ix_super_entry_outdated_tf_time
    ON trade.super_entry_outdated (tf_minutes, time DESC);

CREATE INDEX IF NOT EXISTS ix_super_entry_outdated_score
    ON trade.super_entry_outdated (combined_score DESC, time DESC);

-- Comment
COMMENT ON TABLE trade.super_entry_outdated IS 
    'Archived signals from trade.super_entry_signals that are older than 24 hours. '
    'Hourly job moves signals here. Retention: 60,000 signals per timeframe.';
