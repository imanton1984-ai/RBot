-- database/ddl/040_market_raw_signals.sql
-- Aggregated signals table: one row per candle with all signals stored in a JSONB array.
-- This reduces row count significantly compared to individual signal rows.

CREATE SCHEMA IF NOT EXISTS market;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS market.candle_signals (
    time            TIMESTAMPTZ NOT NULL,
    time_ms         BIGINT      NOT NULL,
    symbol_id       BIGINT      NOT NULL,
    symbol          TEXT        NOT NULL,
    tf_minutes      SMALLINT    NOT NULL,

    -- Aggregated Signals
    -- Format: [{"type": "RSI", "side": -1, "score": 0.95, "value": 85.0}, ...]
    signals         JSONB       NOT NULL DEFAULT '[]'::jsonb,
    
    -- Aggregated Summary
    total_score     REAL        GENERATED ALWAYS AS (
        (SELECT COALESCE(SUM((x->>'score')::real * (x->>'side')::int), 0) 
         FROM jsonb_array_elements(signals) x)
    ) STORED,

    candle_is_final BOOLEAN     NOT NULL DEFAULT TRUE,
    calc_source     SMALLINT    NOT NULL DEFAULT 1,
    event_time_ms   BIGINT,

    features_json   JSONB,
    scores_json     JSONB,
    predictions_json JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
    
    PRIMARY KEY (symbol_id, tf_minutes, time)
);

-- Hypertable
SELECT create_hypertable('market.candle_signals', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '14 days');

-- Auto-compress after 3 days
ALTER TABLE market.candle_signals SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'symbol_id, tf_minutes'
);
SELECT add_compression_policy('market.candle_signals', INTERVAL '3 days');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_candle_signals_sid_tf_time_desc ON market.candle_signals (symbol_id, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS idx_candle_signals_symbol_tf_time_desc ON market.candle_signals (symbol, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS idx_candle_signals_time_ms_desc ON market.candle_signals (time_ms DESC);
CREATE INDEX IF NOT EXISTS idx_candle_signals_total_score_desc ON market.candle_signals (total_score DESC);
CREATE INDEX IF NOT EXISTS idx_candle_signals_signals_gin ON market.candle_signals USING gin (signals);
