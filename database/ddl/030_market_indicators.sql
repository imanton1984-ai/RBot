-- database/ddl/030_market_indicators.sql
-- Wide table format: one row per (symbol, timeframe, time) with columns for each indicator.
-- This replaces the EAV format for better performance and reduced storage.

CREATE SCHEMA IF NOT EXISTS market;

-- TimescaleDB (если у тебя уже есть - ок)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Single wide table for all timeframes instead of multiple tables
CREATE TABLE IF NOT EXISTS market.indicators_wide (
    time            TIMESTAMPTZ NOT NULL,
    time_ms         BIGINT      NOT NULL,
    symbol_id       BIGINT      NOT NULL,
    symbol          TEXT        NOT NULL,
    tf_minutes      SMALLINT    NOT NULL,

    -- Core Indicators (Columns instead of rows)
    rsi             REAL,
    macd            REAL,
    macd_signal     REAL,
    macd_hist       REAL,
    bb_upper        REAL,
    bb_mid          REAL,
    bb_lower        REAL,
    stoch_k         REAL,
    stoch_d         REAL,
    adx             REAL,
    atr             REAL,
    cci             REAL,
    obv             DOUBLE PRECISION,
    vwap            DOUBLE PRECISION,
    ema_20          REAL,
    ema_50          REAL,
    ema_200         REAL,
    
    -- Complex/JSON data stays in JSONB
    sr_levels       JSONB,
    
    candle_is_final BOOLEAN     NOT NULL DEFAULT TRUE,
    calc_source     SMALLINT    NOT NULL DEFAULT 1,
    event_time_ms   BIGINT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
    
    PRIMARY KEY (symbol_id, tf_minutes, time)
);

-- Turn into Hypertable
SELECT create_hypertable('market.indicators_wide', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

-- Compression (Critical for wide tables)
ALTER TABLE market.indicators_wide SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'symbol_id, tf_minutes',
    timescaledb.compress_orderby = 'time DESC'
);

SELECT add_compression_policy('market.indicators_wide', INTERVAL '2 days');

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_indicators_wide_sid_time_desc ON market.indicators_wide (symbol_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_wide_symbol_time_desc ON market.indicators_wide (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_wide_tf_time_desc ON market.indicators_wide (tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_wide_time_ms_desc ON market.indicators_wide (time_ms DESC);
