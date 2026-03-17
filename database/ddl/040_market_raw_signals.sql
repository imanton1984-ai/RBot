-- database/ddl/040_market_raw_signals.sql
CREATE SCHEMA IF NOT EXISTS market;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- NOTE: DO NOT drop raw_signals on restart!
-- Previous DROP TABLE destroyed all computed signals, forcing full recompute.
-- Use CREATE TABLE IF NOT EXISTS for idempotent startup.
-- To force a full reset, run: DROP TABLE IF EXISTS market.raw_signals CASCADE;

CREATE TABLE IF NOT EXISTS market.raw_signals (
    time            TIMESTAMPTZ NOT NULL,
    time_ms         BIGINT      NOT NULL,
    symbol_id       BIGINT      NOT NULL,
    symbol          TEXT        NOT NULL,
    tf_minutes      SMALLINT    NOT NULL,

    indicator_id    SMALLINT    NOT NULL,
    signal_kind     SMALLINT    NOT NULL,
    signal_sub_id   SMALLINT    NOT NULL,

    side            SMALLINT    NOT NULL,
    score           REAL        NOT NULL,
    value           REAL        NOT NULL,
    details         JSONB,

    candle_is_final BOOLEAN     NOT NULL DEFAULT TRUE,
    calc_source     SMALLINT    NOT NULL DEFAULT 1,
    event_time_ms   BIGINT,

    features_json   JSONB,
    scores_json     JSONB,
    predictors_json JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- Исправлено: Один Primary Key, включающий все поля уникальности
    CONSTRAINT raw_signals_pkey PRIMARY KEY (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id)
);

-- Hypertable
SELECT create_hypertable('market.raw_signals', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

-- Indexes
CREATE INDEX IF NOT EXISTS ix_raw_signals_sym_tf_time_desc ON market.raw_signals(symbol_id, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS ix_raw_signals_time_ms ON market.raw_signals(time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_raw_signals_symbol ON market.raw_signals(symbol);

-- -- Compression
-- ALTER TABLE market.raw_signals SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'symbol_id, tf_minutes, indicator_id',
--     timescaledb.compress_orderby = 'time DESC'
-- );
-- SELECT add_compression_policy('market.raw_signals', INTERVAL '7 days');