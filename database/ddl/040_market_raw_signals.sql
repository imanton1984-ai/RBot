-- database/ddl/040_market_raw_signals.sql
-- Таблица сырых сигналов: одна строка = один raw-signal на свечу.
-- Ключ ВКЛЮЧАЕТ signal_sub_id (чтобы не ломать upsert и позволить подтипы).

CREATE SCHEMA IF NOT EXISTS market;
CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS market.raw_signals (
    time              TIMESTAMPTZ NOT NULL,
    time_ms           BIGINT      NOT NULL,

    symbol_id         BIGINT      NOT NULL,
    symbol            TEXT        NOT NULL,
    tf_minutes        SMALLINT    NOT NULL,

    indicator_id      SMALLINT    NOT NULL,
    signal_kind       SMALLINT    NOT NULL,
    signal_sub_id     SMALLINT    NOT NULL DEFAULT 0,

    side              SMALLINT    NOT NULL,
    score             REAL        NOT NULL,
    value             REAL        NOT NULL,

    details           JSONB,

    candle_is_final   BOOLEAN     NOT NULL DEFAULT TRUE,
    calc_source       SMALLINT    NOT NULL DEFAULT 1,
    event_time_ms     BIGINT,

    features_json     JSONB,
    scores_json       JSONB,
    predictions_json  JSONB,

    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at_ms     BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_ms     BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,

    CONSTRAINT raw_signals_pkey PRIMARY KEY (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id)
);

-- Hypertable
DO $$
BEGIN
    PERFORM create_hypertable('market.raw_signals', 'time', if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL '14 days');
EXCEPTION
    WHEN undefined_function THEN
        RAISE NOTICE 'TimescaleDB not available, skipping create_hypertable for raw_signals';
END $$;

-- Индексы
CREATE INDEX IF NOT EXISTS idx_raw_signals_sid_tf_time_desc ON market.raw_signals (symbol_id, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_signals_symbol_tf_time_desc ON market.raw_signals (symbol, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS idx_raw_signals_time_ms_desc ON market.raw_signals (time_ms DESC);
CREATE INDEX IF NOT EXISTS idx_raw_signals_score_desc ON market.raw_signals (score DESC);
