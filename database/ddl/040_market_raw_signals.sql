-- database/ddl/040_market_raw_signals.sql
-- Aggregated signals table: one row per candle with all signals stored in a JSONB array.
-- This reduces row count significantly compared to individual signal rows.

CREATE SCHEMA IF NOT EXISTS market;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- database/ddl/040_market_raw_signals.sql
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
    predictions_json JSONB,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,

    CONSTRAINT raw_signals_pkey PRIMARY KEY (symbol_id, tf_minutes, time, indicator_id, signal_kind, signal_sub_id)
    PRIMARY KEY (symbol_id, tf_minutes, time)
);

-- Превращаем в гипертаблицу TimescaleDB
SELECT create_hypertable('market.raw_signals', 'time', if_not_exists => TRUE);

-- Индексы для быстрой выборки последних сигналов
CREATE INDEX IF NOT EXISTS ix_raw_signals_sym_tf_time_desc ON market.raw_signals(symbol_id, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS ix_raw_signals_time_ms ON market.raw_signals(time_ms);