-- database/ddl/030_market_indicators.sql
-- EAV-таблицы индикаторов: одна строка = один индикатор на свечу.
-- PK гарантирует 1 значение на (symbol_id, time, indicator_name).

CREATE SCHEMA IF NOT EXISTS market;

-- TimescaleDB (если у тебя уже есть - ок)
CREATE EXTENSION IF NOT EXISTS timescaledb;

DO $$
DECLARE
    tf TEXT;
    tf_minutes SMALLINT;
    chunk_txt TEXT;
BEGIN
    FOREACH tf IN ARRAY ARRAY['1m','5m','15m','1h','4h','1d']
    LOOP
        -- tf_minutes
        IF tf = '1m' THEN tf_minutes := 1; chunk_txt := '7 days';
        ELSIF tf = '5m' THEN tf_minutes := 5; chunk_txt := '14 days';
        ELSIF tf = '15m' THEN tf_minutes := 15; chunk_txt := '21 days';
        ELSIF tf = '1h' THEN tf_minutes := 60; chunk_txt := '60 days';
        ELSIF tf = '4h' THEN tf_minutes := 240; chunk_txt := '180 days';
        ELSIF tf = '1d' THEN tf_minutes := 1440; chunk_txt := '730 days';
        ELSE
            RAISE EXCEPTION 'Unknown tf=%', tf;
        END IF;

        EXECUTE format($SQL$
            CREATE TABLE IF NOT EXISTS market.indicators_%s (
                time            TIMESTAMPTZ NOT NULL,
                time_ms         BIGINT      NOT NULL,
                symbol_id       BIGINT      NOT NULL,
                symbol          TEXT        NOT NULL,
                tf_minutes      SMALLINT    NOT NULL DEFAULT %s,

                indicator_name  TEXT        NOT NULL,
                value_float     DOUBLE PRECISION,
                value_json      JSONB,

                candle_is_final BOOLEAN     NOT NULL DEFAULT TRUE,
                calc_source     SMALLINT    NOT NULL DEFAULT 1,
                event_time_ms   BIGINT,

                created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                created_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,
                updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at_ms   BIGINT      NOT NULL DEFAULT (extract(epoch from now())*1000)::bigint,

                CONSTRAINT indicators_%s_pkey PRIMARY KEY (symbol_id, time, indicator_name)
            );
        $SQL$, tf, tf_minutes, tf);

        -- Hypertable (idempotent)
        BEGIN
            EXECUTE format(
                'SELECT create_hypertable(%L, %L, if_not_exists => TRUE, migrate_data => TRUE, chunk_time_interval => INTERVAL %L)',
                'market.indicators_' || tf,
                'time',
                chunk_txt
            );
        EXCEPTION
            WHEN undefined_function THEN
                RAISE NOTICE 'TimescaleDB not available, skipping create_hypertable for indicators_%', tf;
        END;

        -- Индексы под чтение
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_ind_%s_sid_time_desc ON market.indicators_%s (symbol_id, time DESC)', tf, tf);
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_ind_%s_symbol_time_desc ON market.indicators_%s (symbol, time DESC)', tf, tf);
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_ind_%s_name_time_desc ON market.indicators_%s (indicator_name, time DESC)', tf, tf);
        EXECUTE format('CREATE INDEX IF NOT EXISTS idx_ind_%s_time_ms_desc ON market.indicators_%s (time_ms DESC)', tf, tf);
    END LOOP;
END $$;
