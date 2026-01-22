-- 040_market_raw_signals.sql (HARDCORE: time_ms BIGINT + generated time)

-- 1. Создаем функцию синхронизации (если она не была создана ранее или создаем уникальную)
CREATE OR REPLACE FUNCTION market.sync_raw_signals_time()
RETURNS TRIGGER AS $$
BEGIN
    -- Основное время
    NEW.time := to_timestamp(NEW.time_ms / 1000.0);
    
    -- Логика для created_at
    IF NEW.created_at_ms IS NULL THEN
        NEW.created_at := now();
    ELSE
        NEW.created_at := to_timestamp(NEW.created_at_ms / 1000.0);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Создаем таблицу с обычными колонками
CREATE TABLE IF NOT EXISTS market.raw_signals (
    time_ms   BIGINT NOT NULL,
    time      TIMESTAMPTZ NOT NULL, -- Обычная колонка

    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    tf_minutes SMALLINT NOT NULL,

    indicator_id SMALLINT NOT NULL,
    signal_kind  SMALLINT NOT NULL,

    side   SMALLINT NOT NULL,          -- 1 long, -1 short, 0 neutral
    score  REAL NOT NULL,              -- 0..1
    value  REAL,
    details JSONB,

    created_at_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), -- Обычная колонка

    -- ВАЖНО: Добавляем 'time' в PRIMARY KEY для TimescaleDB
    PRIMARY KEY(symbol_id, tf_minutes, indicator_id, signal_kind, time)
);

-- 3. Вешаем триггер
DROP TRIGGER IF EXISTS trg_sync_raw_signals ON market.raw_signals;
CREATE TRIGGER trg_sync_raw_signals 
    BEFORE INSERT OR UPDATE ON market.raw_signals 
    FOR EACH ROW EXECUTE FUNCTION market.sync_raw_signals_time();

-- 4. Создаем гипертаблицу
SELECT create_hypertable('market.raw_signals', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '7 days');

-- 5. Индексы
CREATE INDEX IF NOT EXISTS ix_raw_signals_time_desc ON market.raw_signals(time DESC);
CREATE INDEX IF NOT EXISTS ix_raw_signals_symbol_tf_time_desc ON market.raw_signals(symbol_id, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS ix_raw_signals_score_desc ON market.raw_signals(score DESC);