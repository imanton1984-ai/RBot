-- 020_market_candles_tf.sql (HARDCORE: time_ms BIGINT + generated time timestamptz)

-- ВАЖНО:
-- 1) time_ms = close time в миллисекундах (epoch ms)
-- 2) time = generated STORED timestamptz (для Timescale hypertable)
-- 3) Hypertable по time (generated) - works, time computed once and stored.

-- 1. Создаем функцию для автоматического заполнения времени
CREATE OR REPLACE FUNCTION market.sync_candle_time()
RETURNS TRIGGER AS $$
BEGIN
    -- Конвертируем ms в timestamptz
    NEW.time := to_timestamp(NEW.time_ms / 1000.0);
    
    -- Обработка source_event_time если нужно
    IF NEW.source_event_time_ms IS NOT NULL THEN
        NEW.source_event_time := to_timestamp(NEW.source_event_time_ms / 1000.0);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Переопределяем структуру таблицы
CREATE TABLE IF NOT EXISTS market.candles_1m (
    time_ms   BIGINT NOT NULL,
    time      TIMESTAMPTZ NOT NULL,

    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,

    open   DOUBLE PRECISION NOT NULL,
    high   DOUBLE PRECISION NOT NULL,
    low    DOUBLE PRECISION NOT NULL,
    close  DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,

    source_event_time_ms BIGINT,
    source_event_time    TIMESTAMPTZ,

    -- ИЗМЕНЕНИЕ ЗДЕСЬ: Добавляем 'time' в первичный ключ
    PRIMARY KEY(symbol_id, time)
);

-- 3. Создаем таблицы для других таймфреймов
CREATE TABLE IF NOT EXISTS market.candles_5m  (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_15m (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_1h  (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_4h  (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_1d  (LIKE market.candles_1m INCLUDING ALL);

-- 4. Вешаем триггер на каждую таблицу (чтобы автоматически конвертировать ms в time)
CREATE TRIGGER trg_sync_time_1m BEFORE INSERT OR UPDATE ON market.candles_1m FOR EACH ROW EXECUTE FUNCTION market.sync_candle_time();
CREATE TRIGGER trg_sync_time_5m BEFORE INSERT OR UPDATE ON market.candles_5m FOR EACH ROW EXECUTE FUNCTION market.sync_candle_time();
CREATE TRIGGER trg_sync_time_15m BEFORE INSERT OR UPDATE ON market.candles_15m FOR EACH ROW EXECUTE FUNCTION market.sync_candle_time();
CREATE TRIGGER trg_sync_time_1h BEFORE INSERT OR UPDATE ON market.candles_1h FOR EACH ROW EXECUTE FUNCTION market.sync_candle_time();
CREATE TRIGGER trg_sync_time_4h BEFORE INSERT OR UPDATE ON market.candles_4h FOR EACH ROW EXECUTE FUNCTION market.sync_candle_time();
CREATE TRIGGER trg_sync_time_1d BEFORE INSERT OR UPDATE ON market.candles_1d FOR EACH ROW EXECUTE FUNCTION market.sync_candle_time();

-- 5. Теперь превращаем в гипертаблицы (теперь ошибок не будет)
SELECT create_hypertable('market.candles_1m', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '1 day');
SELECT create_hypertable('market.candles_5m', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '2 days');
SELECT create_hypertable('market.candles_15m','time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '7 days');
SELECT create_hypertable('market.candles_1h', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '30 days');
SELECT create_hypertable('market.candles_4h', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '60 days');
SELECT create_hypertable('market.candles_1d', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '90 days');

-- Индексы остаются без изменений
CREATE INDEX IF NOT EXISTS ix_candles_1m_symbol_time_desc  ON market.candles_1m (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_candles_5m_symbol_time_desc  ON market.candles_5m (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_candles_15m_symbol_time_desc ON market.candles_15m(symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_candles_1h_symbol_time_desc  ON market.candles_1h (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_candles_4h_symbol_time_desc  ON market.candles_4h (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_candles_1d_symbol_time_desc  ON market.candles_1d (symbol_id, time_ms DESC);
