-- 030_market_indicators_tf.sql (HARDCORE: time_ms BIGINT + generated time)

-- 1. Используем уже созданную функцию или создаем аналогичную для индикаторов
-- Если market.sync_candle_time уже создана в 020, можно использовать её.
-- Но для индикаторов добавим логику updated_at:
CREATE OR REPLACE FUNCTION market.sync_indicator_time()
RETURNS TRIGGER AS $$
BEGIN
    -- Основное время
    NEW.time := to_timestamp(NEW.time_ms / 1000.0);
    
    -- Логика для updated_at
    IF NEW.updated_at_ms IS NULL THEN
        NEW.updated_at := now();
    ELSE
        NEW.updated_at := to_timestamp(NEW.updated_at_ms / 1000.0);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Создаем таблицу с обычными колонками (НЕ GENERATED)
CREATE TABLE IF NOT EXISTS market.indicators_1m (
    time_ms   BIGINT NOT NULL,
    time      TIMESTAMPTZ NOT NULL, -- Обычная колонка

    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,

    ema20  REAL, ema50 REAL, ema200 REAL,
    sma    REAL,
    rsi    REAL,
    macd       REAL,
    macd_signal REAL,
    macd_hist   REAL,
    atr REAL,
    adx REAL,
    bb_upper REAL, bb_mid REAL, bb_lower REAL,
    stoch_k REAL, stoch_d REAL,
    vwap REAL,
    obv  REAL,
    cci  REAL,
    williams REAL,
    alli_jaw REAL, alli_teeth REAL, alli_lips REAL,
    sr_levels JSONB,
    poc REAL,

    features_version TEXT NOT NULL DEFAULT 'v1',
    updated_at_ms BIGINT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), -- Обычная колонка

    -- ВАЖНО: time включен в PK для TimescaleDB
    PRIMARY KEY(symbol_id, time)
);

-- 3. Клонируем структуру
CREATE TABLE IF NOT EXISTS market.indicators_5m  (LIKE market.indicators_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.indicators_15m (LIKE market.indicators_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.indicators_1h  (LIKE market.indicators_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.indicators_4h  (LIKE market.indicators_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.indicators_1d  (LIKE market.indicators_1m INCLUDING ALL);

-- 4. Вешаем триггеры
CREATE TRIGGER trg_ind_sync_1m BEFORE INSERT OR UPDATE ON market.indicators_1m FOR EACH ROW EXECUTE FUNCTION market.sync_indicator_time();
CREATE TRIGGER trg_ind_sync_5m BEFORE INSERT OR UPDATE ON market.indicators_5m FOR EACH ROW EXECUTE FUNCTION market.sync_indicator_time();
CREATE TRIGGER trg_ind_sync_15m BEFORE INSERT OR UPDATE ON market.indicators_15m FOR EACH ROW EXECUTE FUNCTION market.sync_indicator_time();
CREATE TRIGGER trg_ind_sync_1h BEFORE INSERT OR UPDATE ON market.indicators_1h FOR EACH ROW EXECUTE FUNCTION market.sync_indicator_time();
CREATE TRIGGER trg_ind_sync_4h BEFORE INSERT OR UPDATE ON market.indicators_4h FOR EACH ROW EXECUTE FUNCTION market.sync_indicator_time();
CREATE TRIGGER trg_ind_sync_1d BEFORE INSERT OR UPDATE ON market.indicators_1d FOR EACH ROW EXECUTE FUNCTION market.sync_indicator_time();

-- 5. Превращаем в гипертаблицы
SELECT create_hypertable('market.indicators_1m', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '1 day');
SELECT create_hypertable('market.indicators_5m', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '2 days');
SELECT create_hypertable('market.indicators_15m','time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '7 days');
SELECT create_hypertable('market.indicators_1h', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '30 days');
SELECT create_hypertable('market.indicators_4h', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '60 days');
SELECT create_hypertable('market.indicators_1d', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '90 days');

-- Индексы (используем time_ms для совместимости с вашими выборками)
CREATE INDEX IF NOT EXISTS ix_ind_1m_symbol_time_desc  ON market.indicators_1m (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_ind_5m_symbol_time_desc  ON market.indicators_5m (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_ind_15m_symbol_time_desc ON market.indicators_15m(symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_ind_1h_symbol_time_desc  ON market.indicators_1h (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_ind_4h_symbol_time_desc  ON market.indicators_4h (symbol_id, time_ms DESC);
CREATE INDEX IF NOT EXISTS ix_ind_1d_symbol_time_desc  ON market.indicators_1d (symbol_id, time_ms DESC);