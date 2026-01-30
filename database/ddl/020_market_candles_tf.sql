-- 020_market_candles_tf.sql (OPTIMIZED: time_ms BIGINT + time timestamptz, no triggers)

-- ВАЖНО:
-- 1) time_ms = close time в миллисекундах (epoch ms)
-- 2) time = timestamptz calculated in Rust code before insertion
-- 3) No triggers to slow down bulk inserts

-- 1. Create the table structure without triggers
CREATE TABLE IF NOT EXISTS market.candles_1m (
    time_ms   BIGINT NOT NULL,
    time      TIMESTAMPTZ NOT NULL, -- Rust will send this field filled

    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,

    open   DOUBLE PRECISION NOT NULL,
    high   DOUBLE PRECISION NOT NULL,
    low    DOUBLE PRECISION NOT NULL,
    close  DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,

    source_event_time_ms BIGINT,
    source_event_time    TIMESTAMPTZ,

    PRIMARY KEY(symbol_id, time)
);

-- 2. Create tables for other timeframes
CREATE TABLE IF NOT EXISTS market.candles_5m  (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_15m (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_1h  (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_4h  (LIKE market.candles_1m INCLUDING ALL);
CREATE TABLE IF NOT EXISTS market.candles_1d  (LIKE market.candles_1m INCLUDING ALL);

-- 3. Create hypertables (TimescaleDB)
SELECT create_hypertable('market.candles_1m', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '1 day');
SELECT create_hypertable('market.candles_5m', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '2 days');
SELECT create_hypertable('market.candles_15m','time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '7 days');
SELECT create_hypertable('market.candles_1h', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '30 days');
SELECT create_hypertable('market.candles_4h', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '60 days');
SELECT create_hypertable('market.candles_1d', 'time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '90 days');

-- 4. Create indexes
CREATE INDEX IF NOT EXISTS ix_candles_1m_symbol_time_desc  ON market.candles_1m (symbol_id, time DESC);
CREATE INDEX IF NOT EXISTS ix_candles_5m_symbol_time_desc  ON market.candles_5m (symbol_id, time DESC);
CREATE INDEX IF NOT EXISTS ix_candles_15m_symbol_time_desc ON market.candles_15m(symbol_id, time DESC);
CREATE INDEX IF NOT EXISTS ix_candles_1h_symbol_time_desc  ON market.candles_1h (symbol_id, time DESC);
CREATE INDEX IF NOT EXISTS ix_candles_4h_symbol_time_desc  ON market.candles_4h (symbol_id, time DESC);
CREATE INDEX IF NOT EXISTS ix_candles_1d_symbol_time_desc  ON market.candles_1d (symbol_id, time DESC);
