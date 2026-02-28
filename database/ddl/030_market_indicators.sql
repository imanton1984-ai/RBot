-- database/ddl/030_market_indicators.sql
CREATE SCHEMA IF NOT EXISTS market;
CREATE EXTENSION IF NOT EXISTS timescaledb;

DROP TABLE IF EXISTS market.indicators_wide CASCADE;

CREATE TABLE IF NOT EXISTS market.indicators_wide (
    time            TIMESTAMPTZ NOT NULL,
    time_ms         BIGINT      NOT NULL,
    symbol_id       BIGINT      NOT NULL,
    symbol          TEXT        NOT NULL,
    tf_minutes      SMALLINT    NOT NULL,

    -- Momentum
    rsi             REAL,
    cci             REAL,
    stoch_k         REAL,
    stoch_d         REAL,
    williams        REAL,
    
    -- Trend
    macd            REAL,
    macd_signal     REAL,
    macd_hist       REAL,
    adx             REAL,
    sma             REAL,
    ema_20          REAL,
    ema_50          REAL,
    ema_200         REAL,
    
    -- Volatility
    bb_upper        REAL,
    bb_mid          REAL,
    bb_lower        REAL,
    atr             REAL,
    
    -- Volume
    obv             DOUBLE PRECISION,
    vwap            DOUBLE PRECISION,
    volume_spike    REAL,
    
    -- Alligator
    alligator_jaw   REAL,
    alligator_teeth REAL,
    alligator_lips  REAL,
    
    -- Other
    trend           SMALLINT,
    trend_short     SMALLINT,
    poc             REAL,
    
    -- New indicators (v2)
    mfi             REAL,        -- Money Flow Index (0-100)
    fibo_pivot      REAL,        -- Fibonacci Pivot
    fibo_r1         REAL,        -- Fibonacci R1
    fibo_s1         REAL,        -- Fibonacci S1
    supertrend      REAL,        -- SuperTrend value
    supertrend_dir  SMALLINT,    -- SuperTrend direction (1=bullish, -1=bearish)
    cmf             REAL,        -- Chaikin Money Flow (-1 to +1)
    
    -- Complex data
    sr_levels       JSONB,
    
    -- Meta
    candle_is_final BOOLEAN     NOT NULL DEFAULT TRUE,
    calc_source     SMALLINT    NOT NULL DEFAULT 1,
    event_time_ms   BIGINT,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    
    PRIMARY KEY (symbol_id, tf_minutes, time)
);

-- Hypertable
SELECT create_hypertable('market.indicators_wide', 'time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '7 days');

-- Indexes
CREATE INDEX IF NOT EXISTS idx_indicators_wide_sym_tf_time ON market.indicators_wide (symbol_id, tf_minutes, time DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_wide_time_ms ON market.indicators_wide (time_ms DESC);
CREATE INDEX IF NOT EXISTS idx_indicators_wide_symbol ON market.indicators_wide(symbol);

-- Compression
ALTER TABLE market.indicators_wide SET (
    timescaledb.compress, 
    timescaledb.compress_segmentby = 'symbol_id, tf_minutes',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('market.indicators_wide', INTERVAL '3 days');