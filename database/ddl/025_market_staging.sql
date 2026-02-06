-- 025_market_staging.sql
-- UNLOGGED staging tables for fast COPY + dedup insert

CREATE UNLOGGED TABLE IF NOT EXISTS market.candles_staging (
  time_ms BIGINT NOT NULL,
  symbol_id BIGINT NOT NULL,
  open DOUBLE PRECISION NOT NULL,
  high DOUBLE PRECISION NOT NULL,
  low  DOUBLE PRECISION NOT NULL,
  close DOUBLE PRECISION NOT NULL,
  volume DOUBLE PRECISION NOT NULL,
  source_event_time_ms BIGINT
);

CREATE UNLOGGED TABLE IF NOT EXISTS market.indicators_staging (
  time_ms BIGINT NOT NULL,
  symbol_id BIGINT NOT NULL,

  ema_20 REAL, ema_50 REAL, ema_200 REAL,
  sma REAL,
  rsi REAL,
  macd REAL, macd_signal REAL, macd_hist REAL,
  atr REAL, adx REAL,
  bb_upper REAL, bb_mid REAL, bb_lower REAL,
  stoch_k REAL, stoch_d REAL,
  vwap REAL, obv REAL, cci REAL, williams REAL,
  alli_jaw REAL, alli_teeth REAL, alli_lips REAL,
  sr_levels JSONB,
  features_version TEXT NOT NULL DEFAULT 'v1',
  updated_at_ms BIGINT
);

-- small indexes help ON CONFLICT/merge
CREATE INDEX IF NOT EXISTS ix_candles_staging_symbol_time ON market.candles_staging(symbol_id, time_ms);
CREATE INDEX IF NOT EXISTS ix_ind_staging_symbol_time     ON market.indicators_staging(symbol_id, time_ms);
