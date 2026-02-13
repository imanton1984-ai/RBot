-- 085_trade_predictors.sql
-- Predictor registry and predictors tables

-- 1) Registry: unified registry for ML and hardcode predictors
CREATE TABLE IF NOT EXISTS trade.predictor_registry (
  predictor_id       bigserial PRIMARY KEY,

  -- what we predict
  aspect             smallint NOT NULL,  -- 1=price_target, 2=level_bounce, 3=level_breakout
  horizon_bars       int      NOT NULL DEFAULT 10,

  -- how we calculate
  calc_source        smallint NOT NULL,  -- 1=hard, 2=ml
  framework          text     NOT NULL,  -- "hardcode" | "xgboost" | "lightgbm"

  name               text     NOT NULL,  -- "price10_v1", "levels_v2"...
  version            text     NOT NULL,  -- semver/date/commit
  code_hash          text     NULL,      -- for hardcode (git commit), for models too

  -- ML artifact (for Rust inference using XGBoost)
  artifact_path      text     NULL,      -- path to .ubj/.json on disk/volume
  artifact_sha256    text     NULL,

  -- feature schema
  feature_schema_id  text     NOT NULL,  -- hash/version of feature schema
  calibration_json   jsonb    NULL,      -- Platt/isotonic/temperature

  -- quality metrics
  metrics_json       jsonb    NULL,      -- auc/mae/hitrate etc.
  trained_from       timestamptz NULL,
  trained_to         timestamptz NULL,

  is_active          boolean  NOT NULL DEFAULT true,
  created_at         timestamptz NOT NULL DEFAULT now(),
  updated_at         timestamptz NOT NULL DEFAULT now(),

  UNIQUE(calc_source, name, version)
);

CREATE INDEX IF NOT EXISTS ix_predictor_registry_active
  ON trade.predictor_registry(is_active, aspect, calc_source);

-- 2) predictors table (unified)
CREATE TABLE IF NOT EXISTS trade.predictors (
  -- technical key
  prediction_id      bigserial,

  -- time axes
  time              timestamptz NOT NULL,
  time_ms           bigint      NOT NULL,

  symbol_id         bigint      NOT NULL,
  symbol            text        NOT NULL,
  tf_minutes        int         NOT NULL,

  horizon_bars      int         NOT NULL DEFAULT 10,

  aspect            smallint    NOT NULL, -- 1=price_target, 2=level_bounce, 3=level_breakout
  calc_source       smallint    NOT NULL, -- 1=hard, 2=ml

  predictor_id      bigint      NOT NULL REFERENCES trade.predictor_registry(predictor_id),

  -- mandatory normalized score
  score_norm        real        NOT NULL,
  CHECK (score_norm >= 0.0 AND score_norm <= 1.0),

  -- universal prediction value:
  -- price_target: target_price
  -- bounce/breakout: probability
  value             double precision NOT NULL,

  -- intervals/uncertainty (useful for price)
  value_low         double precision NULL,
  value_high        double precision NULL,

  -- direction (for price/events)
  side              smallint NULL, -- -1 short/down, 0 neutral, +1 long/up

  -- level (if aspect is about levels)
  level_hash        text NULL,              -- stable level key (from sr_levels json)
  level_kind        smallint NULL,           -- 1 support, 2 resistance
  level_price       double precision NULL,
  level_strength    real NULL,
  level_distance_atr real NULL,

  -- computation context (in spirit of your tables)
  candle_is_final   boolean     NOT NULL DEFAULT true,
  event_time_ms     bigint      NULL,
  created_at        timestamptz NOT NULL DEFAULT now(),
  updated_at        timestamptz NOT NULL DEFAULT now(),

  -- for extensions (raw signals, features, explanations)
  details_json      jsonb       NULL,

  -- to do UPSERT without dancing
  prediction_key    text        NOT NULL,

  PRIMARY KEY(time, symbol_id, tf_minutes, prediction_key, predictor_id),
  
  -- Logical constraints to prevent incorrect data combinations
  -- For price_target (aspect=1), level-related fields should be NULL
  CONSTRAINT chk_price_has_no_levels
    CHECK (aspect <> 1 OR (level_hash IS NULL AND level_kind IS NULL AND level_price IS NULL 
                           AND level_strength IS NULL AND level_distance_atr IS NULL)),
  
  -- For level aspects (aspect=2,3), level-related fields should NOT be NULL
  CONSTRAINT chk_levels_required_for_level_aspects
    CHECK (aspect NOT IN (2, 3) OR (level_hash IS NOT NULL AND level_kind IS NOT NULL AND level_price IS NOT NULL))
);

-- Timescale hypertable
SELECT create_hypertable('trade.predictors', 'time', if_not_exists => TRUE);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS ix_predictors_symbol_tf_time
  ON trade.predictors(symbol_id, tf_minutes, time DESC);

-- Partial index for your 0.80 gate (table will already store only >=0.80, but index speeds up queries)
CREATE INDEX IF NOT EXISTS ix_predictors_aspect_score_hi
  ON trade.predictors(aspect, score_norm DESC, time DESC)
  WHERE score_norm >= 0.80;

-- Index for level-based queries
CREATE INDEX IF NOT EXISTS ix_predictors_level_hash
  ON trade.predictors(level_hash, time DESC)
  WHERE level_hash IS NOT NULL;

-- Index for predictor-based queries
CREATE INDEX IF NOT EXISTS ix_predictors_predictor_time
  ON trade.predictors(predictor_id, time DESC);