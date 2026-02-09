# Proposed Database Schemas for Predictors

These schemas are designed for storing predictions from ML and heuristic models. Each table includes a normalized score (0-1), with insertions only for scores >= 0.80 handled in the persistence logic. Tables are set up as TimescaleDB hypertables for efficient time-series data management.

## ML_future_price

```sql
CREATE TABLE market.ml_future_price (
    time timestamptz NOT NULL,
    symbol text NOT NULL,
    timeframe text NOT NULL,
    predicted_prices float8[] NOT NULL,  -- Array of predicted prices for next 10 candles
    score float8 NOT NULL CHECK (score >= 0 AND score <= 1)
);

SELECT create_hypertable('market.ml_future_price', by_range('time', INTERVAL '1 day'));

CREATE INDEX idx_ml_future_price_symbol_tf_time ON market.ml_future_price (symbol, timeframe, time DESC);
```

## CALC_FUTURE (Heuristic Future Price)

```sql
CREATE TABLE market.calc_future (
    time timestamptz NOT NULL,
    symbol text NOT NULL,
    timeframe text NOT NULL,
    predicted_prices float8[] NOT NULL,  -- Array of predicted prices for next 10 candles
    score float8 NOT NULL CHECK (score >= 0 AND score <= 1)
);

SELECT create_hypertable('market.calc_future', by_range('time', INTERVAL '1 day'));

CREATE INDEX idx_calc_future_symbol_tf_time ON market.calc_future (symbol, timeframe, time DESC);
```

## ML_Bounce_Break

```sql
CREATE TABLE market.ml_bounce_break (
    time timestamptz NOT NULL,
    symbol text NOT NULL,
    timeframe text NOT NULL,
    level float8 NOT NULL,  -- Support/Resistance level
    prob_bounce float8 NOT NULL,
    prob_break float8 NOT NULL,
    score float8 NOT NULL CHECK (score >= 0 AND score <= 1)
);

SELECT create_hypertable('market.ml_bounce_break', by_range('time', INTERVAL '1 day'));

CREATE INDEX idx_ml_bounce_break_symbol_tf_time ON market.ml_bounce_break (symbol, timeframe, time DESC);
```

## CALC_BOUNCE_BREAK (Heuristic Bounce/Break)

```sql
CREATE TABLE market.calc_bounce_break (
    time timestamptz NOT NULL,
    symbol text NOT NULL,
    timeframe text NOT NULL,
    level float8 NOT NULL,  -- Support/Resistance level
    prob_bounce float8 NOT NULL,
    prob_break float8 NOT NULL,
    score float8 NOT NULL CHECK (score >= 0 AND score <= 1)
);

SELECT create_hypertable('market.calc_bounce_break', by_range('time', INTERVAL '1 day'));

CREATE INDEX idx_calc_bounce_break_symbol_tf_time ON market.calc_bounce_break (symbol, timeframe, time DESC);
```

**Notes:**
- All tables use `time` as the partitioning key.
- Data sourced from `indicators_wide` and `raw_signals` views/tables.
- Persistence code should filter insertions where score < 0.80.
- Scores: 0.80 minimum to store, 0.96 high, 1.0 maximum, 0 minimum (though not stored if below 0.80).
- Integrate with existing bulk persistor for efficient batch inserts.
