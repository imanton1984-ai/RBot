-- 061_update_retention_12k.sql
-- Migration: increase retention to support 12000 candles per pair per TF
-- for ML model training (super_entry strategy).
--
-- Run: psql $DATABASE_URL -f database/ddl/061_update_retention_12k.sql
--
-- NOTE: TimescaleDB add_retention_policy with if_not_exists=>TRUE won't update
-- an existing policy. We need to remove old policies first, then add new ones.

-- Remove old retention policies (candles & indicators)
SELECT remove_retention_policy('market.candles_1m',  if_exists=>TRUE);
SELECT remove_retention_policy('market.candles_5m',  if_exists=>TRUE);
SELECT remove_retention_policy('market.candles_15m', if_exists=>TRUE);
SELECT remove_retention_policy('market.candles_1h',  if_exists=>TRUE);
SELECT remove_retention_policy('market.candles_4h',  if_exists=>TRUE);
SELECT remove_retention_policy('market.candles_1d',  if_exists=>TRUE);
SELECT remove_retention_policy('market.indicators_wide', if_exists=>TRUE);

-- Add new policies with expanded retention:
-- 1m:  10 days  (~14400 candles per pair)
-- 5m:  45 days  (~12960 candles per pair)
-- 15m: 130 days (~12480 candles per pair)
-- 1h:  510 days (~12240 candles per pair)
-- 4h:  2000 days (~12000 candles per pair)
-- 1d:  3700 days (~3700 candles per pair, ~10 years)
SELECT add_retention_policy('market.candles_1m',  INTERVAL '10 days');
SELECT add_retention_policy('market.candles_5m',  INTERVAL '45 days');
SELECT add_retention_policy('market.candles_15m', INTERVAL '130 days');
SELECT add_retention_policy('market.candles_1h',  INTERVAL '510 days');
SELECT add_retention_policy('market.candles_4h',  INTERVAL '2000 days');
SELECT add_retention_policy('market.candles_1d',  INTERVAL '3700 days');

-- Indicators: keep long enough to cover the deepest TF (4h = 2000 days)
SELECT add_retention_policy('market.indicators_wide', INTERVAL '2000 days');

-- Verify
SELECT hypertable_name, 
       (config->>'drop_after')::interval AS retention
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention'
  AND config->>'hypertable_name' IN (
    'candles_1m','candles_5m','candles_15m','candles_1h','candles_4h','candles_1d','indicators_wide'
  )
ORDER BY hypertable_name;
