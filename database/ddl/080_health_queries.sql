-- 080_health_queries.sql
-- Use with: psql -tAc "<QUERY>" "$DATABASE_URL"

-- [H00] DB reachable
SELECT 1 AS ok;

-- [H01] Timescale extension installed
SELECT EXISTS (
  SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'
) AS timescale_installed;

-- [H02] Schemas exist
SELECT
  EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = 'market') AS market_schema,
  EXISTS(SELECT 1 FROM pg_namespace WHERE nspname = 'trade')  AS trade_schema;

-- [H03] Core tables exist
SELECT
  to_regclass('market.pairs')              IS NOT NULL AS pairs,
  to_regclass('market.collected_candles')  IS NOT NULL AS collected_candles,
  to_regclass('market.candles_live')       IS NOT NULL AS candles_live,
  to_regclass('market.raw_signals')        IS NOT NULL AS raw_signals,
  to_regclass('trade.final_signals')       IS NOT NULL AS final_signals,
  to_regclass('trade.orders')              IS NOT NULL AS orders,
  to_regclass('trade.positions')           IS NOT NULL AS positions,
  to_regclass('trade.position_events')     IS NOT NULL AS position_events,
  to_regclass('trade.ml_train_examples')   IS NOT NULL AS ml_train_examples;

-- [H04] Candle TF tables exist
SELECT
  to_regclass('market.candles_1m')  IS NOT NULL AS c1m,
  to_regclass('market.candles_5m')  IS NOT NULL AS c5m,
  to_regclass('market.candles_15m') IS NOT NULL AS c15m,
  to_regclass('market.candles_1h')  IS NOT NULL AS c1h,
  to_regclass('market.candles_4h')  IS NOT NULL AS c4h,
  to_regclass('market.candles_1d')  IS NOT NULL AS c1d;

-- [H05] Indicators TF tables exist
SELECT
  to_regclass('market.indicators_1m')  IS NOT NULL AS i1m,
  to_regclass('market.indicators_5m')  IS NOT NULL AS i5m,
  to_regclass('market.indicators_15m') IS NOT NULL AS i15m,
  to_regclass('market.indicators_1h')  IS NOT NULL AS i1h,
  to_regclass('market.indicators_4h')  IS NOT NULL AS i4h,
  to_regclass('market.indicators_1d')  IS NOT NULL AS i1d;

-- [H06] Hypertables present (candles + indicators + raw_signals + final_signals + ml_train_examples + position_events)
-- Expect candles_*/indicators_* + market.raw_signals + trade.final_signals + trade.ml_train_examples + trade.position_events
SELECT
  (SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='market' AND hypertable_name LIKE 'candles_%')    AS candles_ht,
  (SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='market' AND hypertable_name LIKE 'indicators_%') AS indicators_ht,
  (SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='market' AND hypertable_name='raw_signals')       AS raw_signals_ht,
  (SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='trade'  AND hypertable_name='final_signals')     AS final_signals_ht,
  (SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='trade'  AND hypertable_name='ml_train_examples') AS ml_train_examples_ht,
  (SELECT count(*) FROM timescaledb_information.hypertables WHERE hypertable_schema='trade'  AND hypertable_name='position_events')   AS position_events_ht;

-- [H07] Compression enabled for candles/indicators (retention can exist without compression)
SELECT
  hypertable_schema,
  hypertable_name,
  compression_enabled
FROM timescaledb_information.hypertables
WHERE (hypertable_schema='market' AND (hypertable_name LIKE 'candles_%' OR hypertable_name LIKE 'indicators_%'))
ORDER BY hypertable_schema, hypertable_name;

-- [H08] Retention policies jobs existence (Timescale creates background jobs)
-- We search for 'policy_retention' jobs and match hypertables in config JSON.
-- This query returns how many retention jobs are configured for our key tables.
SELECT
  SUM(CASE WHEN (config::text LIKE '%market.candles_1m%' ) THEN 1 ELSE 0 END) AS ret_candles_1m,
  SUM(CASE WHEN (config::text LIKE '%market.candles_5m%' ) THEN 1 ELSE 0 END) AS ret_candles_5m,
  SUM(CASE WHEN (config::text LIKE '%market.candles_15m%') THEN 1 ELSE 0 END) AS ret_candles_15m,
  SUM(CASE WHEN (config::text LIKE '%market.candles_1h%' ) THEN 1 ELSE 0 END) AS ret_candles_1h,
  SUM(CASE WHEN (config::text LIKE '%market.candles_4h%' ) THEN 1 ELSE 0 END) AS ret_candles_4h,
  SUM(CASE WHEN (config::text LIKE '%market.candles_1d%' ) THEN 1 ELSE 0 END) AS ret_candles_1d,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_1m%' ) THEN 1 ELSE 0 END) AS ret_ind_1m,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_5m%' ) THEN 1 ELSE 0 END) AS ret_ind_5m,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_15m%') THEN 1 ELSE 0 END) AS ret_ind_15m,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_1h%' ) THEN 1 ELSE 0 END) AS ret_ind_1h,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_4h%' ) THEN 1 ELSE 0 END) AS ret_ind_4h,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_1d%' ) THEN 1 ELSE 0 END) AS ret_ind_1d,
  SUM(CASE WHEN (config::text LIKE '%market.raw_signals%' ) THEN 1 ELSE 0 END) AS ret_raw_signals,
  SUM(CASE WHEN (config::text LIKE '%trade.final_signals%') THEN 1 ELSE 0 END) AS ret_final_signals,
  SUM(CASE WHEN (config::text LIKE '%trade.ml_train_examples%') THEN 1 ELSE 0 END) AS ret_ml_examples
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_retention';

-- [H09] Compression policies jobs existence
SELECT
  SUM(CASE WHEN (config::text LIKE '%market.candles_1m%' ) THEN 1 ELSE 0 END) AS comp_candles_1m,
  SUM(CASE WHEN (config::text LIKE '%market.candles_5m%' ) THEN 1 ELSE 0 END) AS comp_candles_5m,
  SUM(CASE WHEN (config::text LIKE '%market.candles_15m%') THEN 1 ELSE 0 END) AS comp_candles_15m,
  SUM(CASE WHEN (config::text LIKE '%market.candles_1h%' ) THEN 1 ELSE 0 END) AS comp_candles_1h,
  SUM(CASE WHEN (config::text LIKE '%market.candles_4h%' ) THEN 1 ELSE 0 END) AS comp_candles_4h,
  SUM(CASE WHEN (config::text LIKE '%market.candles_1d%' ) THEN 1 ELSE 0 END) AS comp_candles_1d,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_1m%' ) THEN 1 ELSE 0 END) AS comp_ind_1m,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_5m%' ) THEN 1 ELSE 0 END) AS comp_ind_5m,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_15m%') THEN 1 ELSE 0 END) AS comp_ind_15m,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_1h%' ) THEN 1 ELSE 0 END) AS comp_ind_1h,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_4h%' ) THEN 1 ELSE 0 END) AS comp_ind_4h,
  SUM(CASE WHEN (config::text LIKE '%market.indicators_1d%' ) THEN 1 ELSE 0 END) AS comp_ind_1d
FROM timescaledb_information.jobs
WHERE proc_name = 'policy_compression';
