-- 060_policies.sql
-- retention: target ~12 000 bars per pair per TF for ML training (super_entry)
-- 1m:  12000 мин ≈ 8.3 days  -> 10 days (1m/5m kept as-is for now, can lower if needed)
-- 5m:  12000*5 = 60000 мин ≈ 41.7 days -> 45 days
-- 15m: 12000*15 = 125 days -> 130 days
-- 1h:  12000*60 = 500 days -> 510 days
-- 4h:  12000*240 = 2000 days -> 2000 days
-- 1d:  3700 days -> 3700 days (~10 years)

-- свечи
SELECT add_retention_policy('market.candles_1m', INTERVAL '10 days',   if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_5m', INTERVAL '45 days',   if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_15m',INTERVAL '130 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_1h', INTERVAL '510 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_4h', INTERVAL '2000 days', if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_1d', INTERVAL '3700 days', if_not_exists=>TRUE);

-- индикаторы snapshot (match 4h retention — 2000 days covers all TFs)
SELECT add_retention_policy('market.indicators_wide', INTERVAL '2000 days', if_not_exists=>TRUE);

-- raw_signals (храним меньше, чтобы не пухло)
SELECT add_retention_policy('market.raw_signals', INTERVAL '60 days', if_not_exists=>TRUE);

-- trade.final_signals (можно дольше)
SELECT add_retention_policy('trade.final_signals', INTERVAL '365 days', if_not_exists=>TRUE);

-- ml_train_examples (по желанию; можно держать дольше, но пока ограничим)
SELECT add_retention_policy('trade.ml_train_examples', INTERVAL '365 days', if_not_exists=>TRUE);


-- Compression (сильно экономит место)
-- candles
ALTER TABLE market.candles_1m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.candles_5m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.candles_15m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.candles_1h SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.candles_4h SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.candles_1d SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');

SELECT add_compression_policy('market.candles_1m', INTERVAL '6 hours',  if_not_exists=>TRUE);
SELECT add_compression_policy('market.candles_5m', INTERVAL '1 day',    if_not_exists=>TRUE);
SELECT add_compression_policy('market.candles_15m',INTERVAL '2 days',   if_not_exists=>TRUE);
SELECT add_compression_policy('market.candles_1h', INTERVAL '7 days',   if_not_exists=>TRUE);
SELECT add_compression_policy('market.candles_4h', INTERVAL '14 days',  if_not_exists=>TRUE);
SELECT add_compression_policy('market.candles_1d', INTERVAL '30 days',  if_not_exists=>TRUE);

-- indicators
ALTER TABLE market.indicators_wide SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');


SELECT add_compression_policy('market.indicators_wide', INTERVAL '20 days',  if_not_exists=>TRUE);
