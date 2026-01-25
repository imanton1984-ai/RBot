-- 060_policies.sql
-- retention: стараемся удерживать примерно ~1200 баров на TF (кроме 1d)
-- 1m: 1200 мин ≈ 20h  -> округлим до 2 days, чтобы не было слишком агрессивно
-- 5m: 1200*5 = 6000 мин ≈ 4.2 days -> 7 days
-- 15m: 12.5 days -> 21 days
-- 1h: 50 days -> 90 days
-- 4h: 200 days -> 240 days
-- 1d: 2 years -> 730 days

-- свечи
SELECT add_retention_policy('market.candles_1m', INTERVAL '2 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_5m', INTERVAL '7 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_15m',INTERVAL '21 days', if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_1h', INTERVAL '90 days', if_not_exists=>TRUE);
SELECT add_retention_policy('market.candles_4h', INTERVAL '240 days',if_not_exists=>TRUE);
-- 1d: меняем с 730 на 1095 (3 года), чтобы соответствовать ожиданиям Health Gate
SELECT add_retention_policy('market.candles_1d', INTERVAL '1095 days', if_not_exists=>TRUE);

-- индикаторы snapshot (тот же retention)
SELECT add_retention_policy('market.indicators_1m', INTERVAL '2 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('market.indicators_5m', INTERVAL '7 days',  if_not_exists=>TRUE);
SELECT add_retention_policy('market.indicators_15m',INTERVAL '21 days', if_not_exists=>TRUE);
SELECT add_retention_policy('market.indicators_1h', INTERVAL '90 days', if_not_exists=>TRUE);
SELECT add_retention_policy('market.indicators_4h', INTERVAL '240 days',if_not_exists=>TRUE);
-- индикаторы 1d тоже подтянем для консистентности
SELECT add_retention_policy('market.indicators_1d', INTERVAL '1095 days', if_not_exists=>TRUE);

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
ALTER TABLE market.indicators_1m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.indicators_5m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.indicators_15m SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.indicators_1h SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.indicators_4h SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');
ALTER TABLE market.indicators_1d SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol_id');

SELECT add_compression_policy('market.indicators_1m', INTERVAL '6 hours',  if_not_exists=>TRUE);
SELECT add_compression_policy('market.indicators_5m', INTERVAL '1 day',    if_not_exists=>TRUE);
SELECT add_compression_policy('market.indicators_15m',INTERVAL '2 days',   if_not_exists=>TRUE);
SELECT add_compression_policy('market.indicators_1h', INTERVAL '7 days',   if_not_exists=>TRUE);
SELECT add_compression_policy('market.indicators_4h', INTERVAL '14 days',  if_not_exists=>TRUE);
SELECT add_compression_policy('market.indicators_1d', INTERVAL '30 days',  if_not_exists=>TRUE);
