-- 070_views_all.sql (HARDCORE)

CREATE OR REPLACE VIEW market.candles AS
SELECT 1::SMALLINT AS tf_minutes, time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time
FROM market.candles_1m
UNION ALL
SELECT 5::SMALLINT AS tf_minutes, time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time
FROM market.candles_5m
UNION ALL
SELECT 15::SMALLINT AS tf_minutes, time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time
FROM market.candles_15m
UNION ALL
SELECT 60::SMALLINT AS tf_minutes, time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time
FROM market.candles_1h
UNION ALL
SELECT 240::SMALLINT AS tf_minutes, time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time
FROM market.candles_4h
UNION ALL
SELECT 1440::SMALLINT AS tf_minutes, time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time
FROM market.candles_1d;

CREATE OR REPLACE VIEW market.indicators AS
SELECT 1::SMALLINT AS tf_minutes, * FROM market.indicators_1m
UNION ALL SELECT 5::SMALLINT AS tf_minutes, * FROM market.indicators_5m
UNION ALL SELECT 15::SMALLINT AS tf_minutes, * FROM market.indicators_15m
UNION ALL SELECT 60::SMALLINT AS tf_minutes, * FROM market.indicators_1h
UNION ALL SELECT 240::SMALLINT AS tf_minutes, * FROM market.indicators_4h
UNION ALL SELECT 1440::SMALLINT AS tf_minutes, * FROM market.indicators_1d;

