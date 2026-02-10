-- 070_views_all.sql (HARDCORE)

CREATE OR REPLACE VIEW market.candles AS
SELECT time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time, 1::SMALLINT as tf_minutes
FROM market.candles_1m
UNION ALL
SELECT time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time, 5::SMALLINT as tf_minutes
FROM market.candles_5m
UNION ALL
SELECT time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time, 15::SMALLINT as tf_minutes
FROM market.candles_15m
UNION ALL
SELECT time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time, 60::SMALLINT as tf_minutes
FROM market.candles_1h
UNION ALL
SELECT time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time, 240::SMALLINT as tf_minutes
FROM market.candles_4h
UNION ALL
SELECT time_ms, time, symbol_id, open, high, low, close, volume, source_event_time_ms, source_event_time, 1440::SMALLINT as tf_minutes
FROM market.candles_1d;

CREATE OR REPLACE VIEW market.indicators AS
SELECT * FROM market.indicators_wide;

