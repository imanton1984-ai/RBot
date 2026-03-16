-- create_dataset_indexes.sql
-- Создание индексов для ускорения LEFT JOIN candles ↔ indicators_wide
-- при генерации датасета Super Entry.
--
-- Проблема: запросы по 2-2.5 сек на каждый символ × 125 символов × 6 TF = ~25 мин.
-- Решение: composite index на (symbol_id, tf_minutes, time) для indicators_wide
--           + индексы на (symbol, time) для candle таблиц.

-- ═══════════════════════════════════════════════════════════
-- indicators_wide — ключевой JOIN: ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $2
-- ═══════════════════════════════════════════════════════════
-- NOTE: TimescaleDB hypertables don't support CONCURRENTLY, using regular CREATE INDEX
CREATE INDEX IF NOT EXISTS idx_indicators_wide_sid_tf_time
    ON market.indicators_wide (symbol_id, tf_minutes, time);

-- ═══════════════════════════════════════════════════════════
-- Candle tables — WHERE c.symbol = $1 ORDER BY c.time ASC LIMIT $3
-- ═══════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_candles_15m_symbol_time
    ON market.candles_15m (symbol, time ASC);

CREATE INDEX IF NOT EXISTS idx_candles_1h_symbol_time
    ON market.candles_1h (symbol, time ASC);

CREATE INDEX IF NOT EXISTS idx_candles_4h_symbol_time
    ON market.candles_4h (symbol, time ASC);

CREATE INDEX IF NOT EXISTS idx_candles_1d_symbol_time
    ON market.candles_1d (symbol, time ASC);

CREATE INDEX IF NOT EXISTS idx_candles_1m_symbol_time
    ON market.candles_1m (symbol, time ASC);

CREATE INDEX IF NOT EXISTS idx_candles_5m_symbol_time
    ON market.candles_5m (symbol, time ASC);

-- ═══════════════════════════════════════════════════════════
-- market.pairs — JOIN ON p.symbol = c.symbol
-- ═══════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_pairs_symbol
    ON market.pairs (symbol);

CREATE INDEX IF NOT EXISTS idx_pairs_symbol_active
    ON market.pairs (symbol) WHERE is_active = true;

-- ═══════════════════════════════════════════════════════════
-- Verify indexes created
-- ═══════════════════════════════════════════════════════════
SELECT schemaname, tablename, indexname
FROM pg_indexes
WHERE schemaname = 'market'
  AND (indexname LIKE 'idx_candles%' OR indexname LIKE 'idx_indicators%' OR indexname LIKE 'idx_pairs%')
ORDER BY tablename, indexname;
