-- 090_strategy_support.sql
-- Добавляет поддержку 6 стратегий и обновляет таблицу predictors

-- ═══════════════════════════════════════════════════════════════════════════
-- 1. Обновление таблицы final_signals - добавление колонок стратегии
-- ═══════════════════════════════════════════════════════════════════════════

-- Добавляем колонку strategy_id (1-6 для 6 стратегий)
ALTER TABLE trade.final_signals 
ADD COLUMN IF NOT EXISTS strategy_id SMALLINT NOT NULL DEFAULT 6;

-- Добавляем колонку strategy_name
ALTER TABLE trade.final_signals 
ADD COLUMN IF NOT EXISTS strategy_name TEXT NOT NULL DEFAULT 'level_consensus';

-- Добавляем индекс для фильтрации по стратегиям
CREATE INDEX IF NOT EXISTS ix_final_signals_strategy 
ON trade.final_signals(strategy_id, time DESC);

CREATE INDEX IF NOT EXISTS ix_final_signals_strategy_name 
ON trade.final_signals(strategy_name, time DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- 2. Обновление таблицы predictors - поддержка ML + heuristic отдельно
-- ═══════════════════════════════════════════════════════════════════════════

-- Таблица predictors уже имеет правильную структуру:
-- - aspect: 1=price_target, 2=level_bounce, 3=level_breakout
-- - calc_source: 1=hard (heuristic), 2=ml
-- 
-- Теперь для каждого aspect будет:
-- - PriceTarget: 2 строки (ML + heuristic) + 1 consensus = 3 строки
-- - LevelBounce: 2 строки (ML + heuristic)
-- - LevelBreakout: 2 строки (ML + heuristic)
-- Итого: 7 строк на пару/таймфрейм
--
-- Consensus рассчитывается отдельно и сохраняется как дополнительная строка
-- с calc_source = 3 (consensus) - нужно добавить поддержку

-- Добавляем новое значение для calc_source = 3 (consensus)
-- Примечание: это логическое изменение, не требует изменения схемы
-- CalcSource::Consensus будет иметь as_int() = 3

-- ═══════════════════════════════════════════════════════════════════════════
-- 3. Таблица для хранения результатов backtest по стратегиям
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS trade.backtest_results_by_strategy (
    id              BIGSERIAL,
    signal_time     TIMESTAMPTZ NOT NULL,
    signal_time_ms  BIGINT NOT NULL,
    symbol          TEXT NOT NULL,
    symbol_id       BIGINT NOT NULL,
    tf_minutes      SMALLINT NOT NULL,
    strategy_id     SMALLINT NOT NULL,
    strategy_name   TEXT NOT NULL,
    side            SMALLINT NOT NULL,
    final_score     REAL NOT NULL,
    entry_price     REAL NOT NULL,
    sl_price        REAL NOT NULL,
    tp1_price       REAL NOT NULL,
    tp2_price       REAL,
    tp3_price       REAL,
    outcome         TEXT NOT NULL,
    pnl_pct         DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    bars_to_outcome INT NOT NULL,
    max_favorable   DOUBLE PRECISION NOT NULL,
    max_adverse     DOUBLE PRECISION NOT NULL,
    tp1_hit         BOOLEAN NOT NULL DEFAULT false,
    tp2_hit         BOOLEAN NOT NULL DEFAULT false,
    tp3_hit         BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (symbol_id, tf_minutes, signal_time, strategy_id)
);

SELECT create_hypertable('trade.backtest_results_by_strategy', 'signal_time', if_not_exists => TRUE, chunk_time_interval => INTERVAL '30 days');

CREATE INDEX IF NOT EXISTS ix_backtest_strategy_symbol_time 
ON trade.backtest_results_by_strategy(strategy_id, symbol, signal_time DESC);

CREATE INDEX IF NOT EXISTS ix_backtest_strategy_outcome 
ON trade.backtest_results_by_strategy(strategy_id, outcome, final_score DESC);

-- ═══════════════════════════════════════════════════════════════════════════
-- 4. Справочник стратегий
-- ═══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS trade.strategy_registry (
    strategy_id     SMALLINT PRIMARY KEY,
    strategy_name   TEXT NOT NULL UNIQUE,
    description     TEXT,
    is_active       BOOLEAN NOT NULL DEFAULT true,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Вставляем 6 стратегий
INSERT INTO trade.strategy_registry (strategy_id, strategy_name, description) VALUES
    (1, 'indicator_ml', 'Индикаторы + ML предсказания, равные веса индикаторов'),
    (2, 'indicator_heuristic', 'Индикаторы + эвристические расчеты, равные веса индикаторов'),
    (3, 'indicator_consensus', 'Индикаторы + консенсус предсказаний'),
    (4, 'level_ml', 'Уровни + ML предсказания, текущая логика TP/SL'),
    (5, 'level_heuristic', 'Уровни + эвристические предсказания, текущая логика TP/SL'),
    (6, 'level_consensus', 'Уровни + консенсус (текущая default стратегия)')
ON CONFLICT (strategy_id) DO NOTHING;

-- ═══════════════════════════════════════════════════════════════════════════
-- 5. Обновление таблицы backtest_results - добавление strategy_id
-- ═══════════════════════════════════════════════════════════════════════════

-- Add columns only if table exists
DO $$
BEGIN
    -- Add strategy_id column
    ALTER TABLE trade.backtest_results 
    ADD COLUMN IF NOT EXISTS strategy_id SMALLINT NOT NULL DEFAULT 6;

    -- Add strategy_name column  
    ALTER TABLE trade.backtest_results 
    ADD COLUMN IF NOT EXISTS strategy_name TEXT NOT NULL DEFAULT 'level_consensus';
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE 'Table trade.backtest_results does not exist, skipping column additions';
END $$;

-- Индекс для группировки по стратегиям (только если таблица существует)
DO $$
BEGIN
    CREATE INDEX IF NOT EXISTS ix_backtest_results_strategy 
    ON trade.backtest_results(strategy_id, signal_time DESC);
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE 'Table trade.backtest_results does not exist, skipping index creation';
END $$;

-- ═══════════════════════════════════════════════════════════════════════════
-- 6. Comment для документации
-- ═══════════════════════════════════════════════════════════════════════════

COMMENT ON COLUMN trade.final_signals.strategy_id IS '1=indicator_ml, 2=indicator_heuristic, 3=indicator_consensus, 4=level_ml, 5=level_heuristic, 6=level_consensus';

-- trade.backtest_results is created at runtime by the backtester binary,
-- so add COMMENT only if the table already exists
DO $$
BEGIN
    COMMENT ON COLUMN trade.backtest_results.strategy_id IS '1=indicator_ml, 2=indicator_heuristic, 3=indicator_consensus, 4=level_ml, 5=level_heuristic, 6=level_consensus';
EXCEPTION
    WHEN undefined_table THEN
        RAISE NOTICE 'Table trade.backtest_results does not exist yet, skipping COMMENT';
END $$;
