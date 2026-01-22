-- 050_trade_tables.sql

-- 1. Универсальная функция синхронизации (если еще не создана в прошлых файлах)
CREATE OR REPLACE FUNCTION trade.sync_time_universal()
RETURNS TRIGGER AS $$
BEGIN
    -- Конвертируем основное время
    IF NEW.time_ms IS NOT NULL THEN
        NEW.time := to_timestamp(NEW.time_ms / 1000.0);
    END IF;
    
    -- Конвертируем создано/обновлено
    -- Для created_at
    IF TG_TABLE_NAME IN ('final_signals', 'position_events', 'ml_train_examples') THEN
        IF NEW.created_at_ms IS NULL THEN
            NEW.created_at := now();
        ELSE
            NEW.created_at := to_timestamp(NEW.created_at_ms / 1000.0);
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 2. Таблица final_signals
CREATE TABLE IF NOT EXISTS trade.final_signals (
    time_ms   BIGINT NOT NULL,
    time      TIMESTAMPTZ NOT NULL, -- Обычная

    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    tf_minutes SMALLINT NOT NULL,

    side SMALLINT NOT NULL,
    final_score REAL NOT NULL,
    ml_score    REAL,
    heur_score  REAL,
    model_version TEXT,
    strategy_id  SMALLINT NOT NULL DEFAULT 0,
    entry_price REAL,
    sl_price    REAL,
    tp1_price   REAL, tp2_price REAL, tp3_price REAL,
    reason JSONB,

    signal_id UUID NOT NULL DEFAULT gen_random_uuid(),
    created_at_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- ВАЖНО: 'time' должен быть в PRIMARY KEY и во всех UNIQUE индексах
    PRIMARY KEY(symbol_id, tf_minutes, time),
    UNIQUE(signal_id, time) 
);

-- 3. Таблицы позиций и ордеров (тут нет GENERATED, оставляем как есть)
CREATE TABLE IF NOT EXISTS trade.orders (
    id BIGSERIAL PRIMARY KEY,
    order_id_exchange TEXT,
    client_order_id   TEXT,
    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    side SMALLINT NOT NULL,
    order_type SMALLINT NOT NULL,
    reduce_only BOOLEAN NOT NULL DEFAULT FALSE,
    qty   DOUBLE PRECISION NOT NULL,
    price DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,
    status SMALLINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS trade.positions (
    id BIGSERIAL PRIMARY KEY,
    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    side SMALLINT NOT NULL,
    qty DOUBLE PRECISION NOT NULL,
    entry_price DOUBLE PRECISION NOT NULL,
    leverage REAL NOT NULL DEFAULT 1,
    margin_type SMALLINT NOT NULL DEFAULT 1,
    unrealized_pnl DOUBLE PRECISION,
    realized_pnl   DOUBLE PRECISION,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at TIMESTAMPTZ,
    status SMALLINT NOT NULL,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- 4. Таблица position_events
CREATE TABLE IF NOT EXISTS trade.position_events (
    time_ms BIGINT NOT NULL,
    time    TIMESTAMPTZ NOT NULL,
    position_id BIGINT NOT NULL REFERENCES trade.positions(id) ON DELETE CASCADE,
    event_type SMALLINT NOT NULL,
    price DOUBLE PRECISION,
    qty   DOUBLE PRECISION,
    start_final_score REAL,
    current_final_score REAL,
    start_features JSONB,
    current_features JSONB,
    note TEXT,
    meta JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 5. Таблица ml_train_examples
CREATE TABLE IF NOT EXISTS trade.ml_train_examples (
    time_ms BIGINT NOT NULL,
    time    TIMESTAMPTZ NOT NULL,
    symbol_id BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    tf_minutes SMALLINT NOT NULL,
    features_version TEXT NOT NULL DEFAULT 'v1',
    features REAL[],
    features_json JSONB,
    label SMALLINT,
    outcome JSONB,
    created_at_ms BIGINT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(symbol_id, tf_minutes, time, features_version)
);

-- 6. Вешаем триггеры
CREATE TRIGGER trg_sync_final_signals BEFORE INSERT OR UPDATE ON trade.final_signals FOR EACH ROW EXECUTE FUNCTION trade.sync_time_universal();
CREATE TRIGGER trg_sync_position_events BEFORE INSERT OR UPDATE ON trade.position_events FOR EACH ROW EXECUTE FUNCTION trade.sync_time_universal();
CREATE TRIGGER trg_sync_ml_examples BEFORE INSERT OR UPDATE ON trade.ml_train_examples FOR EACH ROW EXECUTE FUNCTION trade.sync_time_universal();

-- 7. Гипертаблицы
SELECT create_hypertable('trade.final_signals','time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '30 days');
SELECT create_hypertable('trade.position_events','time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '30 days');
SELECT create_hypertable('trade.ml_train_examples','time', if_not_exists=>TRUE, chunk_time_interval=>INTERVAL '30 days');

-- 8. Индексы
CREATE INDEX IF NOT EXISTS ix_final_signals_time_desc ON trade.final_signals(time DESC);
CREATE INDEX IF NOT EXISTS ix_position_events_position_time ON trade.position_events(position_id, time DESC);
CREATE INDEX IF NOT EXISTS ix_ml_examples_time_desc ON trade.ml_train_examples(time DESC);