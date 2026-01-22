-- 010_market_core.sql

-- Пары (universe результат)
CREATE TABLE IF NOT EXISTS market.pairs (
    symbol_id       BIGSERIAL PRIMARY KEY,
    symbol          TEXT NOT NULL UNIQUE,        -- "BTCUSDT"
    base_asset      TEXT NOT NULL,               -- "BTC"
    quote_asset     TEXT NOT NULL,               -- "USDT"
    is_active       BOOLEAN NOT NULL DEFAULT TRUE,

    -- universe/meta
    futures_eligible    BOOLEAN NOT NULL DEFAULT TRUE,
    perpetual_only      BOOLEAN NOT NULL DEFAULT TRUE,
    volume_24h_usdt     DOUBLE PRECISION,
    last_price          DOUBLE PRECISION,
    last_refreshed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),

    -- manual overrides
    manual_allow    BOOLEAN NOT NULL DEFAULT FALSE,
    manual_deny     BOOLEAN NOT NULL DEFAULT FALSE,

    meta            JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS ix_pairs_active ON market.pairs(is_active);
CREATE INDEX IF NOT EXISTS ix_pairs_volume ON market.pairs(volume_24h_usdt DESC);


-- Контрольная точка сбора (последняя закрытая свеча, удобно для gap-fill)
CREATE TABLE IF NOT EXISTS market.collected_candles (
    symbol_id    BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    tf_minutes   SMALLINT NOT NULL,           -- 1,5,15,60,240,1440
    last_closed_time TIMESTAMPTZ,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(symbol_id, tf_minutes)
);


-- “Живая” незакрытая свеча (persist_live_candle=true)
-- 1 строка на (symbol_id, tf)
CREATE TABLE IF NOT EXISTS market.candles_live (
    symbol_id    BIGINT NOT NULL REFERENCES market.pairs(symbol_id) ON DELETE CASCADE,
    tf_minutes   SMALLINT NOT NULL,

    open_time    TIMESTAMPTZ NOT NULL,        -- start time of current candle
    open         DOUBLE PRECISION NOT NULL,
    high         DOUBLE PRECISION NOT NULL,
    low          DOUBLE PRECISION NOT NULL,
    close        DOUBLE PRECISION NOT NULL,
    volume       DOUBLE PRECISION NOT NULL,

    last_event_time TIMESTAMPTZ NOT NULL,     -- время последнего WS update
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY(symbol_id, tf_minutes)
);

CREATE INDEX IF NOT EXISTS ix_candles_live_updated ON market.candles_live(updated_at DESC);
