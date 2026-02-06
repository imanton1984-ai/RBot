-- 045_add_symbol_column.sql
-- Add symbol column to all relevant tables for improved readability

-- Add symbol column to all candle tables
ALTER TABLE market.candles_1m ADD COLUMN IF NOT EXISTS symbol TEXT;
ALTER TABLE market.candles_5m ADD COLUMN IF NOT EXISTS symbol TEXT;
ALTER TABLE market.candles_15m ADD COLUMN IF NOT EXISTS symbol TEXT;
ALTER TABLE market.candles_1h ADD COLUMN IF NOT EXISTS symbol TEXT;
ALTER TABLE market.candles_4h ADD COLUMN IF NOT EXISTS symbol TEXT;
ALTER TABLE market.candles_1d ADD COLUMN IF NOT EXISTS symbol TEXT;
-- Also add to the live candles table
ALTER TABLE market.candles_live ADD COLUMN IF NOT EXISTS symbol TEXT;
ALTER TABLE market.indicators_wide ADD COLUMN IF NOT EXISTS symbol TEXT;

-- Add symbol column to all indicator tables


-- Add symbol column to raw_signals table
ALTER TABLE market.raw_signals ADD COLUMN IF NOT EXISTS symbol TEXT;

-- Update the sync functions to populate the symbol column automatically
CREATE OR REPLACE FUNCTION market.sync_candle_time()
RETURNS TRIGGER AS $$
DECLARE
    sym TEXT;
BEGIN
    -- Основное время
    NEW.time := to_timestamp(NEW.time_ms / 1000.0);

    -- Автоматически заполняем символ, если его нет
    IF NEW.symbol IS NULL OR NEW.symbol = '' THEN
        SELECT symbol INTO sym FROM market.pairs WHERE symbol_id = NEW.symbol_id;
        IF FOUND THEN
            NEW.symbol := sym;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION market.sync_indicator_time()
RETURNS TRIGGER AS $$
DECLARE
    sym TEXT;
BEGIN
    -- Основное время
    NEW.time := to_timestamp(NEW.time_ms / 1000.0);

    -- Логика для updated_at
    IF NEW.updated_at_ms IS NULL THEN
        NEW.updated_at := now();
    ELSE
        NEW.updated_at := to_timestamp(NEW.updated_at_ms / 1000.0);
    END IF;

    -- Автоматически заполняем символ, если его нет
    IF NEW.symbol IS NULL OR NEW.symbol = '' THEN
        SELECT symbol INTO sym FROM market.pairs WHERE symbol_id = NEW.symbol_id;
        IF FOUND THEN
            NEW.symbol := sym;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION market.sync_raw_signals_time()
RETURNS TRIGGER AS $$
DECLARE
    sym TEXT;
BEGIN
    -- Основное время
    NEW.time := to_timestamp(NEW.time_ms / 1000.0);

    -- Логика для created_at
    IF NEW.created_at_ms IS NULL THEN
        NEW.created_at := now();
    ELSE
        NEW.created_at := to_timestamp(NEW.created_at_ms / 1000.0);
    END IF;

    -- Автоматически заполняем символ, если его нет
    IF NEW.symbol IS NULL OR NEW.symbol = '' THEN
        SELECT symbol INTO sym FROM market.pairs WHERE symbol_id = NEW.symbol_id;
        IF FOUND THEN
            NEW.symbol := sym;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create a sync function for live candles
CREATE OR REPLACE FUNCTION market.sync_candles_live_time()
RETURNS TRIGGER AS $$
DECLARE
    sym TEXT;
BEGIN
    -- Automatically populate symbol if it's not provided
    IF NEW.symbol IS NULL OR NEW.symbol = '' THEN
        SELECT symbol INTO sym FROM market.pairs WHERE symbol = NEW.symbol; -- NEW.symbol should already be provided
        IF FOUND THEN
            NEW.symbol := sym;
        END IF;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Populate the symbol column with values from the pairs table
-- For candles tables
UPDATE market.candles_1m SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_1m.symbol_id = p.symbol_id AND (market.candles_1m.symbol IS NULL OR market.candles_1m.symbol = '');

UPDATE market.candles_5m SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_5m.symbol_id = p.symbol_id AND (market.candles_5m.symbol IS NULL OR market.candles_5m.symbol = '');

UPDATE market.candles_15m SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_15m.symbol_id = p.symbol_id AND (market.candles_15m.symbol IS NULL OR market.candles_15m.symbol = '');

UPDATE market.candles_1h SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_1h.symbol_id = p.symbol_id AND (market.candles_1h.symbol IS NULL OR market.candles_1h.symbol = '');

UPDATE market.candles_4h SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_4h.symbol_id = p.symbol_id AND (market.candles_4h.symbol IS NULL OR market.candles_4h.symbol = '');

UPDATE market.candles_1d SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_1d.symbol_id = p.symbol_id AND (market.candles_1d.symbol IS NULL OR market.candles_1d.symbol = '');

-- For live candles table (symbol is already part of the data)
UPDATE market.candles_live SET symbol = p.symbol
FROM market.pairs p
WHERE market.candles_live.symbol = p.symbol AND (market.candles_live.symbol IS NULL OR market.candles_live.symbol = '');

-- For indicators tables
UPDATE market.indicators_wide SET symbol = p.symbol
FROM market.pairs p
WHERE market.indicators_wide.symbol_id = p.symbol_id AND (market.indicators_wide.symbol IS NULL OR market.indicators_wide.symbol = '');


-- For raw_signals table
UPDATE market.raw_signals SET symbol = p.symbol
FROM market.pairs p
WHERE market.raw_signals.symbol_id = p.symbol_id AND (market.raw_signals.symbol IS NULL OR market.raw_signals.symbol = '');

-- Create indexes for the new symbol column to improve query performance
CREATE INDEX IF NOT EXISTS ix_candles_1m_symbol ON market.candles_1m (symbol);
CREATE INDEX IF NOT EXISTS ix_candles_5m_symbol ON market.candles_5m (symbol);
CREATE INDEX IF NOT EXISTS ix_candles_15m_symbol ON market.candles_15m (symbol);
CREATE INDEX IF NOT EXISTS ix_candles_1h_symbol ON market.candles_1h (symbol);
CREATE INDEX IF NOT EXISTS ix_candles_4h_symbol ON market.candles_4h (symbol);
CREATE INDEX IF NOT EXISTS ix_candles_1d_symbol ON market.candles_1d (symbol);
CREATE INDEX IF NOT EXISTS ix_candles_live_symbol ON market.candles_live (symbol);

CREATE INDEX IF NOT EXISTS ix_indicators_wide_symbol ON market.indicators_wide (symbol);

CREATE INDEX IF NOT EXISTS ix_raw_signals_symbol ON market.raw_signals (symbol);