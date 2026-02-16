-- 051_final_signals_symbol.sql
-- Add symbol TEXT column to trade.final_signals for readability
-- (previously only symbol_id was stored, requiring a JOIN to see the pair name)

ALTER TABLE trade.final_signals ADD COLUMN IF NOT EXISTS symbol TEXT;

-- Populate existing rows
UPDATE trade.final_signals SET symbol = p.symbol
FROM market.pairs p
WHERE trade.final_signals.symbol_id = p.symbol_id
  AND (trade.final_signals.symbol IS NULL OR trade.final_signals.symbol = '');

-- Index for fast lookup by symbol
CREATE INDEX IF NOT EXISTS ix_final_signals_symbol ON trade.final_signals(symbol, time DESC);
