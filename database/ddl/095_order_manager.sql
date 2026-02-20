-- 095_order_manager.sql
-- DDL for Order Manager module
-- Extends trade.positions and trade.orders with columns required by
-- signal_scanner → order_executor → position_tracker pipeline.
-- Also creates trade.position_history for closed-trade audit trail.

-- ═══════════════════════════════════════════════════════════
-- 1. EXTEND trade.positions
-- ═══════════════════════════════════════════════════════════

-- Human-readable symbol (denormalized for speed)
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS symbol TEXT;
-- Timeframe of the signal that triggered this position
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS tf_minutes SMALLINT;
-- Candle-time of the signal
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS signal_time TIMESTAMPTZ;
-- Countdown: how many bars left until force-close
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS candles_left SMALLINT;
-- Max bars to hold (from strategy config, default 25)
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS max_hold_bars SMALLINT DEFAULT 25;
-- Stop-loss price
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS sl_price DOUBLE PRECISION;
-- Take-profit price
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS tp_price DOUBLE PRECISION;
-- Actual exit price (filled on close)
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS exit_price DOUBLE PRECISION;
-- Why the position was closed: tp_hit | sl_hit | max_bars | risk_manager | manual
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS close_reason TEXT;
-- ML combined score of the triggering signal
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS combined_score REAL;
-- P(super) from the model
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS p_super REAL;
-- Binance exchange order IDs for the 3 legs (entry/SL/TP)
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS entry_order_id_exchange TEXT;
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS sl_order_id_exchange TEXT;
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS tp_order_id_exchange TEXT;
-- Total fees paid (entry + exit)
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS fees_total DOUBLE PRECISION DEFAULT 0;
-- Client-side order ID for dedup
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS client_order_id TEXT;
-- Realized PnL as percentage
ALTER TABLE trade.positions ADD COLUMN IF NOT EXISTS realized_pnl_pct DOUBLE PRECISION;

-- ═══════════════════════════════════════════════════════════
-- 2. EXTEND trade.orders
-- ═══════════════════════════════════════════════════════════

-- Human-readable symbol
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS symbol TEXT;
-- Timeframe context
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS tf_minutes SMALLINT;
-- Link to managed position
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS position_id BIGINT;
-- Filled price (actual execution price)
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS filled_price DOUBLE PRECISION;
-- Filled quantity
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS filled_qty DOUBLE PRECISION;
-- Commission paid
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS commission DOUBLE PRECISION DEFAULT 0;
-- Commission asset (e.g. "USDT")
ALTER TABLE trade.orders ADD COLUMN IF NOT EXISTS commission_asset TEXT DEFAULT 'USDT';

-- ═══════════════════════════════════════════════════════════
-- 3. POSITION HISTORY (audit trail for closed trades)
-- ═══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS trade.position_history (
    id              BIGSERIAL PRIMARY KEY,
    position_id     BIGINT NOT NULL,
    symbol          TEXT NOT NULL,
    symbol_id       BIGINT NOT NULL,
    tf_minutes      SMALLINT NOT NULL,
    side            SMALLINT NOT NULL,       -- 1=LONG, -1=SHORT
    entry_price     DOUBLE PRECISION NOT NULL,
    exit_price      DOUBLE PRECISION,
    qty             DOUBLE PRECISION NOT NULL,
    leverage        REAL NOT NULL DEFAULT 1,
    sl_price        DOUBLE PRECISION,
    tp_price        DOUBLE PRECISION,
    realized_pnl    DOUBLE PRECISION,
    realized_pnl_pct DOUBLE PRECISION,
    fees_total      DOUBLE PRECISION DEFAULT 0,
    combined_score  REAL,                    -- ML signal quality at entry
    p_super         REAL,
    close_reason    TEXT,                    -- tp_hit | sl_hit | max_bars | risk_manager | manual
    opened_at       TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    duration_bars   SMALLINT,               -- how many bars the position lived
    max_hold_bars   SMALLINT,
    meta            JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- ═══════════════════════════════════════════════════════════
-- 4. INDEXES
-- ═══════════════════════════════════════════════════════════

-- Fast lookup of active positions by status
CREATE INDEX IF NOT EXISTS ix_positions_status
    ON trade.positions (status) WHERE status = 1;

-- Fast lookup by symbol + status
CREATE INDEX IF NOT EXISTS ix_positions_symbol_status
    ON trade.positions (symbol_id, status);

-- Position history indexes
CREATE INDEX IF NOT EXISTS ix_position_history_closed_at
    ON trade.position_history (closed_at DESC);

CREATE INDEX IF NOT EXISTS ix_position_history_symbol
    ON trade.position_history (symbol, closed_at DESC);

CREATE INDEX IF NOT EXISTS ix_position_history_tf
    ON trade.position_history (tf_minutes, closed_at DESC);

CREATE INDEX IF NOT EXISTS ix_position_history_pnl
    ON trade.position_history (realized_pnl_pct DESC)
    WHERE realized_pnl_pct IS NOT NULL;

-- ═══════════════════════════════════════════════════════════
-- 5. COMMENTS
-- ═══════════════════════════════════════════════════════════

COMMENT ON TABLE trade.position_history IS
    'Audit trail for all closed positions managed by Order Manager. '
    'Each row is a frozen snapshot created when a position is closed.';

COMMENT ON COLUMN trade.positions.candles_left IS
    'Countdown of bars until force-close. Starts at max_hold_bars, '
    'decremented every new candle of the position timeframe.';

COMMENT ON COLUMN trade.positions.close_reason IS
    'Reason the position was closed: tp_hit, sl_hit, max_bars, risk_manager, manual';
