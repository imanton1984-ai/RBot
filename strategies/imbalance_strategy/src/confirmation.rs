// strategies/imbalance_strategy/src/confirmation.rs
//
// Entry Confirmation & Trade Management — v4
//
// v4 CHANGES (pullback entry + smart exit):
//   1. NEW: Pullback entry mode — wait for price to pull back after imbalance,
//      then enter on continuation. Gives better entry price + tighter SL.
//   2. SL placed below pullback low (for longs) → natural structure-based stop
//   3. TP based on child ATR multiplied by config tp_atr_mult
//   4. Trailing stop: after 1+ candles in profit, trail SL to breakeven
//   5. Max hold configurable (default 3)
//
// PULLBACK ENTRY LOGIC:
//   After a bullish imbalance on parent TF:
//   1. Wait up to N child candles for a pullback (at least 1 red candle)
//   2. Pullback must be shallow (< pullback_pct of parent body)
//   3. Enter when a green candle closes above the pullback high
//   4. SL = pullback low - buffer
//   5. TP = entry + tp_atr_mult × child_ATR
//
// NO LOOK-AHEAD BIAS:
//   At entry decision, we only see candles that have already closed.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::imbalance::{
    Candle, Direction, ImbalanceCandle, ImbalanceConfig,
    SignalType, TradeDirection,
};

// ═════════════════════════════════════════════════════════════════════════════
// TRADE SIMULATION TYPES
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeOutcome {
    TpHit,
    SlHit,
    Expired,
}

impl std::fmt::Display for TradeOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TpHit => write!(f, "✅TP"),
            Self::SlHit => write!(f, "❌SL"),
            Self::Expired => write!(f, "⏰EXP"),
        }
    }
}

/// A fully simulated trade from an imbalance signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimTrade {
    pub symbol: String,
    pub parent_tf: i32,
    pub child_tf: i32,
    pub direction: Direction,
    pub signal_type: SignalType,
    pub trade_direction: TradeDirection,
    pub imbalance_time: DateTime<Utc>,
    pub entry_time: DateTime<Utc>,
    pub entry_price: f64,
    pub tp_price: f64,
    pub sl_price: f64,
    pub exit_price: f64,
    pub exit_time: DateTime<Utc>,
    pub outcome: TradeOutcome,
    pub hold_candles: usize,
    pub pnl_pct: f64,
    pub score: f64,
    pub parent_move_pct: f64,
    pub parent_body_ratio: f64,
    pub parent_volume_ratio: f64,
}

// ═════════════════════════════════════════════════════════════════════════════
// CONFIRMATION LOGIC
// ═════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum ConfirmationResult {
    Confirmed {
        entry_price: f64,
        entry_time: DateTime<Utc>,
        candle_idx: usize,
        /// SL price derived from pullback structure (None = use ATR-based SL)
        structure_sl: Option<f64>,
    },
    Rejected {
        reason: &'static str,
    },
}

/// Main entry point: check confirmation using the configured mode.
pub fn check_confirmation(
    child_candles: &[Candle],
    imbalance: &ImbalanceCandle,
    config: &ImbalanceConfig,
) -> ConfirmationResult {
    if config.use_pullback_entry {
        check_pullback_confirmation(child_candles, imbalance, config)
    } else {
        check_immediate_confirmation(child_candles, imbalance, config)
    }
}

/// PULLBACK ENTRY: Wait for pullback after imbalance, then enter on continuation.
///
/// This is the v4 entry mode — gives much better entry prices and structure-based SL.
///
/// For LONG after bullish imbalance:
///   1. Look for child candles that pull back (any close below parent close)
///   2. Track the lowest low during pullback
///   3. When a bullish child candle closes above previous candle's high → enter
///   4. SL = pullback low - small buffer
///   5. Pullback must not exceed max_pullback_pct of parent body
fn check_pullback_confirmation(
    child_candles: &[Candle],
    imbalance: &ImbalanceCandle,
    config: &ImbalanceConfig,
) -> ConfirmationResult {
    let first_idx = child_candles.partition_point(|c| c.time <= imbalance.time);

    if first_idx >= child_candles.len() {
        return ConfirmationResult::Rejected {
            reason: "no child candles after imbalance",
        };
    }

    // Temporal coverage check
    let child = &child_candles[first_idx];
    let max_gap = Duration::minutes(imbalance.parent_tf as i64 * 2);
    if child.time - imbalance.time > max_gap {
        return ConfirmationResult::Rejected {
            reason: "child TF data gap",
        };
    }

    let parent_body = (imbalance.close - imbalance.open).abs();
    let max_wait = config.max_entry_wait_candles;
    let last_idx = (first_idx + max_wait).min(child_candles.len().saturating_sub(1));

    // Compute child ATR for pullback depth check
    let lookback = 14.min(first_idx);
    let child_atr = if lookback > 0 {
        let sum: f64 = child_candles[(first_idx - lookback)..first_idx]
            .iter()
            .map(|c| c.high - c.low)
            .sum();
        (sum / lookback as f64).max(imbalance.close * 0.001)
    } else {
        (child.high - child.low).max(imbalance.close * 0.001)
    };

    match imbalance.trade_direction {
        TradeDirection::Long => {
            find_pullback_entry_long(
                child_candles, first_idx, last_idx, imbalance,
                parent_body, child_atr, config,
            )
        }
        TradeDirection::Short => {
            find_pullback_entry_short(
                child_candles, first_idx, last_idx, imbalance,
                parent_body, child_atr, config,
            )
        }
    }
}

/// Find pullback entry for LONG trades.
fn find_pullback_entry_long(
    child_candles: &[Candle],
    first_idx: usize,
    last_idx: usize,
    imbalance: &ImbalanceCandle,
    parent_body: f64,
    child_atr: f64,
    config: &ImbalanceConfig,
) -> ConfirmationResult {
    let mut pullback_low = f64::MAX;
    let mut saw_pullback = false;
    let min_pullback_depth = child_atr * config.min_pullback_atr;
    let reference_price = imbalance.close;

    for i in first_idx..=last_idx {
        if i >= child_candles.len() { break; }
        let c = &child_candles[i];

        // Track pullback low
        if c.low < pullback_low {
            pullback_low = c.low;
        }

        // Check if we've seen a pullback (price dropped from reference)
        let pullback_depth = reference_price - pullback_low;
        if pullback_depth >= min_pullback_depth {
            saw_pullback = true;
        }

        // Check if pullback is too deep (invalidates the setup)
        if parent_body > 1e-12 && pullback_depth > parent_body * config.continuation_pullback_pct {
            return ConfirmationResult::Rejected {
                reason: "pullback too deep for LONG continuation",
            };
        }

        // Also reject if price drops below parent candle open (full retracement)
        if c.close < imbalance.open {
            return ConfirmationResult::Rejected {
                reason: "price dropped below parent open — full retracement",
            };
        }

        // Entry condition: bullish candle after seeing some pullback, OR
        // first candle is strongly bullish (immediate continuation)
        let is_bullish = c.close > c.open;
        let child_range = c.high - c.low;
        let body_ratio = if child_range > 1e-12 { (c.close - c.open).abs() / child_range } else { 0.0 };

        if is_bullish && body_ratio >= config.min_confirm_body_ratio {
            // Option A: We saw a pullback and now resuming
            if saw_pullback {
                let sl_price = pullback_low - child_atr * 0.3;
                return ConfirmationResult::Confirmed {
                    entry_price: c.close,
                    entry_time: c.time,
                    candle_idx: i,
                    structure_sl: Some(sl_price),
                };
            }
            // Option B: Immediate strong continuation (first candle is bullish)
            // Only accept if it's the first candle and body is strong
            if i == first_idx && body_ratio >= 0.40 {
                return ConfirmationResult::Confirmed {
                    entry_price: c.close,
                    entry_time: c.time,
                    candle_idx: i,
                    structure_sl: None, // use ATR-based SL
                };
            }
        }
    }

    // Fallback: if we never saw a meaningful pullback, check if any candle
    // closed higher than the imbalance close (continuation without pullback)
    for i in first_idx..=last_idx {
        if i >= child_candles.len() { break; }
        let c = &child_candles[i];
        if c.close > reference_price && c.close > c.open {
            let child_range = c.high - c.low;
            let body_ratio = if child_range > 1e-12 { (c.close - c.open) / child_range } else { 0.0 };
            if body_ratio >= config.min_confirm_body_ratio {
                return ConfirmationResult::Confirmed {
                    entry_price: c.close,
                    entry_time: c.time,
                    candle_idx: i,
                    structure_sl: None,
                };
            }
        }
    }

    ConfirmationResult::Rejected {
        reason: "no pullback + continuation pattern found for LONG",
    }
}

/// Find pullback entry for SHORT trades.
fn find_pullback_entry_short(
    child_candles: &[Candle],
    first_idx: usize,
    last_idx: usize,
    imbalance: &ImbalanceCandle,
    parent_body: f64,
    child_atr: f64,
    config: &ImbalanceConfig,
) -> ConfirmationResult {
    let mut pullback_high = f64::MIN;
    let mut saw_pullback = false;
    let min_pullback_depth = child_atr * config.min_pullback_atr;
    let reference_price = imbalance.close;

    for i in first_idx..=last_idx {
        if i >= child_candles.len() { break; }
        let c = &child_candles[i];

        if c.high > pullback_high {
            pullback_high = c.high;
        }

        let pullback_depth = pullback_high - reference_price;
        if pullback_depth >= min_pullback_depth {
            saw_pullback = true;
        }

        if parent_body > 1e-12 && pullback_depth > parent_body * config.continuation_pullback_pct {
            return ConfirmationResult::Rejected {
                reason: "pullback too deep for SHORT continuation",
            };
        }

        if c.close > imbalance.open {
            return ConfirmationResult::Rejected {
                reason: "price rose above parent open — full retracement",
            };
        }

        let is_bearish = c.close < c.open;
        let child_range = c.high - c.low;
        let body_ratio = if child_range > 1e-12 { (c.open - c.close).abs() / child_range } else { 0.0 };

        if is_bearish && body_ratio >= config.min_confirm_body_ratio {
            if saw_pullback {
                let sl_price = pullback_high + child_atr * 0.3;
                return ConfirmationResult::Confirmed {
                    entry_price: c.close,
                    entry_time: c.time,
                    candle_idx: i,
                    structure_sl: Some(sl_price),
                };
            }
            if i == first_idx && body_ratio >= 0.40 {
                return ConfirmationResult::Confirmed {
                    entry_price: c.close,
                    entry_time: c.time,
                    candle_idx: i,
                    structure_sl: None,
                };
            }
        }
    }

    for i in first_idx..=last_idx {
        if i >= child_candles.len() { break; }
        let c = &child_candles[i];
        if c.close < reference_price && c.close < c.open {
            let child_range = c.high - c.low;
            let body_ratio = if child_range > 1e-12 { (c.open - c.close) / child_range } else { 0.0 };
            if body_ratio >= config.min_confirm_body_ratio {
                return ConfirmationResult::Confirmed {
                    entry_price: c.close,
                    entry_time: c.time,
                    candle_idx: i,
                    structure_sl: None,
                };
            }
        }
    }

    ConfirmationResult::Rejected {
        reason: "no pullback + continuation pattern found for SHORT",
    }
}

/// IMMEDIATE ENTRY (legacy mode): Enter on first confirming child candle.
fn check_immediate_confirmation(
    child_candles: &[Candle],
    imbalance: &ImbalanceCandle,
    config: &ImbalanceConfig,
) -> ConfirmationResult {
    let first_child_idx = child_candles.partition_point(|c| c.time <= imbalance.time);

    if first_child_idx >= child_candles.len() {
        return ConfirmationResult::Rejected {
            reason: "no child candles after imbalance",
        };
    }

    let child = &child_candles[first_child_idx];
    let max_gap = Duration::minutes(imbalance.parent_tf as i64 * 2);
    if child.time - imbalance.time > max_gap {
        return ConfirmationResult::Rejected {
            reason: "child TF data gap",
        };
    }

    let child_body = child.close - child.open;
    let child_range = child.high - child.low;

    // Reject doji
    if child_range < 1e-12 || child_body.abs() / child_range < config.min_confirm_body_ratio {
        return ConfirmationResult::Rejected {
            reason: "child candle is doji/indecision",
        };
    }

    match imbalance.direction {
        Direction::Bullish => {
            if child.close >= child.open {
                return ConfirmationResult::Confirmed {
                    entry_price: child.close,
                    entry_time: child.time,
                    candle_idx: first_child_idx,
                    structure_sl: None,
                };
            }
            let pullback = (child.open - child.close).max(0.0);
            let parent_body = (imbalance.close - imbalance.open).abs();
            if parent_body > 1e-12 && pullback < parent_body * config.continuation_pullback_pct {
                return ConfirmationResult::Confirmed {
                    entry_price: child.close,
                    entry_time: child.time,
                    candle_idx: first_child_idx,
                    structure_sl: None,
                };
            }
            ConfirmationResult::Rejected {
                reason: "child rejects bullish continuation",
            }
        }
        Direction::Bearish => {
            if child.close <= child.open {
                return ConfirmationResult::Confirmed {
                    entry_price: child.close,
                    entry_time: child.time,
                    candle_idx: first_child_idx,
                    structure_sl: None,
                };
            }
            let pullback = (child.close - child.open).max(0.0);
            let parent_body = (imbalance.close - imbalance.open).abs();
            if parent_body > 1e-12 && pullback < parent_body * config.continuation_pullback_pct {
                return ConfirmationResult::Confirmed {
                    entry_price: child.close,
                    entry_time: child.time,
                    candle_idx: first_child_idx,
                    structure_sl: None,
                };
            }
            ConfirmationResult::Rejected {
                reason: "child rejects bearish continuation",
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// TRADE SIMULATION — v4 with structure-based SL + trailing
// ═════════════════════════════════════════════════════════════════════════════

/// Simulate a trade after entry confirmation.
///
/// v4: Uses structure-based SL if available (from pullback logic),
///     otherwise falls back to child ATR-based SL.
///     Trailing stop: after candle 1, if in profit, move SL to entry.
pub fn simulate_trade(
    child_candles: &[Candle],
    entry_idx: usize,
    imbalance: &ImbalanceCandle,
    entry_price: f64,
    entry_time: DateTime<Utc>,
    structure_sl: Option<f64>,
    config: &ImbalanceConfig,
) -> Option<SimTrade> {
    let max_hold = config.max_hold_candles;

    // ── Compute child ATR ──
    let lookback = 14.min(entry_idx);
    let child_atr = if lookback > 0 {
        let sum: f64 = child_candles[(entry_idx - lookback)..entry_idx]
            .iter()
            .map(|c| c.high - c.low)
            .sum();
        (sum / lookback as f64).max(entry_price * 0.001)
    } else {
        let c = &child_candles[entry_idx];
        (c.high - c.low).max(entry_price * 0.001)
    };

    // ── TP based on child ATR, with percentage caps ──
    let tp_dist_raw = child_atr * config.tp_atr_mult;
    let sl_dist_raw = child_atr * config.sl_atr_mult;

    // Apply percentage caps to prevent catastrophic losses / unrealistic targets
    let max_tp_abs = entry_price * config.max_tp_pct / 100.0;
    let max_sl_abs = entry_price * config.max_sl_pct / 100.0;
    let min_tp_abs = entry_price * config.min_tp_pct / 100.0;
    let min_sl_abs = entry_price * config.min_sl_pct / 100.0;

    let tp_dist = tp_dist_raw.clamp(min_tp_abs, max_tp_abs);

    let (tp_price, initial_sl) = match imbalance.trade_direction {
        TradeDirection::Long => {
            let tp = entry_price + tp_dist;
            let sl_from_structure = structure_sl.unwrap_or(entry_price - sl_dist_raw);
            let sl_dist_actual = (entry_price - sl_from_structure).abs().clamp(min_sl_abs, max_sl_abs);
            let sl = entry_price - sl_dist_actual;
            (tp, sl)
        }
        TradeDirection::Short => {
            let tp = entry_price - tp_dist;
            let sl_from_structure = structure_sl.unwrap_or(entry_price + sl_dist_raw);
            let sl_dist_actual = (sl_from_structure - entry_price).abs().clamp(min_sl_abs, max_sl_abs);
            let sl = entry_price + sl_dist_actual;
            (tp, sl)
        }
    };

    let last_possible = (entry_idx + max_hold).min(child_candles.len().saturating_sub(1));

    if entry_idx + 1 > last_possible {
        return None;
    }

    for bar in (entry_idx + 1)..=last_possible {
        let c = &child_candles[bar];
        let hold = bar - entry_idx;

        // Profit exit: if configured, exit at close after N candles if in profit
        if config.profit_exit_after_candles > 0 && hold >= config.profit_exit_after_candles {
            let prev = &child_candles[bar - 1];
            let in_profit = match imbalance.trade_direction {
                TradeDirection::Long => prev.close > entry_price,
                TradeDirection::Short => prev.close < entry_price,
            };
            if in_profit {
                let exit = prev.close;
                let pnl = match imbalance.trade_direction {
                    TradeDirection::Long => (exit - entry_price) / entry_price * 100.0,
                    TradeDirection::Short => (entry_price - exit) / entry_price * 100.0,
                };
                return Some(make_trade(
                    imbalance, entry_price, entry_time, exit, prev.time,
                    TradeOutcome::TpHit, hold - 1, pnl, tp_price, initial_sl,
                ));
            }
        }

        // Trailing stop: after N candles, if in profit, trail SL to entry (breakeven)
        let current_sl = if hold >= config.trailing_start_candle {
            let prev = &child_candles[bar - 1];
            let in_profit = match imbalance.trade_direction {
                TradeDirection::Long => prev.close > entry_price,
                TradeDirection::Short => prev.close < entry_price,
            };
            if in_profit {
                match imbalance.trade_direction {
                    TradeDirection::Long => initial_sl.max(entry_price),
                    TradeDirection::Short => initial_sl.min(entry_price),
                }
            } else {
                initial_sl
            }
        } else {
            initial_sl
        };

        match imbalance.trade_direction {
            TradeDirection::Long => {
                // Check SL first (conservative — assumes worst case)
                if c.low <= current_sl {
                    let exit = current_sl;
                    let pnl = (exit - entry_price) / entry_price * 100.0;
                    let outcome = if (current_sl - entry_price).abs() < entry_price * 0.001 {
                        TradeOutcome::Expired // breakeven exit
                    } else if pnl < 0.0 {
                        TradeOutcome::SlHit
                    } else {
                        TradeOutcome::Expired
                    };
                    return Some(make_trade(
                        imbalance, entry_price, entry_time, exit, c.time,
                        outcome, hold, pnl, tp_price, initial_sl,
                    ));
                }
                // Check TP
                if c.high >= tp_price {
                    let pnl = (tp_price - entry_price) / entry_price * 100.0;
                    return Some(make_trade(
                        imbalance, entry_price, entry_time, tp_price, c.time,
                        TradeOutcome::TpHit, hold, pnl, tp_price, initial_sl,
                    ));
                }
            }
            TradeDirection::Short => {
                if c.high >= current_sl {
                    let exit = current_sl;
                    let pnl = (entry_price - exit) / entry_price * 100.0;
                    let outcome = if (current_sl - entry_price).abs() < entry_price * 0.001 {
                        TradeOutcome::Expired
                    } else if pnl < 0.0 {
                        TradeOutcome::SlHit
                    } else {
                        TradeOutcome::Expired
                    };
                    return Some(make_trade(
                        imbalance, entry_price, entry_time, exit, c.time,
                        outcome, hold, pnl, tp_price, initial_sl,
                    ));
                }
                if c.low <= tp_price {
                    let pnl = (entry_price - tp_price) / entry_price * 100.0;
                    return Some(make_trade(
                        imbalance, entry_price, entry_time, tp_price, c.time,
                        TradeOutcome::TpHit, hold, pnl, tp_price, initial_sl,
                    ));
                }
            }
        }

        // Max hold reached — exit at close
        if hold >= max_hold {
            let exit = c.close;
            let pnl = match imbalance.trade_direction {
                TradeDirection::Long => (exit - entry_price) / entry_price * 100.0,
                TradeDirection::Short => (entry_price - exit) / entry_price * 100.0,
            };
            return Some(make_trade(
                imbalance, entry_price, entry_time, exit, c.time,
                TradeOutcome::Expired, hold, pnl, tp_price, initial_sl,
            ));
        }
    }

    None
}

fn make_trade(
    imbalance: &ImbalanceCandle,
    entry_price: f64,
    entry_time: DateTime<Utc>,
    exit_price: f64,
    exit_time: DateTime<Utc>,
    outcome: TradeOutcome,
    hold_candles: usize,
    pnl_pct: f64,
    tp_price: f64,
    sl_price: f64,
) -> SimTrade {
    SimTrade {
        symbol: imbalance.symbol.clone(),
        parent_tf: imbalance.parent_tf,
        child_tf: imbalance.child_tf,
        direction: imbalance.direction,
        signal_type: imbalance.signal_type,
        trade_direction: imbalance.trade_direction,
        imbalance_time: imbalance.time,
        entry_time,
        entry_price,
        tp_price,
        sl_price,
        exit_price,
        exit_time,
        outcome,
        hold_candles,
        pnl_pct,
        score: imbalance.score,
        parent_move_pct: imbalance.body_move_pct,
        parent_body_ratio: imbalance.body_ratio,
        parent_volume_ratio: imbalance.volume_ratio,
    }
}

/// Process a full signal: detect imbalance → confirm → simulate trade.
pub fn process_symbol(
    parent_candles: &[Candle],
    child_candles: &[Candle],
    tf_pair: &crate::imbalance::TfPair,
    config: &ImbalanceConfig,
) -> Vec<SimTrade> {
    let mut trades = Vec::new();
    let imbalances = crate::imbalance::scan_for_imbalances(parent_candles, tf_pair, config);
    let mut next_allowed_time: Option<DateTime<Utc>> = None;

    for imb in &imbalances {
        if let Some(min_time) = next_allowed_time {
            if imb.time < min_time {
                continue;
            }
        }

        match check_confirmation(child_candles, imb, config) {
            ConfirmationResult::Confirmed { entry_price, entry_time, candle_idx, structure_sl } => {
                if let Some(trade) = simulate_trade(
                    child_candles, candle_idx, imb, entry_price, entry_time,
                    structure_sl, config,
                ) {
                    next_allowed_time = Some(trade.exit_time);
                    trades.push(trade);
                }
            }
            ConfirmationResult::Rejected { .. } => {}
        }
    }

    trades
}

// ═════════════════════════════════════════════════════════════════════════════
// TESTS
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imbalance::{Direction, ImbalanceCandle, SignalType, TradeDirection};
    use chrono::TimeZone;

    fn make_candle_at(t: DateTime<Utc>, o: f64, h: f64, l: f64, c: f64, v: f64) -> Candle {
        Candle {
            time: t, symbol: "TESTUSDT".to_string(),
            open: o, high: h, low: l, close: c, volume: v,
            rsi: 50.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            adx: 25.0, atr: (h - l) * 0.5,
            bb_upper: c * 1.02, bb_lower: c * 0.98,
            ema_20: c, ema_50: c,
            volume_spike: 1.0, trend: 0.0, trend_short: 0.0,
            supertrend_dir: 1.0, macd_hist: 0.0, cmf: 0.0,
            mfi: 50.0, obv: 0.0,
        }
    }

    fn make_imbalance(time: DateTime<Utc>) -> ImbalanceCandle {
        ImbalanceCandle {
            symbol: "TESTUSDT".to_string(),
            time,
            parent_tf: 240, child_tf: 60,
            direction: Direction::Bullish,
            open: 100.0, high: 116.0, low: 99.5, close: 115.0, volume: 2000.0,
            body_ratio: 0.87, upper_wick_ratio: 0.06, lower_wick_ratio: 0.03,
            body_move_pct: 15.0, range_pct: 16.5, volume_ratio: 2.0,
            rsi: 65.0, stoch_k: 70.0, adx: 30.0,
            trend: 1.0, supertrend_dir: 1.0, macd_hist: 0.5, cmf: 0.3, atr: 2.0,
            signal_type: SignalType::Continuation,
            trade_direction: TradeDirection::Long,
            score: 0.85, tp_distance: 5.0, sl_distance: 4.0,
        }
    }

    #[test]
    fn test_pullback_entry_long() {
        let config = ImbalanceConfig::default();
        let t0 = Utc.with_ymd_and_hms(2025, 1, 1, 4, 0, 0).unwrap();

        let imb = make_imbalance(t0);

        // Build child candles:
        // Before imbalance
        let mut candles = Vec::new();
        for i in 0..15 {
            let t = t0 - Duration::hours(15 - i);
            candles.push(make_candle_at(t, 100.0, 102.0, 98.0, 101.0, 1000.0));
        }
        // After imbalance: pullback then continuation
        // Candle 1: small pullback (bearish)
        candles.push(make_candle_at(t0 + Duration::hours(1), 115.0, 115.5, 113.0, 113.5, 1500.0));
        // Candle 2: bullish continuation (entry point)
        candles.push(make_candle_at(t0 + Duration::hours(2), 113.5, 117.0, 113.0, 116.5, 1800.0));
        // Candle 3: continue up
        candles.push(make_candle_at(t0 + Duration::hours(3), 116.5, 120.0, 116.0, 119.0, 1200.0));

        let result = check_confirmation(&candles, &imb, &config);
        match result {
            ConfirmationResult::Confirmed { entry_price, structure_sl, .. } => {
                assert!(entry_price > 0.0, "Entry price should be positive");
                assert!(structure_sl.is_some(), "Pullback entry should have structure SL");
                if let Some(sl) = structure_sl {
                    assert!(sl < entry_price, "SL should be below entry for LONG");
                }
            }
            ConfirmationResult::Rejected { reason } => {
                panic!("Should confirm pullback entry, but rejected: {}", reason);
            }
        }
    }
}
