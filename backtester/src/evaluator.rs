// backtester/src/evaluator.rs
//
// Core evaluation logic: for each signal, walk forward through future candles
// and determine trade outcome using dynamic take-profit with partial closes.
//
// PARTIAL CLOSE STRATEGY (when TP1/TP2/TP3 all present):
//   - TP1 hit → close 50%, trail SL to just below TP1
//   - TP2 hit → close 30%, trail SL to just below TP2
//   - TP3 hit → close remaining 20%
//   - If SL (original or trailing) hit at any phase → close remaining at SL
//
// FALLBACK (if TP2/TP3 missing):
//   - 100% close at TP1 or SL (old behavior)

use anyhow::Result;
use sqlx::PgPool;

use crate::types::*;

/// Small buffer for breakeven trailing SL (0.1% inside profit)
const BREAKEVEN_BUFFER: f64 = 0.001;

/// Position close fractions for partial take-profit (V4: Updated for better RR)
/// Close less at TP1, more at TP2/TP3 to capture full moves
const TP1_CLOSE_PCT: f64 = 0.50;  // V4: Reduced from 0.70
const TP2_CLOSE_PCT: f64 = 0.30;  // V4: Increased from 0.20
// TP3_CLOSE_PCT = 1.0 - TP1 - TP2 = 0.20 (V4: Increased from 0.10)

/// Calculate PnL percentage from entry to exit
#[inline]
fn pnl_from(entry: f64, exit: f64, is_long: bool) -> f64 {
    if is_long {
        (exit - entry) / entry
    } else {
        (entry - exit) / entry
    }
}

/// After TP1: move SL to BREAKEVEN (entry + small buffer).
/// This makes the remaining position RISK-FREE.
/// For LONG: SL = entry * 1.001 (slightly above entry → guaranteed small profit)
/// For SHORT: SL = entry * 0.999
#[inline]
fn trail_sl_to_breakeven(entry: f64, is_long: bool) -> f64 {
    if is_long {
        entry * (1.0 + BREAKEVEN_BUFFER)
    } else {
        entry * (1.0 - BREAKEVEN_BUFFER)
    }
}

/// After TP2: move SL to just below TP1 (lock in TP1-level profit).
/// For LONG: SL = tp1 * 0.999
/// For SHORT: SL = tp1 * 1.001
#[inline]
fn trail_sl_to_tp1(tp1: f64, is_long: bool) -> f64 {
    if is_long {
        tp1 * (1.0 - BREAKEVEN_BUFFER)
    } else {
        tp1 * (1.0 + BREAKEVEN_BUFFER)
    }
}

pub struct SignalEvaluator {
    pool: PgPool,
    /// Max number of candles to look forward before declaring "expired"
    timeout_bars: usize,
}

impl SignalEvaluator {
    pub fn new(pool: PgPool, timeout_bars: usize) -> Self {
        Self { pool, timeout_bars }
    }

    /// Evaluate a single signal against future candle data.
    /// Returns None if not enough future data exists.
    pub async fn evaluate(&self, signal: &SignalForBacktest) -> Result<Option<BacktestResult>> {
        let entry = signal.entry_price.unwrap_or(0.0) as f64;
        let sl = signal.sl_price.unwrap_or(0.0) as f64;
        let tp1 = signal.tp1_price.unwrap_or(0.0) as f64;
        let tp2 = signal.tp2_price.map(|v| v as f64);
        let tp3 = signal.tp3_price.map(|v| v as f64);

        if entry <= 0.0 || sl <= 0.0 || tp1 <= 0.0 {
            return Ok(None);
        }

        // Fetch future candles after signal time
        let candles = self.fetch_future_candles(
            signal.symbol_id,
            signal.tf_minutes,
            signal.time,
            self.timeout_bars,
        ).await?;

        if candles.is_empty() {
            return Ok(None);
        }

        let is_long = signal.side > 0;

        // Use partial close strategy only when all 3 TPs are available
        let use_partial = tp2.is_some() && tp3.is_some();

        let mut max_favorable: f64 = 0.0;
        let mut max_adverse: f64 = 0.0;
        let mut bars_used: usize = 0;

        // Position tracking
        let mut remaining_pct: f64 = 1.0;
        let mut realized_pnl: f64 = 0.0;
        let mut current_sl: f64 = sl;
        let mut tp1_hit = false;
        let mut tp2_hit = false;
        let mut tp3_hit = false;
        let mut highest_tp: u8 = 0;
        let mut last_exit_price: f64 = entry;
        let mut final_outcome: Option<Outcome> = None;

        for (i, candle) in candles.iter().enumerate() {
            bars_used = i + 1;

            // Track max favorable/adverse excursion (MFE/MAE)
            let favorable = if is_long {
                (candle.high - entry) / entry
            } else {
                (entry - candle.low) / entry
            };
            let adverse = if is_long {
                (entry - candle.low) / entry
            } else {
                (candle.high - entry) / entry
            };

            if favorable > max_favorable { max_favorable = favorable; }
            if adverse > max_adverse { max_adverse = adverse; }

            if !use_partial {
                // =====================================================
                // FALLBACK: 100% position, TP1-only (old behavior)
                // =====================================================
                let sl_hit = if is_long { candle.low <= current_sl } else { candle.high >= current_sl };
                let tp1_reached = if is_long { candle.high >= tp1 } else { candle.low <= tp1 };

                if sl_hit && tp1_reached {
                    // Both hit in same candle — use open direction heuristic
                    let opened_adverse = if is_long {
                        candle.open < entry
                    } else {
                        candle.open > entry
                    };

                    if opened_adverse {
                        realized_pnl = pnl_from(entry, sl, is_long);
                        last_exit_price = sl;
                        final_outcome = Some(Outcome::Loss { exit_price: sl });
                    } else {
                        realized_pnl = pnl_from(entry, tp1, is_long);
                        last_exit_price = tp1;
                        tp1_hit = true;
                        highest_tp = 1;
                        final_outcome = Some(Outcome::Win { tp_level: 1, exit_price: tp1 });
                    }
                    remaining_pct = 0.0;
                    break;
                }

                if sl_hit {
                    realized_pnl = pnl_from(entry, sl, is_long);
                    last_exit_price = sl;
                    remaining_pct = 0.0;
                    final_outcome = Some(Outcome::Loss { exit_price: sl });
                    break;
                }

                if tp1_reached {
                    realized_pnl = pnl_from(entry, tp1, is_long);
                    last_exit_price = tp1;
                    tp1_hit = true;
                    highest_tp = 1;
                    remaining_pct = 0.0;
                    final_outcome = Some(Outcome::Win { tp_level: 1, exit_price: tp1 });
                    break;
                }
            } else {
                // =====================================================
                // PARTIAL CLOSE with dynamic TP/SL (50% / 30% / 20%)
                // =====================================================
                let tp2_val = tp2.unwrap();
                let tp3_val = tp3.unwrap();

                if !tp1_hit {
                    // --- Phase 1: Waiting for TP1, original SL active ---
                    let sl_hit = if is_long { candle.low <= current_sl } else { candle.high >= current_sl };
                    let tp1_reached = if is_long { candle.high >= tp1 } else { candle.low <= tp1 };

                    if sl_hit && tp1_reached {
                        // Same-candle conflict: use open direction vs entry
                        let opened_adverse = if is_long {
                            candle.open < entry
                        } else {
                            candle.open > entry
                        };

                        if opened_adverse {
                            // SL hit first → full loss
                            realized_pnl = pnl_from(entry, current_sl, is_long);
                            remaining_pct = 0.0;
                            last_exit_price = current_sl;
                            final_outcome = Some(Outcome::Loss { exit_price: current_sl });
                            break;
                        } else {
                            // TP1 hit first → partial close 50%
                            realized_pnl += TP1_CLOSE_PCT * pnl_from(entry, tp1, is_long);
                            remaining_pct -= TP1_CLOSE_PCT;
                            tp1_hit = true;
                            highest_tp = 1;
                            current_sl = trail_sl_to_breakeven(entry, is_long);
                            last_exit_price = tp1;
                            // Skip further checks this candle (TP just hit, trailing SL just set)
                            continue;
                        }
                    } else if sl_hit {
                        // SL hit before any TP
                        realized_pnl = pnl_from(entry, current_sl, is_long);
                        remaining_pct = 0.0;
                        last_exit_price = current_sl;
                        final_outcome = Some(Outcome::Loss { exit_price: current_sl });
                        break;
                    } else if tp1_reached {
                        // TP1 hit, no SL conflict
                        realized_pnl += TP1_CLOSE_PCT * pnl_from(entry, tp1, is_long);
                        remaining_pct -= TP1_CLOSE_PCT;
                        tp1_hit = true;
                        highest_tp = 1;
                        current_sl = trail_sl_to_breakeven(entry, is_long);
                        last_exit_price = tp1;
                        continue;
                    }
                } else if !tp2_hit {
                    // --- Phase 2: TP1 hit, waiting for TP2, trailing SL below TP1 ---
                    let sl_hit = if is_long { candle.low <= current_sl } else { candle.high >= current_sl };
                    let tp2_reached = if is_long { candle.high >= tp2_val } else { candle.low <= tp2_val };

                    if sl_hit && tp2_reached {
                        // Same-candle conflict: use open direction vs TP1 reference
                        let opened_toward_sl = if is_long {
                            candle.open < tp1
                        } else {
                            candle.open > tp1
                        };

                        if opened_toward_sl {
                            // Trailing SL hit first → close remaining at trailing SL
                            realized_pnl += remaining_pct * pnl_from(entry, current_sl, is_long);
                            remaining_pct = 0.0;
                            last_exit_price = current_sl;
                            final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                            break;
                        } else {
                            // TP2 hit first → partial close 30%
                            realized_pnl += TP2_CLOSE_PCT * pnl_from(entry, tp2_val, is_long);
                            remaining_pct -= TP2_CLOSE_PCT;
                            tp2_hit = true;
                            highest_tp = 2;
                            current_sl = trail_sl_to_tp1(tp1, is_long);
                            last_exit_price = tp2_val;
                            continue;
                        }
                    } else if sl_hit {
                        // Trailing SL hit → close remaining
                        realized_pnl += remaining_pct * pnl_from(entry, current_sl, is_long);
                        remaining_pct = 0.0;
                        last_exit_price = current_sl;
                        final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                        break;
                    } else if tp2_reached {
                        // TP2 hit, no SL conflict
                        realized_pnl += TP2_CLOSE_PCT * pnl_from(entry, tp2_val, is_long);
                        remaining_pct -= TP2_CLOSE_PCT;
                        tp2_hit = true;
                        highest_tp = 2;
                        current_sl = trail_sl_to_tp1(tp1, is_long);
                        last_exit_price = tp2_val;
                        continue;
                    }
                } else if !tp3_hit {
                    // --- Phase 3: TP1+TP2 hit, waiting for TP3, trailing SL below TP2 ---
                    let sl_hit = if is_long { candle.low <= current_sl } else { candle.high >= current_sl };
                    let tp3_reached = if is_long { candle.high >= tp3_val } else { candle.low <= tp3_val };

                    if sl_hit && tp3_reached {
                        // Same-candle conflict: use open direction vs TP2 reference
                        let opened_toward_sl = if is_long {
                            candle.open < tp2_val
                        } else {
                            candle.open > tp2_val
                        };

                        if opened_toward_sl {
                            // Trailing SL hit first → close remaining
                            realized_pnl += remaining_pct * pnl_from(entry, current_sl, is_long);
                            remaining_pct = 0.0;
                            last_exit_price = current_sl;
                            final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                            break;
                        } else {
                            // TP3 hit → close remaining 20%
                            realized_pnl += remaining_pct * pnl_from(entry, tp3_val, is_long);
                            remaining_pct = 0.0;
                            tp3_hit = true;
                            highest_tp = 3;
                            last_exit_price = tp3_val;
                            final_outcome = Some(Outcome::Win { tp_level: 3, exit_price: tp3_val });
                            break;
                        }
                    } else if sl_hit {
                        // Trailing SL hit → close remaining
                        realized_pnl += remaining_pct * pnl_from(entry, current_sl, is_long);
                        remaining_pct = 0.0;
                        last_exit_price = current_sl;
                        final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                        break;
                    } else if tp3_reached {
                        // TP3 hit → close remaining 20%
                        realized_pnl += remaining_pct * pnl_from(entry, tp3_val, is_long);
                        remaining_pct = 0.0;
                        tp3_hit = true;
                        highest_tp = 3;
                        last_exit_price = tp3_val;
                        final_outcome = Some(Outcome::Win { tp_level: 3, exit_price: tp3_val });
                        break;
                    }
                }
                // If tp3_hit, remaining_pct should be 0.0, loop will end
            }
        }

        // Handle remaining position at timeout (expired)
        if remaining_pct > 0.0 {
            let last_close = candles.last().map(|c| c.close).unwrap_or(entry);
            realized_pnl += remaining_pct * pnl_from(entry, last_close, is_long);
            last_exit_price = last_close;

            if final_outcome.is_none() {
                if tp1_hit {
                    // TP1 (and maybe TP2) was hit but remaining portion expired
                    final_outcome = Some(Outcome::Win {
                        tp_level: highest_tp,
                        exit_price: last_exit_price,
                    });
                } else {
                    final_outcome = Some(Outcome::Expired { last_price: last_close });
                }
            }
        }

        let outcome = final_outcome.unwrap_or_else(|| {
            let last_price = candles.last().map(|c| c.close).unwrap_or(entry);
            Outcome::Expired { last_price }
        });

        Ok(Some(BacktestResult {
            signal_time: signal.time,
            signal_time_ms: signal.time_ms,
            symbol: signal.symbol.clone(),
            symbol_id: signal.symbol_id,
            tf_minutes: signal.tf_minutes,
            side: signal.side,
            final_score: signal.final_score,
            ml_score: signal.ml_score,
            heur_score: signal.heur_score,
            entry_price: signal.entry_price.unwrap_or(0.0),
            sl_price: signal.sl_price.unwrap_or(0.0),
            tp1_price: signal.tp1_price.unwrap_or(0.0),
            tp2_price: signal.tp2_price,
            tp3_price: signal.tp3_price,
            price10_score: signal.price10_score,
            bounce_prob: signal.bounce_prob,
            bounce_score: signal.bounce_score,
            breakout_prob: signal.breakout_prob,
            breakout_score: signal.breakout_score,
            outcome,
            pnl_pct: realized_pnl,
            exit_price: Some(last_exit_price),
            bars_to_outcome: bars_used,
            max_favorable,
            max_adverse,
            tp1_hit,
            tp2_hit,
            tp3_hit,
            reason_json: signal.reason.clone().unwrap_or(serde_json::json!({})),
        }))
    }

    /// Fetch candles AFTER signal_time for the given symbol and timeframe
    async fn fetch_future_candles(
        &self,
        symbol_id: i64,
        tf_minutes: i16,
        after_time: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<CandleRow>> {
        let table = match tf_minutes {
            1 => "market.candles_1m",
            5 => "market.candles_5m",
            15 => "market.candles_15m",
            60 => "market.candles_1h",
            240 => "market.candles_4h",
            _ => "market.candles_1h",
        };

        let sql = format!(
            "SELECT time, open, high, low, close FROM {} \
             WHERE symbol_id = $1 AND time > $2 \
             ORDER BY time ASC LIMIT $3",
            table
        );

        let rows = sqlx::query_as::<_, CandleRow>(&sql)
            .bind(symbol_id)
            .bind(after_time)
            .bind(limit as i64)
            .fetch_all(&self.pool)
            .await?;

        Ok(rows)
    }
}
