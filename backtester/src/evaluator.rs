// backtester/src/evaluator.rs
//
// Core evaluation logic: for each signal, walk forward through future candles
// and determine if TP1/TP2/TP3 or SL was hit first.

use anyhow::Result;
use sqlx::PgPool;

use crate::types::*;

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
            return Ok(None); // No future data
        }

        let is_long = signal.side > 0;

        let mut max_favorable: f64 = 0.0;
        let mut max_adverse: f64 = 0.0;
        let mut outcome: Option<Outcome> = None;
        let mut bars_used: usize = 0;

        for (i, candle) in candles.iter().enumerate() {
            bars_used = i + 1;

            // Track max favorable/adverse excursion
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

            // Check SL hit first (within the same candle, SL takes priority if both hit)
            let sl_hit = if is_long {
                candle.low <= sl
            } else {
                candle.high >= sl
            };

            // Check TP hits (highest TP first for best outcome tracking)
            let tp3_hit = tp3.map(|t| if is_long { candle.high >= t } else { candle.low <= t }).unwrap_or(false);
            let tp2_hit = tp2.map(|t| if is_long { candle.high >= t } else { candle.low <= t }).unwrap_or(false);
            let tp1_hit = if is_long { candle.high >= tp1 } else { candle.low <= tp1 };

            // Priority: If both SL and TP hit in same candle, check open direction
            // Simplification: assume SL checked first if the candle opened adversely
            if sl_hit && (tp1_hit || tp2_hit || tp3_hit) {
                // Both hit in same candle - check which was more likely hit first
                // If the candle opened on the adverse side, SL probably hit first
                let opened_adverse = if is_long {
                    candle.open < entry
                } else {
                    candle.open > entry
                };

                if opened_adverse {
                    outcome = Some(Outcome::Loss { exit_price: sl });
                } else if tp3_hit {
                    outcome = Some(Outcome::Win { tp_level: 3, exit_price: tp3.unwrap() });
                } else if tp2_hit {
                    outcome = Some(Outcome::Win { tp_level: 2, exit_price: tp2.unwrap() });
                } else {
                    outcome = Some(Outcome::Win { tp_level: 1, exit_price: tp1 });
                }
                break;
            }

            if sl_hit {
                outcome = Some(Outcome::Loss { exit_price: sl });
                break;
            }

            if tp3_hit {
                outcome = Some(Outcome::Win { tp_level: 3, exit_price: tp3.unwrap() });
                break;
            }
            if tp2_hit {
                outcome = Some(Outcome::Win { tp_level: 2, exit_price: tp2.unwrap() });
                break;
            }
            if tp1_hit {
                outcome = Some(Outcome::Win { tp_level: 1, exit_price: tp1 });
                break;
            }
        }

        // If no outcome within timeout, mark as expired
        let outcome = outcome.unwrap_or_else(|| {
            let last_price = candles.last().map(|c| c.close).unwrap_or(entry);
            Outcome::Expired { last_price }
        });

        // Calculate PnL
        let exit_price = match &outcome {
            Outcome::Win { exit_price, .. } => Some(*exit_price),
            Outcome::Loss { exit_price } => Some(*exit_price),
            Outcome::Expired { last_price } => Some(*last_price),
        };

        let pnl_pct = match exit_price {
            Some(ep) => {
                if is_long { (ep - entry) / entry }
                else { (entry - ep) / entry }
            }
            None => 0.0,
        };

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
            pnl_pct,
            exit_price,
            bars_to_outcome: bars_used,
            max_favorable,
            max_adverse,
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
            1440 => "market.candles_1d",
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
