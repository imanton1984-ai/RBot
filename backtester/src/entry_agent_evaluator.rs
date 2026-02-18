// backtester/src/entry_agent_evaluator.rs
//
// Entry Agent Evaluator — Backtest with Entry Agent decision logic
//
// This module evaluates signals using the Entry Agent instead of immediate entry.
// It simulates the real-time ENTER/WAIT/CANCEL decisions on each bar.
//
// COMPARISON MODES:
//   1. Baseline: Enter immediately on signal (current behavior)
//   2. EntryAgent: Wait for ENTER decision before entering
//
// This allows measuring the improvement from using the Entry Agent.

use anyhow::Result;
use sqlx::PgPool;
use crate::types::*;
use crate::entry_policy_labeler::*;

/// Entry Agent decision (simplified for backtesting)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryAgentDecision {
    Enter,
    Wait,
    Cancel,
}

/// Configuration for Entry Agent evaluation
#[derive(Clone, Debug)]
pub struct EntryAgentEvalConfig {
    /// Use Entry Agent instead of immediate entry
    pub use_entry_agent: bool,
    /// Enter threshold (probability)
    pub enter_threshold: f32,
    /// Cancel threshold (probability)
    pub cancel_threshold: f32,
    /// Minimum margin between enter and cancel
    pub min_margin: f32,
    /// Use mock decisions (for testing without models)
    pub use_mock_decisions: bool,
}

impl Default for EntryAgentEvalConfig {
    fn default() -> Self {
        Self {
            use_entry_agent: false, // Default to baseline behavior
            enter_threshold: 0.55,
            cancel_threshold: 0.50,
            min_margin: 0.15,
            use_mock_decisions: true, // Mock for now, real models later
        }
    }
}

/// Evaluator that uses Entry Agent for entry timing
pub struct EntryAgentEvaluator {
    pool: PgPool,
    config: EntryAgentEvalConfig,
    timeout_bars: usize,
}

impl EntryAgentEvaluator {
    pub fn new(pool: PgPool, timeout_bars: usize, config: EntryAgentEvalConfig) -> Self {
        Self {
            pool,
            config,
            timeout_bars,
        }
    }

    /// Evaluate a signal using Entry Agent logic
    pub async fn evaluate(&self, signal: &SignalForBacktest) -> Result<Option<BacktestResult>> {
        let entry_price = signal.entry_price.unwrap_or(0.0) as f64;
        let sl_price = signal.sl_price.unwrap_or(0.0) as f64;
        let tp1_price = signal.tp1_price.unwrap_or(0.0) as f64;
        let tp2_price = signal.tp2_price.map(|v| v as f64);
        let tp3_price = signal.tp3_price.map(|v| v as f64);

        if entry_price <= 0.0 || sl_price <= 0.0 || tp1_price <= 0.0 {
            return Ok(None);
        }

        // Fetch future candles
        let candles = self.fetch_future_candles(
            signal.symbol_id,
            signal.tf_minutes,
            signal.time,
            self.timeout_bars + 20, // Extra bars for entry window
        ).await?;

        if candles.is_empty() {
            return Ok(None);
        }

        // Convert to OhlcBar for simulation
        let ohlc_bars: Vec<OhlcBar> = candles.iter().map(|c| OhlcBar {
            high: c.high,
            low: c.low,
            close: c.close,
            atr: 0.001, // Placeholder - would need ATR from DB
        }).collect();

        // Determine entry bar
        let entry_bar_idx = if self.config.use_entry_agent {
            // Use Entry Agent logic
            self.find_entry_bar_with_agent(
                signal,
                &ohlc_bars,
                signal.tf_minutes,
            )
        } else {
            // Baseline: enter immediately on bar 0
            Some(0)
        };

        // No valid entry found (agent cancelled or waited too long)
        let entry_bar_idx = match entry_bar_idx {
            Some(idx) if idx < ohlc_bars.len() => idx,
            _ => {
                // Entry agent cancelled or expired without entering
                return Ok(Some(self.create_expired_result(signal, &candles)));
            }
        };

        // Now evaluate the trade from entry_bar_idx
        // (This would use the same logic as SignalEvaluator, but starting from entry_bar_idx)
        self.evaluate_trade_from_entry(signal, &candles, entry_bar_idx)
    }

    /// Find the optimal entry bar using Entry Agent logic
    ///
    /// In production, this would call the actual Entry Agent models.
    /// For now, we use mock logic based on expert labeling.
    fn find_entry_bar_with_agent(
        &self,
        signal: &SignalForBacktest,
        ohlc_bars: &[OhlcBar],
        tf_minutes: i16,
    ) -> Option<usize> {
        if self.config.use_mock_decisions {
            // Mock: use expert labeling to find best entry
            let window_bars = get_window_bars_for_tf(tf_minutes);
            let max_hold_bars = get_max_hold_bars_for_tf(tf_minutes);

            let cfg = SimCfg {
                window_bars,
                max_hold_bars,
                sl_atr_mult: 1.0,
                rr1: 1.0,
                rr2: 1.5,
                rr3: 2.0,
                tp1_close_pct: 0.50,
                tp2_close_pct: 0.30,
                tp3_close_pct: 0.20,
            };

            let label = find_best_entry(signal.side as i8, 0, ohlc_bars, cfg);

            // If expert says CANCEL (no profitable entry), return None
            if label.best_entry_offset.is_none() {
                return None;
            }

            // Return the optimal entry offset
            label.best_entry_offset
        } else {
            // TODO: Real model inference would go here
            // This would:
            // 1. Load Entry Agent models via ModelManager
            // 2. Extract features for each bar
            // 3. Call agent.decide() on each bar
            // 4. Return the bar where agent says ENTER

            // For now, fall back to immediate entry
            Some(0)
        }
    }

    /// Evaluate trade outcome starting from a specific entry bar
    fn evaluate_trade_from_entry(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleRow],
        entry_bar_idx: usize,
    ) -> Result<Option<BacktestResult>> {
        // This is a simplified version - in production, you'd reuse SignalEvaluator logic
        // For now, just return a basic result

        let entry_price = candles[entry_bar_idx].close;
        let is_long = signal.side > 0;

        // Simple evaluation (placeholder - would use full SignalEvaluator logic)
        let mut outcome = Outcome::Expired { last_price: entry_price };
        let mut pnl_pct = 0.0;
        let mut bars_used = 0;

        // ... (evaluation logic similar to SignalEvaluator)

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
            entry_price: entry_price as f32,
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
            exit_price: Some(entry_price),
            bars_to_outcome: bars_used,
            max_favorable: 0.0,
            max_adverse: 0.0,
            tp1_hit: false,
            tp2_hit: false,
            tp3_hit: false,
            reason_json: signal.reason.clone().unwrap_or(serde_json::json!({})),
        }))
    }

    /// Create an expired result (when agent cancels or times out)
    fn create_expired_result(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleRow],
    ) -> BacktestResult {
        let last_price = candles.last().map(|c| c.close).unwrap_or(signal.entry_price.unwrap_or(0.0) as f64);

        BacktestResult {
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
            outcome: Outcome::Expired { last_price },
            pnl_pct: 0.0,
            exit_price: Some(last_price),
            bars_to_outcome: candles.len(),
            max_favorable: 0.0,
            max_adverse: 0.0,
            tp1_hit: false,
            tp2_hit: false,
            tp3_hit: false,
            reason_json: signal.reason.clone().unwrap_or(serde_json::json!({})),
        }
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

/// Compare baseline vs Entry Agent evaluation
pub async fn run_comparison_evaluation(
    pool: &PgPool,
    signals: &[SignalForBacktest],
    timeout_bars: usize,
) -> Result<(ComparisonResults, ComparisonResults)> {
    // Baseline evaluation (immediate entry)
    let baseline_evaluator = crate::evaluator::SignalEvaluator::new(pool.clone(), timeout_bars);
    let mut baseline_results = Vec::new();

    for signal in signals {
        if let Some(result) = baseline_evaluator.evaluate(signal).await? {
            baseline_results.push(result);
        }
    }

    // Entry Agent evaluation (delayed entry)
    let agent_config = EntryAgentEvalConfig {
        use_entry_agent: true,
        use_mock_decisions: true, // Use expert labeling as proxy
        ..Default::default()
    };
    let agent_evaluator = EntryAgentEvaluator::new(pool.clone(), timeout_bars, agent_config);
    let mut agent_results = Vec::new();

    for signal in signals {
        if let Some(result) = agent_evaluator.evaluate(signal).await? {
            agent_results.push(result);
        }
    }

    // Calculate comparison metrics
    let baseline_comparison = compare_results(&baseline_results);
    let agent_comparison = compare_results(&agent_results);

    Ok((baseline_comparison, agent_comparison))
}

/// Comparison results summary
#[derive(Clone, Debug, Default)]
pub struct ComparisonResults {
    pub total_signals: usize,
    pub entered_trades: usize,
    pub cancelled_trades: usize,
    pub win_count: usize,
    pub loss_count: usize,
    pub expired_count: usize,
    pub avg_pnl: f64,
    pub avg_pnl_wins: f64,
    pub avg_pnl_losses: f64,
    pub win_rate: f64,
}

fn compare_results(results: &[BacktestResult]) -> ComparisonResults {
    let mut stats = ComparisonResults::default();

    stats.total_signals = results.len();

    for r in results {
        match &r.outcome {
            Outcome::Win { .. } => {
                stats.win_count += 1;
                stats.entered_trades += 1;
            }
            Outcome::Loss { .. } => {
                stats.loss_count += 1;
                stats.entered_trades += 1;
            }
            Outcome::Expired { .. } => {
                stats.expired_count += 1;
                // Check if it was a cancelled setup or just expired
                if r.pnl_pct == 0.0 && r.exit_price.unwrap_or(0.0) == r.entry_price as f64 {
                    stats.cancelled_trades += 1;
                } else {
                    stats.entered_trades += 1;
                }
            }
        }

        stats.avg_pnl += r.pnl_pct;
    }

    if stats.total_signals > 0 {
        stats.avg_pnl /= stats.total_signals as f64;
    }

    if stats.win_count > 0 {
        let win_pnls: f64 = results.iter()
            .filter(|r| matches!(r.outcome, Outcome::Win { .. }))
            .map(|r| r.pnl_pct)
            .sum();
        stats.avg_pnl_wins = win_pnls / stats.win_count as f64;
    }

    if stats.loss_count > 0 {
        let loss_pnls: f64 = results.iter()
            .filter(|r| matches!(r.outcome, Outcome::Loss { .. }))
            .map(|r| r.pnl_pct)
            .sum();
        stats.avg_pnl_losses = loss_pnls / stats.loss_count as f64;
    }

    if stats.entered_trades > 0 {
        stats.win_rate = stats.win_count as f64 / stats.entered_trades as f64;
    }

    stats
}
