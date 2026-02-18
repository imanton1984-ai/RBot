// backtester/src/entry_agent_real_evaluator.rs
//
// Real Entry Agent Evaluator - Loads models and runs actual inference
//
// This module evaluates signals using the trained Entry Agent models.
// It makes real ENTER/WAIT/CANCEL decisions on each bar.

use anyhow::Result;
use sqlx::{PgPool, Row};
use predictors::entry_policy::{EntryAgent, EntryAgentConfig, EntryDecision};
use predictors::ml::model_manager::ModelManager;
use crate::types::*;
use crate::entry_policy_labeler::*;

/// Real Entry Agent Evaluator with model loading
pub struct RealEntryAgentEvaluator {
    agent: EntryAgent,
    model_manager: ModelManager,
    pool: PgPool,
    timeout_bars: usize,
    use_gpu: bool,
}

impl RealEntryAgentEvaluator {
    pub fn new(pool: &PgPool, timeout_bars: usize, use_gpu: bool) -> Result<Self> {
        // Create model manager
        let mut model_manager = ModelManager::new(use_gpu);

        // Load entry policy models for all timeframes
        let timeframes = vec![1, 5, 15, 60, 240];

        // Try to load models (may not exist yet)
        // Get models directory - check multiple possible locations
        let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| {
            // Try to find models directory relative to executable or current dir
            if std::path::Path::new("models").exists() {
                "models".to_string()
            } else if std::path::Path::new("../models").exists() {
                "../models".to_string()
            } else {
                "/home/anton/Desktop/Rust_trader/models".to_string()
            }
        });
        
        let enter_template = format!("{}/entry_enter_v1_tf{{tf}}.ubj", models_dir);
        let cancel_template = format!("{}/entry_cancel_v1_tf{{tf}}.ubj", models_dir);
        
        tracing::info!("Loading entry_enter models from: {}", enter_template);
        let enter_result = model_manager.load_models_for_timeframes(
            "entry_enter",
            &enter_template,
            &timeframes,
            use_gpu,
        );
        if let Err(e) = enter_result {
            tracing::warn!("Failed to load entry_enter models: {}", e);
        }

        tracing::info!("Loading entry_cancel models from: {}", cancel_template);
        let cancel_result = model_manager.load_models_for_timeframes(
            "entry_cancel",
            &cancel_template,
            &timeframes,
            use_gpu,
        );
        if let Err(e) = cancel_result {
            tracing::warn!("Failed to load entry_cancel models: {}", e);
        }
        
        // Configure entry agent
        let config = EntryAgentConfig {
            enter_threshold: 0.55,
            cancel_threshold: 0.50,
            min_margin: 0.15,
            default_window_bars: 10,
        };
        
        let agent = EntryAgent::new(config);
        
        Ok(Self {
            agent,
            model_manager,
            pool: pool.clone(),
            timeout_bars,
            use_gpu,
        })
    }
    
    /// Check if models are loaded
    pub fn has_models(&self) -> bool {
        self.model_manager.has_model("entry_enter_tf1") &&
        self.model_manager.has_model("entry_cancel_tf1")
    }
    
    /// Evaluate a signal using Entry Agent with real model inference
    pub async fn evaluate(&self, signal: &SignalForBacktest) -> Result<Option<EntryAgentTradeResult>> {
        let entry_price = signal.entry_price.unwrap_or(0.0) as f64;
        let sl_price = signal.sl_price.unwrap_or(0.0) as f64;
        let tp1_price = signal.tp1_price.unwrap_or(0.0) as f64;
        
        if entry_price <= 0.0 || sl_price <= 0.0 || tp1_price <= 0.0 {
            return Ok(None);
        }
        
        // Fetch future candles with indicators for feature extraction
        let candles = self.fetch_candles_with_indicators(
            signal.symbol_id,
            signal.tf_minutes,
            signal.time,
            self.timeout_bars + 20, // Extra bars for entry window
        ).await?;
        
        if candles.len() < 2 {
            return Ok(None);
        }

        // Get window size for this TF
        let window_bars = EntryAgent::get_window_bars_for_tf(signal.tf_minutes as i32);

        // Run Entry Agent decision loop
        let mut decision_bar: Option<usize> = None;
        let mut final_decision = EntryDecision::Wait { confidence: 0.0 };

        for bar_idx in 0..window_bars.min(candles.len()) {
            // Extract features for this bar
            let features = self.extract_features(signal, &candles, bar_idx)?;

            // Get decision from Entry Agent
            let decision = self.agent.decide(
                &self.model_manager,
                signal.tf_minutes as i32,
                features,
                bar_idx as u16,
                (window_bars - bar_idx) as u16,
                self.use_gpu,
            )?;
            
            match decision {
                EntryDecision::Enter { .. } => {
                    decision_bar = Some(bar_idx);
                    final_decision = decision;
                    break; // Enter on this bar
                }
                EntryDecision::Cancel { .. } => {
                    // Agent says cancel - don't enter
                    return Ok(Some(self.create_cancelled_result(signal, bar_idx)));
                }
                EntryDecision::Wait { .. } => {
                    // Continue to next bar
                    final_decision = decision;
                }
            }
        }
        
        // No ENTER decision - setup expired or waited through window
        if decision_bar.is_none() {
            return Ok(Some(self.create_expired_result(signal, &candles, window_bars)));
        }
        
        // Enter on decision_bar - evaluate the trade
        let entry_idx = decision_bar.unwrap();
        self.evaluate_trade_from_entry(signal, &candles, entry_idx)
    }
    
    /// Extract features for a specific bar
    fn extract_features(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleWithIndicators],
        bar_idx: usize,
    ) -> Result<Vec<f32>> {
        let reason = signal.reason.as_ref().map(|r| r.clone()).unwrap_or(serde_json::json!({}));
        let get = |k: &str| reason.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let debug_get = |k: &str| {
            reason.get("debug").and_then(|d| d.get(k)).and_then(|v| v.as_f64())
                .or_else(|| reason.get(k).and_then(|v| v.as_f64()))
                .unwrap_or(0.0)
        };
        
        let ep = signal.entry_price.unwrap_or(0.0) as f64;
        let sl_pct = if ep > 0.0 { (signal.sl_price.unwrap_or(0.0) as f64 - ep).abs() / ep } else { 0.0 };
        let tp1_pct = if ep > 0.0 { (signal.tp1_price.unwrap_or(0.0) as f64 - ep).abs() / ep } else { 0.0 };
        let tp2_pct = signal.tp2_price.map(|t| if ep > 0.0 { (t as f64 - ep).abs() / ep } else { 0.0 }).unwrap_or(0.0);
        let tp3_pct = signal.tp3_price.map(|t| if ep > 0.0 { (t as f64 - ep).abs() / ep } else { 0.0 }).unwrap_or(0.0);
        let rr = if sl_pct > 0.0 { tp1_pct / sl_pct } else { 0.0 };
        let rr2 = if sl_pct > 0.0 { tp2_pct / sl_pct } else { 0.0 };
        
        let level_aware = reason.get("level_aware").and_then(|v| v.as_bool()).unwrap_or(false);
        let quality_mult = get("quality_multiplier");
        let quality_grade_val = match reason.get("quality_grade").and_then(|v| v.as_str()).unwrap_or("?") {
            "A" => 4.0, "B" => 3.0, "C" => 2.0, "D" => 1.0, _ => 0.0,
        };
        let original_score = get("original_score");
        
        let pred = debug_get("predictors_score");
        let raw = debug_get("raw_signals_score");
        let ind = debug_get("indicators_score");
        let mkt = debug_get("market_score");
        let pred_vs_raw = if raw > 0.0001 { pred / raw } else { 0.0 };
        let pred_vs_ind = if ind > 0.0001 { pred / ind } else { 0.0 };
        let scores = [pred, raw, ind, mkt];
        let mean_s = scores.iter().sum::<f64>() / 4.0;
        let var_s = scores.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / 4.0;
        let component_std = var_s.sqrt();
        let component_min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let ml_s = signal.ml_score.unwrap_or(0.0) as f64;
        let heur_s = signal.heur_score.unwrap_or(0.0) as f64;
        let ml_heur_gap = (ml_s - heur_s).abs();
        let score_per_risk = if sl_pct > 0.0 { signal.final_score as f64 / sl_pct } else { 0.0 };
        
        let atr_pct = get("atr_pct");
        let market_factor = get("market_factor");
        let score_factor = get("score_factor");
        
        // Build feature vector (must match training schema)
        let mut features = vec![
            signal.tf_minutes as f32,
            signal.side as f32,
            signal.final_score,
            ml_s as f32,
            heur_s as f32,
            ep as f32,
            sl_pct as f32,
            tp1_pct as f32,
            tp2_pct as f32,
            tp3_pct as f32,
            rr as f32,
            rr2 as f32,
            pred as f32,
            raw as f32,
            ind as f32,
            mkt as f32,
            debug_get("coverage_score") as f32,
            debug_get("consensus_score") as f32,
            signal.price10_score.unwrap_or(0.0) as f32,
            signal.bounce_prob.unwrap_or(0.0) as f32,
            signal.bounce_score.unwrap_or(0.0) as f32,
            signal.breakout_prob.unwrap_or(0.0) as f32,
            signal.breakout_score.unwrap_or(0.0) as f32,
            debug_get("trend_strength") as f32,
            debug_get("momentum_strength") as f32,
            debug_get("volatility_regime") as f32,
            debug_get("volume_spike_score") as f32,
            if level_aware { 1.0 } else { 0.0 },
            get("market_quality_score") as f32,
            quality_mult as f32,
            quality_grade_val as f32,
            original_score as f32,
            pred_vs_raw as f32,
            pred_vs_ind as f32,
            component_std as f32,
            component_min as f32,
            ml_heur_gap as f32,
            score_per_risk as f32,
            atr_pct as f32,
            market_factor as f32,
            score_factor as f32,
        ];
        
        // Note: elapsed and remaining are added by EntryAgent::decide()
        
        Ok(features)
    }
    
    /// Evaluate trade outcome starting from a specific entry bar
    fn evaluate_trade_from_entry(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleWithIndicators],
        entry_bar_idx: usize,
    ) -> Result<Option<EntryAgentTradeResult>> {
        if entry_bar_idx >= candles.len() {
            return Ok(None);
        }
        
        let entry_price = candles[entry_bar_idx].close;
        let is_long = signal.side > 0;
        
        // Use same evaluation logic as SignalEvaluator
        // (simplified here - would reuse full logic in production)
        
        let sl_price = signal.sl_price.unwrap_or(0.0) as f64;
        let tp1_price = signal.tp1_price.unwrap_or(0.0) as f64;
        let tp2_price = signal.tp2_price.map(|v| v as f64);
        let tp3_price = signal.tp3_price.map(|v| v as f64);
        
        let mut outcome = Outcome::Expired { last_price: entry_price };
        let mut pnl_pct = 0.0;
        let mut bars_used = 0;
        let mut tp1_hit = false;
        let mut tp2_hit = false;
        let mut tp3_hit = false;
        
        // Simple evaluation (placeholder - would use full SignalEvaluator logic)
        for (i, candle) in candles.iter().enumerate().skip(entry_bar_idx) {
            bars_used = i - entry_bar_idx + 1;
            
            let sl_hit = if is_long { candle.low <= sl_price } else { candle.high >= sl_price };
            let tp1_reached = if is_long { candle.high >= tp1_price } else { candle.low <= tp1_price };
            
            if sl_hit && !tp1_reached {
                pnl_pct = (sl_price - entry_price) / entry_price * (signal.side as f64) * 100.0;
                outcome = Outcome::Loss { exit_price: sl_price };
                break;
            }
            
            if tp1_reached {
                pnl_pct = (tp1_price - entry_price) / entry_price * (signal.side as f64) * 100.0;
                tp1_hit = true;
                outcome = Outcome::Win { tp_level: 1, exit_price: tp1_price };
                break;
            }
        }
        
        Ok(Some(EntryAgentTradeResult {
            signal_time: signal.time,
            signal_time_ms: signal.time_ms,
            symbol: signal.symbol.clone(),
            symbol_id: signal.symbol_id,
            tf_minutes: signal.tf_minutes,
            side: signal.side,
            final_score: signal.final_score,
            entry_bar_offset: entry_bar_idx as u16,
            entry_price: entry_price as f32,
            outcome,
            pnl_pct,
            bars_to_entry: entry_bar_idx as u16,
            bars_to_outcome: bars_used as u16,
            tp1_hit,
            tp2_hit,
            tp3_hit,
        }))
    }
    
    /// Create cancelled result
    fn create_cancelled_result(
        &self,
        signal: &SignalForBacktest,
        cancel_bar: usize,
    ) -> EntryAgentTradeResult {
        EntryAgentTradeResult {
            signal_time: signal.time,
            signal_time_ms: signal.time_ms,
            symbol: signal.symbol.clone(),
            symbol_id: signal.symbol_id,
            tf_minutes: signal.tf_minutes,
            side: signal.side,
            final_score: signal.final_score,
            entry_bar_offset: u16::MAX, // Indicates no entry
            entry_price: 0.0,
            outcome: Outcome::Expired { last_price: signal.entry_price.unwrap_or(0.0) as f64 },
            pnl_pct: 0.0,
            bars_to_entry: cancel_bar as u16,
            bars_to_outcome: cancel_bar as u16,
            tp1_hit: false,
            tp2_hit: false,
            tp3_hit: false,
        }
    }
    
    /// Create expired result
    fn create_expired_result(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleWithIndicators],
        window_bars: usize,
    ) -> EntryAgentTradeResult {
        let last_price = candles.last().map(|c| c.close).unwrap_or(signal.entry_price.unwrap_or(0.0) as f64);
        
        EntryAgentTradeResult {
            signal_time: signal.time,
            signal_time_ms: signal.time_ms,
            symbol: signal.symbol.clone(),
            symbol_id: signal.symbol_id,
            tf_minutes: signal.tf_minutes,
            side: signal.side,
            final_score: signal.final_score,
            entry_bar_offset: u16::MAX,
            entry_price: 0.0,
            outcome: Outcome::Expired { last_price },
            pnl_pct: 0.0,
            bars_to_entry: window_bars as u16,
            bars_to_outcome: window_bars as u16,
            tp1_hit: false,
            tp2_hit: false,
            tp3_hit: false,
        }
    }
    
    /// Fetch candles with indicators for feature extraction
    async fn fetch_candles_with_indicators(
        &self,
        symbol_id: i64,
        tf_minutes: i16,
        after_time: chrono::DateTime<chrono::Utc>,
        limit: usize,
    ) -> Result<Vec<CandleWithIndicators>> {
        let candle_table = match tf_minutes {
            1 => "market.candles_1m",
            5 => "market.candles_5m",
            15 => "market.candles_15m",
            60 => "market.candles_1h",
            240 => "market.candles_4h",
            _ => "market.candles_1h",
        };
        
        let sql = format!(
            "SELECT c.time, c.high, c.low, c.close, \
                    COALESCE(i.atr, 0.001) as atr, COALESCE(i.rsi, 50.0) as rsi \
             FROM {} c \
             LEFT JOIN market.indicators_wide i \
               ON c.symbol_id = i.symbol_id AND c.time = i.time AND i.tf_minutes = $4 \
             WHERE c.symbol_id = $1 AND c.time > $2 \
             ORDER BY c.time ASC LIMIT $3",
            candle_table
        );
        
        let rows = sqlx::query(&sql)
            .bind(symbol_id)
            .bind(after_time)
            .bind(limit as i64)
            .bind(tf_minutes as i16)
            .fetch_all(&self.pool)
            .await?;
        
        let mut candles = Vec::with_capacity(rows.len());
        for row in rows {
            candles.push(CandleWithIndicators {
                time: row.try_get("time")?,
                open: 0.0, // Not fetched
                high: row.try_get("high")?,
                low: row.try_get("low")?,
                close: row.try_get("close")?,
                atr: row.try_get::<f32, _>("atr").unwrap_or(0.001) as f64,
                rsi: row.try_get::<f32, _>("rsi").unwrap_or(50.0) as f64,
            });
        }
        
        Ok(candles)
    }
}

/// Candle with indicators for feature extraction
#[derive(Debug, Clone)]
pub struct CandleWithIndicators {
    pub time: chrono::DateTime<chrono::Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub atr: f64,
    pub rsi: f64,
}

impl CandleWithIndicators {
    pub fn low(&self) -> f64 { self.low }
    pub fn high(&self) -> f64 { self.high }
}

/// Trade result from Entry Agent evaluation
#[derive(Debug, Clone)]
pub struct EntryAgentTradeResult {
    pub signal_time: chrono::DateTime<chrono::Utc>,
    pub signal_time_ms: i64,
    pub symbol: String,
    pub symbol_id: i64,
    pub tf_minutes: i16,
    pub side: i16,
    pub final_score: f32,
    /// Bar offset where entry occurred (u16::MAX if no entry)
    pub entry_bar_offset: u16,
    pub entry_price: f32,
    pub outcome: Outcome,
    pub pnl_pct: f64,
    pub bars_to_entry: u16,
    pub bars_to_outcome: u16,
    pub tp1_hit: bool,
    pub tp2_hit: bool,
    pub tp3_hit: bool,
}

/// Comparison statistics
#[derive(Debug, Clone, Default)]
pub struct EntryAgentComparison {
    pub baseline_total: usize,
    pub baseline_entered: usize,
    pub baseline_wins: usize,
    pub baseline_avg_pnl: f64,
    
    pub agent_total: usize,
    pub agent_entered: usize,
    pub agent_wins: usize,
    pub agent_avg_pnl: f64,
    pub agent_cancelled: usize,
    pub agent_expired: usize,
}

impl EntryAgentComparison {
    pub fn print(&self) {
        let baseline_win_rate: f64 = if self.baseline_entered > 0 {
            self.baseline_wins as f64 / self.baseline_entered as f64 * 100.0
        } else { 0.0 };

        let agent_win_rate: f64 = if self.agent_entered > 0 {
            self.agent_wins as f64 / self.agent_entered as f64 * 100.0
        } else { 0.0 };

        let trades_change: f64 = if self.baseline_entered > 0 {
            (self.agent_entered as i64 - self.baseline_entered as i64) as f64 / self.baseline_entered as f64 * 100.0
        } else { 0.0 };

        let win_rate_change: f64 = agent_win_rate - baseline_win_rate;
        let pnl_change: f64 = (self.agent_avg_pnl - self.baseline_avg_pnl) * 100.0;

        println!("\n");
        println!("╔═══════════════════════════════════════════════════════════╗");
        println!("║     ENTRY AGENT COMPARISON: Baseline vs EntryAgent        ║");
        println!("╚═══════════════════════════════════════════════════════════╝");
        println!("\n");

        println!("┌─────────────────────────────────────────────────────────────┐");
        println!("│ Metric              │ Baseline  │ EntryAgent │ Improvement │");
        println!("├─────────────────────┼───────────┼────────────┼─────────────┤");
        println!("│ Total Signals       │ {:>9} │ {:>10} │   {:>6}   │",
            self.baseline_total, self.agent_total, "0");
        println!("│ Entered Trades      │ {:>9} │ {:>10} │ {:>+6.0}%  │",
            self.baseline_entered, self.agent_entered, trades_change);
        println!("│ Win Rate            │ {:>7.1}% │ {:>9.1}% │ {:>+6.1}%  │",
            baseline_win_rate, agent_win_rate, win_rate_change);
        println!("│ Avg PnL             │ {:>8.4}% │ {:>9.4}% │ {:>+7.4}% │",
            self.baseline_avg_pnl * 100.0, self.agent_avg_pnl * 100.0, pnl_change);
        println!("├─────────────────────┴───────────┴────────────┼─────────────┤");
        println!("│ Cancelled Setups    │ {:>10} │              │             │", self.agent_cancelled);
        println!("│ Expired (no entry)  │ {:>10} │              │             │", self.agent_expired);
        println!("└──────────────────────────────────────────────┴─────────────┘");
        println!("\n");
    }
}
