// backtester/src/entry_agent_real_evaluator.rs
//
// Real Entry Agent Evaluator - Loads models and runs actual inference
//
// This module evaluates signals using the trained Entry Agent models.
// It makes real ENTER/WAIT/CANCEL decisions on each bar and evaluates
// trades using the SAME partial-close logic as the baseline evaluator.
//
// KEY IMPROVEMENTS:
//   1. Full partial-close TP1/TP2/TP3 evaluation (matches evaluator.rs exactly)
//   2. Per-TF breakdown with TP distribution in comparison output
//   3. Proper online learning: agent sees indicators on each bar
//   4. CUDA-accelerated labeling when available

use anyhow::Result;
use sqlx::{PgPool, Row};
use std::collections::HashMap;
use predictors::entry_policy::{EntryAgent, EntryAgentConfig, EntryDecision};
use predictors::ml::model_manager::ModelManager;
use crate::types::*;
use crate::entry_policy_labeler::*;

/// Small buffer for breakeven trailing SL (0.1% inside profit)
const BREAKEVEN_BUFFER: f64 = 0.001;

/// Position close fractions for partial take-profit (must match evaluator.rs!)
const TP1_CLOSE_PCT: f64 = 0.70;
const TP2_CLOSE_PCT: f64 = 0.20;

/// Calculate PnL percentage from entry to exit
#[inline]
fn pnl_from(entry: f64, exit: f64, is_long: bool) -> f64 {
    if is_long {
        (exit - entry) / entry
    } else {
        (entry - exit) / entry
    }
}

/// After TP1: move SL to BREAKEVEN
#[inline]
fn trail_sl_to_breakeven(entry: f64, is_long: bool) -> f64 {
    if is_long {
        entry * (1.0 + BREAKEVEN_BUFFER)
    } else {
        entry * (1.0 - BREAKEVEN_BUFFER)
    }
}

/// After TP2: move SL to just below TP1
#[inline]
fn trail_sl_to_tp1(tp1: f64, is_long: bool) -> f64 {
    if is_long {
        tp1 * (1.0 - BREAKEVEN_BUFFER)
    } else {
        tp1 * (1.0 + BREAKEVEN_BUFFER)
    }
}

/// Real Entry Agent Evaluator with model loading
pub struct RealEntryAgentEvaluator {
    agent: EntryAgent,
    model_manager: ModelManager,
    pool: PgPool,
    timeout_bars: usize,
    use_gpu: bool,
    /// Fallback to expert labeling when models are not loaded
    fallback_to_expert: bool,
}

impl RealEntryAgentEvaluator {
    pub fn new(pool: &PgPool, timeout_bars: usize, use_gpu: bool) -> Result<Self> {
        // Create model manager
        let mut model_manager = ModelManager::new(use_gpu);

        // Load entry policy models for all timeframes
        let timeframes = vec![1, 5, 15, 60, 240];

        // Try to load models (may not exist yet)
        let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| {
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
        
        // Check if models exist; if not, we'll fall back to expert labeling
        let has_models = model_manager.has_model("entry_enter_tf1") &&
                         model_manager.has_model("entry_cancel_tf1");
        
        Ok(Self {
            agent,
            model_manager,
            pool: pool.clone(),
            timeout_bars,
            use_gpu,
            fallback_to_expert: !has_models,
        })
    }
    
    /// Check if models are loaded
    pub fn has_models(&self) -> bool {
        !self.fallback_to_expert
    }
    
    /// Returns true if using expert labeling fallback (no trained models)
    pub fn is_using_expert_fallback(&self) -> bool {
        self.fallback_to_expert
    }
    
    /// Evaluate a signal using Entry Agent logic.
    /// Returns an EntryAgentTradeResult with full partial-close PnL.
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
            self.timeout_bars + 20,
        ).await?;
        
        if candles.len() < 2 {
            return Ok(None);
        }

        // Get window size for this TF
        let window_bars = EntryAgent::get_window_bars_for_tf(signal.tf_minutes as i32);

        // Determine entry bar using either model inference or expert labeling
        let entry_bar_idx = if self.fallback_to_expert {
            self.find_entry_bar_expert(signal, &candles, window_bars)
        } else {
            self.find_entry_bar_with_model(signal, &candles, window_bars)?
        };
        
        match entry_bar_idx {
            EntryBarResult::Enter(idx) => {
                // Evaluate trade from the chosen entry bar using full partial-close logic
                self.evaluate_trade_full(signal, &candles, idx)
            }
            EntryBarResult::Cancel(bar) => {
                Ok(Some(self.create_cancelled_result(signal, bar)))
            }
            EntryBarResult::Expired => {
                Ok(Some(self.create_expired_result(signal, &candles, window_bars)))
            }
        }
    }
    
    /// Find entry bar using expert labeling (optimal PnL search)
    fn find_entry_bar_expert(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleWithIndicators],
        window_bars: usize,
    ) -> EntryBarResult {
        let max_hold_bars = EntryAgent::get_max_hold_bars_for_tf(signal.tf_minutes as i32);
        
        let ohlc_bars: Vec<OhlcBar> = candles.iter().map(|c| OhlcBar {
            high: c.high,
            low: c.low,
            close: c.close,
            atr: c.atr.max(1e-9),
        }).collect();
        
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
        
        let label = find_best_entry(signal.side as i8, 0, &ohlc_bars, cfg);
        
        match label.best_entry_offset {
            Some(offset) if offset < candles.len() => EntryBarResult::Enter(offset),
            Some(_) => EntryBarResult::Expired,
            None => EntryBarResult::Cancel(0),
        }
    }
    
    /// Find entry bar using trained model inference
    fn find_entry_bar_with_model(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleWithIndicators],
        window_bars: usize,
    ) -> Result<EntryBarResult> {
        for bar_idx in 0..window_bars.min(candles.len()) {
            let features = self.extract_features(signal, candles, bar_idx)?;
            
            let decision = self.agent.decide(
                &self.model_manager,
                signal.tf_minutes as i32,
                features,
                bar_idx as u16,
                (window_bars.saturating_sub(bar_idx)) as u16,
                self.use_gpu,
            )?;
            
            match decision {
                EntryDecision::Enter { .. } => {
                    return Ok(EntryBarResult::Enter(bar_idx));
                }
                EntryDecision::Cancel { .. } => {
                    return Ok(EntryBarResult::Cancel(bar_idx));
                }
                EntryDecision::Wait { .. } => {
                    // continue to next bar
                }
            }
        }
        
        Ok(EntryBarResult::Expired)
    }
    
    /// Full trade evaluation with partial-close logic
    /// This MUST match evaluator.rs exactly to produce comparable results.
    fn evaluate_trade_full(
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
        
        let sl = signal.sl_price.unwrap_or(0.0) as f64;
        let tp1 = signal.tp1_price.unwrap_or(0.0) as f64;
        let tp2 = signal.tp2_price.map(|v| v as f64);
        let tp3 = signal.tp3_price.map(|v| v as f64);
        
        if entry_price <= 0.0 || sl <= 0.0 || tp1 <= 0.0 {
            return Ok(None);
        }
        
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
        let mut last_exit_price: f64 = entry_price;
        let mut final_outcome: Option<Outcome> = None;
        
        // Evaluate from entry_bar_idx onward, limited by timeout_bars
        let max_bars = (entry_bar_idx + self.timeout_bars).min(candles.len());
        
        for i in entry_bar_idx..max_bars {
            bars_used = i - entry_bar_idx + 1;
            
            let candle_high = candles[i].high;
            let candle_low = candles[i].low;
            let candle_open = candles[i].open;
            let candle_close = candles[i].close;
            
            // Track MFE/MAE
            let favorable = if is_long {
                (candle_high - entry_price) / entry_price
            } else {
                (entry_price - candle_low) / entry_price
            };
            let adverse = if is_long {
                (entry_price - candle_low) / entry_price
            } else {
                (candle_high - entry_price) / entry_price
            };
            if favorable > max_favorable { max_favorable = favorable; }
            if adverse > max_adverse { max_adverse = adverse; }
            
            if !use_partial {
                // === FALLBACK: 100% position, TP1-only ===
                let sl_hit = if is_long { candle_low <= current_sl } else { candle_high >= current_sl };
                let tp1_reached = if is_long { candle_high >= tp1 } else { candle_low <= tp1 };
                
                if sl_hit && tp1_reached {
                    let opened_adverse = if is_long { candle_open < entry_price } else { candle_open > entry_price };
                    if opened_adverse {
                        realized_pnl = pnl_from(entry_price, sl, is_long);
                        last_exit_price = sl;
                        final_outcome = Some(Outcome::Loss { exit_price: sl });
                    } else {
                        realized_pnl = pnl_from(entry_price, tp1, is_long);
                        last_exit_price = tp1;
                        tp1_hit = true;
                        highest_tp = 1;
                        final_outcome = Some(Outcome::Win { tp_level: 1, exit_price: tp1 });
                    }
                    remaining_pct = 0.0;
                    break;
                }
                
                if sl_hit {
                    realized_pnl = pnl_from(entry_price, sl, is_long);
                    last_exit_price = sl;
                    remaining_pct = 0.0;
                    final_outcome = Some(Outcome::Loss { exit_price: sl });
                    break;
                }
                
                if tp1_reached {
                    realized_pnl = pnl_from(entry_price, tp1, is_long);
                    last_exit_price = tp1;
                    tp1_hit = true;
                    highest_tp = 1;
                    remaining_pct = 0.0;
                    final_outcome = Some(Outcome::Win { tp_level: 1, exit_price: tp1 });
                    break;
                }
            } else {
                // === PARTIAL CLOSE with dynamic TP/SL (70% / 20% / 10%) ===
                let tp2_val = tp2.unwrap();
                let tp3_val = tp3.unwrap();
                
                if !tp1_hit {
                    let sl_hit = if is_long { candle_low <= current_sl } else { candle_high >= current_sl };
                    let tp1_reached = if is_long { candle_high >= tp1 } else { candle_low <= tp1 };
                    
                    if sl_hit && tp1_reached {
                        let opened_adverse = if is_long { candle_open < entry_price } else { candle_open > entry_price };
                        if opened_adverse {
                            realized_pnl = pnl_from(entry_price, current_sl, is_long);
                            remaining_pct = 0.0;
                            last_exit_price = current_sl;
                            final_outcome = Some(Outcome::Loss { exit_price: current_sl });
                            break;
                        } else {
                            realized_pnl += TP1_CLOSE_PCT * pnl_from(entry_price, tp1, is_long);
                            remaining_pct -= TP1_CLOSE_PCT;
                            tp1_hit = true;
                            highest_tp = 1;
                            current_sl = trail_sl_to_breakeven(entry_price, is_long);
                            last_exit_price = tp1;
                            continue;
                        }
                    } else if sl_hit {
                        realized_pnl = pnl_from(entry_price, current_sl, is_long);
                        remaining_pct = 0.0;
                        last_exit_price = current_sl;
                        final_outcome = Some(Outcome::Loss { exit_price: current_sl });
                        break;
                    } else if tp1_reached {
                        realized_pnl += TP1_CLOSE_PCT * pnl_from(entry_price, tp1, is_long);
                        remaining_pct -= TP1_CLOSE_PCT;
                        tp1_hit = true;
                        highest_tp = 1;
                        current_sl = trail_sl_to_breakeven(entry_price, is_long);
                        last_exit_price = tp1;
                        continue;
                    }
                } else if !tp2_hit {
                    let sl_hit = if is_long { candle_low <= current_sl } else { candle_high >= current_sl };
                    let tp2_reached = if is_long { candle_high >= tp2_val } else { candle_low <= tp2_val };
                    
                    if sl_hit && tp2_reached {
                        let opened_toward_sl = if is_long { candle_open < tp1 } else { candle_open > tp1 };
                        if opened_toward_sl {
                            realized_pnl += remaining_pct * pnl_from(entry_price, current_sl, is_long);
                            remaining_pct = 0.0;
                            last_exit_price = current_sl;
                            final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                            break;
                        } else {
                            realized_pnl += TP2_CLOSE_PCT * pnl_from(entry_price, tp2_val, is_long);
                            remaining_pct -= TP2_CLOSE_PCT;
                            tp2_hit = true;
                            highest_tp = 2;
                            current_sl = trail_sl_to_tp1(tp1, is_long);
                            last_exit_price = tp2_val;
                            continue;
                        }
                    } else if sl_hit {
                        realized_pnl += remaining_pct * pnl_from(entry_price, current_sl, is_long);
                        remaining_pct = 0.0;
                        last_exit_price = current_sl;
                        final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                        break;
                    } else if tp2_reached {
                        realized_pnl += TP2_CLOSE_PCT * pnl_from(entry_price, tp2_val, is_long);
                        remaining_pct -= TP2_CLOSE_PCT;
                        tp2_hit = true;
                        highest_tp = 2;
                        current_sl = trail_sl_to_tp1(tp1, is_long);
                        last_exit_price = tp2_val;
                        continue;
                    }
                } else if !tp3_hit {
                    let sl_hit = if is_long { candle_low <= current_sl } else { candle_high >= current_sl };
                    let tp3_reached = if is_long { candle_high >= tp3_val } else { candle_low <= tp3_val };
                    
                    if sl_hit && tp3_reached {
                        let opened_toward_sl = if is_long { candle_open < tp2_val } else { candle_open > tp2_val };
                        if opened_toward_sl {
                            realized_pnl += remaining_pct * pnl_from(entry_price, current_sl, is_long);
                            remaining_pct = 0.0;
                            last_exit_price = current_sl;
                            final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                            break;
                        } else {
                            realized_pnl += remaining_pct * pnl_from(entry_price, tp3_val, is_long);
                            remaining_pct = 0.0;
                            tp3_hit = true;
                            highest_tp = 3;
                            last_exit_price = tp3_val;
                            final_outcome = Some(Outcome::Win { tp_level: 3, exit_price: tp3_val });
                            break;
                        }
                    } else if sl_hit {
                        realized_pnl += remaining_pct * pnl_from(entry_price, current_sl, is_long);
                        remaining_pct = 0.0;
                        last_exit_price = current_sl;
                        final_outcome = Some(Outcome::Win { tp_level: highest_tp, exit_price: current_sl });
                        break;
                    } else if tp3_reached {
                        realized_pnl += remaining_pct * pnl_from(entry_price, tp3_val, is_long);
                        remaining_pct = 0.0;
                        tp3_hit = true;
                        highest_tp = 3;
                        last_exit_price = tp3_val;
                        final_outcome = Some(Outcome::Win { tp_level: 3, exit_price: tp3_val });
                        break;
                    }
                }
            }
        }
        
        // Handle remaining position at timeout (expired)
        if remaining_pct > 0.0 {
            let last_close = candles.get(max_bars.saturating_sub(1)).map(|c| c.close).unwrap_or(entry_price);
            realized_pnl += remaining_pct * pnl_from(entry_price, last_close, is_long);
            last_exit_price = last_close;
            
            if final_outcome.is_none() {
                if tp1_hit {
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
            let last_price = candles.last().map(|c| c.close).unwrap_or(entry_price);
            Outcome::Expired { last_price }
        });
        
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
            sl_price: signal.sl_price.unwrap_or(0.0),
            tp1_price: signal.tp1_price.unwrap_or(0.0),
            tp2_price: signal.tp2_price,
            tp3_price: signal.tp3_price,
            outcome,
            pnl_pct: realized_pnl,
            exit_price: Some(last_exit_price),
            bars_to_entry: entry_bar_idx as u16,
            bars_to_outcome: bars_used as u16,
            max_favorable,
            max_adverse,
            tp1_hit,
            tp2_hit,
            tp3_hit,
        }))
    }
    
    /// Extract features for a specific bar
    fn extract_features(
        &self,
        signal: &SignalForBacktest,
        candles: &[CandleWithIndicators],
        bar_idx: usize,
    ) -> Result<Vec<f32>> {
        let reason = signal.reason.as_ref().cloned().unwrap_or(serde_json::json!({}));
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
        
        // Add bar-specific features from market data
        let bar = if bar_idx < candles.len() { &candles[bar_idx] } else { &candles[candles.len() - 1] };
        let bar_close_vs_entry = if ep > 0.0 { (bar.close - ep) / ep } else { 0.0 };
        let bar_rsi = bar.rsi;
        let bar_atr = bar.atr;
        
        // Phase 1: Impulse phase detection from bar-level RSI
        let impulse_phase = {
            let r = bar.rsi;
            if signal.side > 0 {
                if r < 55.0 { 0.0 } else if r < 70.0 { 0.33 } else { 0.66 }
            } else if signal.side < 0 {
                if r > 45.0 { 0.0 } else if r > 30.0 { 0.33 } else { 0.66 }
            } else { 0.5 }
        };
        let rsi_slope = ((bar.rsi - 50.0) / 30.0).clamp(-1.0, 1.0);
        let momentum_accel = 0.0f64; // Approximation without MACD per-bar
        let volume_impulse = 0.0f32; // Not available per-bar in backtester
        
        // Phase 2: SR distance from reason JSON
        let sr_get = |k: &str| reason.get(k).and_then(|v| v.as_f64())
            .or_else(|| reason.get("debug").and_then(|d| d.get(k)).and_then(|v| v.as_f64()))
            .unwrap_or(-1.0);
        let nearest_support_dist_atr = sr_get("nearest_support_dist_atr");
        let nearest_resistance_dist_atr = sr_get("nearest_resistance_dist_atr");
        let sr_position = if nearest_support_dist_atr >= 0.0 && nearest_resistance_dist_atr >= 0.0 {
            let total = nearest_support_dist_atr + nearest_resistance_dist_atr;
            if total > 0.0 { nearest_support_dist_atr / total } else { 0.5 }
        } else { 0.5 };
        
        // Phase 3: EMA/BB from reason
        let ema_stack = sr_get("ema_stack").max(-1.0);
        let price_vs_emas = sr_get("price_vs_emas").max(-1.0);
        let bb_position = 0.5f64;
        let bb_width = (bar.atr * 2.0 / bar.close.max(1e-9)).clamp(0.0, 0.1);
        
        let features = vec![
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
            // Bar-specific features (online learning context)
            bar_close_vs_entry as f32,
            bar_rsi as f32,
            bar_atr as f32,
            // Phase 1: Impulse + Momentum features
            impulse_phase as f32,
            momentum_accel as f32,
            rsi_slope as f32,
            volume_impulse,
            // Phase 2: SR distance features
            nearest_support_dist_atr as f32,
            nearest_resistance_dist_atr as f32,
            sr_position as f32,
            // Phase 3: EMA/BB features
            ema_stack as f32,
            price_vs_emas as f32,
            bb_position as f32,
            bb_width as f32,
        ];
        
        Ok(features)
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
            entry_bar_offset: u16::MAX,
            entry_price: 0.0,
            sl_price: signal.sl_price.unwrap_or(0.0),
            tp1_price: signal.tp1_price.unwrap_or(0.0),
            tp2_price: signal.tp2_price,
            tp3_price: signal.tp3_price,
            outcome: Outcome::Expired { last_price: signal.entry_price.unwrap_or(0.0) as f64 },
            pnl_pct: 0.0,
            exit_price: None,
            bars_to_entry: cancel_bar as u16,
            bars_to_outcome: cancel_bar as u16,
            max_favorable: 0.0,
            max_adverse: 0.0,
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
            sl_price: signal.sl_price.unwrap_or(0.0),
            tp1_price: signal.tp1_price.unwrap_or(0.0),
            tp2_price: signal.tp2_price,
            tp3_price: signal.tp3_price,
            outcome: Outcome::Expired { last_price },
            pnl_pct: 0.0,
            exit_price: Some(last_price),
            bars_to_entry: window_bars as u16,
            bars_to_outcome: window_bars as u16,
            max_favorable: 0.0,
            max_adverse: 0.0,
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
            "SELECT c.time, c.open, c.high, c.low, c.close, \
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
                open: row.try_get("open").unwrap_or(0.0),
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

/// Result of entry bar search
enum EntryBarResult {
    /// Agent says ENTER at this bar index
    Enter(usize),
    /// Agent says CANCEL at this bar
    Cancel(usize),
    /// Window expired without entry
    Expired,
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
    pub sl_price: f32,
    pub tp1_price: f32,
    pub tp2_price: Option<f32>,
    pub tp3_price: Option<f32>,
    pub outcome: Outcome,
    pub pnl_pct: f64,
    pub exit_price: Option<f64>,
    pub bars_to_entry: u16,
    pub bars_to_outcome: u16,
    pub max_favorable: f64,
    pub max_adverse: f64,
    pub tp1_hit: bool,
    pub tp2_hit: bool,
    pub tp3_hit: bool,
}

impl EntryAgentTradeResult {
    /// Whether this result represents an actual trade (not cancelled/expired without entry)
    pub fn was_entered(&self) -> bool {
        self.entry_bar_offset != u16::MAX
    }
    
    /// Whether this was cancelled by the agent
    pub fn was_cancelled(&self) -> bool {
        !self.was_entered() && self.bars_to_entry < 100
    }
}

// ============================================================================
// Per-TF statistics for comparison tables
// ============================================================================

/// Per-TF statistics used in comparison
#[derive(Debug, Clone, Default)]
pub struct TfStats {
    pub total: usize,
    pub wins: usize,
    pub losses: usize,
    pub expired: usize,
    pub pnl_sum: f64,
    pub tp1_count: usize,
    pub tp2_count: usize,
    pub tp3_count: usize,
    pub win_pnl_sum: f64,
    pub loss_pnl_sum: f64,
    pub pnls: Vec<f64>,
}

impl TfStats {
    pub fn win_rate(&self) -> f64 {
        let entered = self.wins + self.losses + self.expired;
        if entered > 0 { self.wins as f64 / entered as f64 * 100.0 } else { 0.0 }
    }
    
    pub fn avg_pnl(&self) -> f64 {
        if self.total > 0 { self.pnl_sum / self.total as f64 * 100.0 } else { 0.0 }
    }
    
    pub fn avg_win_pnl(&self) -> f64 {
        if self.wins > 0 { self.win_pnl_sum / self.wins as f64 * 100.0 } else { 0.0 }
    }
    
    pub fn avg_loss_pnl(&self) -> f64 {
        if self.losses > 0 { self.loss_pnl_sum / self.losses as f64 * 100.0 } else { 0.0 }
    }
    
    pub fn sharpe(&self) -> f64 {
        if self.pnls.is_empty() { return 0.0; }
        let mean = self.pnl_sum / self.pnls.len() as f64;
        let var = self.pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / self.pnls.len() as f64;
        if var > 0.0 { mean / var.sqrt() } else { 0.0 }
    }
}

/// Full comparison statistics with per-TF breakdown
#[derive(Debug, Clone, Default)]
pub struct EntryAgentComparison {
    // Baseline aggregate
    pub baseline_total: usize,
    pub baseline_entered: usize,
    pub baseline_wins: usize,
    pub baseline_losses: usize,
    pub baseline_expired: usize,
    pub baseline_avg_pnl: f64,
    pub baseline_by_tf: HashMap<i16, TfStats>,
    
    // Agent aggregate
    pub agent_total: usize,
    pub agent_entered: usize,
    pub agent_wins: usize,
    pub agent_losses: usize,
    pub agent_expired_no_entry: usize,
    pub agent_cancelled: usize,
    pub agent_avg_pnl: f64,
    pub agent_by_tf: HashMap<i16, TfStats>,
}

impl EntryAgentComparison {
    /// Build comparison from baseline BacktestResults and agent EntryAgentTradeResults
    pub fn from_results(
        baseline_results: &[BacktestResult],
        agent_results: &[EntryAgentTradeResult],
    ) -> Self {
        let mut comp = Self::default();
        
        // ---- Baseline ----
        comp.baseline_total = baseline_results.len();
        for r in baseline_results {
            let tf_stats = comp.baseline_by_tf.entry(r.tf_minutes).or_default();
            tf_stats.total += 1;
            tf_stats.pnl_sum += r.pnl_pct;
            tf_stats.pnls.push(r.pnl_pct);
            
            if r.tp1_hit { tf_stats.tp1_count += 1; }
            if r.tp2_hit { tf_stats.tp2_count += 1; }
            if r.tp3_hit { tf_stats.tp3_count += 1; }
            
            match &r.outcome {
                Outcome::Win { .. } => {
                    comp.baseline_wins += 1;
                    comp.baseline_entered += 1;
                    tf_stats.wins += 1;
                    tf_stats.win_pnl_sum += r.pnl_pct;
                }
                Outcome::Loss { .. } => {
                    comp.baseline_losses += 1;
                    comp.baseline_entered += 1;
                    tf_stats.losses += 1;
                    tf_stats.loss_pnl_sum += r.pnl_pct;
                }
                Outcome::Expired { .. } => {
                    comp.baseline_expired += 1;
                    tf_stats.expired += 1;
                }
            }
        }
        comp.baseline_avg_pnl = if comp.baseline_total > 0 {
            baseline_results.iter().map(|r| r.pnl_pct).sum::<f64>() / comp.baseline_total as f64
        } else { 0.0 };
        
        // ---- Agent ----
        comp.agent_total = agent_results.len();
        for r in agent_results {
            let tf_stats = comp.agent_by_tf.entry(r.tf_minutes).or_default();
            tf_stats.total += 1;
            
            if r.was_entered() {
                comp.agent_entered += 1;
                tf_stats.pnl_sum += r.pnl_pct;
                tf_stats.pnls.push(r.pnl_pct);
                
                if r.tp1_hit { tf_stats.tp1_count += 1; }
                if r.tp2_hit { tf_stats.tp2_count += 1; }
                if r.tp3_hit { tf_stats.tp3_count += 1; }
                
                match &r.outcome {
                    Outcome::Win { .. } => {
                        comp.agent_wins += 1;
                        tf_stats.wins += 1;
                        tf_stats.win_pnl_sum += r.pnl_pct;
                    }
                    Outcome::Loss { .. } => {
                        comp.agent_losses += 1;
                        tf_stats.losses += 1;
                        tf_stats.loss_pnl_sum += r.pnl_pct;
                    }
                    Outcome::Expired { .. } => {
                        tf_stats.expired += 1;
                    }
                }
            } else if r.was_cancelled() {
                comp.agent_cancelled += 1;
            } else {
                comp.agent_expired_no_entry += 1;
            }
        }
        comp.agent_avg_pnl = if comp.agent_entered > 0 {
            agent_results.iter().filter(|r| r.was_entered()).map(|r| r.pnl_pct).sum::<f64>() / comp.agent_entered as f64
        } else { 0.0 };
        
        comp
    }
    
    pub fn print(&self) {
        let baseline_win_rate = if self.baseline_entered > 0 {
            self.baseline_wins as f64 / self.baseline_entered as f64 * 100.0
        } else { 0.0 };
        
        let agent_win_rate = if self.agent_entered > 0 {
            self.agent_wins as f64 / self.agent_entered as f64 * 100.0
        } else { 0.0 };
        
        let trades_change = if self.baseline_entered > 0 {
            (self.agent_entered as i64 - self.baseline_entered as i64) as f64 / self.baseline_entered as f64 * 100.0
        } else { 0.0 };
        
        let win_rate_change = agent_win_rate - baseline_win_rate;
        let pnl_change = self.agent_avg_pnl * 100.0 - self.baseline_avg_pnl * 100.0;
        
        // ============================================================
        // HEADER
        // ============================================================
        println!("\n");
        println!("╔══════════════════════════════════════════════════════════════════════════════╗");
        println!("║              ENTRY AGENT COMPARISON: Baseline vs EntryAgent                  ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");
        
        // ============================================================
        // AGGREGATE SUMMARY
        // ============================================================
        println!("┌─────────────────────────────────────────────────────────────────────────────┐");
        println!("│ {:20} │ {:>10} │ {:>11} │ {:>12} │", "Metric", "Baseline", "EntryAgent", "Improvement");
        println!("├──────────────────────┼────────────┼─────────────┼──────────────┤");
        println!("│ {:20} │ {:>10} │ {:>11} │ {:>12} │",
            "Total Signals", self.baseline_total, self.agent_total, "-");
        println!("│ {:20} │ {:>10} │ {:>11} │ {:>+11.0}% │",
            "Entered Trades", self.baseline_entered, self.agent_entered, trades_change);
        println!("│ {:20} │ {:>9.1}% │ {:>10.1}% │ {:>+11.1}% │",
            "Win Rate", baseline_win_rate, agent_win_rate, win_rate_change);
        println!("│ {:20} │ {:>9.4}% │ {:>10.4}% │ {:>+10.4}% │",
            "Avg PnL (entered)", self.baseline_avg_pnl * 100.0, self.agent_avg_pnl * 100.0, pnl_change);
        println!("│ {:20} │ {:>10} │ {:>11} │ {:>12} │",
            "Wins", self.baseline_wins, self.agent_wins, "-");
        println!("│ {:20} │ {:>10} │ {:>11} │ {:>12} │",
            "Losses", self.baseline_losses, self.agent_losses, "-");
        println!("├──────────────────────┴────────────┴─────────────┼──────────────┤");
        println!("│ Cancelled by Agent   │ {:>10}                │              │", self.agent_cancelled);
        println!("│ Expired (no entry)   │ {:>10}                │              │", self.agent_expired_no_entry);
        println!("└─────────────────────────────────────────────────┴──────────────┘\n");
        
        // ============================================================
        // PER-TF BASELINE SUMMARY
        // ============================================================
        println!("======== BASELINE (Immediate Entry) ========");
        println!("{:<6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10} {:>10}",
            "TF", "Total", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL", "Sharpe");
        
        let mut tfs: Vec<i16> = self.baseline_by_tf.keys().copied().collect();
        tfs.sort();
        
        for &tf in &tfs {
            if let Some(stats) = self.baseline_by_tf.get(&tf) {
                let tf_name = tf_to_name(tf);
                println!("{:<6} {:>6} {:>6} {:>6} {:>8} {:>7.1}% {:>9.4}% {:>10.3}",
                    tf_name, stats.total, stats.wins, stats.losses, stats.expired,
                    stats.win_rate(), stats.avg_pnl(), stats.sharpe());
            }
        }
        println!("-------------------------------");
        println!("TOTAL: {} signals, {} wins ({:.1}%), avg PnL: {:.4}%",
            self.baseline_total, self.baseline_wins,
            if self.baseline_entered > 0 { self.baseline_wins as f64 / self.baseline_entered as f64 * 100.0 } else { 0.0 },
            self.baseline_avg_pnl * 100.0);
        println!("==================================\n");
        
        // ============================================================
        // PER-TF ENTRY AGENT SUMMARY
        // ============================================================
        println!("======== ENTRY AGENT (Optimized Entry) ========");
        println!("{:<6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>10} {:>10}",
            "TF", "Enter", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL", "Sharpe");
        
        for &tf in &tfs {
            if let Some(stats) = self.agent_by_tf.get(&tf) {
                let tf_name = tf_to_name(tf);
                let entered = stats.wins + stats.losses + stats.expired;
                println!("{:<6} {:>6} {:>6} {:>6} {:>8} {:>7.1}% {:>9.4}% {:>10.3}",
                    tf_name, entered, stats.wins, stats.losses, stats.expired,
                    stats.win_rate(), stats.avg_pnl(), stats.sharpe());
            }
        }
        println!("-------------------------------");
        println!("TOTAL: {} entered, {} wins ({:.1}%), avg PnL: {:.4}%",
            self.agent_entered, self.agent_wins,
            if self.agent_entered > 0 { self.agent_wins as f64 / self.agent_entered as f64 * 100.0 } else { 0.0 },
            self.agent_avg_pnl * 100.0);
        println!("==================================\n");
        
        // ============================================================
        // PER-TF IMPROVEMENT DELTA
        // ============================================================
        println!("======== IMPROVEMENT DELTA (Agent - Baseline) ========");
        println!("{:<6} {:>10} {:>10} {:>10} {:>12}",
            "TF", "ΔWinRate", "ΔAvgPnL", "ΔAvgWin", "ΔAvgLoss");
        
        for &tf in &tfs {
            let b = self.baseline_by_tf.get(&tf);
            let a = self.agent_by_tf.get(&tf);
            if let (Some(b), Some(a)) = (b, a) {
                let tf_name = tf_to_name(tf);
                let dwr = a.win_rate() - b.win_rate();
                let dpnl = a.avg_pnl() - b.avg_pnl();
                let dwp = a.avg_win_pnl() - b.avg_win_pnl();
                let dlp = a.avg_loss_pnl() - b.avg_loss_pnl();
                println!("{:<6} {:>+9.1}% {:>+9.4}% {:>+9.4}% {:>+11.4}%",
                    tf_name, dwr, dpnl, dwp, dlp);
            }
        }
        println!("=====================================================\n");
        
        // ============================================================
        // TP LEVEL DISTRIBUTION — BASELINE
        // ============================================================
        println!("======== TP LEVEL DISTRIBUTION — BASELINE ========");
        println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12}",
            "TF", "TP1_hit", "TP2_hit", "TP3_hit", "SL_only", "AvgPnL+", "AvgPnL-");
        for &tf in &tfs {
            if let Some(stats) = self.baseline_by_tf.get(&tf) {
                let tf_name = tf_to_name(tf);
                println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>9.4}% {:>10.4}%",
                    tf_name, stats.tp1_count, stats.tp2_count, stats.tp3_count,
                    stats.losses,
                    stats.avg_win_pnl(), stats.avg_loss_pnl());
            }
        }
        println!("=====================================================\n");
        
        // ============================================================
        // TP LEVEL DISTRIBUTION — ENTRY AGENT
        // ============================================================
        println!("======== TP LEVEL DISTRIBUTION — ENTRY AGENT ========");
        println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12}",
            "TF", "TP1_hit", "TP2_hit", "TP3_hit", "SL_only", "AvgPnL+", "AvgPnL-");
        for &tf in &tfs {
            if let Some(stats) = self.agent_by_tf.get(&tf) {
                let tf_name = tf_to_name(tf);
                println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>9.4}% {:>10.4}%",
                    tf_name, stats.tp1_count, stats.tp2_count, stats.tp3_count,
                    stats.losses,
                    stats.avg_win_pnl(), stats.avg_loss_pnl());
            }
        }
        println!("=====================================================\n");
    }
}

/// Helper: convert tf_minutes to human-readable name
fn tf_to_name(tf: i16) -> &'static str {
    match tf { 1 => "1m", 5 => "5m", 15 => "15m", 60 => "1h", 240 => "4h", _ => "??" }
}
