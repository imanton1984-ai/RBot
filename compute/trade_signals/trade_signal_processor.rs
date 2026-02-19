// compute/trade_signals/trade_signal_processor.rs
//
// TradeSignalStage — final pipeline stage that converts predictions into trade signals.
//
// Pipeline flow:
//   FeatureSnapshot → PredictorsPipeline → predictions → TradeSignalStage → trade.final_signals
//
// This stage:
// 1. Receives completed prediction batches from PredictorsPipeline
// 2. Computes BTC MarketParams (with caching via MarketParamsCalculator)
// 3. Runs TradeSignalCalculator (which internally uses FinalScorer)
// 4. If score >= 0.96 and signal produced → sends PersistRecord::TradeSignal to BulkPersistor
// 5. Entry Agent annotates signal with entry_decision (ENTER/WAIT/CANCEL) for execution layer

use anyhow::Result;
use serde_json::{json, Value};
use sqlx::PgPool;
use tokio::sync::mpsc;
use common::Symbol;

use crate::predictors::types::{PredictionAspect, CalcSource};
use crate::predictors::entry_policy::{EntryAgent, EntryAgentConfig, EntryDecision};
use crate::predictors::ml::model_manager::ModelManager;
use crate::predictors::signal_quality::heuristic_scorer::HeuristicQualityScorer;
use crate::predictors::signal_quality::types::SignalFeatures;
pub use crate::predictors::pipeline::TradeSignalInput;

use super::final_score::FinalScorer;
use super::market_params_calculator::MarketParamsCalculator;
use super::trade_signal_calculator::{TradeSignalCalculator, TradeSignal};

/// Final pipeline stage that produces trade signals from prediction data.
pub struct TradeSignalStage {
    db_pool: PgPool,
    market_params_calc: MarketParamsCalculator,
    final_scorer: FinalScorer,
    trade_calc: TradeSignalCalculator,
    quality_scorer: HeuristicQualityScorer,
    entry_agent: Option<EntryAgent>,
    entry_model_manager: Option<ModelManager>,
    use_entry_agent_gpu: bool,
    bulk_sender: mpsc::Sender<database_lib::PersistRecord>,
    prediction_rx: mpsc::UnboundedReceiver<TradeSignalInput>,
}

impl TradeSignalStage {
    pub fn new(
        db_pool: PgPool,
        market_params_calc: MarketParamsCalculator,
        bulk_sender: mpsc::Sender<database_lib::PersistRecord>,
        prediction_rx: mpsc::UnboundedReceiver<TradeSignalInput>,
        min_score: f64,
    ) -> Self {
        tracing::info!("Initializing TradeSignalStage with min_score: {}", min_score);
        
        // Try to load Entry Agent models
        let use_gpu = std::env::var("ENTRY_AGENT_USE_GPU")
            .unwrap_or_default()
            .to_lowercase()
            .parse::<bool>()
            .unwrap_or(false);
        
        let (entry_agent, entry_mm) = Self::try_load_entry_agent(use_gpu);
        
        Self {
            db_pool,
            market_params_calc,
            final_scorer: FinalScorer::new(min_score),
            trade_calc: TradeSignalCalculator::new(min_score),
            quality_scorer: HeuristicQualityScorer::new(),
            entry_agent,
            entry_model_manager: entry_mm,
            use_entry_agent_gpu: use_gpu,
            bulk_sender,
            prediction_rx,
        }
    }
    
    /// Attempt to load Entry Agent models. Returns (agent, model_manager) if successful.
    fn try_load_entry_agent(use_gpu: bool) -> (Option<EntryAgent>, Option<ModelManager>) {
        let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| {
            if std::path::Path::new("models").exists() {
                "models".to_string()
            } else if std::path::Path::new("../models").exists() {
                "../models".to_string()
            } else {
                "/home/anton/Desktop/Rust_trader/models".to_string()
            }
        });
        
        let mut mm = ModelManager::new(use_gpu);
        let timeframes = vec![1, 5, 15, 60, 240];
        
        let enter_template = format!("{}/entry_enter_v1_tf{{tf}}.ubj", models_dir);
        let cancel_template = format!("{}/entry_cancel_v1_tf{{tf}}.ubj", models_dir);
        
        let enter_ok = mm.load_models_for_timeframes("entry_enter", &enter_template, &timeframes, use_gpu).is_ok();
        let cancel_ok = mm.load_models_for_timeframes("entry_cancel", &cancel_template, &timeframes, use_gpu).is_ok();
        
        if enter_ok && cancel_ok && mm.has_model("entry_enter_tf1") && mm.has_model("entry_cancel_tf1") {
            tracing::info!(target: "trade_signal_stage", "Entry Agent models loaded — will annotate signals with entry timing");
            let config = EntryAgentConfig {
                enter_threshold: 0.55,
                cancel_threshold: 0.50,
                min_margin: 0.15,
                default_window_bars: 10,
            };
            (Some(EntryAgent::new(config)), Some(mm))
        } else {
            tracing::info!(target: "trade_signal_stage", "Entry Agent models not found — signals will use immediate entry");
            (None, None)
        }
    }

    /// Main loop: receive prediction inputs, score, and produce trade signals.
    pub async fn run(mut self) -> Result<()> {
        tracing::info!(target: "trade_signal_stage", "TradeSignalStage started");
        let mut signals_produced: u64 = 0;
        let mut inputs_processed: u64 = 0;

        while let Some(input) = self.prediction_rx.recv().await {
            inputs_processed += 1;

            if let Err(e) = self.process_input(input, &mut signals_produced).await {
                tracing::error!(
                    target: "trade_signal_stage",
                    "Error processing trade signal input: {}", e
                );
            }

            if inputs_processed % 1000 == 0 {
                tracing::info!(
                    target: "trade_signal_stage",
                    "TradeSignalStage: processed {} inputs, produced {} signals",
                    inputs_processed, signals_produced
                );
            }
        }

        tracing::info!(
            target: "trade_signal_stage",
            "TradeSignalStage shutting down: processed {} inputs, produced {} signals",
            inputs_processed, signals_produced
        );

        Ok(())
    }

    async fn process_input(
        &self,
        input: TradeSignalInput,
        signals_produced: &mut u64,
    ) -> Result<()> {
        if input.predictions.is_empty() {
            return Ok(());
        }

        // Diagnostic: sample-log inputs so we can verify predictions reach the stage
        static INPUT_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let cnt = INPUT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if cnt % 10000 == 0 {
            tracing::info!(
                target: "trade_signal_stage",
                "DIAG input #{}: {} tf={} close={:.4} preds={} side_summary={}",
                cnt, input.symbol, input.tf_minutes, input.close_price,
                input.predictions.len(),
                input.raw_signals_summary.get("side").and_then(|v| v.as_i64()).unwrap_or(0)
            );
        }

        // 1. Compute BTC market params (cached, graceful fallback)
        let market_params = self
            .market_params_calc
            .get_or_compute(&self.db_pool, input.timestamp)
            .await?;

        // 2. Run TradeSignalCalculator (which internally uses FinalScorer)
        let symbol = Symbol::from(input.symbol.clone());
        let signal_opt = self
            .trade_calc
            .build_trade_signal(
                &self.db_pool,
                &self.final_scorer,
                Some(&market_params),
                input.timestamp,
                input.time_ms,
                input.symbol_id,
                &symbol,
                input.tf_minutes,
                input.close_price,
                &input.raw_signals_summary,
                &input.predictions,
            )
            .await?;

        // 3. If signal was produced, apply quality scoring + entry agent + persist
        if let Some(mut signal) = signal_opt {
            // Apply Signal Quality Scorer
            let features = SignalFeatures::from_reason_json(
                &signal.breakdown_json,
                &signal.symbol,
                signal.tf_minutes,
                signal.side as i16,
                signal.entry,
                signal.stop_loss,
                signal.tp1,
                signal.tp2,
                signal.tp3,
                None,
                None,
                None, None, None, None, None, None,
            );

            let quality = self.quality_scorer.score(&features);
            let original_score = signal.final_score;
            let win_prob = quality.breakdown.combined_quality.clamp(0.0, 0.99);

            // Add quality info to breakdown
            if let Value::Object(ref mut obj) = signal.breakdown_json {
                obj.insert("original_score".to_string(), json!(original_score));
                obj.insert("win_prob".to_string(), json!(win_prob));
                obj.insert("quality_grade".to_string(), json!(quality.grade.as_str()));
                obj.insert("quality_multiplier".to_string(), json!(quality.quality_multiplier));
                obj.insert("combined_quality".to_string(), json!(quality.breakdown.combined_quality));
                obj.insert("heuristic_quality".to_string(), json!(quality.breakdown.heuristic_quality));
                obj.insert("ml_quality".to_string(), json!(quality.breakdown.ml_quality));
            }

            // ── Entry Agent: annotate signal with entry timing decision ──
            // Runs on the CURRENT bar features; execution layer uses this to decide entry timing.
            let entry_decision_str = if let (Some(agent), Some(mm)) = (&self.entry_agent, &self.entry_model_manager) {
                let entry_features = build_entry_agent_features(&signal, &input);
                match agent.decide(
                    mm,
                    signal.tf_minutes as i32,
                    entry_features,
                    0,  // elapsed = 0 (first bar)
                    EntryAgent::get_window_bars_for_tf(signal.tf_minutes as i32) as u16,
                    self.use_entry_agent_gpu,
                ) {
                    Ok(decision) => {
                        let (decision_str, confidence) = match &decision {
                            EntryDecision::Enter { confidence } => ("ENTER", *confidence),
                            EntryDecision::Wait { confidence } => ("WAIT", *confidence),
                            EntryDecision::Cancel { confidence } => ("CANCEL", *confidence),
                        };
                        
                        if let Value::Object(ref mut obj) = signal.breakdown_json {
                            obj.insert("entry_decision".to_string(), json!(decision_str));
                            obj.insert("entry_confidence".to_string(), json!(confidence));
                            obj.insert("entry_window_bars".to_string(),
                                json!(EntryAgent::get_window_bars_for_tf(signal.tf_minutes as i32)));
                        }
                        
                        decision_str.to_string()
                    }
                    Err(e) => {
                        tracing::debug!(target: "trade_signal_stage",
                            "Entry Agent error for {} tf={}: {} — defaulting to ENTER",
                            signal.symbol, signal.tf_minutes, e);
                        "ENTER".to_string()
                    }
                }
            } else {
                "ENTER".to_string()
            };

            let record = trade_signal_to_persist_record(&signal, &input);

            if let Err(e) = self.bulk_sender.send(record).await {
                tracing::error!(
                    target: "trade_signal_stage",
                    "Failed to send TradeSignal to BulkPersistor: {}", e
                );
                return Ok(());
            }

            *signals_produced += 1;

            tracing::warn!(
                target: "trade_signal_stage",
                "Trade signal: {} {} tf={} side={} score={:.4} quality={:.2} entry_decision={} entry={:.6} sl={:.6} tp1={:.6}",
                signal.symbol, signal.time, signal.tf_minutes,
                signal.side, signal.final_score,
                quality.breakdown.combined_quality, entry_decision_str,
                signal.entry, signal.stop_loss, signal.tp1
            );
        }

        Ok(())
    }
}

/// Build feature vector for Entry Agent from signal + prediction context.
fn build_entry_agent_features(signal: &TradeSignal, input: &TradeSignalInput) -> Vec<f32> {
    let reason = &signal.breakdown_json;
    let get = |k: &str| reason.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
    let debug_get = |k: &str| {
        reason.get("debug").and_then(|d| d.get(k)).and_then(|v| v.as_f64())
            .or_else(|| reason.get(k).and_then(|v| v.as_f64()))
            .unwrap_or(0.0)
    };
    
    let ep = signal.entry;
    let sl_pct = if ep > 0.0 { (signal.stop_loss - ep).abs() / ep } else { 0.0 };
    let tp1_pct = if ep > 0.0 { (signal.tp1 - ep).abs() / ep } else { 0.0 };
    let tp2_pct = if ep > 0.0 { (signal.tp2 - ep).abs() / ep } else { 0.0 };
    let tp3_pct = if ep > 0.0 { (signal.tp3 - ep).abs() / ep } else { 0.0 };
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
    
    let mut ml_s = 0.0f64;
    let mut heur_s = 0.0f64;
    for p in &input.predictions {
        if p.aspect == PredictionAspect::PriceTarget {
            match p.calc_source {
                CalcSource::Ml => { ml_s = p.score_norm as f64; }
                CalcSource::Hard => { heur_s = p.score_norm as f64; }
            }
        }
    }
    let ml_heur_gap = (ml_s - heur_s).abs();
    let score_per_risk = if sl_pct > 0.0 { signal.final_score / sl_pct } else { 0.0 };
    
    let bounce_prob = input.predictions.iter()
        .find(|p| p.aspect == PredictionAspect::LevelBounce)
        .map(|p| p.value as f32).unwrap_or(0.0);
    let bounce_score = input.predictions.iter()
        .find(|p| p.aspect == PredictionAspect::LevelBounce)
        .map(|p| p.score_norm).unwrap_or(0.0);
    let breakout_prob = input.predictions.iter()
        .find(|p| p.aspect == PredictionAspect::LevelBreakout)
        .map(|p| p.value as f32).unwrap_or(0.0);
    let breakout_score = input.predictions.iter()
        .find(|p| p.aspect == PredictionAspect::LevelBreakout)
        .map(|p| p.score_norm).unwrap_or(0.0);
    let price10_score = input.predictions.iter()
        .find(|p| p.aspect == PredictionAspect::PriceTarget)
        .map(|p| p.score_norm).unwrap_or(0.0);
    
    let atr_pct = get("atr_pct");
    let market_factor = get("market_factor");
    let score_factor = get("score_factor");
    
    // Phase 1-3: Extract new impulse/SR features from raw_signals_summary
    let raw_get = |k: &str| input.raw_signals_summary.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
    
    vec![
        signal.tf_minutes as f32,
        signal.side as f32,
        signal.final_score as f32,
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
        price10_score,
        bounce_prob,
        bounce_score,
        breakout_prob,
        breakout_score,
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
        // Phase 1: Impulse + Momentum features (from raw_signals_summary)
        raw_get("impulse_phase") as f32,
        raw_get("momentum_acceleration") as f32,
        raw_get("rsi_slope") as f32,
        raw_get("volume_impulse_confirm") as f32,
        // Phase 2: SR distance features
        raw_get("nearest_support_dist_atr") as f32,
        raw_get("nearest_resistance_dist_atr") as f32,
        raw_get("sr_position") as f32,
        // Phase 3: EMA/BB features
        raw_get("ema_stack") as f32,
        raw_get("price_vs_emas") as f32,
        raw_get("bb_position") as f32,
        raw_get("bb_width") as f32,
    ]
}

/// Build a minimal raw_signals_summary from FeatureView indicators.
/// This is a stopgap until the raw signals pipeline provides a real summary.
pub fn build_raw_signals_summary_from_indicators(
    close: f32,
    atr: f32,
    rsi: f32,
    macd_hist: f32,
    trend_short: f32,
    volume_spike: f32,
    adx: f32,
) -> Value {
    let atr_pct = if close > 0.0 { (atr as f64) / (close as f64) } else { 0.0 };

    // Normalize indicators to 0..1 scores
    let trend_strength = ((adx as f64 - 15.0) / 25.0).clamp(0.0, 1.0);

    let momentum_strength = {
        let rsi_dev = ((rsi as f64) - 50.0).abs() / 50.0;
        let macd_norm = (macd_hist as f64).abs().min(1.0);
        (0.6 * rsi_dev + 0.4 * macd_norm).clamp(0.0, 1.0)
    };

    let volatility_regime = ((atr_pct - 0.006) / 0.034).clamp(0.0, 1.0);
    let volume_spike_score = (volume_spike as f64).clamp(0.0, 1.0);

    // Determine dominant side from momentum
    let dominant_side: i64 = if rsi > 55.0 && macd_hist > 0.0 { 1 }
        else if rsi < 45.0 && macd_hist < 0.0 { -1 }
        else { 0 };

    // "Best" scores for raw signal buckets (derived from indicators)
    let best_momentum = momentum_strength;
    let best_volume = volume_spike_score;
    // Derive best_levels from confluence instead of hardcoding 0.5
    let best_levels = ((trend_strength * 0.5 + momentum_strength * 0.3 + volume_spike_score * 0.2) * 1.1).clamp(0.0, 1.0);
    let best_raw = ((trend_strength + momentum_strength) / 2.0).clamp(0.0, 1.0);

    json!({
        "atr": atr,
        "atr_pct": atr_pct,
        "side": dominant_side,
        "dominant_side": dominant_side,
        "trend_strength": trend_strength,
        "momentum_strength": momentum_strength,
        "volatility_regime": volatility_regime,
        "volume_spike_score": volume_spike_score,
        "best_raw_signal_score": best_raw,
        "best_levels_score": best_levels,
        "best_momentum_score": best_momentum,
        "best_volume_score": best_volume,
        // All indicators are computed by our pipeline; feature_coverage should be 1.0.
        // Previous value 0.7 crushed coverage_score in FinalScorer.
        "feature_coverage": 1.0,
        "trend_short": trend_short,
    })
}

/// Convert a TradeSignal into a PersistRecord::TradeSignal for bulk persistence.
fn trade_signal_to_persist_record(
    signal: &TradeSignal,
    input: &TradeSignalInput,
) -> database_lib::PersistRecord {
    // Extract prediction aspect scores for the DB columns
    let mut price10_target: Option<f64> = None;
    let mut price10_score: Option<f32> = None;
    let mut bounce_prob: Option<f32> = None;
    let mut bounce_score: Option<f32> = None;
    let mut breakout_prob: Option<f32> = None;
    let mut breakout_score: Option<f32> = None;
    let mut ml_score: Option<f32> = None;
    let mut heur_score: Option<f32> = None;

    for p in &input.predictions {
        match p.aspect {
            PredictionAspect::PriceTarget => {
                // After consensus fusion, a prediction may have calc_source of the "winner".
                // Try to extract original scores from details_json if consensus was applied.
                let (orig_ml, orig_hard) = extract_consensus_scores(&p.details_json);

                match p.calc_source {
                    CalcSource::Ml => {
                        price10_target = Some(p.value);
                        price10_score = Some(p.score_norm);
                        ml_score = Some(p.score_norm);
                        // If consensus was applied, also set heur from original
                        if let Some(hs) = orig_hard {
                            heur_score = Some(hs as f32);
                        }
                    }
                    CalcSource::Hard => {
                        if price10_target.is_none() {
                            price10_target = Some(p.value);
                            price10_score = Some(p.score_norm);
                        }
                        heur_score = Some(p.score_norm);
                        // If consensus was applied, also set ml from original
                        if let Some(ms) = orig_ml {
                            ml_score = Some(ms as f32);
                        }
                    }
                }
            }
            PredictionAspect::LevelBounce => {
                bounce_prob = Some(p.value as f32);
                bounce_score = Some(p.score_norm);
            }
            PredictionAspect::LevelBreakout => {
                breakout_prob = Some(p.value as f32);
                breakout_score = Some(p.score_norm);
            }
        }
    }

    database_lib::PersistRecord::TradeSignal {
        symbol: Symbol::from(signal.symbol.clone()),
        timeframe: signal.tf_minutes,
        time_ms: signal.time_ms,
        side: signal.side as i16,
        final_score: signal.final_score as f32,
        ml_score,
        heur_score,
        entry_price: Some(signal.entry as f32),
        sl_price: Some(signal.stop_loss as f32),
        tp1_price: Some(signal.tp1 as f32),
        tp2_price: Some(signal.tp2 as f32),
        tp3_price: Some(signal.tp3 as f32),
        reason: Some(signal.breakdown_json.clone()),
        price10_target,
        price10_score,
        bounce_prob,
        bounce_score,
        breakout_prob,
        breakout_score,
    }
}

/// Extract original ML and hard scores from consensus details_json
/// (when consensus engine fused two predictors, it stores original scores)
fn extract_consensus_scores(details: &Option<serde_json::Value>) -> (Option<f64>, Option<f64>) {
    let details = match details {
        Some(d) => d,
        None => return (None, None),
    };

    let consensus = details.get("consensus_applied").and_then(|v| v.as_bool()).unwrap_or(false);
    if !consensus {
        return (None, None);
    }

    let ml = details.get("original_ml_score").and_then(|v| v.as_f64());
    let hard = details.get("original_hard_score").and_then(|v| v.as_f64());
    (ml, hard)
}
