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

use anyhow::Result;
use serde_json::{json, Value};
use sqlx::PgPool;
use tokio::sync::mpsc;
use common::Symbol;

use crate::predictors::types::{PredictionAspect, CalcSource};
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
    bulk_sender: mpsc::Sender<database_lib::PersistRecord>,
    prediction_rx: mpsc::UnboundedReceiver<TradeSignalInput>,
}

impl TradeSignalStage {
    pub fn new(
        db_pool: PgPool,
        market_params_calc: MarketParamsCalculator,
        bulk_sender: mpsc::Sender<database_lib::PersistRecord>,
        prediction_rx: mpsc::UnboundedReceiver<TradeSignalInput>,
        min_score: f64,  // Accept min_score as parameter
    ) -> Self {
        // Log for debugging
        tracing::info!("Initializing TradeSignalStage with min_score: {}", min_score);
        
        Self {
            db_pool,
            market_params_calc,
            final_scorer: FinalScorer::new(min_score),
            trade_calc: TradeSignalCalculator::new(min_score),
            bulk_sender,
            prediction_rx,
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

        // 3. If signal was produced, persist it
        if let Some(signal) = signal_opt {
            let record = trade_signal_to_persist_record(&signal, &input);

            if let Err(e) = self.bulk_sender.send(record).await {
                tracing::error!(
                    target: "trade_signal_stage",
                    "Failed to send TradeSignal to BulkPersistor: {}", e
                );
                return Ok(());
            }

            *signals_produced += 1;

            tracing::info!(
                target: "trade_signal_stage",
                "Trade signal produced: {} {} tf={} side={} score={:.4} entry={:.6} sl={:.6} tp1={:.6}",
                signal.symbol, signal.time, signal.tf_minutes,
                signal.side, signal.final_score, signal.entry,
                signal.stop_loss, signal.tp1
            );
        }

        Ok(())
    }
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
