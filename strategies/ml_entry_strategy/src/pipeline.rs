// strategies/ml_entry_strategy/src/pipeline.rs
//
// Pipeline for Super Entry Strategy (v2 — ML-only, Direction v4)
//
// Orchestrates the full flow:
//   1. Load candles + indicators from DB
//   2. Build feature vectors:
//      a. 128 features for P(super) model
//      b. v4 pattern features for Direction model (CNN-like sliding window)
//   3. Run model inference
//   4. Score predictions (apply P(super) threshold + per-TF direction confidence gate)
//   5. Generate trade signals
//
// v2 changes (ML-only):
//   - Direction v4 pattern model replaces v3 (32-feature regression)
//   - Removed: heuristic cross-TF filter, danger zone filter, overheated filter
//   - Only P(super) and direction confidence affect signal generation
//   - Per-TF confidence thresholds: 15m≥0.75, 1h≥0.70, 4h≥0.65
//
// Can be used both for:
//   - Historical backtesting (batch processing all candles)
//   - Real-time inference (single candle at a time)

use anyhow::Result;
use sqlx::PgPool;
use tracing::{info, warn, debug};

use crate::config::SuperEntryConfig;
use crate::dataset::{CandleWithIndicators, fetch_candles_with_indicators, fetch_active_symbols};
use crate::model::{SuperEntryModelManager, SuperEntryPrediction};
use crate::scorer::{SuperEntryScorer, SuperEntryDecision};
use crate::signal_generator::{SignalGenerator, SuperEntrySignal};
use crate::direction::DirectionConfig;
use crate::direction::features::{
    compute_direction_v3_features, resolve_btc_context, resolve_htf_context,
    compute_pattern_features, DIRECTION_V3_FEATURE_COUNT,
};

/// Result of processing a single candle through the pipeline
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The generated signal (if any)
    pub signal: Option<SuperEntrySignal>,
    /// The raw prediction from the model
    pub prediction: Option<SuperEntryPrediction>,
    /// The scorer's decision
    pub decision: Option<SuperEntryDecision>,
    /// Index of the candle in the series
    pub candle_index: usize,
    /// DEPRECATED: kept for backward compat, always None
    pub agrees_count: Option<usize>,
}

/// Super Entry Pipeline
///
/// Full pipeline: DB → features → model → scorer → signal
pub struct SuperEntryPipeline {
    config: SuperEntryConfig,
    model_manager: SuperEntryModelManager,
    scorer: SuperEntryScorer,
    signal_generator: SignalGenerator,
    dir_config: DirectionConfig,
}

impl SuperEntryPipeline {
    /// Create a new pipeline with loaded models.
    pub fn new(config: SuperEntryConfig, use_gpu: bool) -> Result<Self> {
        let model_manager = SuperEntryModelManager::new(config.clone(), use_gpu)?;
        let scorer = SuperEntryScorer::from_strategy_config(&config);
        let signal_generator = SignalGenerator::new(config.clone());
        let dir_config = DirectionConfig::from_env();

        Ok(Self {
            config,
            model_manager,
            scorer,
            signal_generator,
            dir_config,
        })
    }

    /// Check if the pipeline has any models loaded
    pub fn has_models(&self) -> bool {
        self.model_manager.has_models()
    }

    /// Process a series of candles for a given symbol/timeframe.
    ///
    /// Skips the first `warmup_bars` candles (used for indicator warmup).
    pub fn process_candles(
        &self,
        candles: &[CandleWithIndicators],
        tf_minutes: i32,
        use_gpu: bool,
    ) -> Result<Vec<PipelineResult>> {
        self.process_candles_with_htf(candles, tf_minutes, use_gpu, None)
    }

    /// Process candles with optional HTF (Higher Timeframe) context.
    ///
    /// When `htf_candles` is provided:
    ///   - HTF features in 128-set are populated
    ///   - Direction v3 htf_supertrend_dir is populated (fallback only)
    pub fn process_candles_with_htf(
        &self,
        candles: &[CandleWithIndicators],
        tf_minutes: i32,
        use_gpu: bool,
        htf_candles: Option<&[CandleWithIndicators]>,
    ) -> Result<Vec<PipelineResult>> {
        self.process_candles_full(candles, tf_minutes, use_gpu, htf_candles, None)
    }

    /// Process candles with HTF and BTC context.
    pub fn process_candles_full(
        &self,
        candles: &[CandleWithIndicators],
        tf_minutes: i32,
        use_gpu: bool,
        htf_candles: Option<&[CandleWithIndicators]>,
        btc_candles: Option<&[CandleWithIndicators]>,
    ) -> Result<Vec<PipelineResult>> {
        if !self.model_manager.has_model_for_tf(tf_minutes) {
            warn!("No model for TF {}m, skipping", tf_minutes);
            return Ok(Vec::new());
        }

        let warmup = self.config.warmup_bars;
        if candles.len() <= warmup {
            debug!("Not enough candles for TF {}m (need > {})", tf_minutes, warmup);
            return Ok(Vec::new());
        }

        let n = candles.len();
        let process_start = warmup;
        let batch_size = n - process_start;

        let has_dir_v4 = self.model_manager.has_direction_v4_for_tf(tf_minutes);
        let has_dir_v3 = self.model_manager.has_direction_v3_for_tf(tf_minutes);

        // Build feature matrix for batch inference — zero-copy: write f32 directly
        let ncol = crate::config::total_feature_count();
        let mut features_flat: Vec<f32> = Vec::with_capacity(batch_size * ncol);

        // Direction v4 pattern features — if v4 model is loaded
        let dir_v4_ncol = self.dir_config.total_features();
        let mut dir_v4_flat: Vec<f32> = if has_dir_v4 {
            Vec::with_capacity(batch_size * dir_v4_ncol)
        } else {
            Vec::new()
        };

        // Direction v3 features (32 × batch_size) — fallback if v4 not available
        let mut dir_v3_flat: Vec<f32> = if !has_dir_v4 && has_dir_v3 {
            Vec::with_capacity(batch_size * DIRECTION_V3_FEATURE_COUNT)
        } else {
            Vec::new()
        };

        for i in process_start..n {
            // ═══ 128 super_entry features ═══
            let c = &candles[i];
            // Raw indicators (33 features) — matches INDICATOR_FEATURES order
            features_flat.push(c.rsi as f32);
            features_flat.push(c.cci as f32);
            features_flat.push(c.stoch_k as f32);
            features_flat.push(c.stoch_d as f32);
            features_flat.push(c.williams as f32);
            features_flat.push(c.macd as f32);
            features_flat.push(c.macd_signal as f32);
            features_flat.push(c.macd_hist as f32);
            features_flat.push(c.adx as f32);
            features_flat.push(c.sma as f32);
            features_flat.push(c.ema_20 as f32);
            features_flat.push(c.ema_50 as f32);
            features_flat.push(c.ema_200 as f32);
            features_flat.push(c.bb_upper as f32);
            features_flat.push(c.bb_mid as f32);
            features_flat.push(c.bb_lower as f32);
            features_flat.push(c.atr as f32);
            features_flat.push(c.obv as f32);
            features_flat.push(c.vwap as f32);
            features_flat.push(c.volume_spike as f32);
            features_flat.push(c.trend as f32);
            features_flat.push(c.trend_short as f32);
            features_flat.push(c.poc as f32);
            features_flat.push(c.alligator_jaw as f32);
            features_flat.push(c.alligator_teeth as f32);
            features_flat.push(c.alligator_lips as f32);
            features_flat.push(c.mfi as f32);
            features_flat.push(c.fibo_pivot as f32);
            features_flat.push(c.fibo_r1 as f32);
            features_flat.push(c.fibo_s1 as f32);
            features_flat.push(c.supertrend as f32);
            features_flat.push(c.supertrend_dir as f32);
            features_flat.push(c.cmf as f32);

            // Derived features (19 features)
            let close = c.close;
            let safe_div = |a: f64, b: f64| -> f32 {
                if b.abs() > 1e-12 { (a / b) as f32 } else { 0.0f32 }
            };
            features_flat.push((c.rsi / 100.0) as f32);
            features_flat.push((c.cci / 200.0) as f32);
            features_flat.push((c.stoch_k / 100.0) as f32);
            features_flat.push(((c.williams + 100.0) / 100.0) as f32);
            let bb_range = c.bb_upper - c.bb_lower;
            features_flat.push(if bb_range.abs() > 1e-12 {
                ((close - c.bb_lower) / bb_range) as f32
            } else { 0.5f32 });
            features_flat.push(safe_div(bb_range, close) * 100.0);
            features_flat.push(safe_div(c.atr, close) * 100.0);
            features_flat.push(safe_div(close - c.sma, close) * 100.0);
            features_flat.push(safe_div(close - c.ema_20, close) * 100.0);
            features_flat.push(safe_div(close - c.ema_50, close) * 100.0);
            features_flat.push(safe_div(close - c.ema_200, close) * 100.0);
            features_flat.push(safe_div(close - c.vwap, close) * 100.0);
            features_flat.push(safe_div(c.macd_hist, close) * 1000.0);
            features_flat.push(0.0f32); // obv_change_pct (N/A)
            features_flat.push(if c.volume_spike > 2.0 { 1.0f32 } else { 0.0f32 });
            features_flat.push((c.mfi / 100.0) as f32);
            features_flat.push(safe_div(close - c.fibo_pivot, close) * 100.0);
            features_flat.push(safe_div(close - c.supertrend, close) * 100.0);
            features_flat.push(safe_div(c.alligator_jaw - c.alligator_lips, close) * 100.0);

            // Dynamic temporal features (76 features)
            let htf_candle_ref = htf_candles.and_then(|htf| {
                let target_time = candles[i].time;
                let idx = htf.partition_point(|c| c.time <= target_time);
                if idx > 0 { Some(&htf[idx - 1]) } else { None }
            });
            let dyn_feats = crate::dataset::compute_dynamic_features_with_htf(candles, i, htf_candle_ref);
            for v in &dyn_feats {
                features_flat.push(*v as f32);
            }

            // ═══ Direction v4 pattern features (CNN-like sliding window) ═══
            if has_dir_v4 {
                match compute_pattern_features(candles, i, &self.dir_config) {
                    Some(feats) => {
                        for v in &feats {
                            dir_v4_flat.push(*v as f32);
                        }
                    }
                    None => {
                        // Not enough history for pattern window — pad with zeros
                        for _ in 0..dir_v4_ncol {
                            dir_v4_flat.push(0.0f32);
                        }
                    }
                }
            }

            // ═══ Direction v3 features (32) — fallback if v4 not available ═══
            if !has_dir_v4 && has_dir_v3 {
                let btc_ctx = btc_candles
                    .and_then(|btc| resolve_btc_context(btc, candles[i].time));
                let htf_ctx = htf_candles
                    .and_then(|htf| resolve_htf_context(htf, candles[i].time));

                let dir_feats = compute_direction_v3_features(
                    candles, i,
                    btc_ctx.as_ref(),
                    htf_ctx.as_ref(),
                );
                for v in &dir_feats {
                    dir_v3_flat.push(*v as f32);
                }
            }
        }

        // Batch inference
        let dir_v3_ref = if !has_dir_v4 && has_dir_v3 && !dir_v3_flat.is_empty() {
            Some(dir_v3_flat.as_slice())
        } else {
            None
        };

        let dir_v4_ref = if has_dir_v4 && !dir_v4_flat.is_empty() {
            Some(dir_v4_flat.as_slice())
        } else {
            None
        };

        let predictions = self.model_manager.predict_batch(
            tf_minutes,
            &features_flat,
            batch_size,
            ncol,
            dir_v3_ref,
            dir_v4_ref,
            dir_v4_ncol,
            use_gpu,
        )?;

        // Score each prediction — only ML models (P(super) + direction confidence)
        let mut results = Vec::with_capacity(batch_size);
        for (idx, pred) in predictions.iter().enumerate() {
            let candle_idx = process_start + idx;
            let candle = &candles[candle_idx];

            // Score with per-TF direction confidence threshold
            let decision = self.scorer.score(pred, tf_minutes);

            let signal = self.signal_generator.generate(
                &decision,
                &candle.symbol,
                candle.symbol_id,
                tf_minutes,
                candle.time,
                candle.close,
                candle.atr,
            );

            results.push(PipelineResult {
                signal,
                prediction: Some(*pred),
                decision: Some(decision),
                candle_index: candle_idx,
                agrees_count: None, // DEPRECATED
            });
        }

        Ok(results)
    }

    /// Process a single candle (for real-time use).
    ///
    /// # Arguments
    /// * `candle_history` - Slice of recent candles
    /// * `candle` - The candle to generate features for (must be the last in history)
    /// * `tf_minutes` - Timeframe in minutes
    /// * `use_gpu` - Whether to use GPU
    pub fn process_single_with_context(
        &self,
        candle_history: Option<&[CandleWithIndicators]>,
        candle: &CandleWithIndicators,
        tf_minutes: i32,
        use_gpu: bool,
    ) -> Result<PipelineResult> {
        if !self.model_manager.has_model_for_tf(tf_minutes) {
            return Ok(PipelineResult {
                signal: None,
                prediction: None,
                decision: None,
                candle_index: 0,
                agrees_count: None,
            });
        }

        let mut features: Vec<f32> = candle.full_features().into_iter().map(|v| v as f32).collect();

        // Add dynamic features using candle history for lookback context
        let dyn_feats = match candle_history {
            Some(history) if !history.is_empty() => {
                let last_idx = history.len() - 1;
                crate::dataset::compute_dynamic_features(history, last_idx)
            }
            _ => vec![0.0f64; crate::config::dynamic_feature_count()],
        };
        features.extend(dyn_feats.iter().map(|&v| v as f32));

        // Compute direction v4 pattern features (preferred)
        let dir_v4_feats = if self.model_manager.has_direction_v4_for_tf(tf_minutes) {
            match candle_history {
                Some(history) if history.len() >= self.dir_config.window_size => {
                    let last_idx = history.len() - 1;
                    compute_pattern_features(history, last_idx, &self.dir_config)
                        .map(|feats| feats.iter().map(|&v| v as f32).collect::<Vec<f32>>())
                }
                _ => None,
            }
        } else {
            None
        };

        // Fallback: compute direction v3 features if v4 not available
        let dir_v3_feats = if dir_v4_feats.is_none()
            && self.model_manager.has_direction_v3_for_tf(tf_minutes)
        {
            match candle_history {
                Some(history) if history.len() > 50 => {
                    let last_idx = history.len() - 1;
                    let feats = compute_direction_v3_features(history, last_idx, None, None);
                    Some(feats)
                }
                _ => None,
            }
        } else {
            None
        };

        let dir_v4_ncol = self.dir_config.total_features();
        let prediction = self.model_manager.predict(
            tf_minutes,
            &features,
            dir_v3_feats.as_deref(),
            dir_v4_feats.as_deref(),
            dir_v4_ncol,
            use_gpu,
        )?;

        let result = match prediction {
            Some(pred) => {
                // Score with per-TF direction confidence threshold
                let decision = self.scorer.score(&pred, tf_minutes);

                let signal = self.signal_generator.generate(
                    &decision,
                    &candle.symbol,
                    candle.symbol_id,
                    tf_minutes,
                    candle.time,
                    candle.close,
                    candle.atr,
                );

                PipelineResult {
                    signal,
                    prediction: Some(pred),
                    decision: Some(decision),
                    candle_index: 0,
                    agrees_count: None,
                }
            }
            None => PipelineResult {
                signal: None,
                prediction: None,
                decision: None,
                candle_index: 0,
                agrees_count: None,
            },
        };

        Ok(result)
    }

    /// Per-TF candle limit for full pipeline run.
    fn candle_limit_for_tf(tf_minutes: i32) -> usize {
        match tf_minutes {
            1 => 5000,
            5 => 12000,
            15 => 12000,
            60 => 12000,
            240 => 12000,
            1440 => 3700,
            _ => 5000,
        }
    }

    /// Run the full pipeline for all symbols and timeframes.
    pub async fn run_all(
        &self,
        pool: &PgPool,
        use_gpu: bool,
    ) -> Result<Vec<SuperEntrySignal>> {
        let symbols = fetch_active_symbols(pool).await?;
        info!("Processing {} symbols", symbols.len());

        let mut all_signals = Vec::new();

        for &tf in SuperEntryConfig::timeframes() {
            if !self.model_manager.has_model_for_tf(tf) {
                continue;
            }

            let mut tf_signals = 0;
            let mut tf_total = 0;
            let limit = Self::candle_limit_for_tf(tf);

            for symbol in &symbols {
                let candles = fetch_candles_with_indicators(pool, symbol, tf, limit).await?;

                if candles.len() < self.config.warmup_bars + self.config.lookahead_bars {
                    continue;
                }

                let results = self.process_candles(&candles, tf, use_gpu)?;

                for result in results {
                    tf_total += 1;
                    if let Some(signal) = result.signal {
                        tf_signals += 1;
                        all_signals.push(signal);
                    }
                }
            }

            info!(
                "TF {}m: {} signals from {} candles ({:.1}% coverage)",
                tf, tf_signals, tf_total,
                if tf_total > 0 { tf_signals as f64 / tf_total as f64 * 100.0 } else { 0.0 }
            );
        }

        info!("Total signals generated: {}", all_signals.len());
        Ok(all_signals)
    }

    /// Get reference to config
    pub fn config(&self) -> &SuperEntryConfig {
        &self.config
    }
}
