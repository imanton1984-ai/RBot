// strategies/super_level_strategy/src/pipeline.rs
//
// Pipeline for Super Level Strategy
//
// Orchestrates: DB → levels → features → 5 models → scorer → signal
//
// Отличие от super_entry: сначала вычисляем уровни по касаниям,
// потом генерируем level features, потом ML inference на всех 5 моделях.

use anyhow::Result;
use sqlx::PgPool;
use tracing::{info, warn, debug};

use crate::config::SuperLevelConfig;
use crate::dataset::{
    compute_levels, compute_level_features, compute_dynamic_features,
    PriceLevel,
};
use crate::model::{SuperLevelModelManager, SuperLevelPrediction};
use crate::scorer::{SuperLevelScorer, SuperLevelDecision};
use crate::signal_generator::{SignalGenerator, SuperLevelSignal};
use ml_entry_strategy::dataset::CandleWithIndicators;

/// Результат обработки одной свечи
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub signal: Option<SuperLevelSignal>,
    pub prediction: Option<SuperLevelPrediction>,
    pub decision: Option<SuperLevelDecision>,
    pub candle_index: usize,
}

/// Super Level Pipeline
pub struct SuperLevelPipeline {
    config: SuperLevelConfig,
    model_manager: SuperLevelModelManager,
    scorer: SuperLevelScorer,
    signal_generator: SignalGenerator,
}

impl SuperLevelPipeline {
    pub fn new(config: SuperLevelConfig, use_gpu: bool) -> Result<Self> {
        let model_manager = SuperLevelModelManager::new(config.clone(), use_gpu)?;
        let scorer = SuperLevelScorer::from_strategy_config(&config);
        let signal_generator = SignalGenerator::new(config.clone());

        Ok(Self {
            config,
            model_manager,
            scorer,
            signal_generator,
        })
    }

    pub fn has_models(&self) -> bool {
        self.model_manager.has_models()
    }

    /// Process all candles for a symbol/tf.
    ///
    /// Steps:
    ///   1. Compute levels from history (recompute every 50 bars)
    ///   2. Build feature matrix (indicators + derived + level + dynamic)
    ///   3. Batch inference on all 5 models
    ///   4. Score each prediction
    ///   5. Generate signals
    pub fn process_candles(
        &self,
        candles: &[CandleWithIndicators],
        tf_minutes: i32,
    ) -> Result<Vec<PipelineResult>> {
        if !self.model_manager.has_model_for_tf(tf_minutes) {
            warn!("No models for TF {}m, skipping", tf_minutes);
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
        let ncol = crate::config::total_feature_count();

        // ── Step 1: Precompute levels (refreshed every 50 bars) ──
        let level_params = &self.config.level_params;
        let mut level_cache: Vec<(usize, Vec<PriceLevel>)> = Vec::new(); // (start_idx, levels)

        // Compute levels at each 50-bar boundary
        let mut t = process_start;
        while t < n {
            let formation_start = t.saturating_sub(level_params.formation_bars);
            let levels = compute_levels(candles, formation_start, t, level_params);
            level_cache.push((t, levels));
            t += 50;
        }

        // ── Step 2: Build feature matrix ──
        let mut features_flat: Vec<f32> = Vec::with_capacity(batch_size * ncol);
        let mut level_idx_for_bar = 0;

        for i in process_start..n {
            // Find appropriate level cache entry
            while level_idx_for_bar + 1 < level_cache.len()
                && level_cache[level_idx_for_bar + 1].0 <= i
            {
                level_idx_for_bar += 1;
            }
            let current_levels = &level_cache[level_idx_for_bar].1;

            let c = &candles[i];

            // Raw indicators (33)
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

            // Derived features (19)
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
            features_flat.push(0.0f32); // obv_change_pct
            features_flat.push(if c.volume_spike > 2.0 { 1.0f32 } else { 0.0f32 });
            features_flat.push((c.mfi / 100.0) as f32);
            features_flat.push(safe_div(close - c.fibo_pivot, close) * 100.0);
            features_flat.push(safe_div(close - c.supertrend, close) * 100.0);
            features_flat.push(safe_div(c.alligator_jaw - c.alligator_lips, close) * 100.0);

            // Level features (18)
            let level_feats = compute_level_features(candles, i, current_levels);
            for v in &level_feats {
                features_flat.push(*v as f32);
            }

            // Dynamic features (34)
            let dyn_feats = compute_dynamic_features(candles, i);
            for v in &dyn_feats {
                features_flat.push(*v as f32);
            }
        }

        // ── Step 3: Batch inference (all 5 models) ──
        let predictions = self.model_manager.predict_batch(
            tf_minutes,
            &features_flat,
            batch_size,
            ncol,
        )?;

        // ── Step 4-5: Score + Signal generation ──
        let mut results = Vec::with_capacity(batch_size);
        for (idx, pred) in predictions.iter().enumerate() {
            let candle_idx = process_start + idx;
            let candle = &candles[candle_idx];

            let decision = self.scorer.score(pred);

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
            });
        }

        Ok(results)
    }

    /// Run for all symbols and timeframes
    pub async fn run_all(
        &self,
        pool: &PgPool,
    ) -> Result<Vec<SuperLevelSignal>> {
        let symbols = crate::dataset::fetch_active_symbols(pool).await?;
        info!("Processing {} symbols", symbols.len());

        let mut all_signals = Vec::new();

        for &tf in SuperLevelConfig::timeframes() {
            if !self.model_manager.has_model_for_tf(tf) {
                continue;
            }

            let limit = candle_limit_for_tf(tf);
            let grouped = crate::dataset::fetch_all_candles_for_tf(pool, tf, limit).await?;

            let mut tf_signals = 0;
            let mut tf_total = 0;

            for (_symbol, candles) in &grouped {
                if candles.len() < self.config.warmup_bars + self.config.lookahead_bars {
                    continue;
                }

                let results = self.process_candles(candles, tf)?;
                for res in results {
                    tf_total += 1;
                    if let Some(sig) = res.signal {
                        tf_signals += 1;
                        all_signals.push(sig);
                    }
                }
            }

            info!("TF {}m: {} signals from {} candles ({:.1}%)",
                tf, tf_signals, tf_total,
                if tf_total > 0 { tf_signals as f64 / tf_total as f64 * 100.0 } else { 0.0 });
        }

        info!("Total SuperLevel signals: {}", all_signals.len());
        Ok(all_signals)
    }

    pub fn config(&self) -> &SuperLevelConfig {
        &self.config
    }
}

/// Per-TF candle limit
pub fn candle_limit_for_tf(tf_minutes: i32) -> usize {
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
