// strategies/ml_entry_strategy/src/pipeline.rs
//
// Pipeline for Super Entry Strategy
//
// Orchestrates the full flow:
//   1. Load candles + indicators from DB
//   2. Build feature vectors
//   3. Run model inference (P(super), P(direction))
//   4. Score predictions (apply thresholds, overheated filter)
//   5. Generate trade signals
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
use crate::scorer::{SuperEntryScorer, SuperEntryDecision, OverheatedFeatures};
use crate::signal_generator::{SignalGenerator, SuperEntrySignal};

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
}

/// Super Entry Pipeline
///
/// Full pipeline: DB → features → model → scorer → signal
pub struct SuperEntryPipeline {
    config: SuperEntryConfig,
    model_manager: SuperEntryModelManager,
    scorer: SuperEntryScorer,
    signal_generator: SignalGenerator,
}

impl SuperEntryPipeline {
    /// Create a new pipeline with loaded models.
    ///
    /// # Arguments
    /// * `config` - Strategy configuration
    /// * `use_gpu` - Whether to use GPU for model inference
    pub fn new(config: SuperEntryConfig, use_gpu: bool) -> Result<Self> {
        let model_manager = SuperEntryModelManager::new(config.clone(), use_gpu)?;
        let scorer = SuperEntryScorer::from_strategy_config(&config);
        let signal_generator = SignalGenerator::new(config.clone());

        Ok(Self {
            config,
            model_manager,
            scorer,
            signal_generator,
        })
    }

    /// Check if the pipeline has any models loaded
    pub fn has_models(&self) -> bool {
        self.model_manager.has_models()
    }

    /// Process a series of candles for a given symbol/timeframe.
    ///
    /// Skips the first `warmup_bars` candles (used for indicator warmup).
    /// For each candle after warmup:
    ///   1. Extract features
    ///   2. Run inference
    ///   3. Score
    ///   4. Generate signal (if super)
    ///
    /// # Returns
    /// Vec of PipelineResult for each processed candle
    pub fn process_candles(
        &self,
        candles: &[CandleWithIndicators],
        tf_minutes: i32,
        use_gpu: bool,
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

        // Build feature matrix for batch inference
        let ncol = crate::config::total_feature_count();
        let mut features_flat: Vec<f32> = Vec::with_capacity(batch_size * ncol);

        for i in process_start..n {
            let feat = candles[i].full_features();
            for &v in &feat {
                features_flat.push(v as f32);
            }
        }

        // Batch inference
        let predictions = self.model_manager.predict_batch(
            tf_minutes,
            &features_flat,
            batch_size,
            ncol,
            use_gpu,
        )?;

        // Score each prediction
        let mut results = Vec::with_capacity(batch_size);
        for (idx, pred) in predictions.iter().enumerate() {
            let candle_idx = process_start + idx;
            let candle = &candles[candle_idx];

            // Build overheated features
            let bb_range = candle.bb_upper - candle.bb_lower;
            let bb_position = if bb_range.abs() > 1e-12 {
                (candle.close - candle.bb_lower) / bb_range
            } else {
                0.5
            };
            let atr_pct = if candle.close > 0.0 {
                candle.atr / candle.close * 100.0
            } else {
                0.0
            };

            let oh = OverheatedFeatures {
                rsi: candle.rsi,
                stoch_k: candle.stoch_k,
                cci: candle.cci,
                williams: candle.williams,
                bb_position,
                atr_pct,
            };

            let decision = self.scorer.score(pred, Some(&oh));

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

    /// Process a single candle (for real-time use).
    pub fn process_single(
        &self,
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
            });
        }

        let features: Vec<f32> = candle.full_features().into_iter().map(|v| v as f32).collect();

        let prediction = self.model_manager.predict(tf_minutes, &features, use_gpu)?;

        let result = match prediction {
            Some(pred) => {
                let bb_range = candle.bb_upper - candle.bb_lower;
                let bb_position = if bb_range.abs() > 1e-12 {
                    (candle.close - candle.bb_lower) / bb_range
                } else {
                    0.5
                };
                let atr_pct = if candle.close > 0.0 {
                    candle.atr / candle.close * 100.0
                } else {
                    0.0
                };

                let oh = OverheatedFeatures {
                    rsi: candle.rsi,
                    stoch_k: candle.stoch_k,
                    cci: candle.cci,
                    williams: candle.williams,
                    bb_position,
                    atr_pct,
                };

                let decision = self.scorer.score(&pred, Some(&oh));

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
                }
            }
            None => PipelineResult {
                signal: None,
                prediction: None,
                decision: None,
                candle_index: 0,
            },
        };

        Ok(result)
    }

    /// Run the full pipeline for all symbols and timeframes.
    ///
    /// Fetches data from DB, processes each (symbol, tf) pair,
    /// and returns all generated signals grouped by TF.
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

            for symbol in &symbols {
                let candles = fetch_candles_with_indicators(pool, symbol, tf, 1000).await?;

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
