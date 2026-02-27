// strategies/ewmac_strategy/src/pipeline.rs
//
// Pipeline for EWMAC Strategy
//
// Orchestrates the full flow:
//   1. Load candles from DB
//   2. Compute EWMAC signals (EMA crossovers, ATR normalization)
//   3. Generate trade signals (apply thresholds, ATR-based SL/TP)
//
// Can be used both for:
//   - Historical backtesting (batch processing all candles)
//   - Real-time inference (incremental updates)
//
// Unlike ml_entry_strategy, EWMAC doesn't need ML models — it's purely rule-based.

use anyhow::Result;
use sqlx::PgPool;
use tracing::{info, debug};

use crate::config::EwmacConfig;
use crate::dataset::{Candle, fetch_all_candles_for_tf, fetch_active_symbols};
use crate::ewmac::{EwmacCalculator, EwmacResult};
use crate::signal_generator::{SignalGenerator, EwmacSignal};

/// Result of processing a single candle through the pipeline
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The generated signal (if any)
    pub signal: Option<EwmacSignal>,
    /// The EWMAC computation result
    pub ewmac_result: Option<EwmacResult>,
    /// Index of the candle in the series
    pub candle_index: usize,
}

/// EWMAC Pipeline
///
/// Full pipeline: DB → EWMAC calculation → signal generation
pub struct EwmacPipeline {
    config: EwmacConfig,
    signal_generator: SignalGenerator,
}

impl EwmacPipeline {
    /// Create a new pipeline.
    ///
    /// No model loading required (EWMAC is rule-based).
    pub fn new(config: EwmacConfig) -> Self {
        let signal_generator = SignalGenerator::new(config.clone());
        Self { config, signal_generator }
    }

    /// Process a series of candles for a given symbol/timeframe.
    ///
    /// Creates a fresh EwmacCalculator, feeds all candles through it,
    /// and generates signals for candles after warmup.
    ///
    /// Applies cooldown: after generating a signal, suppresses same-direction
    /// signals for `cooldown_bars` candles to prevent signal spam in trends.
    pub fn process_candles(
        &self,
        candles: &[Candle],
        tf_minutes: i32,
    ) -> Result<Vec<PipelineResult>> {
        if candles.len() < self.config.warmup_bars {
            debug!("Not enough candles for TF {}m (need >= {})", tf_minutes, self.config.warmup_bars);
            return Ok(Vec::new());
        }

        let mut calculator = EwmacCalculator::new(&self.config);
        let mut results = Vec::with_capacity(candles.len().saturating_sub(self.config.warmup_bars));

        // Cooldown tracking: last signal index per direction
        let cooldown = self.config.cooldown_bars;
        let mut last_long_signal_idx: Option<usize> = None;
        let mut last_short_signal_idx: Option<usize> = None;

        for (i, candle) in candles.iter().enumerate() {
            let ewmac_result = calculator.update(candle.high, candle.low, candle.close);

            if let Some(ref result) = ewmac_result {
                let mut signal = self.signal_generator.generate(
                    result,
                    &candle.symbol,
                    candle.symbol_id,
                    tf_minutes,
                    candle.time,
                    candle.close,
                );

                // Apply cooldown: suppress if same direction signal was recently generated
                if cooldown > 0 {
                    if let Some(ref sig) = signal {
                        let suppressed = if sig.side == 1 {
                            // LONG — check cooldown from last long
                            last_long_signal_idx.map_or(false, |last_i| i - last_i < cooldown)
                        } else {
                            // SHORT — check cooldown from last short
                            last_short_signal_idx.map_or(false, |last_i| i - last_i < cooldown)
                        };

                        if suppressed {
                            signal = None;
                        } else {
                            // Record this signal
                            if sig.side == 1 {
                                last_long_signal_idx = Some(i);
                            } else {
                                last_short_signal_idx = Some(i);
                            }
                        }
                    }
                }

                results.push(PipelineResult {
                    signal,
                    ewmac_result: Some(*result),
                    candle_index: i,
                });
            }
        }

        Ok(results)
    }

    /// Run the full pipeline for all symbols and timeframes.
    ///
    /// Fetches data from DB, processes each (symbol, tf) pair,
    /// and returns all generated signals.
    pub async fn run_all(
        &self,
        pool: &PgPool,
    ) -> Result<Vec<EwmacSignal>> {
        let _symbols = fetch_active_symbols(pool).await?;
        info!("Processing EWMAC for all active symbols");

        let mut all_signals = Vec::new();

        for &tf in EwmacConfig::timeframes() {
            let grouped = match fetch_all_candles_for_tf(pool, tf, 1000).await {
                Ok(g) => g,
                Err(e) => {
                    tracing::warn!("Failed to fetch candles for TF {}m: {}", tf, e);
                    continue;
                }
            };

            let mut tf_signals = 0;
            let mut tf_total = 0;

            for (_symbol, candles) in &grouped {
                if candles.len() < self.config.warmup_bars + 1 {
                    continue;
                }

                let results = self.process_candles(candles, tf)?;
                tf_total += results.len();

                for result in results {
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

        info!("Total EWMAC signals generated: {}", all_signals.len());
        Ok(all_signals)
    }

    /// Get reference to config
    pub fn config(&self) -> &EwmacConfig {
        &self.config
    }
}
