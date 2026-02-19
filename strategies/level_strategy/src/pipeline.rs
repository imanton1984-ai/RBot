// strategies/level_strategy/src/pipeline.rs
//
// Level Strategy Pipeline
//
// Orchestrates the full flow:
//   1. Load candles + indicators from DB
//   2. Build feature vectors
//   3. Run predictors (ML + heuristic)
//   4. Apply consensus (gate + fuse)
//   5. Generate trade signals
//
// This is a wrapper around compute/predictors pipeline
// providing a strategy-specific interface.

use anyhow::Result;
use sqlx::PgPool;
use tracing::{info, warn};

use crate::config::LevelStrategyConfig;

/// High-level Level Strategy Pipeline
///
/// Wraps the predictors pipeline and provides a simple interface
pub struct LevelStrategyPipeline {
    config: LevelStrategyConfig,
}

impl LevelStrategyPipeline {
    /// Create a new pipeline instance
    pub fn new(config: LevelStrategyConfig) -> Self {
        Self { config }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self::new(LevelStrategyConfig::from_env())
    }

    /// Get reference to config
    pub fn config(&self) -> &LevelStrategyConfig {
        &self.config
    }

    /// Check if the strategy is enabled (has models)
    pub fn has_models(&self) -> bool {
        // Check if price and levels models exist for at least one TF
        let tfs = Self::timeframes();
        for &tf in tfs {
            let price_path = self.config.price_model_path(tf);
            let levels_path = self.config.levels_model_path(tf);
            if std::path::Path::new(&price_path).exists()
                && std::path::Path::new(&levels_path).exists()
            {
                return true;
            }
        }
        false
    }

    /// Supported timeframes
    pub fn timeframes() -> &'static [i32] {
        &[1, 5, 15, 60, 240]
    }

    /// Run the pipeline on historical data
    ///
    /// This processes all symbols and timeframes from the database
    /// and generates predictions and trade signals.
    pub async fn run_history(&self, pool: &PgPool) -> Result<usize> {
        info!("Level Strategy: running history pipeline");

        if !self.has_models() {
            warn!("Level Strategy: no models found, skipping history");
            return Ok(0);
        }

        // The actual pipeline is in compute/predictors
        // This is a placeholder - the real work is done by compute_history
        info!("Level Strategy: history pipeline complete (via compute_history)");
        Ok(0)
    }

    /// Run the pipeline in realtime mode
    ///
    /// This processes incoming candles and generates predictions/signals
    pub async fn run_realtime(&self, pool: &PgPool) -> Result<usize> {
        info!("Level Strategy: running realtime pipeline");

        if !self.has_models() {
            warn!("Level Strategy: no models found, skipping realtime");
            return Ok(0);
        }

        // The actual pipeline is in compute/predictors
        // This is a placeholder - the real work is done by compute_realtime
        info!("Level Strategy: realtime pipeline complete (via compute_realtime)");
        Ok(0)
    }
}
