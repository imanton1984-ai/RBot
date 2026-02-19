// strategies/level_strategy/src/strategy.rs
//
// High-level Strategy Interface for Level Strategy
//
// Provides a unified API for:
//   - Running the strategy on historical data (backtest)
//   - Enabling/disabling via flags
//   - Integration with the existing system

use anyhow::Result;
use sqlx::PgPool;
use tracing::info;

use crate::config::LevelStrategyConfig;
use crate::pipeline::LevelStrategyPipeline;

/// High-level Level Strategy
pub struct LevelStrategy {
    pipeline: LevelStrategyPipeline,
    enabled: bool,
}

impl LevelStrategy {
    /// Create a new strategy instance
    pub fn new(config: LevelStrategyConfig) -> Result<Self> {
        let pipeline = LevelStrategyPipeline::new(config);
        let enabled = pipeline.has_models();

        if !enabled {
            tracing::warn!("Level Strategy: no models loaded, strategy disabled");
        } else {
            info!("Level Strategy: initialized successfully");
        }

        Ok(Self { pipeline, enabled })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self> {
        Self::new(LevelStrategyConfig::from_env())
    }

    /// Check if the strategy is enabled (has loaded models)
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Manually enable/disable the strategy
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Run the strategy on all available data from the database
    pub async fn run(&self, pool: &PgPool) -> Result<usize> {
        if !self.enabled {
            info!("Level Strategy is disabled, returning empty");
            return Ok(0);
        }

        self.pipeline.run_history(pool).await
    }

    /// Get reference to the pipeline
    pub fn pipeline(&self) -> &LevelStrategyPipeline {
        &self.pipeline
    }

    /// Get reference to config
    pub fn config(&self) -> &LevelStrategyConfig {
        self.pipeline.config()
    }
}
