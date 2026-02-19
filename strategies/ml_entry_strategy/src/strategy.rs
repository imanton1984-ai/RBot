// strategies/ml_entry_strategy/src/strategy.rs
//
// High-level Strategy Interface for Super Entry
//
// Provides a unified API for:
//   - Running the strategy on historical data (backtest)
//   - Enabling/disabling via flags
//   - Integration with the existing system

use anyhow::Result;
use sqlx::PgPool;
use tracing::info;

use crate::config::SuperEntryConfig;
use crate::pipeline::SuperEntryPipeline;
use crate::signal_generator::SuperEntrySignal;

/// High-level Super Entry Strategy
///
/// Wraps the pipeline and provides a simple interface for
/// running the strategy from external callers (backtester, real-time runner).
pub struct SuperEntryStrategy {
    pipeline: SuperEntryPipeline,
    enabled: bool,
}

impl SuperEntryStrategy {
    /// Create a new strategy instance.
    ///
    /// # Arguments
    /// * `config` - Strategy configuration
    /// * `use_gpu` - Whether to use GPU for inference
    pub fn new(config: SuperEntryConfig, use_gpu: bool) -> Result<Self> {
        let pipeline = SuperEntryPipeline::new(config, use_gpu)?;
        let enabled = pipeline.has_models();

        if !enabled {
            tracing::warn!("Super Entry Strategy: no models loaded, strategy disabled");
        } else {
            info!("Super Entry Strategy: initialized successfully");
        }

        Ok(Self { pipeline, enabled })
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self> {
        let config = SuperEntryConfig::from_env();
        let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
            .unwrap_or_default()
            .parse::<bool>()
            .unwrap_or(false);

        Self::new(config, use_gpu)
    }

    /// Check if the strategy is enabled (has loaded models)
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Manually enable/disable the strategy
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Run the strategy on all available data from the database.
    ///
    /// Returns all generated super entry signals.
    pub async fn run(&self, pool: &PgPool) -> Result<Vec<SuperEntrySignal>> {
        if !self.enabled {
            info!("Super Entry Strategy is disabled, returning empty signals");
            return Ok(Vec::new());
        }

        let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
            .unwrap_or_default()
            .parse::<bool>()
            .unwrap_or(false);

        self.pipeline.run_all(pool, use_gpu).await
    }

    /// Get reference to the pipeline
    pub fn pipeline(&self) -> &SuperEntryPipeline {
        &self.pipeline
    }

    /// Get reference to config
    pub fn config(&self) -> &SuperEntryConfig {
        self.pipeline.config()
    }
}
