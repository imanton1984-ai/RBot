// strategies/ewmac_strategy/src/strategy.rs
//
// High-level Strategy Interface for EWMAC
//
// Provides a unified API for:
//   - Running the strategy on historical data (backtest)
//   - Enabling/disabling via flags
//   - Integration with the existing system

use anyhow::Result;
use sqlx::PgPool;
use tracing::info;

use crate::config::EwmacConfig;
use crate::pipeline::EwmacPipeline;
use crate::signal_generator::EwmacSignal;

/// High-level EWMAC Strategy
///
/// Wraps the pipeline and provides a simple interface for
/// running the strategy from external callers (backtester, real-time runner).
pub struct EwmacStrategy {
    pipeline: EwmacPipeline,
    enabled: bool,
}

impl EwmacStrategy {
    /// Create a new strategy instance.
    pub fn new(config: EwmacConfig) -> Self {
        let pipeline = EwmacPipeline::new(config);
        info!("EWMAC Strategy: initialized successfully (rule-based, no models needed)");

        Self { pipeline, enabled: true }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        let config = EwmacConfig::from_env();
        Self::new(config)
    }

    /// Check if the strategy is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Manually enable/disable the strategy
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Run the strategy on all available data from the database.
    ///
    /// Returns all generated EWMAC signals.
    pub async fn run(&self, pool: &PgPool) -> Result<Vec<EwmacSignal>> {
        if !self.enabled {
            info!("EWMAC Strategy is disabled, returning empty signals");
            return Ok(Vec::new());
        }

        self.pipeline.run_all(pool).await
    }

    /// Get reference to the pipeline
    pub fn pipeline(&self) -> &EwmacPipeline {
        &self.pipeline
    }

    /// Get reference to config
    pub fn config(&self) -> &EwmacConfig {
        self.pipeline.config()
    }
}
