// strategies/level_strategy/src/signal_generator.rs
//
// Level Strategy Signal Generator
//
// Generates trade signals from predictions

use anyhow::Result;

/// Level Strategy Trade Signal
#[derive(Debug, Clone)]
pub struct LevelStrategySignal {
    pub symbol: String,
    pub symbol_id: i64,
    pub tf_minutes: i32,
    pub time_ms: i64,
    pub side: i8,  // 1 = LONG, -1 = SHORT
    pub entry_price: f64,
    pub tp_price: f64,
    pub sl_price: f64,
    pub score: f64,
    pub confidence: f64,
}

/// Level Strategy Signal Generator
pub struct SignalGenerator {
    config: crate::config::LevelStrategyConfig,
}

impl SignalGenerator {
    pub fn new(config: crate::config::LevelStrategyConfig) -> Self {
        Self { config }
    }

    /// Generate a trade signal from predictions
    ///
    /// This is a placeholder - the actual signal generation
    /// is done in compute/scorer/trade_signal_calculator.rs
    pub fn generate(
        &self,
        _symbol: &str,
        _symbol_id: i64,
        _tf_minutes: i32,
        _time_ms: i64,
        _close_price: f64,
        _predictions: &[predictors::types::PredictionRow],
    ) -> Option<LevelStrategySignal> {
        // The actual signal generation logic is in compute module
        // This is a strategy-specific wrapper
        None
    }
}
