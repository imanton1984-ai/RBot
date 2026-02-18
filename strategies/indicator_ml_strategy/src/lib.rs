// strategies/indicator_ml_strategy/src/lib.rs

pub mod scorer;
pub mod trade_signal_calculator;

pub use scorer::{IndicatorMlScorer, IndicatorMlScoreBreakdown, SetupKind, IndicatorWeights};
pub use trade_signal_calculator::{TradeSignal, TradeSignalCalculator, TradeSide};

/// Strategy ID для indicator_ml_strategy
pub const STRATEGY_ID: i16 = 1;

/// Strategy name
pub const STRATEGY_NAME: &str = "indicator_ml";
