// strategies/level_ml_strategy/src/lib.rs

pub mod scorer;
pub mod trade_signal_calculator;

pub use scorer::{LevelMlScorer, LevelMlScoreBreakdown, SetupKind};
pub use trade_signal_calculator::{TradeSignal, TradeSignalCalculator};

pub const STRATEGY_ID: i16 = 4;
pub const STRATEGY_NAME: &str = "level_ml";
