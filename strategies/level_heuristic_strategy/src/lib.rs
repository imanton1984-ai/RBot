// strategies/level_heuristic_strategy/src/lib.rs

pub mod scorer;
pub mod trade_signal_calculator;

pub use scorer::{LevelHeuristicScorer, LevelHeuristicScoreBreakdown, SetupKind};
pub use trade_signal_calculator::{TradeSignal, TradeSignalCalculator};

pub const STRATEGY_ID: i16 = 5;
pub const STRATEGY_NAME: &str = "level_heuristic";
