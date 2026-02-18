// strategies/indicator_heuristic_strategy/src/lib.rs

pub mod scorer;
pub mod trade_signal_calculator;

pub use scorer::{IndicatorHeuristicScorer, IndicatorHeuristicScoreBreakdown, SetupKind, IndicatorWeights};
pub use trade_signal_calculator::{TradeSignal, TradeSignalCalculator};

pub const STRATEGY_ID: i16 = 2;
pub const STRATEGY_NAME: &str = "indicator_heuristic";
