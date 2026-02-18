// strategies/indicator_consensus_strategy/src/lib.rs
//
// Indicator Consensus Strategy - использует консенсус между ML и heuristic

pub mod scorer;
pub mod trade_signal_calculator;

pub use scorer::{IndicatorConsensusScorer, IndicatorConsensusScoreBreakdown, SetupKind};
pub use trade_signal_calculator::{TradeSignal, TradeSignalCalculator};

pub const STRATEGY_ID: i16 = 3;
pub const STRATEGY_NAME: &str = "indicator_consensus";
