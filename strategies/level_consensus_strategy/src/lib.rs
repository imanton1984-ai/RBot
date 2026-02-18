// strategies/level_consensus_strategy/src/lib.rs
//
// Level Consensus Strategy - копия текущей логики final_scorer / trade_signal_calculator

pub mod scorer;
pub mod trade_signal_calculator;

pub use scorer::{LevelConsensusScorer, LevelConsensusScoreBreakdown, SetupKind};
pub use trade_signal_calculator::{TradeSignal, TradeSignalCalculator};

pub const STRATEGY_ID: i16 = 6;
pub const STRATEGY_NAME: &str = "level_consensus";
