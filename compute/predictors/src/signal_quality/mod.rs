// signal_quality/mod.rs
//
// Signal Quality Scorer — "meta-predictor" that evaluates final trade signals.
//
// This module adds a SECOND scoring layer on top of trade.final_signals:
//   1. HeuristicQualityScorer — rule-based quality assessment from signal features
//   2. MlQualityScorer — XGBoost model trained on backtested signal outcomes
//
// Both produce a quality_multiplier in [0.5, 1.5] that augments the final_score.
// This multiplier is intentionally capped to prevent over-reliance on a single model.
//
// INTEGRATION (future):
//   After backtester module is implemented and model is trained:
//   1. Add to PredictorsPipeline after TradeSignalStage
//   2. Or use in StrategyModule to filter/rank signals before order execution
//
// NOT YET WIRED INTO PIPELINE — standalone code ready for integration.

pub mod types;
pub mod heuristic_scorer;
pub mod ml_scorer;
pub mod trainer;
