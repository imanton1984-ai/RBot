// Scoring / trade signal modules (files live outside src/ — use #[path])

#[path = "../../scorer/final_score.rs"]
pub mod final_score;

#[path = "../../market_parameters/market_params_calculator.rs"]
pub mod market_params_calculator;

#[path = "../../trade_signals/trade_signal_calculator.rs"]
pub mod trade_signal_calculator;

#[path = "../../trade_signals/trade_signal_processor.rs"]
pub mod trade_signal_processor;
