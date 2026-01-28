pub mod pairs;

// candles.rs позже подключишь, когда дойдём до свечей
// pub mod candles;

pub use pairs::{refresh_universe_pairs, UniversePairsResult};
