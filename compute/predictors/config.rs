use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionsConfig {
    pub enabled: bool,
    pub horizon_bars: usize,
    pub min_store_score: f64,
    pub min_final_score: f64,
    pub prefer_ml: bool,
    pub max_levels_per_side: usize,
    pub use_cuda: bool,
}