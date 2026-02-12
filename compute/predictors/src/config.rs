use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictorsConfig {
    pub enabled: bool,
    pub horizon_bars: usize,
    pub min_store_score: f64,
    pub min_final_score: f64,
    pub prefer_ml: bool,
    pub max_levels_per_side: usize,
    pub use_cuda: bool,
    pub use_gpu_history: bool,   // true on GPU machine
    pub use_gpu_realtime: bool,  // usually false
    pub model_path_price: String,   // "models/price_v1_tf{tf}.ubj"
    pub model_path_levels: String,  // "models/levels_v1_tf{tf}.ubj"
    pub ml_batch_size: usize,
}