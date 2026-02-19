// strategies/level_strategy/src/config.rs
//
// Configuration for Level Strategy
//
// Contains:
//   - Horizon bars for predictions
//   - Min scores for storing predictions
//   - Model paths for price/levels predictors
//   - Trade signal parameters

use serde::{Deserialize, Serialize};

/// Configuration for the Level Strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LevelStrategyConfig {
    /// Prediction horizon in bars (default: 10)
    pub horizon_bars: usize,

    /// Minimum score to store predictions (default: 0.50)
    pub min_store_score: f64,

    /// Minimum final score for trade signals (default: 0.60)
    pub min_final_score: f64,

    /// Prefer ML models over heuristic predictors
    pub prefer_ml: bool,

    /// Maximum levels per side to consider
    pub max_levels_per_side: usize,

    /// Use CUDA for computation
    pub use_cuda: bool,

    /// Use GPU for history processing
    pub use_gpu_history: bool,

    /// Use GPU for realtime processing
    pub use_gpu_realtime: bool,

    /// Price prediction model path template
    pub model_path_price: String,

    /// Levels prediction model path template
    pub model_path_levels: String,

    /// ML batch size for inference
    pub ml_batch_size: usize,
}

impl Default for LevelStrategyConfig {
    fn default() -> Self {
        Self {
            horizon_bars: 10,
            min_store_score: 0.50,
            min_final_score: 0.60,
            prefer_ml: true,
            max_levels_per_side: 2,
            use_cuda: false,
            use_gpu_history: false,
            use_gpu_realtime: false,
            model_path_price: "models/price_v1_tf{tf}.ubj".to_string(),
            model_path_levels: "models/levels_v1_tf{tf}.ubj".to_string(),
            ml_batch_size: 4096,
        }
    }
}

impl LevelStrategyConfig {
    /// Load config from environment variables
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("LEVEL_HORIZON_BARS") {
            if let Ok(n) = v.parse() { cfg.horizon_bars = n; }
        }
        if let Ok(v) = std::env::var("LEVEL_MIN_STORE_SCORE") {
            if let Ok(n) = v.parse() { cfg.min_store_score = n; }
        }
        if let Ok(v) = std::env::var("MIN_FINAL_SCORE") {
            if let Ok(n) = v.parse() { cfg.min_final_score = n; }
        }
        if let Ok(v) = std::env::var("LEVEL_PREFER_ML") {
            cfg.prefer_ml = v == "true" || v == "1";
        }
        if let Ok(v) = std::env::var("LEVEL_MAX_LEVELS") {
            if let Ok(n) = v.parse() { cfg.max_levels_per_side = n; }
        }
        if let Ok(v) = std::env::var("LEVEL_USE_CUDA") {
            cfg.use_cuda = v == "true" || v == "1";
        }
        if let Ok(v) = std::env::var("LEVEL_USE_GPU_HISTORY") {
            cfg.use_gpu_history = v == "true" || v == "1";
        }
        if let Ok(v) = std::env::var("LEVEL_USE_GPU_REALTIME") {
            cfg.use_gpu_realtime = v == "true" || v == "1";
        }
        if let Ok(v) = std::env::var("LEVEL_ML_BATCH_SIZE") {
            if let Ok(n) = v.parse() { cfg.ml_batch_size = n; }
        }

        cfg
    }

    /// Supported timeframes for level strategy
    pub fn timeframes() -> &'static [i32] {
        &[1, 5, 15, 60, 240]
    }

    /// Resolve model path for a given timeframe
    pub fn price_model_path(&self, tf_minutes: i32) -> String {
        self.model_path_price.replace("{tf}", &tf_minutes.to_string())
    }

    /// Resolve levels model path for a given timeframe
    pub fn levels_model_path(&self, tf_minutes: i32) -> String {
        self.model_path_levels.replace("{tf}", &tf_minutes.to_string())
    }
}
