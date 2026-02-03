use serde::{Deserialize, Serialize};
use common::{Symbol, Timeframe};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureValue {
    Float(f64),
    Json(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowSpec {
    pub length: usize,
    pub warmup: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchTensor {
    pub close: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub volume: Vec<f64>,
    pub timestamps: Vec<i64>,
    pub symbols: Vec<Symbol>,
    pub timeframes: Vec<Timeframe>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeConfig {
    pub batch_size: usize,
    pub max_concurrent_jobs: usize,
    pub use_cuda: bool,
    pub cuda_device_id: Option<usize>,
}

impl Default for ComputeConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            max_concurrent_jobs: 4,
            use_cuda: false,
            cuda_device_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputBuffer {
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
    pub timestamps: Vec<i64>,
}