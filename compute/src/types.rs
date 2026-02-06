use serde::{Deserialize, Serialize};
use common::Timeframe;

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
    pub symbol_ids: Vec<u32>,  // Changed from Vec<Symbol> to Vec<u32> for efficiency
    pub timeframe: Timeframe,  // Changed from Vec<Timeframe> to single Timeframe enum
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

// === ADD BELOW your existing types in compute/src/types.rs ===

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FeatureColumn {
    F64 { name: String, values: Vec<f64> },
    Json { name: String, values: Vec<serde_json::Value> },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FeatureBatch {
    pub timestamps: Vec<i64>,
    pub columns: Vec<FeatureColumn>,
}

impl FeatureBatch {
    pub fn new(timestamps: Vec<i64>) -> Self {
        Self { timestamps, columns: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.timestamps.len()
    }

    pub fn push_f64(&mut self, name: impl Into<String>, values: Vec<f64>) {
        debug_assert_eq!(values.len(), self.timestamps.len(), "column len != timestamps len");
        self.columns.push(FeatureColumn::F64 { name: name.into(), values });
    }

    pub fn push_json(&mut self, name: impl Into<String>, values: Vec<serde_json::Value>) {
        debug_assert_eq!(values.len(), self.timestamps.len(), "column len != timestamps len");
        self.columns.push(FeatureColumn::Json { name: name.into(), values });
    }

    /// O(cols) lookup — cols мало (20-40), это нормально и НЕ выделяет память.
    pub fn get_f64(&self, name: &str) -> Option<&[f64]> {
        for c in &self.columns {
            if let FeatureColumn::F64 { name: n, values } = c {
                if n == name {
                    return Some(values.as_slice());
                }
            }
        }
        None
    }

    pub fn get_json(&self, name: &str) -> Option<&[serde_json::Value]> {
        for c in &self.columns {
            if let FeatureColumn::Json { name: n, values } = c {
                if n == name {
                    return Some(values.as_slice());
                }
            }
        }
        None
    }
}