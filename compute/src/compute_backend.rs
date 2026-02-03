use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{CandleWindow, FeatureValue};

#[derive(Debug, Clone)]
pub struct ComputeJob {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub window_start: i64,
    pub window_end: i64,
    pub indicators: Vec<String>,
    pub candle_window: Option<CandleWindow>,
}

#[derive(Debug, Clone)]
pub struct ComputeResult {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,
    pub features: std::collections::HashMap<String, FeatureValue>,
}

#[derive(Debug, Clone)]
pub struct FeatureWindow {
    pub features: Vec<ComputeResult>,
    pub start_time: i64,
    pub end_time: i64,
    pub candle_window: Option<CandleWindow>,
}

impl FeatureWindow {
    pub fn new(start_time: i64, end_time: i64, candle_window: Option<CandleWindow>) -> Self {
        Self {
            features: Vec::new(),
            start_time,
            end_time,
            candle_window,
        }
    }

    pub fn get_indicator_values(&self, indicator_name: &str) -> Vec<f64> {
        let mut values = Vec::new();
        for result in &self.features {
            if let Some(feature_value) = result.features.get(indicator_name) {
                if let FeatureValue::Float(val) = feature_value {
                    values.push(*val);
                }
            }
        }
        values
    }
}

#[async_trait::async_trait]
pub trait ComputeBackend: Send + Sync {
    async fn compute_indicators(
        &self,
        jobs: Vec<ComputeJob>,
    ) -> Result<Vec<Arc<FeatureWindow>>, Box<dyn std::error::Error + Send + Sync>>;

    async fn compute_single_indicator(
        &self,
        symbol: Symbol,
        timeframe: Timeframe,
        prices: &[f64],
        indicator_name: &str,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>>;
}

pub enum ComputeBackendType {
    Cuda,
    Cpu,
}

pub struct ComputeBackendManager {
    backend: Arc<dyn ComputeBackend>,
}

impl ComputeBackendManager {
    pub fn new(backend_type: ComputeBackendType) -> Self {
        let backend: Arc<dyn ComputeBackend> = match backend_type {
            ComputeBackendType::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    Arc::new(crate::cuda_backend::CudaBackend::new())
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Arc::new(crate::cpu_backend::CpuBackend::new())
                }
            }
            ComputeBackendType::Cpu => Arc::new(crate::cpu_backend::CpuBackend::new()),
        };

        Self { backend }
    }

    pub fn get_backend(&self) -> Arc<dyn ComputeBackend> {
        self.backend.clone()
    }
}