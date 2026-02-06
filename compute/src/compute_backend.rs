use std::sync::Arc;
use std::collections::HashMap;
use common::{Symbol, Timeframe};
use crate::{CandleWindow, FeatureBatch, FeatureValue};

#[derive(Debug, Clone)]
pub struct ComputeJob {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub window_start: i64,
    pub window_end: i64,
    pub indicators: Vec<String>,
    pub candle_window: Option<CandleWindow>,
}

/// Legacy (оставляем для совместимости, но в fast-path больше НЕ создаём)
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ComputeResult {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,
    pub features: HashMap<String, FeatureValue>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FeatureWindow {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub start_time: i64,
    pub end_time: i64,

    /// Держим свечи как раньше (это ок). Но НЕ держим per-bar feature maps.
    pub candle_window: Option<CandleWindow>,

    /// Новое columnar-хранилище фич
    pub batch: FeatureBatch,

    /// Legacy: по умолчанию None (не заполняем, чтобы не жрать RAM)
    pub legacy_features: Option<Vec<ComputeResult>>,
}

impl FeatureWindow {
    pub fn new(start_time: i64, end_time: i64, candle_window: Option<CandleWindow>) -> Self {
        Self {
            symbol: Symbol::new(""), // Will be set appropriately
            timeframe: Timeframe::M1, // Will be set appropriately
            start_time,
            end_time,
            candle_window,
            batch: FeatureBatch::new(Vec::new()), // Empty batch initially
            legacy_features: None,
        }
    }

    pub fn get_indicator_values(&self, indicator_name: &str) -> Vec<f64> {
        // Try to get from the new columnar format first
        if let Some(values) = self.batch.get_f64(indicator_name) {
            values.to_vec()
        } else {
            // Fallback to legacy format if needed
            let mut values = Vec::new();
            if let Some(legacy_features) = &self.legacy_features {
                for result in legacy_features {
                    if let Some(feature_value) = result.features.get(indicator_name) {
                        if let FeatureValue::Float(val) = feature_value {
                            values.push(*val);
                        }
                    }
                }
            }
            values
        }
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