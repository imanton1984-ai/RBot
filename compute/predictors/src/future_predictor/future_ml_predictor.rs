// compute/predictors/future_price/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::ml::onnx_runtime::OnnxRunner; // Предполагаем, что runner тут
use std::sync::Arc;
use crate::ml::model_pool::ModelPool;

pub struct FuturePriceMl {
    runner: OnnxRunner,
}

impl FuturePriceMl {
    pub fn new(model_path: &str, use_cuda: bool, pool: Arc<ModelPool>) -> Result<Self> {
        let runner = OnnxRunner::new(model_path, use_cuda, pool)?;
        Ok(Self { runner })
    }

    pub async fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureVector) -> Result<Option<(Vec<f64>, f64)>> {
        // Преобразование Vec<f32> в формат для ONNX (slice)
        let prices = self.runner.run(&features.values)?;

        if !prices.is_empty() {
            let avg_price = prices.iter().sum::<f32>() as f64 / prices.len() as f64;
            let prices_f64: Vec<f64> = prices.iter().map(|&x| x as f64).collect();
            Ok(Some((prices_f64, avg_price.abs())))
        } else {
            Ok(None)
        }
    }
}