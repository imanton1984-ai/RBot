// compute/predictors/level_predictor/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::ml::onnx_runtime::OnnxRunner;
use crate::ml::model_pool::ModelPool;
use std::sync::Arc;

pub struct LevelPredictorMl {
    runner: OnnxRunner,
}

impl LevelPredictorMl {
    pub fn new(model_path: &str, use_cuda: bool, pool: Arc<ModelPool>) -> Result<Self> {
        let runner = OnnxRunner::new(model_path, use_cuda, pool)?;
        Ok(Self { runner })
    }

    pub async fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureVector, _level_price: f64) -> Result<Option<(f64, f64, f64)>> {
        // Преобразование Vec<f32> в формат для ONNX (slice)
        let outputs = self.runner.run(&features.values)?;

        if outputs.len() >= 2 {
            let prob_bounce = outputs[0] as f64;
            let prob_break = outputs[1] as f64;
            let conf = ((prob_bounce + prob_break) / 2.0).min(1.0);
            Ok(Some((prob_bounce, prob_break, conf)))
        } else {
            Ok(None)
        }
    }
}