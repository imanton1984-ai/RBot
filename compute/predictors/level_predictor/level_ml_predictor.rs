// compute/predictors/level_predictor/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::predictors::ml::onnx_runtime::OnnxRunner;

pub struct LevelPredictorMl {
    runner: OnnxRunner,
    model_name: String,
}

impl LevelPredictorMl {
    pub fn new(model_path: &str, use_cuda: bool) -> Result<Self> {
        let mut runner = OnnxRunner::new();
        if std::path::Path::new(model_path).exists() {
            runner.initialize_models(&[crate::predictors::ml::onnx_runtime::ModelConfig::new("level_model", model_path, "LightGBM")], use_cuda)?;
        }
        Ok(Self { runner, model_name: "level_model".to_string() })
    }

    pub async fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureVector, level_price: f64) -> Result<Option<(f64, f64, f64)>> {
        if !self.runner.model_manager.has_model(&self.model_name) {
            return Ok(None); // Модель не загружена
        }

        // Преобразование Vec<f32> в формат для ONNX (slice)
        let (prob_bounce, prob_break, conf) = self.runner.run_level_prediction(&self.model_name, &features.values, level_price)?;
        
        if conf >= 0.80 {
            Ok(Some((prob_bounce, prob_break, conf)))
        } else {
            Ok(None)
        }
    }
}