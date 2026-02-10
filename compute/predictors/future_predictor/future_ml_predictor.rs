// compute/predictors/future_price/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::predictors::ml::onnx_runtime::OnnxRunner; // Предполагаем, что runner тут

pub struct FuturePriceMl {
    runner: OnnxRunner,
    model_name: String,
}

impl FuturePriceMl {
    pub fn new(model_path: &str, use_cuda: bool) -> Result<Self> {
        let mut runner = OnnxRunner::new();
        // Если файла нет, можно сделать warn и не грузить, но лучше fail fast или graceful degradation
        if std::path::Path::new(model_path).exists() {
            runner.initialize_models(&[crate::predictors::ml::onnx_runtime::ModelConfig::new("price_model", model_path, "XGBoost")], use_cuda)?;
        }
        Ok(Self { runner, model_name: "price_model".to_string() })
    }

    pub async fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureVector) -> Result<Option<(Vec<f64>, f64)>> {
        if !self.runner.model_manager.has_model(&self.model_name) {
            return Ok(None); // Модель не загружена
        }

        // Преобразование Vec<f32> в формат для ONNX (slice)
        let (prices, conf) = self.runner.run_price_prediction(&self.model_name, &features.values)?;
        
        if conf >= 0.80 {
            Ok(Some((prices, conf)))
        } else {
            Ok(None)
        }
    }
}