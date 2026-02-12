// compute/predictors/level_predictor/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::ml::model_manager::ModelManager;
use std::sync::Arc;

pub struct LevelPredictorMl {
    mm: Arc<ModelManager>,
    model_key: String,
    use_gpu: bool,
}

impl LevelPredictorMl {
    pub fn new(model_key: impl Into<String>, use_gpu: bool, mm: Arc<ModelManager>) -> Result<Self> {
        Ok(Self { mm, model_key: model_key.into(), use_gpu })
    }

    pub async fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureVector, _level_price: f64) -> Result<Option<(f64, f64, f64)>> {
        let out = self.mm.predict_one(&self.model_key, &features.values, self.use_gpu)?;
        let Some(outputs) = out else { return Ok(None); };

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