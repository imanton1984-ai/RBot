// compute/predictors/future_price/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::ml::model_manager::ModelManager;
use std::sync::Arc;

pub struct FuturePriceMl {
    mm: Arc<ModelManager>,
    model_key: String,
    use_gpu: bool,
}

impl FuturePriceMl {
    pub fn new(model_key: impl Into<String>, use_gpu: bool, mm: Arc<ModelManager>) -> Result<Self> {
        Ok(Self { mm, model_key: model_key.into(), use_gpu })
    }

    pub async fn predict(&self, _symbol: &str, _timeframe: &str, features: &FeatureVector) -> Result<Option<(Vec<f64>, f64)>> {
        // XGBoost: 1-row prediction
        let out = self.mm.predict_one(&self.model_key, &features.values, self.use_gpu)?;
        let Some(prices) = out else { return Ok(None); };

        if !prices.is_empty() {
            let avg = prices.iter().copied().sum::<f32>() as f64 / prices.len() as f64;
            let prices_f64: Vec<f64> = prices.into_iter().map(|x| x as f64).collect();
            Ok(Some((prices_f64, avg.abs())))
        } else {
            Ok(None)
        }
    }
}