// compute/predictors/level_predictor/ml_predictor.rs

use anyhow::Result;
use crate::feature_view::FeatureVector;
use crate::ml::model_manager::ModelManager;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ML_PRED_SAMPLE: AtomicU64 = AtomicU64::new(0);

pub struct LevelPredictorMl {
    mm: Arc<ModelManager>,
    model_key: String,
    use_gpu: bool,
}

impl LevelPredictorMl {
    pub fn new(model_key: impl Into<String>, use_gpu: bool, mm: Arc<ModelManager>) -> Result<Self> {
        Ok(Self { mm, model_key: model_key.into(), use_gpu })
    }

    pub async fn predict(&self, symbol: &str, timeframe: &str, features: &FeatureVector, _level_price: f64) -> Result<Option<(f64, f64, f64)>> {
        let sample_n = ML_PRED_SAMPLE.fetch_add(1, Ordering::Relaxed);
        let do_log = sample_n % 200 == 0; // каждые 200 вызовов

        let t0 = Instant::now();
        let res = self.mm.predict_one(&self.model_key, &features.values, self.use_gpu);
        let ms = t0.elapsed().as_millis();

        if do_log {
            tracing::info!(target: "compute_predictors",
                "ML predict sample: sym={} tf={} key={} use_gpu={} vec_len={} ms={}",
                symbol, timeframe, self.model_key, self.use_gpu, features.values.len(), ms
            );
        }

        let out = res?;
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