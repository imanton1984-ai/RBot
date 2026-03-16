// strategies/super_level_strategy/src/model.rs
//
// Model Manager for Super Level Strategy (5 XGBoost models)
//
// Models:
//   1. slvl_level_v1_tf{X}.ubj   — P(strong_level) — quality of nearest level
//   2. slvl_entry_v1_tf{X}.ubj   — P(good_entry)   — is this a good entry point?
//   3. slvl_dir_v1_tf{X}.ubj     — P(LONG)         — direction prediction
//   4. slvl_bb_v1_tf{X}.ubj      — P(bounce)       — bounce or break?
//   5. slvl_eval_v1_tf{X}.ubj    — P(win)          — final evaluator

use anyhow::Result;
use tracing::{info, warn};
use std::collections::HashMap;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};
use crate::config::SuperLevelConfig;

/// All 5 predictions from the model ensemble
#[derive(Debug, Clone, Copy)]
pub struct SuperLevelPrediction {
    /// P(strong level) — качество уровня (0..1)
    pub p_level: f32,
    /// P(good entry) — качество точки входа (0..1)
    pub p_entry: f32,
    /// P(LONG) — вероятность лонга (>0.5 = LONG, <0.5 = SHORT)
    pub p_long: f32,
    /// P(bounce) — вероятность отскока (>0.5 = bounce, <0.5 = break)
    pub p_bounce: f32,
    /// P(win) — финальная оценка от evaluator (0..1)
    pub p_eval: f32,
    /// Derived direction
    pub direction: i8,
    /// Derived scenario
    pub is_bounce: bool,
}

/// Loaded models for one TF (any can be None if not trained yet)
struct TfModels {
    level_model: Option<Booster>,
    entry_model: Option<Booster>,
    dir_model: Option<Booster>,
    bounce_break_model: Option<Booster>,
    evaluator_model: Option<Booster>,
}

/// Super Level Model Manager
pub struct SuperLevelModelManager {
    models: HashMap<i32, TfModels>,
    config: SuperLevelConfig,
}

impl SuperLevelModelManager {
    /// Load all 5 models per TF. Missing models are OK — they'll return 0.5.
    pub fn new(config: SuperLevelConfig, use_gpu: bool) -> Result<Self> {
        let mut models = HashMap::new();
        let timeframes = SuperLevelConfig::timeframes();

        let device = if use_gpu { Device::Cuda } else { Device::Cpu };
        info!("Loading super_level models (device={:?})...", device);

        for &tf in timeframes {
            let level_path = config.model_path(&config.level_model_template, tf);
            let entry_path = config.model_path(&config.entry_model_template, tf);
            let dir_path = config.model_path(&config.direction_model_template, tf);
            let bb_path = config.model_path(&config.bounce_break_model_template, tf);
            let eval_path = config.model_path(&config.evaluator_model_template, tf);

            let load_model = |path: &str, name: &str| -> Option<Booster> {
                if !std::path::Path::new(path).exists() {
                    warn!("Model not found: {} — {} will use default", path, name);
                    return None;
                }
                match Booster::load(path, device) {
                    Ok(b) => {
                        info!("✅ Loaded {} TF {}m: {}", name, tf, path);
                        Some(b)
                    }
                    Err(e) => {
                        if use_gpu {
                            // Fallback to CPU
                            Booster::load(path, Device::Cpu).ok().map(|b| {
                                info!("✅ Loaded {} TF {}m: {} (CPU fallback)", name, tf, path);
                                b
                            })
                        } else {
                            warn!("❌ Failed to load {}: {}", path, e);
                            None
                        }
                    }
                }
            };

            let level_model = load_model(&level_path, "level");
            let entry_model = load_model(&entry_path, "entry");
            let dir_model = load_model(&dir_path, "direction");
            let bounce_break_model = load_model(&bb_path, "bounce_break");
            let evaluator_model = load_model(&eval_path, "evaluator");

            // Insert even if some models are missing — we'll use defaults
            let has_any = level_model.is_some() || entry_model.is_some()
                || dir_model.is_some() || bounce_break_model.is_some()
                || evaluator_model.is_some();

            if has_any {
                models.insert(tf, TfModels {
                    level_model, entry_model, dir_model,
                    bounce_break_model, evaluator_model,
                });
            }

            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        let loaded_tfs = models.len();
        let total_models: usize = models.values().map(|m| {
            [&m.level_model, &m.entry_model, &m.dir_model,
             &m.bounce_break_model, &m.evaluator_model]
                .iter().filter(|o| o.is_some()).count()
        }).sum();
        info!("SuperLevel models loaded: {} TFs, {} total models", loaded_tfs, total_models);

        Ok(Self { models, config })
    }

    pub fn has_models(&self) -> bool {
        !self.models.is_empty()
    }

    pub fn has_model_for_tf(&self, tf_minutes: i32) -> bool {
        self.models.contains_key(&tf_minutes)
    }

    pub fn available_timeframes(&self) -> Vec<i32> {
        let mut tfs: Vec<i32> = self.models.keys().copied().collect();
        tfs.sort();
        tfs
    }

    /// Single prediction for one feature row across all 5 models
    fn predict_one_model(
        booster: Option<&Booster>,
        features: &[f32],
        ncol: usize,
        default: f32,
    ) -> f32 {
        match booster {
            Some(b) => {
                b.predict_dense_cpu(features, 1, ncol, ModelKind::Regressor1)
                    .ok()
                    .and_then(|v| v.first().copied())
                    .unwrap_or(default)
                    .clamp(0.0, 1.0)
            }
            None => default,
        }
    }

    /// Batch prediction for multiple rows across all 5 models
    fn predict_batch_model(
        booster: Option<&Booster>,
        features: &[f32],
        nrow: usize,
        ncol: usize,
        default: f32,
    ) -> Vec<f32> {
        match booster {
            Some(b) => {
                b.predict_dense_cpu(features, nrow, ncol, ModelKind::Regressor1)
                    .unwrap_or_else(|_| vec![default; nrow])
                    .into_iter()
                    .map(|v| v.clamp(0.0, 1.0))
                    .collect()
            }
            None => vec![default; nrow],
        }
    }

    /// Predict all 5 models for a batch of feature rows
    pub fn predict_batch(
        &self,
        tf_minutes: i32,
        features_batch: &[f32],
        nrow: usize,
        ncol: usize,
    ) -> Result<Vec<SuperLevelPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let p_level_vec = Self::predict_batch_model(tf_models.level_model.as_ref(), features_batch, nrow, ncol, 0.5);
        let p_entry_vec = Self::predict_batch_model(tf_models.entry_model.as_ref(), features_batch, nrow, ncol, 0.5);
        let p_long_vec = Self::predict_batch_model(tf_models.dir_model.as_ref(), features_batch, nrow, ncol, 0.5);
        let p_bounce_vec = Self::predict_batch_model(tf_models.bounce_break_model.as_ref(), features_batch, nrow, ncol, 0.5);
        let p_eval_vec = Self::predict_batch_model(tf_models.evaluator_model.as_ref(), features_batch, nrow, ncol, 0.5);

        let mut predictions = Vec::with_capacity(nrow);
        for i in 0..nrow {
            let p_long = p_long_vec[i];
            let p_bounce = p_bounce_vec[i];
            predictions.push(SuperLevelPrediction {
                p_level: p_level_vec[i],
                p_entry: p_entry_vec[i],
                p_long,
                p_bounce,
                p_eval: p_eval_vec[i],
                direction: if p_long >= 0.5 { 1 } else { -1 },
                is_bounce: p_bounce >= 0.5,
            });
        }

        Ok(predictions)
    }

    /// Predict for a single feature row
    pub fn predict(
        &self,
        tf_minutes: i32,
        features: &[f32],
    ) -> Result<Option<SuperLevelPrediction>> {
        let tf_models = match self.models.get(&tf_minutes) {
            Some(m) => m,
            None => return Ok(None),
        };

        let ncol = features.len();
        let p_level = Self::predict_one_model(tf_models.level_model.as_ref(), features, ncol, 0.5);
        let p_entry = Self::predict_one_model(tf_models.entry_model.as_ref(), features, ncol, 0.5);
        let p_long = Self::predict_one_model(tf_models.dir_model.as_ref(), features, ncol, 0.5);
        let p_bounce = Self::predict_one_model(tf_models.bounce_break_model.as_ref(), features, ncol, 0.5);
        let p_eval = Self::predict_one_model(tf_models.evaluator_model.as_ref(), features, ncol, 0.5);

        Ok(Some(SuperLevelPrediction {
            p_level,
            p_entry,
            p_long,
            p_bounce,
            p_eval,
            direction: if p_long >= 0.5 { 1 } else { -1 },
            is_bounce: p_bounce >= 0.5,
        }))
    }

    pub fn config(&self) -> &SuperLevelConfig {
        &self.config
    }
}
