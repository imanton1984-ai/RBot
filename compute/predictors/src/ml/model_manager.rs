use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, warn};

use crate::feature_schema::FeatureSchema;

use super::xgb_runtime::{Booster, Device, ModelKind};

pub struct ModelManager {
    // ключ: price_tf5 / level_tf1440 и т.д.
    models_cpu: HashMap<String, (Booster, ModelKind)>,
    models_gpu: HashMap<String, (Booster, ModelKind)>,
    pub schemas: HashMap<String, FeatureSchema>,
}

impl ModelManager {
    pub fn new(use_cuda: bool) -> Self {
        let manager = Self {
            models_cpu: HashMap::new(),
            models_gpu: HashMap::new(),
            schemas: HashMap::new(),
        };
        
        // Load models based on CUDA availability
        if use_cuda {
            info!("CUDA is enabled, will attempt to load GPU models");
        } else {
            info!("CUDA is disabled, will only load CPU models");
        }
        
        manager
    }

    pub fn load_models_for_timeframes(
        &mut self,
        model_type: &str,                // "price" | "levels"
        model_template: &str,            // "models/price_v1_tf{tf}.ubj"
        timeframes: &[i32],
        enable_gpu: bool,
    ) -> Result<()> {
        info!("Loading {} models for {:?}...", model_type, timeframes);

        let kind = match model_type {
            "price" => ModelKind::Regressor1,
            "levels" => ModelKind::BinaryProb2,
            other => {
                warn!("Unknown model_type '{}', defaulting to Regressor1", other);
                ModelKind::Regressor1
            }
        };

        for &tf in timeframes {
            let key = format!("{}_tf{}", model_type, tf);
            let path = model_template.replace("{tf}", &tf.to_string());
            let schema_path = path.replace(".ubj", ".json"); // schema рядом

            if std::path::Path::new(&schema_path).exists() {
                let schema = FeatureSchema::from_json_file(&schema_path)?;
                self.schemas.insert(key.clone(), schema);
            } else {
                warn!("Schema not found: {}", schema_path);
            }

            if std::path::Path::new(&path).exists() {
                info!("Loading model: {} -> {}", key, path);

                let cpu = Booster::load(&path, Device::Cpu)?;
                self.models_cpu.insert(key.clone(), (cpu, kind));

                if enable_gpu {
                    match Booster::load(&path, Device::Cuda) {
                        Ok(gpu) => {
                            self.models_gpu.insert(key.clone(), (gpu, kind));
                            info!("Successfully loaded GPU model: {}", key);
                        }
                        Err(e) => {
                            warn!("Failed to load GPU model for {}: {}, falling back to CPU", key, e);
                        }
                    }
                }
            } else {
                warn!("Model file not found: {}", path);
            }
        }

        Ok(())
    }

    /// Batch predict: inputs = row-major [nrow * ncol]
    pub fn predict_batch(&self, key: &str, inputs: &[f32], nrow: usize, ncol: usize, use_gpu: bool) -> Result<Option<Vec<f32>>> {
        let map = if use_gpu { &self.models_gpu } else { &self.models_cpu };

        if let Some((booster, kind)) = map.get(key) {
            // When booster is loaded with device="cuda", XGBoost will use GPU internally
            // even when data comes from CPU memory
            let out = booster.predict_dense_cpu(inputs, nrow, ncol, *kind)?;
            Ok(Some(out))
        } else {
            // Fallback to CPU if GPU model not available
            if let Some((cpu_booster, kind)) = self.models_cpu.get(key) {
                let out = cpu_booster.predict_dense_cpu(inputs, nrow, ncol, *kind)?;
                Ok(Some(out))
            } else {
                Ok(None)
            }
        }
    }

    /// Single-row helper (realtime)
    pub fn predict_one(&self, key: &str, features: &[f32], use_gpu: bool) -> Result<Option<Vec<f32>>> {
        self.predict_batch(key, features, 1, features.len(), use_gpu)
    }
    
    /// Check if a model exists for a given key
    pub fn has_model(&self, key: &str) -> bool {
        self.models_cpu.contains_key(key) || self.models_gpu.contains_key(key)
    }
    
    /// Get schema for a given key
    pub fn get_schema(&self, key: &str) -> Option<&FeatureSchema> {
        self.schemas.get(key)
    }
}