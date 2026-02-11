use super::onnx_runtime::OnnxRunner;
use super::model_pool::ModelPool;
use crate::feature_schema::FeatureSchema;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, warn};

pub struct ModelManager {
    models: HashMap<String, OnnxRunner>,
    schemas: HashMap<String, FeatureSchema>,
    use_cuda: bool,
    pool: Arc<ModelPool>,
}

impl ModelManager {
    pub fn new(use_cuda: bool, pool: Arc<ModelPool>) -> Self {
        Self {
            models: HashMap::new(),
            schemas: HashMap::new(),
            use_cuda,
            pool,
        }
    }

    /// Загружает модель по указанному пути и присваивает ей имя (ключ)
    pub fn load_model(&mut self, key: &str, path: &str) -> Result<()> {
        if std::path::Path::new(path).exists() {
            info!("Loading model '{}' from {}", key, path);
            let runner = OnnxRunner::new(path, self.use_cuda, self.pool.clone())?;
            self.models.insert(key.to_string(), runner);

            // Load schema
            let schema_path = path.replace(".onnx", ".json");
            if std::path::Path::new(&schema_path).exists() {
                let schema = FeatureSchema::from_json_file(&schema_path)?;
                self.schemas.insert(key.to_string(), schema);
            } else {
                warn!("Schema file not found at {}", schema_path);
            }

        } else {
            warn!("Model file not found at {}. ML prediction for '{}' will be disabled.", path, key);
        }
        Ok(())
    }

    pub fn get_schema(&self, key: &str) -> Option<&FeatureSchema> {
        self.schemas.get(key)
    }

    /// Проверяет, загружена ли модель
    pub fn has_model(&self, key: &str) -> bool {
        self.models.contains_key(key)
    }

    /// Запускает инференс для конкретной модели
    pub fn predict(&self, key: &str, features: &[f32]) -> Result<Option<Vec<f32>>> {
        if let Some(runner) = self.models.get(key) {
            let result = runner.run(features)?;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Загружает модели для всех таймфреймов по заданному шаблону
    pub fn load_models_for_timeframes(&mut self, model_type: &str, base_path: &str, timeframes: &[i32]) -> Result<()> {
        info!("Starting to load {} models for timeframes: {:?}", model_type, timeframes);
        
        for &tf in timeframes {
            let key = format!("{}_tf{}", model_type, tf);
            let path = base_path.replace(".onnx", &format!("_tf{}.onnx", tf));
            
            info!("Attempting to load {} model: {} from path: {}", model_type, key, path);
            
            if std::path::Path::new(&path).exists() {
                info!("Loading {} model for timeframe {}m: {} from {}", model_type, tf, key, path);
                let runner = OnnxRunner::new(&path, self.use_cuda, self.pool.clone())?;
                self.models.insert(key.clone(), runner);

                // Load schema
                let schema_path = path.replace(".onnx", ".json");
                info!("Attempting to load schema for {} model from: {}", model_type, schema_path);
                
                if std::path::Path::new(&schema_path).exists() {
                    let schema = FeatureSchema::from_json_file(&schema_path)?;
                    self.schemas.insert(key.clone(), schema);
                    info!("Successfully loaded schema for {} model: {}", model_type, key);
                } else {
                    warn!("Schema file not found at {} for {} model", schema_path, key);
                }
                
                info!("Successfully loaded {} model: {} with schema present: {}", model_type, key, self.schemas.contains_key(&key));
            } else {
                warn!("Model file not found at {} for timeframe {}m. ML prediction for '{}' will be disabled.", path, tf, key);
            }
        }
        
        info!("Completed loading {} models. Total {} models loaded.", model_type, self.models.len());
        Ok(())
    }

    /// Получает модель по типу и таймфрейму
    pub fn get_model_for_timeframe(&self, model_type: &str, timeframe_minutes: i32) -> Option<&OnnxRunner> {
        let key = format!("{}_tf{}", model_type, timeframe_minutes);
        self.models.get(&key)
    }

    /// Получает схему по типу и таймфрейму
    pub fn get_schema_for_timeframe(&self, model_type: &str, timeframe_minutes: i32) -> Option<&FeatureSchema> {
        let key = format!("{}_tf{}", model_type, timeframe_minutes);
        self.schemas.get(&key)
    }
}