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
}