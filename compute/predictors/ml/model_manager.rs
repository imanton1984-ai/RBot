use super::onnx_runtime::OnnxRunner;
use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, warn};

pub struct ModelManager {
    models: HashMap<String, OnnxRunner>,
    use_cuda: bool,
}

impl ModelManager {
    pub fn new(use_cuda: bool) -> Self {
        Self {
            models: HashMap::new(),
            use_cuda,
        }
    }

    /// Загружает модель по указанному пути и присваивает ей имя (ключ)
    pub fn load_model(&mut self, key: &str, path: &str) -> Result<()> {
        if std::path::Path::new(path).exists() {
            info!("Loading model '{}' from {}", key, path);
            let runner = OnnxRunner::new(path, self.use_cuda)?;
            self.models.insert(key.to_string(), runner);
        } else {
            warn!("Model file not found at {}. ML prediction for '{}' will be disabled.", path, key);
        }
        Ok(())
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