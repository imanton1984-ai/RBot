use anyhow::{Context, Result};
use ndarray::{Array, Array2, Axis};
use ort::{Value, Session};
use std::sync::Arc;
use tracing::{info, warn, error};
use super::model_pool::ModelPool;

pub struct OnnxRunner {
    session: Arc<Session>,
}

impl OnnxRunner {
    /// Загружает модель из файла. Пытается использовать CUDA, если доступно.
    pub fn new(model_path: &str, use_cuda: bool, pool: Arc<ModelPool>) -> Result<Self> {
        let session = pool.get_or_load(model_path, use_cuda)?;
        Ok(Self { session })
    }

    /// Выполняет предсказание.
    /// features: вектор фичей (размерность [1, n_features])
    /// Возвращает вектор предсказаний.
    pub fn run(&self, features: &[f32]) -> Result<Vec<f32>> {
        // 1. Создаем тензор из входных данных [Batch Size, Features]
        let input_array = Array2::from_shape_vec((1, features.len()), features.to_vec())?;
        let input_tensor = Value::from_array(input_array)?;

        // 2. Запускаем сессию
        let outputs = self.session.run(ort::inputs![input_tensor]?)?;

        // 3. Извлекаем результат (предполагаем, что выход 0 - это то, что нам нужно)
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        
        // Преобразуем в Vec<f32>
        let result: Vec<f32> = output_tensor.view().iter().cloned().collect();
        Ok(result)
    }
}