use anyhow::{Result};
use ndarray::{Array2};
use ort::{session::Session, value::Value};
use std::sync::{Arc, Mutex};
use super::model_pool::ModelPool;

pub struct OnnxRunner {
    session: Arc<Mutex<Session>>,
}

impl OnnxRunner {
    /// Загружает модель из файла. Пытается использовать CUDA, если доступно.
    pub fn new(model_path: &str, use_cuda: bool, pool: Arc<ModelPool>) -> Result<Self> {
        let _session = pool.get_or_load(model_path, use_cuda)?;  // Just to validate the model exists
        // Since we can't clone Session, we'll create a new one from the same model file
        let new_session = Session::builder()?.commit_from_file(model_path)?;
        let session = Arc::new(Mutex::new(new_session));
        Ok(Self { session })
    }

    /// Выполняет предсказание.
    /// features: вектор фичей (размерность [1, n_features])
    /// Возвращает вектор предсказаний.
    pub fn run(&self, features: &[f32]) -> Result<Vec<f32>> {
        // 1. Создаем тензор из входных данных [Batch Size, Features]
        let input_array = Array2::from_shape_vec((1, features.len()), features.to_vec())?;

        // 2. Get the input name first
        let input_name = {
            let session_guard = self.session.lock().unwrap();
            session_guard.inputs()[0].name().to_string() // Clone the name to avoid borrowing issues
        };

        // 3. Create the input value
        let input_value = Value::from_array((input_array.shape().to_vec(), input_array.as_slice().unwrap().to_vec()))?;

        // 4. Run the session
        let mut session_guard = self.session.lock().unwrap();
        let outputs = session_guard.run(ort::inputs![input_name.as_str() => input_value])?;

        // 5. Извлекаем результат (предполагаем, что выход 0 - это то, что нам нужно)
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;

        let result: Vec<f32> = output_tensor.1.to_vec(); // .1 is the data slice &[f32]
        Ok(result)
    }
}