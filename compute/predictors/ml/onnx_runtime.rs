use anyhow::{Context, Result};
use ndarray::{Array, Array2, Axis};
use ort::{
    GraphOptimizationLevel, Session, SessionBuilder, Value,
    ExecutionProvider,
};
use std::sync::Arc;
use tracing::{info, warn, error};

pub struct OnnxRunner {
    session: Arc<Session>,
}

impl OnnxRunner {
    /// Загружает модель из файла. Пытается использовать CUDA, если доступно.
    pub fn new(model_path: &str, use_cuda: bool) -> Result<Self> {
        let mut builder = Session::builder()?;
        
        builder = builder.with_optimization_level(GraphOptimizationLevel::Level3)?;
        builder = builder.with_intra_threads(4)?;

        if use_cuda {
            // Пытаемся подключить CUDA
            match builder.clone().with_execution_providers([ExecutionProvider::CUDA(Default::default())]) {
                Ok(b) => {
                    builder = b;
                    info!("CUDA Execution Provider enabled for model: {}", model_path);
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA provider: {}. Falling back to CPU.", e);
                }
            }
        }

        let session = builder.commit_from_file(model_path)
            .context(format!("Failed to load ONNX model from {}", model_path))?;

        Ok(Self {
            session: Arc::new(session),
        })
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