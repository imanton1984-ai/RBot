use anyhow::{Result, Context};
use ndarray::Array2;
use ort::execution_providers::CUDAExecutionProvider;
use ort::session::Session;
use ort::value::Value;
use std::sync::{Arc, Mutex};
use super::model_pool::ModelPool;

pub struct OnnxRunner {
    session: Arc<Mutex<Session>>,
}

impl OnnxRunner {
    pub fn new(model_path: &str, use_cuda: bool, _pool: Arc<ModelPool>) -> Result<Self> {
        let builder = if use_cuda {
            let cuda_provider = CUDAExecutionProvider::default();
            match Session::builder()?.with_execution_providers([cuda_provider.build()]) {
                Ok(b) => {
                    tracing::info!("CUDA provider registered for model: {}", model_path);
                    b
                },
                Err(e) => {
                    tracing::warn!("Failed to register CUDA provider for ONNX, falling back to CPU: {}", e);
                    Session::builder()?
                }
            }
        } else {
            Session::builder()?
        };

        let session = builder.commit_from_file(model_path)
            .context(format!("Failed to load model from {}", model_path))?;
            
        let session = Arc::new(Mutex::new(session));
        Ok(Self { session })
    }

    pub fn run(&self, features: &[f32]) -> Result<Vec<f32>> {
        let input_array = Array2::from_shape_vec((1, features.len()), features.to_vec())?;
        
        let mut session_guard = self.session.lock().unwrap();
        let input_name = session_guard.inputs()[0].name().to_string();
        let input_value = Value::from_array((input_array.shape().to_vec(), input_array.as_slice().unwrap().to_vec()))?;
        
        let outputs = session_guard.run(ort::inputs![input_name.as_str() => input_value])?;

        let mut best_result: Option<Vec<f32>> = None;
        if outputs.len() > 1 {
             if let Ok(output_tensor) = outputs[1].try_extract_tensor::<f32>() {
                let vec = output_tensor.1.to_vec();
                if vec.len() > 1 { best_result = Some(vec); }
            }
        }
        if let Some(res) = best_result { return Ok(res); }
        
        if let Ok(output_tensor) = outputs[0].try_extract_tensor::<f32>() {
             return Ok(output_tensor.1.to_vec());
        }
        
        anyhow::bail!("Output extraction failed")
    }
}