// compute/predictions/ml/onnx_runtime.rs

use anyhow::Result;
use ort::{Session, GraphOptimizationLevel, ExecutionProvider, AllocatorType};
use std::collections::HashMap;

/// ONNX model manager for loading and running models
pub struct OnnxModelManager {
    models: HashMap<String, Session>,
}

impl OnnxModelManager {
    /// Creates a new ONNX model manager
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Loads an ONNX model from the given path
    pub fn load_model(&mut self, model_name: &str, model_path: &str, use_gpu: bool) -> Result<()> {
        let mut session_builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_allocator(AllocatorType::Arena)?;

        if use_gpu {
            // Try to use CUDA if available
            if let Err(_) = session_builder.with_execution_providers([ExecutionProvider::CUDA(Default::default())]) {
                tracing::warn!("CUDA not available, falling back to CPU for model: {}", model_path);
            }
        }

        let session = session_builder.commit_from_file(model_path)?;
        self.models.insert(model_name.to_string(), session);

        Ok(())
    }

    /// Runs inference on the specified model
    pub fn run_inference(&self, model_name: &str, inputs: &[ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>]) -> Result<Vec<ndarray::ArrayD<f32>>> {
        let session = self.models.get(model_name)
            .ok_or_else(|| anyhow::anyhow!("Model {} not found", model_name))?;

        // Convert inputs to ort values
        let mut ort_inputs = Vec::new();
        for (i, input) in inputs.iter().enumerate() {
            let input_shape: Vec<i64> = input.shape().iter().map(|&x| x as i64).collect();
            let input_values: Vec<f32> = input.as_slice().unwrap().to_vec();

            let ort_input = ort::inputs![
                input_values => input_shape
            ]?;

            ort_inputs.push((session.inputs_get_names()[i].to_string(), ort_input));
        }

        // Run the model
        let outputs = session.run(ort_inputs.into_iter().map(|(name, value)| (name.as_str(), value)).collect())?;

        // Convert outputs to ndarray
        let mut result = Vec::new();
        for output in outputs {
            let tensor = output.try_extract::<f32>()?;
            let array = tensor.try_extract()?.into_owned();
            result.push(array);
        }

        Ok(result)
    }

    /// Checks if a model is loaded
    pub fn has_model(&self, model_name: &str) -> bool {
        self.models.contains_key(model_name)
    }

    /// Gets a reference to a model
    pub fn get_model(&self, model_name: &str) -> Option<&Session> {
        self.models.get(model_name)
    }

    /// Gets the number of loaded models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }
}

/// ONNX model runner for specific prediction tasks
pub struct OnnxRunner {
    model_manager: OnnxModelManager,
}

impl OnnxRunner {
    /// Creates a new ONNX runner
    pub fn new() -> Self {
        Self {
            model_manager: OnnxModelManager::new(),
        }
    }

    /// Initializes the runner with models
    pub fn initialize_models(&mut self, models_config: &[ModelConfig], use_gpu: bool) -> Result<()> {
        for config in models_config {
            self.model_manager.load_model(&config.name, &config.path, use_gpu)?;
        }
        Ok(())
    }

    /// Runs price prediction model
    pub fn run_price_prediction(&self, model_name: &str, features: &[f32]) -> Result<(Vec<f64>, f64)> {
        // Convert features to 2D array (batch_size=1, feature_count=features.len())
        let input_array = ndarray::arr2(&[features]); // Shape: [1, feature_count]

        let outputs = self.model_manager.run_inference(model_name, &[input_array])?;

        if outputs.is_empty() {
            anyhow::bail!("Model {} returned no outputs", model_name);
        }

        // Process outputs - assuming first output is predicted prices, second is confidence
        let mut predicted_prices = Vec::new();
        let mut confidence = 0.5; // Default confidence

        for (i, output) in outputs.iter().enumerate() {
            let output_slice = output.view();

            if i == 0 {
                // First output: predicted prices for next N candles
                for &value in output_slice.iter() {
                    predicted_prices.push(value as f64);
                }
            } else if i == 1 {
                // Second output: confidence score
                if let Some(&conf) = output_slice.first() {
                    confidence = conf as f64;
                }
            }
        }

        // Ensure we have at least one predicted price
        if predicted_prices.is_empty() {
            // If no prices were returned, derive from the input features
            // This is a fallback - in practice, the model should return predictions
            predicted_prices.push(features[0] as f64); // Use first feature as proxy
        }

        Ok((predicted_prices, confidence.clamp(0.0, 1.0)))
    }

    /// Runs level prediction model
    pub fn run_level_prediction(
        &self,
        model_name: &str,
        features: &[f32],
        level_price: f64,
    ) -> Result<(f64, f64, f64)> { // (prob_bounce, prob_break, confidence)
        // Convert features to 2D array (batch_size=1, feature_count=features.len())
        let input_array = ndarray::arr2(&[features]); // Shape: [1, feature_count]

        let outputs = self.model_manager.run_inference(model_name, &[input_array])?;

        if outputs.is_empty() {
            anyhow::bail!("Model {} returned no outputs", model_name);
        }

        let mut prob_bounce = 0.5;
        let mut prob_break = 0.5;
        let mut confidence = 0.5;

        for (i, output) in outputs.iter().enumerate() {
            let output_slice = output.view();

            match i {
                0 => {
                    // First output: bounce probability
                    if let Some(&prob) = output_slice.first() {
                        prob_bounce = prob as f64;
                    }
                }
                1 => {
                    // Second output: break probability
                    if let Some(&prob) = output_slice.first() {
                        prob_break = prob as f64;
                    }
                }
                2 => {
                    // Third output: confidence
                    if let Some(&conf) = output_slice.first() {
                        confidence = conf as f64;
                    }
                }
                _ => {} // Ignore additional outputs
            }
        }

        Ok((
            prob_bounce.clamp(0.0, 1.0),
            prob_break.clamp(0.0, 1.0),
            confidence.clamp(0.0, 1.0)
        ))
    }

    /// Runs quantile regression model to get prediction bands
    pub fn run_quantile_prediction(&self, model_name: &str, features: &[f32]) -> Result<(f64, f64, f64, f64)> {
        // Convert features to 2D array (batch_size=1, feature_count=features.len())
        let input_array = ndarray::arr2(&[features]); // Shape: [1, feature_count]

        let outputs = self.model_manager.run_inference(model_name, &[input_array])?;

        if outputs.is_empty() {
            anyhow::bail!("Model {} returned no outputs", model_name);
        }

        let mut q10 = features[0] as f64; // Default to first feature
        let mut q50 = features[0] as f64;
        let mut q90 = features[0] as f64;
        let mut confidence = 0.5;

        for (i, output) in outputs.iter().enumerate() {
            let output_slice = output.view();

            match i {
                0 => {
                    // First output: q10 (10th percentile)
                    if let Some(&val) = output_slice.first() {
                        q10 = val as f64;
                    }
                }
                1 => {
                    // Second output: q50 (median)
                    if let Some(&val) = output_slice.first() {
                        q50 = val as f64;
                    }
                }
                2 => {
                    // Third output: q90 (90th percentile)
                    if let Some(&val) = output_slice.first() {
                        q90 = val as f64;
                    }
                }
                3 => {
                    // Fourth output: confidence
                    if let Some(&conf) = output_slice.first() {
                        confidence = conf as f64;
                    }
                }
                _ => {} // Ignore additional outputs
            }
        }

        Ok((q10, q50, q90, confidence.clamp(0.0, 1.0)))
    }
}

/// Configuration for a model
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub name: String,
    pub path: String,
    pub input_shape: Vec<usize>,
    pub output_shapes: Vec<Vec<usize>>,
    pub description: String,
}

impl ModelConfig {
    pub fn new(name: &str, path: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            path: path.to_string(),
            input_shape: vec![], // Will be determined at runtime
            output_shapes: vec![], // Will be determined at runtime
            description: description.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_creation() {
        let config = ModelConfig::new("test_model", "/path/to/model.onnx", "Test model");
        assert_eq!(config.name, "test_model");
        assert_eq!(config.path, "/path/to/model.onnx");
        assert_eq!(config.description, "Test model");
        assert!(config.input_shape.is_empty());
        assert!(config.output_shapes.is_empty());
    }

    #[test]
    fn test_onnx_runner_creation() {
        let runner = OnnxRunner::new();
        assert_eq!(runner.model_manager.model_count(), 0);
    }
}