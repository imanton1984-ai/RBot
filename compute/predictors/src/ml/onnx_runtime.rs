use anyhow::{Result, Context};
use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;
use std::sync::{Arc, Mutex};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut};

use super::model_pool::ModelPool;

pub struct OnnxRunner {
    session: Arc<Mutex<Session>>,
}

impl OnnxRunner {
    pub fn new(base_model_path: &str, use_cuda: bool, _pool: Arc<ModelPool>) -> Result<Self> {
        // Determine the actual model path based on CUDA availability and model existence
        let actual_model_path = if use_cuda {
            // If CUDA is requested, try the GPU-optimized version first
            let gpu_model_path = base_model_path.trim_end_matches(".onnx").to_string() + "_gpu.onnx";
            if std::path::Path::new(&gpu_model_path).exists() {
                tracing::info!("Using GPU-optimized model: {}", gpu_model_path);
                gpu_model_path
            } else {
                // If GPU-optimized model doesn't exist, fall back to regular model
                tracing::info!("GPU-optimized model not found, falling back to: {}", base_model_path);
                base_model_path.to_string()
            }
        } else {
            // If CUDA is not requested, use the regular model
            base_model_path.to_string()
        };

        let session = if use_cuda {
            // Try to use CUDA execution provider
            let cuda_provider = ort::execution_providers::CUDAExecutionProvider::default();
            match ort::Session::builder()?
                .with_execution_providers([cuda_provider.build()])
            {
                Ok(builder) => {
                    tracing::info!("CUDA provider registered for model: {}", actual_model_path);
                    builder.commit_from_file(&actual_model_path)?
                },
                Err(e) => {
                    tracing::warn!("Failed to register CUDA provider for ONNX, falling back to CPU: {}", e);
                    ort::Session::builder()?
                        .commit_from_file(&actual_model_path)?
                }
            }
        } else {
            ort::Session::builder()?
                .commit_from_file(&actual_model_path)?
        };

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

    /// Zero-copy prediction method that works with CudaSlice from cudarc
    /// This method uses ONNX Runtime's I/O binding to avoid memory copies between CPU and GPU
    pub fn predict_with_cuda_slice(
        &self,
        input_slice: &CudaSlice<f32>,
        output_slice: &mut CudaSlice<f32>,
        batch_size: usize
    ) -> Result<()> {
        let session_guard = self.session.lock().unwrap();
        
        // Get the device pointers from the CudaSlices
        let input_ptr = input_slice.device_ptr() as *const f32;
        let output_ptr = output_slice.device_ptr_mut() as *mut f32;
        
        // Define the tensor shape (assuming [batch_size, feature_count])
        let feature_count = input_slice.len() / batch_size;
        let input_shape = vec![batch_size as i64, feature_count as i64];
        let output_shape = vec![batch_size as i64, feature_count as i64]; // Adjust as needed
        
        // Create IO binding for zero-copy execution
        let mut io_binding = session_guard.io_binding()?;
        
        // Get input/output names
        let input_name = session_guard.inputs()[0].name().to_string();
        let output_name = session_guard.outputs()[0].name().to_string();
        
        // Bind input tensor to the GPU memory
        // This is the key part for zero-copy: we bind directly to GPU memory addresses
        unsafe {
            // Note: This API may vary depending on the ort-rs version
            // The exact method names depend on the specific version of ort crate
            io_binding.bind_input_from_device_memory(
                &input_name,
                input_ptr as *const std::ffi::c_void,
                &input_shape,
            )?;
            
            // Bind output tensor to the GPU memory
            io_binding.bind_output_to_device_memory(
                &output_name,
                output_ptr as *mut std::ffi::c_void,
                &output_shape,
            )?;
        }
        
        // Execute the model with zero-copy
        session_guard.run_with_iobinding(&mut io_binding)?;
        
        Ok(())
    }
    
    /// Alternative method for zero-copy prediction using raw device pointers
    pub fn predict_batch_zero_copy(
        &self,
        input_device_ptr: *const f32,  // Pointer to GPU memory
        output_device_ptr: *mut f32,   // Pointer to GPU memory for output
        batch_size: usize,
        feature_count: usize
    ) -> Result<()> {
        let session_guard = self.session.lock().unwrap();
        
        // Define the tensor shapes
        let input_shape = vec![batch_size as i64, feature_count as i64];
        let output_shape = vec![batch_size as i64, feature_count as i64]; // Adjust as needed
        
        // Create IO binding for zero-copy execution
        let mut io_binding = session_guard.io_binding()?;
        
        // Get input/output names
        let input_name = session_guard.inputs()[0].name().to_string();
        let output_name = session_guard.outputs()[0].name().to_string();
        
        // Bind input tensor to the provided GPU memory address
        unsafe {
            io_binding.bind_input_from_device_memory(
                &input_name,
                input_device_ptr as *const std::ffi::c_void,
                &input_shape,
            )?;
            
            // Bind output tensor to the provided GPU memory address
            io_binding.bind_output_to_device_memory(
                &output_name,
                output_device_ptr as *mut std::ffi::c_void,
                &output_shape,
            )?;
        }
        
        // Execute the model with zero-copy
        session_guard.run_with_iobinding(&mut io_binding)?;
        
        Ok(())
    }
}