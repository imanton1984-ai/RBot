use anyhow::Result;
use ndarray::Array2;
use ort::{
    AsPointer,
    session::Session,
    value::{Value, Tensor},
    memory::{MemoryInfo, AllocationDevice, MemoryType},
    tensor::{IntoTensorElementType, Shape},
    ortsys,
    sys,
};
use std::sync::{Arc, Mutex};
use cudarc::driver::{CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice};
use std::{mem::size_of, ptr, ffi::c_void};

use super::model_pool::ModelPool;

pub struct OnnxRunner {
    session: Arc<Mutex<Session>>,
}

/// Create an `ort` tensor that *borrows* an existing buffer (CPU or CUDA) without copying.
///
/// This replaces the removed `Tensor::from_raw` API in `ort >= 2.0.0-rc.11`.
///
/// # Safety
/// - `data` must be valid for `shape.num_elements()` elements of `f32`.
/// - The underlying buffer must outlive the returned `Tensor` (and any inference run using it).
unsafe fn tensor_f32_from_raw(
    mem_info: &MemoryInfo,
    data: *mut c_void,
    shape: impl Into<Shape>,
) -> ort::Result<Tensor<f32>> {
    let shape: Shape = shape.into();
    let mut raw: *mut sys::OrtValue = ptr::null_mut();

    // В rc.11 эта форма ожидает `; nonNull(...)` в аргументах макроса.
    // После `nonNull(raw)` макрос делает raw = NonNull<...> (shadowing).
    ortsys![
        unsafe CreateTensorWithDataAsOrtValue(
            mem_info.ptr(),
            data,
            shape.num_elements() * size_of::<f32>(),
            shape.as_ptr(),
            shape.len(),
            <f32 as IntoTensorElementType>::into_tensor_element_type().into(),
            &mut raw
        )?;
        nonNull(raw)
    ];

    // IMPORTANT: тут raw уже NonNull<OrtValue>
    Ok(Tensor::<f32>::from_ptr(raw, None))
}

impl OnnxRunner {
    pub fn new(base_model_path: &str, use_cuda: bool, _pool: Arc<ModelPool>) -> Result<Self> {
        let actual_model_path = if use_cuda {
            let gpu_model_path = base_model_path.trim_end_matches(".onnx").to_string() + "_gpu.onnx";
            if std::path::Path::new(&gpu_model_path).exists() {
                tracing::info!("Using GPU-optimized model: {}", gpu_model_path);
                gpu_model_path
            } else {
                tracing::info!("GPU-optimized model not found, falling back to: {}", base_model_path);
                base_model_path.to_string()
            }
        } else {
            base_model_path.to_string()
        };

        let session = if use_cuda {
            let cuda_provider = ort::execution_providers::CUDAExecutionProvider::default();
            match Session::builder()?.with_execution_providers([cuda_provider.build()]) {
                Ok(builder) => {
                    tracing::info!("CUDA provider registered for model: {}", actual_model_path);
                    builder.commit_from_file(&actual_model_path)?
                }
                Err(e) => {
                    tracing::warn!("Failed to register CUDA provider for ONNX, falling back to CPU: {}", e);
                    Session::builder()?.commit_from_file(&actual_model_path)?
                }
            }
        } else {
            Session::builder()?.commit_from_file(&actual_model_path)?
        };

        Ok(Self { session: Arc::new(Mutex::new(session)) })
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
                if vec.len() > 1 {
                    best_result = Some(vec);
                }
            }
        }
        if let Some(res) = best_result {
            return Ok(res);
        }

        if let Ok(output_tensor) = outputs[0].try_extract_tensor::<f32>() {
            return Ok(output_tensor.1.to_vec());
        }

        anyhow::bail!("Output extraction failed")
    }

    /// Zero-copy prediction method that works with CudaSlice from cudarc
    pub fn predict_with_cuda_slice(
        &self,
        input_slice: &CudaSlice<f32>,
        output_slice: &mut CudaSlice<f32>,
        batch_size: usize,
    ) -> Result<()> {
        let mut session_guard = self.session.lock().unwrap();

        // cudarc дает адрес как u64
        let input_ptr = (*input_slice.device_ptr()) as *mut c_void;
        let output_ptr = (*output_slice.device_ptr_mut()) as *mut c_void;

        let feature_count = input_slice.len() / batch_size;

        let mut io_binding = session_guard.create_binding()?;

        let input_name = session_guard.inputs()[0].name().to_string();
        let output_name = session_guard.outputs()[0].name().to_string();

        // Важно держать mem_info живым на время run_binding, раз мы используем external memory.
        let mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            ort::memory::AllocatorType::Device,
            MemoryType::Default,
        )?;

        let input_tensor = unsafe {
            tensor_f32_from_raw(&mem_info, input_ptr, [batch_size, feature_count])
                .map_err(anyhow::Error::from)?
        };
        io_binding.bind_input(&input_name, &input_tensor)?;

        let output_tensor = unsafe {
            tensor_f32_from_raw(&mem_info, output_ptr, [batch_size, 1usize])
                .map_err(anyhow::Error::from)?
        };
        io_binding.bind_output(&output_name, output_tensor)?;

        session_guard.run_binding(&io_binding)?;

        Ok(())
    }

    /// Alternative method: raw device pointers
    pub fn predict_batch_zero_copy(
        &self,
        input_device_ptr: *const f32,
        output_device_ptr: *mut f32,
        batch_size: usize,
        feature_count: usize,
    ) -> Result<()> {
        let mut session_guard = self.session.lock().unwrap();

        let mut io_binding = session_guard.create_binding()?;

        let input_name = session_guard.inputs()[0].name().to_string();
        let output_name = session_guard.outputs()[0].name().to_string();

        let mem_info = MemoryInfo::new(
            AllocationDevice::CUDA,
            0,
            ort::memory::AllocatorType::Device,
            MemoryType::Default,
        )?;

        let input_tensor = unsafe {
            tensor_f32_from_raw(&mem_info, input_device_ptr as *mut c_void, [batch_size, feature_count])
                .map_err(anyhow::Error::from)?
        };
        io_binding.bind_input(&input_name, &input_tensor)?;

        let output_tensor = unsafe {
            tensor_f32_from_raw(&mem_info, output_device_ptr as *mut c_void, [batch_size, 1usize])
                .map_err(anyhow::Error::from)?
        };
        io_binding.bind_output(&output_name, output_tensor)?;

        session_guard.run_binding(&io_binding)?;

        Ok(())
    }
}
