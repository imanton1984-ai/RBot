use anyhow::{anyhow, Result};
use std::ffi::{CStr, CString};
use std::ptr;
use tracing::info; // <= add

use super::xgb_sys::*;

#[derive(Clone, Copy, Debug)]
pub enum Device {
    Cpu,
    Cuda,
}

#[derive(Clone, Copy, Debug)]
pub enum ModelKind {
    Regressor1,     // output: [value]
    BinaryProb2,    // output: [1-p, p]
}

fn last_err() -> String {
    unsafe {
        let p = XGBGetLastError();
        if p.is_null() {
            "xgboost: unknown error".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().to_string()
        }
    }
}

fn check(code: i32) -> Result<()> {
    if code == 0 { Ok(()) } else { Err(anyhow!(last_err())) }
}

pub struct DMatrix {
    h: DMatrixHandle,
}
impl Drop for DMatrix {
    fn drop(&mut self) {
        unsafe { let _ = XGDMatrixFree(self.h); }
    }
}

pub struct Booster {
    h: BoosterHandle,
}

// SAFETY: XGBoost handles are thread-safe and can be sent between threads
unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Drop for Booster {
    fn drop(&mut self) {
        unsafe { let _ = XGBoosterFree(self.h); }
    }
}

impl Booster {
    pub fn load(model_path: &str, device: Device) -> Result<Self> {
        unsafe {
            let mut h: BoosterHandle = ptr::null_mut();
            check(XGBoosterCreate(ptr::null(), 0, &mut h))?;

            let path = CString::new(model_path)?;
            check(XGBoosterLoadModel(h, path.as_ptr()))?;

            let (k, v) = match device {
                Device::Cpu => ("device", "cpu"),
                Device::Cuda => ("device", "cuda"),
            };
            let (k2, v2) = match device {
                Device::Cpu => ("predictor", "cpu_predictor"),
                Device::Cuda => ("predictor", "gpu_predictor"),
            };

            info!(
                target: "compute_predictors",
                "XGB Booster loaded: path={}, {}={}, {}={}",
                model_path, k, v, k2, v2
            );

            // Современный способ: device = "cuda"/"cpu" (см c-api-demo)
            // :contentReference[oaicite:4]{index=4} — этот параметр реально применяется.
            let k_param = CString::new(k)?;
            let v_param = CString::new(v)?;
            check(XGBoosterSetParam(h, k_param.as_ptr(), v_param.as_ptr()))?;

            // Устанавливаем predictor для лучшей GPU оптимизации
            let k2_param = CString::new(k2)?;
            let v2_param = CString::new(v2)?;
            check(XGBoosterSetParam(h, k2_param.as_ptr(), v2_param.as_ptr()))?;

            // Разумно также задать nthread для CPU, но оставим внешней настройкой позже.
            Ok(Self { h })
        }
    }

    pub fn predict_dense_cpu(&self, data: &[f32], nrow: usize, ncol: usize, kind: ModelKind) -> Result<Vec<f32>> {
        unsafe {
            let mut dm: DMatrixHandle = ptr::null_mut();
            check(XGDMatrixCreateFromMat(
                data.as_ptr(),
                nrow as bst_ulong,
                ncol as bst_ulong,
                f32::NAN,
                &mut dm,
            ))?;
            let dm = DMatrix { h: dm };
            self.predict_from_dmatrix(&dm, nrow, kind)
        }
    }

    #[cfg(feature = "cuda")]
    pub fn predict_dense_cuda_array_interface(
        &self,
        device_ptr_u64: u64,
        nrow: usize,
        ncol: usize,
        kind: ModelKind,
    ) -> Result<Vec<f32>> {
        use std::ffi::CString;

        // Create CUDA array interface JSON - proper format for XGBoost
        // Format: {"data": [ptr, readonly], "shape": [...], "typestr": "...", "version": 3}
        let data_json = format!(
            r#"{{
              "data": [{}, true],
              "shape": [{}, {}],
              "typestr": "<f4",
              "version": 3
            }}"#,
            device_ptr_u64, nrow, ncol
        );

        // XGBoost принимает NaN в config (как в c-api-demo)
        let cfg_json = r#"{"missing": NaN, "nthread": 0}"#;

        unsafe {
            let mut dm: DMatrixHandle = ptr::null_mut();
            let data_c = CString::new(data_json)?;
            let cfg_c = CString::new(cfg_json)?;
            check(XGDMatrixCreateFromCudaArrayInterface(data_c.as_ptr(), cfg_c.as_ptr(), &mut dm))?;
            let dm = DMatrix { h: dm };
            self.predict_from_dmatrix(&dm, nrow, kind)
        }
    }

    fn predict_from_dmatrix(&self, dm: &DMatrix, nrow: usize, kind: ModelKind) -> Result<Vec<f32>> {
        // Конфиг предикта из c-api-demo
        // :contentReference[oaicite:6]{index=6}
        let cfg = r#"{"type":0,"training":false,"iteration_begin":0,"iteration_end":0,"strict_shape":true}"#;
        let cfg = CString::new(cfg)?;

        unsafe {
            let mut out_shape: *const bst_ulong = ptr::null();
            let mut out_dim: bst_ulong = 0;
            let mut out_result: *const f32 = ptr::null();

            check(XGBoosterPredictFromDMatrix(
                self.h,
                dm.h,
                cfg.as_ptr(),
                &mut out_shape,
                &mut out_dim,
                &mut out_result,
            ))?;

            if out_shape.is_null() || out_result.is_null() {
                return Err(anyhow!("xgboost returned null output"));
            }

            let dim = out_dim as usize;
            let shape = std::slice::from_raw_parts(out_shape, dim);

            // Обычно shape = [nrow] для бинарной вероятности, или [nrow, k] для multi.
            let out_len: usize = shape.iter().copied().map(|v| v as usize).product();
            let raw = std::slice::from_raw_parts(out_result, out_len);

            match kind {
                ModelKind::Regressor1 => Ok(raw.to_vec()),
                ModelKind::BinaryProb2 => {
                    // XGBoost binary:logistic: usually gives p(class=1) as length nrow.
                    // Convert to expected format [1-p, p] for each row.
                    if out_len != nrow {
                        // if it's already 2-column format - just return as is
                        return Ok(raw.to_vec());
                    }
                    let mut v = Vec::with_capacity(nrow * 2);
                    for &p in raw {
                        let p = p.max(0.0).min(1.0);
                        v.push(1.0 - p);
                        v.push(p);
                    }
                    Ok(v)
                }
            }
        }
    }

    /// GPU prediction method that accepts raw device pointer for zero-copy
    #[cfg(feature = "cuda")]
    pub fn predict_from_cuda_array(
        &self,
        device_ptr: u64,
        nrow: usize,
        ncol: usize,
        kind: ModelKind,
    ) -> Result<Vec<f32>> {
        use std::ffi::CString;

        // Create CUDA array interface JSON - proper format for XGBoost
        // Format: {"data": [ptr, readonly], "shape": [...], "typestr": "...", "version": 3}
        let data_json = format!(
            r#"{{
              "data": [{}, true],
              "shape": [{}, {}],
              "typestr": "<f4",
              "version": 3
            }}"#,
            device_ptr, nrow, ncol
        );

        let cfg_json = r#"{"missing": NaN, "nthread": 0}"#;

        unsafe {
            let mut dm: DMatrixHandle = ptr::null_mut();
            let data_c = CString::new(data_json)?;
            let cfg_c = CString::new(cfg_json)?;
            check(XGDMatrixCreateFromCudaArrayInterface(
                data_c.as_ptr(),
                cfg_c.as_ptr(),
                &mut dm,
            ))?;
            let dm = DMatrix { h: dm };
            self.predict_from_dmatrix(&dm, nrow, kind)
        }
    }
}