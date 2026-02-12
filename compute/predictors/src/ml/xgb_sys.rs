use std::ffi::c_char;
use std::os::raw::c_int;
use std::ffi::c_void;

pub type bst_ulong = u64;

pub type DMatrixHandle = *mut c_void;
pub type BoosterHandle = *mut c_void;

#[link(name = "xgboost")]
extern "C" {
    pub fn XGBGetLastError() -> *const c_char;

    pub fn XGDMatrixCreateFromMat(
        data: *const f32,
        nrow: bst_ulong,
        ncol: bst_ulong,
        missing: f32,
        out: *mut DMatrixHandle,
    ) -> c_int;

    pub fn XGDMatrixCreateFromCudaArrayInterface(
        data_json: *const c_char,
        config_json: *const c_char,
        out: *mut DMatrixHandle,
    ) -> c_int;

    pub fn XGDMatrixFree(handle: DMatrixHandle) -> c_int;

    pub fn XGBoosterCreate(
        dmats: *const DMatrixHandle,
        len: bst_ulong,
        out: *mut BoosterHandle,
    ) -> c_int;

    pub fn XGBoosterFree(handle: BoosterHandle) -> c_int;

    pub fn XGBoosterLoadModel(handle: BoosterHandle, fname: *const c_char) -> c_int;

    pub fn XGBoosterSetParam(handle: BoosterHandle, name: *const c_char, value: *const c_char) -> c_int;

    // shape/dims/result are owned by XGBoost; you must copy immediately.
    pub fn XGBoosterPredictFromDMatrix(
        handle: BoosterHandle,
        dmat: DMatrixHandle,
        config: *const c_char,
        out_shape: *mut *const bst_ulong,
        out_dim: *mut bst_ulong,
        out_result: *mut *const f32,
    ) -> c_int;
}