pub mod xgb_sys;
pub mod xgb_runtime;
pub mod model_manager;
// Keep for potential future use
pub mod model_pool;

// Remove OnnxRunner as it's no longer needed
pub use model_pool::ModelPool;
