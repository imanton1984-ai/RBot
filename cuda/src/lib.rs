use tracing;
pub mod indicator_kernels;
pub use indicator_kernels::IndicatorKernelRunner;

use std::sync::Arc;
use cudarc::driver::{CudaDevice};
use cudarc::nvrtc::Ptx;
use anyhow::Result;

// Глобальный контекст устройства
pub struct CudaContext {
    pub device: Arc<CudaDevice>,
}

impl CudaContext {
    pub fn new() -> Result<Self> {
        // Инициализирует устройство 0
        let device = CudaDevice::new(0)?;
        
        // Загружаем PTX (компилируется build.rs и кладется в OUT_DIR)
        let indicators_ptx = include_str!(concat!(env!("OUT_DIR"), "/indicators.ptx"));
        device.load_ptx(Ptx::from_src(indicators_ptx), "indicators", &["sma_kernel", "ema_kernel", "rsi_kernel", "macd_kernel", "adx_kernel", "atr_kernel", "bb_kernel", "obv_kernel", "cci_kernel", "stochastic_kernel", "vwap_kernel", "williams_r_kernel", "alligator_kernel"])?;

        let predictors_ptx = include_str!(concat!(env!("OUT_DIR"), "/predictors.ptx"));
        device.load_ptx(Ptx::from_src(predictors_ptx), "predictors", &["rsi_divergence_predictor_kernel", "sr_level_predictor_kernel", "momentum_reversal_predictor_kernel", "heuristic_combiner_kernel", "heuristic_to_signal_kernel"])?;

        let raw_signals_ptx = include_str!(concat!(env!("OUT_DIR"), "/raw_signals.ptx"));
        device.load_ptx(
            Ptx::from_src(raw_signals_ptx), 
            "raw_signals", 
            &["calculate_raw_signals_kernel"]
        )?;
        
        Ok(Self { device })
    }
}

// Singleton для доступа к контексту
use std::sync::OnceLock;
static CUDA_CONTEXT: OnceLock<Option<CudaContext>> = OnceLock::new();

pub fn get_cuda_device() -> Option<Arc<CudaDevice>> {
    let ctx = CUDA_CONTEXT.get_or_init(|| {
        match CudaContext::new() {
            Ok(c) => {
                tracing::info!("CUDA initialized successfully");
                Some(c)
            },
            Err(e) => {
                tracing::warn!("Failed to initialize CUDA (fallback to CPU): {}", e);
                None
            }
        }
    });
    ctx.as_ref().map(|c| c.device.clone())
}