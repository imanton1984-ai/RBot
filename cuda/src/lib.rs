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
        // В продакшене лучше вшить его через include_str! или include_bytes!
        // Но cudarc умеет грузить PTX текст.
        
        // Вариант А: Загрузка во время компиляции (надежнее)
        let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/indicators.ptx"));
        device.load_ptx(Ptx::from_src(ptx_src), "indicators", &["sma_kernel", "ema_kernel", "rsi_kernel", "macd_kernel", "adx_kernel", "atr_kernel", "bb_kernel", "obv_kernel", "cci_kernel", "stochastic_kernel", "vwap_kernel", "williams_r_kernel", "alligator_kernel"])?;

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