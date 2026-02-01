use std::sync::Arc;
use tokio::sync::RwLock;

pub struct DeviceManager {
    context: Arc<RwLock<Option<super::CudaContext>>>,
}

impl DeviceManager {
    pub fn new() -> Self {
        Self {
            context: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn initialize(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let cuda_context = super::CudaContext::new().await?;
        let mut ctx_guard = self.context.write().await;
        *ctx_guard = Some(cuda_context);
        Ok(())
    }

    pub async fn get_context(&self) -> Option<super::CudaContext> {
        let ctx_guard = self.context.read().await;
        ctx_guard.clone()
    }

    pub async fn is_available(&self) -> bool {
        if let Some(ctx) = self.get_context().await {
            ctx.has_devices()
        } else {
            false
        }
    }

    pub async fn get_device_info(&self) -> Vec<super::DeviceInfo> {
        if let Some(ctx) = self.get_context().await {
            ctx.device_info.clone()
        } else {
            Vec::new()
        }
    }
}

// Global device manager instance
lazy_static::lazy_static! {
    static ref DEVICE_MANAGER: DeviceManager = DeviceManager::new();
}

pub async fn get_device_manager() -> &'static DeviceManager {
    &DEVICE_MANAGER
}

pub async fn initialize_global() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    DEVICE_MANAGER.initialize().await
}

pub async fn is_cuda_available() -> bool {
    DEVICE_MANAGER.is_available().await
}