pub mod device_manager;
pub mod kernel_runner;
pub mod memory_manager;
pub mod indicator_kernels;

pub use device_manager::*;
pub use kernel_runner::*;
pub use memory_manager::*;
pub use indicator_kernels::*;

use std::process::Command;

// Simplified CUDA context for now (placeholder)
#[derive(Clone)]
pub struct CudaContext {
    pub device_count: usize,
    pub active_device: usize,
    pub device_info: Vec<DeviceInfo>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: usize,
    pub name: String,
    pub compute_capability: (u32, u32),
    pub global_memory: u64,
    pub multiprocessor_count: u32,
}

impl CudaContext {
    pub async fn new() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Check if nvidia-smi is available to determine if CUDA is installed
        let output = Command::new("which")
            .arg("nvidia-smi")
            .output();

        let device_count = match output {
            Ok(output) => {
                if output.status.success() {
                    // Try to get actual device count using nvidia-smi
                    let nvidia_output = Command::new("nvidia-smi")
                        .arg("--query-gpu=count")
                        .arg("--format=csv,noheader,nounits")
                        .output();

                    match nvidia_output {
                        Ok(out) => {
                            let count_str = String::from_utf8_lossy(&out.stdout);
                            count_str.trim().parse().unwrap_or(0)
                        }
                        Err(_) => 0
                    }
                } else {
                    0
                }
            }
            Err(_) => 0
        };

        let device_info = if device_count > 0 {
            // For now, create a mock device info - in a real implementation,
            // we would query actual device properties
            vec![DeviceInfo {
                id: 0,
                name: "NVIDIA GPU".to_string(),
                compute_capability: (7, 5), // Common compute capability
                global_memory: 8 * 1024 * 1024 * 1024, // 8GB as example
                multiprocessor_count: 68, // Example value
            }]
        } else {
            vec![]
        };

        Ok(CudaContext {
            device_count,
            active_device: 0,
            device_info,
        })
    }

    pub fn has_devices(&self) -> bool {
        self.device_count > 0
    }

    pub fn get_best_device(&self) -> Option<&DeviceInfo> {
        self.device_info.first()
    }
}