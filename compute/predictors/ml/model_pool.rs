use std::sync::Arc;
use ort::{Session, Environment, GraphOptimizationLevel, ExecutionProvider};
use dashmap::DashMap;
use tracing::{info, warn};

pub struct ModelPool {
    env: Arc<Environment>,
    sessions: DashMap<String, Arc<Session>>,
}

impl ModelPool {
    pub fn new() -> Self {
        let env = Arc::new(Environment::builder().with_name("ORT_ENV").build().unwrap());
        Self {
            env,
            sessions: DashMap::new(),
        }
    }

    pub fn get_or_load(&self, path: &str, use_cuda: bool) -> anyhow::Result<Arc<Session>> {
        if let Some(s) = self.sessions.get(path) {
            return Ok(s.clone());
        }

        let mut builder = Session::builder_with_environment(self.env.clone())?
            .with_optimization_level(GraphOptimizationLevel::Level3)?;

        if use_cuda {
            if let Ok(b) = builder.clone().with_execution_providers([ExecutionProvider::CUDA(Default::default())]) {
                builder = b;
                info!("CUDA Execution Provider enabled for model: {}", path);
            } else {
                warn!("Failed to initialize CUDA provider. Falling back to CPU for model: {}", path);
            }
        }
        
        let session = builder.with_model_from_file(path)?;
        let shared = Arc::new(session);
        self.sessions.insert(path.to_string(), shared.clone());
        Ok(shared)
    }
}

impl Default for ModelPool {
    fn default() -> Self {
        Self::new()
    }
}
