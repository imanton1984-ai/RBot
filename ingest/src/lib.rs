use std::sync::Arc;
use tokio::sync::RwLock;

pub mod backfill;
pub mod candle_builder;
pub mod config;
pub mod gap_fill;
pub mod health;
pub mod producer;
pub mod universe;
pub mod ws_manager;

#[derive(Clone)]
pub struct StageState {
    pub stage: Arc<RwLock<String>>,
}

impl StageState {
    pub fn new(init: &str) -> Self {
        Self {
            stage: Arc::new(RwLock::new(init.to_string())),
        }
    }

    pub async fn get(&self) -> String {
        self.stage.read().await.clone()
    }

    pub async fn set(&self, s: &str) {
        *self.stage.write().await = s.to_string();
    }
}

#[derive(Clone)]
pub struct AppState {
    pub stage: health::StageState,
    pub cfg: Arc<config::IngestConfig>,
    pub db_url: Arc<String>,
    pub rest_base: Arc<String>,
    pub universe_cfg_path: Arc<String>,
}
