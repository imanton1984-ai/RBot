use std::sync::Arc;

pub mod config;
pub mod ws_manager;
pub mod candle_builder;
pub mod backfill;
pub mod gap_fill;

pub mod health;
pub mod universe;

#[derive(Clone)]
pub struct AppState {
    pub stage: health::StageState,
    pub db_url: Arc<String>,
    pub rest_base: Arc<String>,
    pub universe_cfg_path: Arc<String>,
}
