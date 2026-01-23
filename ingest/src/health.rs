use axum::{extract::State, http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde_json::json;

use crate::AppState;

#[derive(Debug, Clone, Serialize)]
pub struct StageView {
    pub stage: String,
    pub updated_at_ms: i64,
    pub details: String,
}

#[derive(Clone)]
pub struct StageState {
    inner: Arc<RwLock<StageView>>,
}

impl StageState {
    pub fn new(init_stage: &str) -> Self {
        Self {
            inner: Arc::new(RwLock::new(StageView {
                stage: init_stage.to_string(),
                updated_at_ms: now_ms(),
                details: String::new(),
            })),
        }
    }

    pub fn set(&self, stage: &str, details: impl Into<String>) {
        let mut w = self.inner.write().unwrap();
        w.stage = stage.to_string();
        w.updated_at_ms = now_ms();
        w.details = details.into();
    }

    pub fn get(&self) -> StageView {
        self.inner.read().unwrap().clone()
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

pub async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

pub async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    let stage = state.stage.get().await;
    let ok = matches!(
        stage.as_str(),
        "PAIRS_READY" | "LOADING_CANDLES" | "BACKFILL_CANDLES_READY" | "RUN"
    );
    if ok {
        (StatusCode::OK, "READY").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "NOT_READY").into_response()
    }
}

pub async fn stagez(State(state): State<AppState>) -> impl IntoResponse {
    let stage = state.stage.get().await;
    Json(json!({ "stage": stage }))
}



