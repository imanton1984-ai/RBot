use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use anyhow::Result;
use tracing::{error, info};

use ingest::health::{healthz, readyz, stagez, StageState};
use ingest::{universe, AppState};


use axum::{routing::get, Router};
use ingest::{AppState, StageState};



mod config;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cfg = Arc::new(config::IngestConfig::load()?);

    let stage = StageState::new("INIT");
    let state = AppState { stage: stage.clone(), cfg: cfg.clone() };

    // 1) universe refresh (у тебя уже есть)
    {
        let st = state.clone();
        tokio::spawn(async move {
            if let Err(e) = ingest::universe::run_universe_loop(st).await {
                tracing::error!("universe loop crashed: {:?}", e);
            }
        });
    }

    // 2) candles pipeline
    {
        let st = state.clone();
        tokio::spawn(async move {
            if let Err(e) = candles_pipeline(st).await {
                tracing::error!("candles pipeline crashed: {:?}", e);
            }
        });
    }

    // http
    let app = Router::new()
        .route("/healthz", get(ingest::health::healthz))
        .route("/readyz", get(ingest::health::readyz))
        .route("/stagez", get(ingest::health::stagez))
        .with_state(state);

    let port = 8081u16; // либо вытаскивай из cfg.ports.market_ingest если хочешь
    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    tracing::info!("ingest listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app.into_make_service()).await?;
    Ok(())
}

async fn wait_for_stage(stage: &StageState, target: &str) {
    loop {
        let s = stage.get().stage; // StageView
        if s == target {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

async fn candles_pipeline(state: AppState) -> anyhow::Result<()> {
    wait_for_stage(&state.stage, "PAIRS_READY").await;

    state.stage.set("LOADING_CANDLES", "starting backfill");

    let (symbols, last_map) = ingest::backfill::run_backfill(state.cfg.clone()).await?;

    state.stage.set("BACKFILL_CANDLES_READY", "backfill done");

    ingest::ws_manager::run_ws_and_poll(state.cfg.clone(), symbols, last_map).await?;
    Ok(())
}

async fn universe_refresh(
    axum::extract::State(st): axum::extract::State<AppState>,
) -> impl axum::response::IntoResponse {
    match universe::refresh_pairs(&st.db_url, &st.rest_base, &st.universe_cfg_path).await {
        Ok(count) => {
            st.stage.set("PAIRS_READY", format!("pairs loaded: {count}"));
            (
                axum::http::StatusCode::OK,
                axum::Json(serde_json::json!({ "ok": true, "pairs": count })),
            )
        }
        Err(e) => {
            st.stage.set("STUCK", format!("universe refresh failed: {e:#}"));
            (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                axum::Json(serde_json::json!({ "ok": false, "error": format!("{e:#}") })),
            )
        }
    }
}

async fn universe_startup(st: AppState) -> Result<()> {
    st.stage.set("LOADING_PAIRS", "startup refresh");

    let mut attempt: u32 = 0;
    loop {
        attempt += 1;
        match universe::refresh_pairs(&st.db_url, &st.rest_base, &st.universe_cfg_path).await {
            Ok(count) => {
                st.stage.set("PAIRS_READY", format!("pairs loaded: {count}"));
                info!("PAIRS_READY (pairs={count})");
                return Ok(());
            }
            Err(e) => {
                st.stage.set("LOADING_PAIRS", format!("attempt {attempt}: {e:#}"));
                error!("attempt {attempt}: {e:#}");
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        }
    }
}




