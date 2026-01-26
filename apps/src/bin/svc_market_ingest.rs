use anyhow::{Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use std::net::SocketAddr;
use tokio::signal;
use tracing::{info, warn};

use market_data::config::IngestConfig;
use market_data::{backfill, universe, ws_manager, AppState};

#[tokio::main]
async fn main() -> Result<()> {
    apps::init_tracing();

    // IngestConfig::load() -> Arc<IngestConfig>
    let cfg_arc = IngestConfig::load().context("IngestConfig::load failed")?;
    let state = AppState::new(cfg_arc.as_ref().clone()).context("AppState::new failed")?;

    let addr: SocketAddr = format!("0.0.0.0:{}", state.cfg.health_port)
        .parse()
        .context("bad bind addr")?;

    info!(
        "svc_market_ingest starting: db={} brokers={} port={}",
        state.cfg.db_url, state.cfg.redpanda_brokers, state.cfg.health_port
    );

    // ВАЖНО:
    // НЕ используем market_data::health::{healthz,readyz,stagez}, т.к. market_data тянет axum 0.7,
    // а apps — axum 0.8. Это и давало ошибку Handler.
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/stagez", get(stagez))
        .with_state(state.clone());

    let listener = tokio::net::TcpListener::bind(addr).await.context("bind failed")?;

    // pipeline в фоне
    let pipeline_state = state.clone();
    let pipeline_handle = tokio::spawn(async move {
        if let Err(e) = run_pipeline(pipeline_state).await {
            warn!("pipeline fatal: {e:#}");
        }
    });

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("http serve failed")?;

    // если HTTP упал/остановился — останавливаем pipeline
    pipeline_handle.abort();
    Ok(())
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn readyz(State(st): State<AppState>) -> impl IntoResponse {
    let v = st.stage.get();
    if v.stage == "RUN" || v.stage.ends_with("_READY") {
        (StatusCode::OK, "READY")
    } else if v.stage == "ERROR" {
        (StatusCode::SERVICE_UNAVAILABLE, "ERROR")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "NOT_READY")
    }
}

async fn stagez(State(st): State<AppState>) -> impl IntoResponse {
    let v = st.stage.get();
    Json(v)
}

async fn run_pipeline(state: AppState) -> Result<()> {
    state
        .stage
        .set("STARTING", "Initializing pipeline components...");

    // 1) universe → pairs
    state.stage.set("LOADING_PAIRS", "Refreshing universe...");
    let active_pairs = universe::refresh_universe_once(&state.cfg.db_url, &state.rest).await?;
    state
        .stage
        .set("PAIRS_READY", format!("Universe ready, active_pairs={active_pairs}"));

    // 2) symbols list (для ws_manager)
    let symbols = universe::fetch_active_symbols(&state.cfg.db_url).await?;
    if symbols.is_empty() {
        warn!("No active symbols found in DB (market.pairs is_active=true)");
    }

    // 3) backfill (даёт last_close_map)
    state
        .stage
        .set("LOADING_CANDLES", format!("Backfill for {} symbols...", symbols.len()));

    let (published, last_map) = backfill::run_backfill(state.cfg.clone(), state.rest.clone(), state.producer.clone()).await?;

    state.stage.set(
        "BACKFILL_CANDLES_READY",
        format!("Backfill done. published={published}, last_map={}", last_map.len()),
    );

    // 4) realtime ws + poll
    state.stage.set("RUN", "WebSocket streams + poll loop active");

    ws_manager::run_ws_and_poll(
        state.cfg.clone(),
        state.rest.clone(),
        state.producer.clone(),
        symbols,
        last_map,
    )
    .await?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let term = async {
        let mut sig =
            signal::unix::signal(signal::unix::SignalKind::terminate()).expect("sigterm handler");
        let _ = sig.recv().await;
    };

    #[cfg(not(unix))]
    let term = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = term => {}
    }
}
