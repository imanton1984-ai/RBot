use anyhow::{Context, Result};
use axum::{routing::get, Router};
use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{error, info};

use ingest::backfill;
use ingest::config::IngestConfig;
use ingest::health::{healthz, readyz, stagez, StageState};
use ingest::universe;
use ingest::ws_manager;
use ingest::AppState;
use api_binance::RateLimiter;
use ingest::http_gate::HttpGate;

#[tokio::main]
async fn main() -> Result<()> {
    // Логи
    tracing_subscriber::fmt::init();

    // Конфиг ingest (с env overrides внутри)
    let cfg = Arc::new(IngestConfig::load().context("IngestConfig::load failed")?);

    let limiter = Arc::new(RateLimiter::new(
        cfg.http_soft_rps,
        cfg.http_soft_burst,
    ));
    let gate = Arc::new(HttpGate::new());

    // Stage state
    let stage = StageState::new("STARTING");

    // App state для HTTP и фоновых задач
    let state = AppState {
        stage: stage.clone(),
        cfg: cfg.clone(),
        db_url: Arc::new(cfg.db_url.clone()),
        rest_base: Arc::new(cfg.rest_base_url.clone()),
        universe_cfg_path: Arc::new(cfg.universe_cfg_path.clone()),
        http_limiter: limiter,
        http_gate: gate,
    };

    // 1) STARTUP: universe refresh (обязательный)
    state.stage.set("UNIVERSE_STARTUP", "refreshing market.pairs (startup)");
    let n = universe::universe_startup(&state)
        .await
        .context("universe_startup failed")?;
    state
        .stage
        .set("PAIRS_READY", format!("pairs active: {}", n));

    // 2) Periodic universe refresh (не трогает стадии RUN/и т.д.)
    {
        let st = state.clone();
        tokio::spawn(async move {
            if let Err(e) = universe::run_universe_loop(st).await {
                error!("universe loop crashed: {:#}", e);
            }
        });
    }

    // 3) Candles pipeline (backfill -> ws/poll)
    {
        let st = state.clone();
        tokio::spawn(async move {
            if let Err(e) = candles_pipeline(st).await {
                error!("candles_pipeline crashed: {:#}", e);
            }
        });
    }

    // HTTP endpoints
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/stagez", get(stagez))
        .with_state(state);

    let port = 8081u16;
    let addr: SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .context("bad listen addr")?;

    info!("ingest listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("bind listener failed")?;

    // axum 0.7 style
    axum::serve(listener, app)
        .await
        .context("axum serve failed")?;

    Ok(())
}

async fn candles_pipeline(state: AppState) -> Result<()> {
    // Backfill
    state.stage.set("LOADING_CANDLES", "starting backfill");
    let (symbols, last_map) = backfill::run_backfill(state.cfg.clone())
        .await
        .context("run_backfill failed")?;

    state.stage.set(
        "BACKFILL_CANDLES_READY",
        format!("symbols: {}, last_map: {}", symbols.len(), last_map.len()),
    );

    // WS + Poll loop (обычно вечный)
    state.stage.set("RUN", "starting ws/poll");
    ws_manager::run_ws_and_poll(state.cfg.clone(), symbols, last_map)
        .await
        .context("run_ws_and_poll failed")?;

    Ok(())
}





