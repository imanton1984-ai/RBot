use anyhow::{Context, Result};
use axum::{routing::get, Json, Router};
use serde::Serialize;
use std::{collections::HashMap, net::SocketAddr, sync::Arc};
use tokio::sync::RwLock;
use tracing::{info, warn};

use connections::{BinanceRestClient, RestRateLimitCfg};

use crate::config::IngestConfig;
use crate::producer::build_producer;
use crate::{backfill, universe, ws_manager};
use crate::health::now_ms;

mod config;
mod producer;
mod candle_builder;
mod gap_fill;
mod health;
mod backfill;
mod ws_manager;
mod universe;

#[derive(Debug, Clone, Serialize)]
struct StageState {
    stage: String,
    updated_at_ms: i64,
    details: HashMap<String, String>,
}

impl StageState {
    fn new(stage: &str) -> Self {
        Self {
            stage: stage.to_string(),
            updated_at_ms: now_ms(),
            details: HashMap::new(),
        }
    }
}

#[derive(Clone)]
struct AppState {
    cfg: Arc<IngestConfig>,
    stage: Arc<RwLock<StageState>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "market_data=info,connections=info".into()),
        )
        .init();

    let cfg = IngestConfig::load().context("IngestConfig::load failed")?;
    info!("market_data starting: db={} brokers={} port={}", cfg.db_url, cfg.redpanda_brokers, cfg.health_port);

    let stage = Arc::new(RwLock::new(StageState::new("STARTING")));
    let state = AppState { cfg: cfg.clone(), stage: stage.clone() };

    // http (health/stage)
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/stagez", get(stagez))
        .with_state(state.clone());

    let addr: SocketAddr = format!("0.0.0.0:{}", cfg.health_port).parse().unwrap();
    info!("http listening on {addr}");
    tokio::spawn(async move {
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    });

    // pipeline task
    tokio::spawn(async move {
        if let Err(e) = run_pipeline(state).await {
            warn!("pipeline fatal: {e:#}");
            let mut g = state.stage.write().await;
            g.stage = "ERROR".to_string();
            g.updated_at_ms = now_ms();
            g.details.insert("error".into(), format!("{e:#}"));
        }
    });

    futures::future::pending::<()>().await;
    Ok(())
}

async fn run_pipeline(state: AppState) -> Result<()> {
    // rest client with soft rate-limit (защита от 418)
    let rest = BinanceRestClient::new(RestRateLimitCfg {
        soft_rps: state.cfg.http_soft_rps,
        soft_burst: state.cfg.http_soft_burst,
    });

    // producer
    let producer = build_producer(&state.cfg.redpanda_brokers).context("build_producer failed")?;

    // 1) universe → pairs
    {
        let mut g = state.stage.write().await;
        *g = StageState::new("LOADING_PAIRS");
    }
    let active_pairs = universe::refresh_universe_once(&state.cfg.db_url, &rest).await?;
    {
        let mut g = state.stage.write().await;
        *g = StageState::new("PAIRS_READY");
        g.details.insert("active_pairs".into(), active_pairs.to_string());
    }

    // 2) backfill
    {
        let mut g = state.stage.write().await;
        *g = StageState::new("LOADING_CANDLES");
    }
    let (published, last_map) = backfill::run_backfill(state.cfg.clone(), rest.clone(), producer.clone()).await?;
    {
        let mut g = state.stage.write().await;
        *g = StageState::new("BACKFILL_CANDLES_READY");
        g.details.insert("published".into(), published.to_string());
        g.details.insert("last_map".into(), last_map.len().to_string());
    }

    // 3) realtime ws + poll
    {
        let mut g = state.stage.write().await;
        *g = StageState::new("RUN");
    }

    // symbols снова из DB уже не читаем — backfill вернул их? (если хочешь — можно вернуть symbols отдельно)
    // Здесь проще: перечитать активные через backfill::load_active_symbols, но сейчас не экспортируем.
    // Поэтому делаем лёгкий запрос прямо тут:
    let symbols = {
        let (client, conn) = tokio_postgres::connect(&state.cfg.db_url, tokio_postgres::NoTls).await?;
        tokio::spawn(async move { let _ = conn.await; });
        let rows = client.query(
            "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol",
            &[]
        ).await?;
        rows.into_iter().map(|r| r.get::<_, String>(0)).collect::<Vec<_>>()
    };

    ws_manager::run_ws_and_poll(state.cfg.clone(), rest.clone(), producer.clone(), symbols, last_map).await?;
    Ok(())
}

async fn healthz() -> Json<HashMap<&'static str, &'static str>> {
    Json(HashMap::from([("ok", "true")]))
}

async fn readyz(axum::extract::State(st): axum::extract::State<AppState>) -> Json<HashMap<&'static str, String>> {
    let s = st.stage.read().await;
    let ok = matches!(
        s.stage.as_str(),
        "PAIRS_READY" | "LOADING_CANDLES" | "BACKFILL_CANDLES_READY" | "RUN"
    );
    Json(HashMap::from([
        ("ok", ok.to_string()),
        ("stage", s.stage.clone()),
    ]))
}

async fn stagez(axum::extract::State(st): axum::extract::State<AppState>) -> Json<StageState> {
    let s = st.stage.read().await;
    Json(s.clone())
}







