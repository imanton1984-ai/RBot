use anyhow::{Context, Result};
use axum::{http::StatusCode, response::IntoResponse, routing::get, Router};
use std::net::SocketAddr;
use tokio::signal;
use tracing::{info, warn};

fn env_u16(key: &str) -> Option<u16> {
    std::env::var(key).ok()?.parse().ok()
}

#[tokio::main]
async fn main() -> Result<()> {
    apps::init_tracing();

    // единый стиль env: SVC_HEALTH_PORT (но поддержим и HEALTH_PORT как запасной)
    let port = env_u16("SVC_HEALTH_PORT")
        .or_else(|| env_u16("HEALTH_PORT"))
        .unwrap_or(9001);

    let addr: SocketAddr = format!("0.0.0.0:{port}")
        .parse()
        .context("bad bind addr")?;

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz));

    info!("svc_health listening on http://{addr}");

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("http serve failed")?;

    Ok(())
}

async fn healthz() -> impl IntoResponse {
    (StatusCode::OK, "OK")
}

async fn readyz() -> impl IntoResponse {
    // svc_health сам по себе ничего не греет — он “готов” если поднялся.
    (StatusCode::OK, "READY")
}

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
    warn!("shutdown signal received");
}

