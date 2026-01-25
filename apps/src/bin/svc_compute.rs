use std::{net::SocketAddr, time::Duration};

use axum::{http::StatusCode, routing::get, Router};
use tokio::signal;
use tracing::{info, warn};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let port = env_u16("COMPUTE_CORE_PORT").unwrap_or(9010);

    // Тут позже появится реальная вычислительная петля:
    // - чтение свечей/индикаторов из DB
    // - расчёт индикаторов/фич
    // - публикация snapshots в Kafka
    // Сейчас — чистый каркас.

    let app = Router::new()
        .route("/healthz", get(|| async { (StatusCode::OK, "OK") }))
        .route("/readyz", get(|| async { (StatusCode::OK, "READY") }));

    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    info!("compute_core listening on http://{addr}");

    axum::serve(tokio::net::TcpListener::bind(addr).await?, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    let _ = signal::ctrl_c().await;
    warn!("shutdown signal received");
    tokio::time::sleep(Duration::from_millis(150)).await;
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .with_target(true)
        .with_thread_ids(true)
        .with_thread_names(true)
        .try_init();
}

fn env_u16(key: &str) -> Option<u16> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}
