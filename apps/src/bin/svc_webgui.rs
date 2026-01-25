use anyhow::Result;
use axum::{http::StatusCode, routing::get, Router};
use std::{net::SocketAddr};
use tracing::info;

use apps::init_tracing;

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();

    let port: u16 = std::env::var("WEBGUI_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9040);

    let app = Router::new()
        .route("/healthz", get(|| async { (StatusCode::OK, "OK") }))
        .route("/", get(|| async { (StatusCode::OK, "webgui placeholder (TODO)") }));

    let addr: SocketAddr = format!("0.0.0.0:{port}").parse()?;
    info!("svc_webgui listening on http://{addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}
