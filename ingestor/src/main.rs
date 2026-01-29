use anyhow::Result;
use dotenvy::dotenv;
use tracing::info;
use tracing_subscriber::EnvFilter;

use ingestor::{run_candles_ingest, refresh_universe_pairs};

fn install_rustls_provider() {
    // Выбираем провайдера явно. Иначе при включенных ring+aws-lc-rs будет panic.
    rustls::crypto::ring::default_provider()
         .install_default()
         .expect("install_default failed");
}

#[tokio::main]
async fn main() -> Result<()> {
    install_rustls_provider();

    dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,ingestor=info,connections=info")),
        )
        .init();

    info!("Refreshing universe pairs...");
    refresh_universe_pairs().await?;

    info!("Starting ingest (candles monolith)...");
    run_candles_ingest().await
}

