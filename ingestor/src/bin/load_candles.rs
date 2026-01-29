use anyhow::Result;
use dotenvy::dotenv;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Starting historical candle loading...");

    // Load historical candles
    ingestor::candles::load_historical_candles().await?;
    info!("Historical candle loading completed");

    // Start realtime candle ingestion
    info!("Starting realtime candle ingestion...");
    // Note: In a production setup, this would run continuously
    // For now, we'll just start it and let it run in the background
    tokio::spawn(async {
        if let Err(e) = ingestor::candles::start_realtime_candle_ingestion().await {
            tracing::error!("Realtime candle ingestion error: {}", e);
        }
    });

    // Keep the program running
    tokio::signal::ctrl_c().await.expect("Failed to listen for ctrl+c");
    info!("Received shutdown signal");

    Ok(())
}