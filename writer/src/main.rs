mod candle_writer;

use anyhow::Result;
use candle_writer::CandleWriter;
use dotenvy::dotenv;
use tracing::info;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    info!("Starting candle writer service...");

    let writer = CandleWriter::new().await?;

    // Run the candle writer with timeframe-specific listening
    writer.start_listening_by_timeframe().await?;

    Ok(())
}