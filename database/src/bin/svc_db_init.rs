use anyhow::Result;
use dotenvy::dotenv;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    dotenv().ok();

    // такой же дефолт, как у тебя в тесте
    let database_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    // Convert the error type to be compatible with anyhow::Result
    match database_lib::initialize_database(&database_url).await {
        Ok(()) => Ok(()),
        Err(e) => Err(anyhow::anyhow!("Database initialization failed: {}", e)),
    }
}
