use anyhow::Result;
use connections::{BinanceRestClient, DatabaseManager, KafkaManager, RestRateLimitCfg};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(false).init();

    let db_url = env::var("DATABASE_URL")?;
    let brokers = env::var("KAFKA_BROKERS")?;
    let topic = env::var("KAFKA_HEALTH_TOPIC").unwrap_or_else(|_| "health_check".to_string());
    let binance_rest = env::var("BINANCE_REST_URL").unwrap_or_else(|_| "https://fapi.binance.com".to_string());

    let db = DatabaseManager::new(&db_url).await?;
    let kafka = KafkaManager::new(&brokers, &topic)?;
    let rest = BinanceRestClient::new(binance_rest, RestRateLimitCfg::default())?;

    // минимальный preflight
    if !db.health_check().await? { anyhow::bail!("DB health failed"); }
    if !kafka.health_check().await? { anyhow::bail!("Kafka health failed"); }
    let _ = rest.futures_exchange_info().await?;

    println!("conncheck OK");
    Ok(())
}
