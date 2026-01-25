use anyhow::Result;
use tracing::{info, warn};

use crate::{BinanceRestClient, DatabaseManager, KafkaManager};

pub async fn preflight(db: &DatabaseManager, kafka: &KafkaManager, rest: &BinanceRestClient) -> Result<()> {
    info!("preflight: db...");
    if !db.health_check().await? {
        anyhow::bail!("preflight: db health_check failed");
    }

    info!("preflight: kafka...");
    if !kafka.health_check().await? {
        anyhow::bail!("preflight: kafka health_check failed");
    }

    info!("preflight: binance rest exchangeInfo...");
    let _ = rest.futures_exchange_info().await?; // один лёгкий вызов
    warn!("preflight: OK");

    Ok(())
}
