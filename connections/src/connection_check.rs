use crate::{
    binance_api::BinanceApi,
    binance_websocket::BinanceWsConnection,
    database::DatabaseConnection,
    redpanda::RedpandaConnection
};
use anyhow::Result;
use tracing::{info, error};
use tokio::try_join;

pub async fn check_all_connections() -> Result<()> {
    info!("Starting parallel system health check...");

    // 1. Проверка БД
    let db_check = async {
        let db = DatabaseConnection::new(Default::default()).await?;
        db.ping().await.map_err(|e| anyhow::anyhow!("Database unreachable: {e}"))
    };

    // 2. Проверка Binance REST API
    let api_check = async {
        let api = BinanceApi::new()?;
        api.ping().await.map_err(|e| anyhow::anyhow!("Binance API unreachable: {e}"))
    };

    // 3. Проверка Redpanda (Kafka)
    let rp_check = async {
        let mut rp = RedpandaConnection::new_from_env().await?;
        rp.connect_producer().await?;
        rp.ping().await.map_err(|e| anyhow::anyhow!("Redpanda unreachable: {e}"))
    };

    // 4. Проверка WebSocket (попытка подключения)
    let ws_check = async {
        let ws = BinanceWsConnection::new_from_env().await?;
        ws.check().await.map_err(|e| anyhow::anyhow!("Binance WS handshake failed: {e}"))
    };

    // Выполняем всё параллельно. Если хоть один упадет, вернется Err.
    match try_join!(db_check, api_check, rp_check, ws_check) {
        Ok(_) => {
            info!("✓ All systems operational");
            Ok(())
        }
        Err(e) => {
            error!("✗ System check failed: {}", e);
            anyhow::bail!("Health check failed: {}", e)
        }
    }
}