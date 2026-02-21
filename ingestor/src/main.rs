use anyhow::Result;
use dotenvy::dotenv;
use tracing::{info, warn, error};
use tracing_subscriber::EnvFilter;

use ingestor::{run_candles_ingest, refresh_universe_pairs, has_active_pairs_in_db};

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
    match refresh_universe_pairs().await {
        Ok(result) => {
            info!(
                "Universe pairs refreshed OK: selected={}, active={}",
                result.selected_cnt, result.active_cnt
            );
        }
        Err(e) => {
            // Refresh не удался. Проверяем, есть ли уже пары в БД.
            // Если есть — продолжаем со старыми (инкрементальный запуск).
            // Если нет — падаем, т.к. без пар работать невозможно.
            error!("Failed to refresh universe pairs: {:#}", e);

            match has_active_pairs_in_db().await {
                Ok(true) => {
                    warn!(
                        "Using existing active pairs from DB (refresh failed, but DB is not empty). \
                         Pairs will be refreshed on next periodic cycle."
                    );
                }
                Ok(false) => {
                    return Err(e.context(
                        "No active pairs in DB and refresh failed — cannot start without pairs"
                    ));
                }
                Err(db_err) => {
                    error!("Cannot check DB for existing pairs: {:#}", db_err);
                    return Err(e.context(
                        "Refresh failed and DB check also failed"
                    ));
                }
            }
        }
    }

    info!("Starting ingest (candles monolith)...");
    run_candles_ingest().await
}

