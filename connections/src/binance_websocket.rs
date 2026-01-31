use anyhow::Result;
use futures::{StreamExt};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};
use tokio_postgres;
use tokio_postgres::NoTls;

#[derive(Clone, Debug)]
pub struct PairInfo {
    pub symbol_id: i64,
    pub symbol: String,
}

#[derive(Clone, Debug)]
pub struct BinanceWsConfig {
    pub base_url: String,
    pub streams: Vec<String>,
    pub ping_interval: Duration,
    pub reconnect_base: Duration,
    pub reconnect_max: Duration,
}

impl BinanceWsConfig {
    pub async fn new_from_env_with_db_fallback() -> Result<Self> {
        let base_url = std::env::var("BINANCE_WS_BASE_URL")
            .unwrap_or_else(|_| "wss://fstream.binance.com/stream".to_string());

        // Try to get streams from environment variable first
        let streams_raw = std::env::var("BINANCE_WS_STREAMS").unwrap_or_else(|_| "".to_string());
        let mut streams = streams_raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        // If no streams are specified via environment variable, fetch active pairs from database
        if streams.is_empty() {
            info!("No streams specified via BINANCE_WS_STREAMS, fetching active pairs from database...");
            match Self::fetch_active_pairs_from_db().await {
                Ok(active_pairs) => {
                    // Create streams for kline_1m and trade for each active pair
                    for pair in active_pairs {
                        let lower_symbol = pair.symbol.to_lowercase();
                        streams.push(format!("{}@kline_1m", lower_symbol));
                        streams.push(format!("{}@trade", lower_symbol));
                    }
                    info!("Loaded {} streams from database", streams.len());
                }
                Err(e) => {
                    warn!("Failed to fetch active pairs from database: {}. Using default BTCUSDT streams.", e);
                    // Fallback to default streams
                    streams.push("btcusdt@trade".to_string());
                    streams.push("btcusdt@kline_1m".to_string());
                }
            }
        }

        Ok(BinanceWsConfig {
            base_url,
            streams,
            ping_interval: Duration::from_secs(30),
            reconnect_base: Duration::from_secs(1),
            reconnect_max: Duration::from_secs(60),
        })
    }

    async fn fetch_active_pairs_from_db() -> Result<Vec<PairInfo>> {
        let db_url = std::env::var("DATABASE_URL")
            .or_else(|_| std::env::var("POSTGRES_CONNECTION_STRING"))
            .or_else(|_| std::env::var("POSTGRES_URL"))
            .unwrap_or_else(|_| "postgresql://postgres:@localhost:5433/timescaledb_binance".to_string());

        let (client, connection) = tokio_postgres::connect(&db_url, NoTls).await?;

        // Spawn the connection handling
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                eprintln!("PostgreSQL connection error: {}", e);
            }
        });

        let rows = client
            .query(
                "SELECT symbol_id, symbol
                 FROM market.pairs
                 WHERE is_active = TRUE
                 ORDER BY symbol_id",
                &[],
            )
            .await?;

        let mut pairs = Vec::with_capacity(rows.len());
        for row in rows {
            pairs.push(PairInfo {
                symbol_id: row.get(0),
                symbol: row.get(1),
            });
        }

        Ok(pairs)
    }
}

pub struct BinanceWsConnection {
    cfg: BinanceWsConfig,
}

impl BinanceWsConnection {
    pub async fn new_from_env() -> Result<Self> {
        let cfg = BinanceWsConfig::new_from_env_with_db_fallback().await?;
        Ok(Self { cfg })
    }

    pub async fn check(&self) -> Result<()> {
        let url = self.build_url(&self.cfg.streams)?;
        let (mut ws_stream, _) = connect_async(&url).await?;
        ws_stream.close(None).await?;
        Ok(())
    }

    pub async fn run_forever(&self, tx: mpsc::Sender<bytes::Bytes>, status_tx: watch::Sender<bool>) -> Result<()> {
        let url = self.build_url(&self.cfg.streams)?;
        let mut backoff = self.cfg.reconnect_base;

        loop {
            info!("Connecting to Binance WS: {}", url);
            match connect_async(&url).await {
                Ok((ws_stream, _)) => {
                    info!("Binance WS connected");
                    let _ = status_tx.send(true);
                    backoff = self.cfg.reconnect_base;

                    let (mut _ws_write, mut ws_read) = ws_stream.split();

                    while let Some(msg) = ws_read.next().await {
                        match msg {
                            Ok(Message::Text(text)) => {
                                // Превращаем Utf8Bytes в Bytes (zero-copy)
                                if let Err(e) = tx.send(text.into()).await {
                                    error!("Receiver dropped: {e}");
                                    return Ok(());
                                }
                            }
                            Ok(Message::Binary(bin)) => {
                                // bin уже является Bytes
                                if let Err(e) = tx.send(bin).await {
                                    error!("Receiver dropped: {e}");
                                    return Ok(());
                                }
                            }
                            Ok(Message::Ping(_)) => {}
                            Ok(Message::Close(_)) => break,
                            Err(e) => {
                                warn!("Binance WS recv error: {e}");
                                break;
                            }
                            _ => {}
                        }
                    }
                    let _ = status_tx.send(false);
                }
                Err(e) => {
                    warn!("Connect failed: {e}. Retrying in {:?}", backoff);
                    let _ = status_tx.send(false);
                }
            }
            tokio::time::sleep(backoff).await;
            backoff = std::cmp::min(backoff * 2, self.cfg.reconnect_max);
        }
    }

    fn build_url(&self, streams: &[String]) -> Result<String> {
        let joined = if streams.is_empty() { "btcusdt@trade".to_string() } else { streams.join("/") };
        Ok(format!("{}?streams={}", self.cfg.base_url, joined))
    }
}


