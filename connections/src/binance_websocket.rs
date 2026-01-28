use anyhow::Result;
use futures::{StreamExt};
use std::time::Duration;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::{connect_async, tungstenite::Message};
use tracing::{error, info, warn};

#[derive(Clone, Debug)]
pub struct BinanceWsConfig {
    pub base_url: String,
    pub streams: Vec<String>,
    pub ping_interval: Duration,
    pub reconnect_base: Duration,
    pub reconnect_max: Duration,
}

impl Default for BinanceWsConfig {
    fn default() -> Self {
        let base_url = std::env::var("BINANCE_WS_BASE_URL")
            .unwrap_or_else(|_| "wss://fstream.binance.com/stream".to_string());
        let streams_raw = std::env::var("BINANCE_WS_STREAMS").unwrap_or_else(|_| "".to_string());
        let streams = streams_raw
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>();

        BinanceWsConfig {
            base_url,
            streams,
            ping_interval: Duration::from_secs(30),
            reconnect_base: Duration::from_secs(1),
            reconnect_max: Duration::from_secs(60),
        }
    }
}

pub struct BinanceWsConnection {
    cfg: BinanceWsConfig,
}

impl BinanceWsConnection {
    pub async fn new_from_env() -> Result<Self> {
        Ok(Self { cfg: BinanceWsConfig::default() })
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


