use anyhow::{anyhow, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tracing::{info, warn};

#[derive(Clone)]
pub struct MarketWsConfig {
    pub ws_base_url: String,
    pub ping_interval: Duration,
    pub reconnect_backoff: Duration,
    pub reconnect_backoff_max: Duration,
}

impl MarketWsConfig {
    pub fn futures_default(ws_base_url: impl Into<String>) -> Self {
        Self {
            ws_base_url: ws_base_url.into(),
            ping_interval: Duration::from_secs(15),
            reconnect_backoff: Duration::from_millis(500),
            reconnect_backoff_max: Duration::from_millis(15_000),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketWs {
    cfg: MarketWsConfig,
    streams: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MarketWsEvent {
    pub stream: String,
    pub data: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct CombinedMsg {
    stream: String,
    data: serde_json::Value,
}

impl MarketWs {
    pub fn new(cfg: MarketWsConfig, streams: Vec<String>) -> Self {
        Self { cfg, streams }
    }

    fn combined_stream_url(&self) -> String {
        // wss://fstream.binance.com/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m
        let base = self.cfg.ws_base_url.trim_end_matches('/');
        let joined = self.streams.join("/");
        format!("{base}/stream?streams={joined}")
    }

    /// Стартует таску, которая:
    /// - коннектится
    /// - держит ping
    /// - на дисконнекте делает reconnect с backoff
    /// - пушит MarketWsEvent в rx
    pub fn spawn(self, chan_size: usize) -> mpsc::Receiver<MarketWsEvent> {
        let (tx, rx) = mpsc::channel(chan_size);
        tokio::spawn(async move {
            if let Err(e) = self.run_loop(tx).await {
                warn!("ws market loop stopped: {e:#}");
            }
        });
        rx
    }

    async fn run_loop(&self, tx: mpsc::Sender<MarketWsEvent>) -> Result<()> {
        let mut backoff = self.cfg.reconnect_backoff;

        loop {
            let url = self.combined_stream_url();
            info!("ws connect: {}", url);

            match tokio_tungstenite::connect_async(&url).await {
                Ok((mut ws, _resp)) => {
                    backoff = self.cfg.reconnect_backoff;

                    let mut ping = tokio::time::interval(self.cfg.ping_interval);

                    loop {
                        tokio::select! {
                            _ = ping.tick() => {
                                // keepalive
                                let _ = ws.send(Message::Ping(Vec::new())).await;
                            }
                            msg = ws.next() => {
                                match msg {
                                    Some(Ok(Message::Text(txt))) => {
                                        if let Ok(cm) = serde_json::from_str::<CombinedMsg>(&txt) {
                                            if tx.send(MarketWsEvent{ stream: cm.stream, data: cm.data }).await.is_err() {
                                                // consumer dropped => stop
                                                return Ok(());
                                            }
                                        }
                                    }
                                    Some(Ok(Message::Binary(_))) => {}
                                    Some(Ok(Message::Pong(_))) => {}
                                    Some(Ok(Message::Close(_))) => break,
                                    Some(Ok(_)) => {}
                                    Some(Err(e)) => {
                                        warn!("ws error: {e}");
                                        break;
                                    }
                                    None => break,
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("ws connect failed: {e}");
                }
            }

            // reconnect
            tokio::time::sleep(backoff).await;
            backoff = std::cmp::min(backoff * 2, self.cfg.reconnect_backoff_max);
        }
    }
}
