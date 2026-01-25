use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use itertools::Itertools;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, RwLock};
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::connect_async;
use tracing::{error, info, warn};

#[derive(Debug, Clone)]
pub enum BinanceWsEvent {
    Connected,
    Disconnected,
    Message { stream: String, data: Value },
    Error(String),
}

#[derive(Debug, Clone)]
pub struct WsCfg {
    pub reconnect_delay: Duration,
    pub ping_interval: Duration,
    /// сколько streams на 1 соединение (шард)
    pub max_streams_per_connection: usize,
}

impl Default for WsCfg {
    fn default() -> Self {
        Self {
            reconnect_delay: Duration::from_secs(5),
            ping_interval: Duration::from_secs(20),
            max_streams_per_connection: 180,
        }
    }
}

/// Клиент держит список подписок и умеет шардиться.
/// Важно: мы используем комбинированный URL (/stream?streams=a/b/c).
pub struct BinanceWsClient {
    base_url: String,
    cfg: WsCfg,
    event_tx: broadcast::Sender<BinanceWsEvent>,
    subscriptions: Arc<RwLock<HashMap<String, bool>>>,
}

impl BinanceWsClient {
    pub fn new(base_url: impl Into<String>, cfg: WsCfg) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(2048);
        Ok(Self {
            base_url: base_url.into(),
            cfg,
            event_tx,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn subscribe(&self, stream: String) -> Result<()> {
        let mut subs = self.subscriptions.try_write()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        subs.insert(stream, true);
        Ok(())
    }

    pub fn unsubscribe(&self, stream: &str) -> Result<()> {
        let mut subs = self.subscriptions.try_write()
            .map_err(|e| anyhow::anyhow!("lock poisoned: {e}"))?;
        subs.remove(stream);
        Ok(())
    }

    pub fn events(&self) -> broadcast::Receiver<BinanceWsEvent> {
        self.event_tx.subscribe()
    }

    pub async fn connect_sharded(&self) -> Result<()> {
        let subs = self.subscriptions.read().await;
        let active: Vec<String> = subs.iter()
            .filter_map(|(s, on)| if *on { Some(s.clone()) } else { None })
            .collect();

        if active.is_empty() {
            info!("WS: no active streams");
            return Ok(());
        }

        for (idx, chunk) in active.iter().chunks(self.cfg.max_streams_per_connection).into_iter().enumerate() {
            let streams: Vec<String> = chunk.cloned().collect();
            let base = self.base_url.clone();
            let tx = self.event_tx.clone();
            let cfg = self.cfg.clone();

            tokio::spawn(async move {
                loop {
                    if let Err(e) = run_ws_shard(idx, &base, &streams, tx.clone(), cfg.clone()).await {
                        warn!("WS shard #{idx} error: {e}; reconnecting in {:?}", cfg.reconnect_delay);
                        sleep(cfg.reconnect_delay).await;
                    }
                }
            });
        }

        Ok(())
    }
}

async fn run_ws_shard(
    idx: usize,
    base_url: &str,
    streams: &[String],
    event_tx: broadcast::Sender<BinanceWsEvent>,
    cfg: WsCfg,
) -> Result<()> {
    let joined = streams.join("/");
    let url = format!("{}/stream?streams={}", base_url.trim_end_matches('/'), joined);

    info!("WS shard #{idx}: connecting {} streams", streams.len());
    let (ws, _) = connect_async(&url).await?;
    let _ = event_tx.send(BinanceWsEvent::Connected);

    let (mut write, mut read) = ws.split();
    let ping_every = cfg.ping_interval;

    let mut ping = tokio::time::interval(ping_every);

    loop {
        tokio::select! {
            _ = ping.tick() => {
                let _ = write.send(Message::Ping(Vec::new())).await;
            }
            msg = read.next() => {
                match msg {
                    Some(Ok(Message::Text(txt))) => {
                        // combined формат: {"stream":"btcusdt@kline_1m","data":{...}}
                        if let Ok(v) = serde_json::from_str::<Value>(&txt) {
                            if let (Some(s), Some(d)) = (v.get("stream").and_then(|x| x.as_str()), v.get("data")) {
                                let _ = event_tx.send(BinanceWsEvent::Message{ stream: s.to_string(), data: d.clone() });
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        let _ = event_tx.send(BinanceWsEvent::Disconnected);
                        break;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        error!("WS shard #{idx}: {e}");
                        let _ = event_tx.send(BinanceWsEvent::Error(format!("{e}")));
                        break;
                    }
                    None => {
                        let _ = event_tx.send(BinanceWsEvent::Disconnected);
                        break;
                    }
                }
            }
        }
    }

    Ok(())
}
