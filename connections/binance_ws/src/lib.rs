use anyhow::Result;
use futures_util::{SinkExt, StreamExt};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use tokio::sync::{broadcast, mpsc, Mutex, RwLock};
use tokio::time::{sleep, timeout};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tracing::{debug, error, info, warn};

#[derive(Debug, Clone)]
pub enum BinanceWsEvent {
    Connected,
    Disconnected,
    Message { stream: String, data: Value },
    Error(String),
}

pub struct BinanceWsClient {
    base_url: String,
    event_tx: broadcast::Sender<BinanceWsEvent>,
    subscriptions: Arc<RwLock<HashMap<String, bool>>>,
}

impl BinanceWsClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(100);
        Ok(Self {
            base_url: base_url.into(),
            event_tx,
            subscriptions: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn subscribe(&self, stream: String) -> Result<()> {
        let mut subs = self.subscriptions.try_write().map_err(|e| {
            anyhow::anyhow!("Failed to acquire write lock: {}", e)
        })?;
        subs.insert(stream, true);
        Ok(())
    }

    pub fn unsubscribe(&self, stream: String) -> Result<()> {
        let mut subs = self.subscriptions.try_write().map_err(|e| {
            anyhow::anyhow!("Failed to acquire write lock: {}", e)
        })?;
        subs.remove(&stream);
        Ok(())
    }

    pub async fn connect(&self) -> Result<()> {
        let url = format!("{}{}", self.base_url.trim_end_matches('/'), "/stream");

        loop {
            match self.connect_with_retry(&url).await {
                Ok(_) => {
                    info!("WebSocket connection established");
                    break;
                }
                Err(e) => {
                    warn!("Connection failed: {}. Retrying in 5 seconds...", e);
                    sleep(Duration::from_secs(5)).await;
                }
            }
        }

        Ok(())
    }

    async fn connect_with_retry(&self, url: &str) -> Result<()> {
        let (ws_stream, _) = connect_async(url).await.map_err(|e| {
            error!("Failed to connect to WebSocket: {}", e);
            anyhow::anyhow!("WebSocket connection failed: {}", e)
        })?;

        info!("Connected to Binance WebSocket at: {}", url);

        // Split the stream into sink and stream components
        let (write_sink, read_stream) = ws_stream.split();
        
        // Wrap the sink in an Arc<Mutex<>> to share between tasks
        use tokio::sync::Mutex;
        let shared_sink = Arc::new(Mutex::new(write_sink));

        // Send initial subscription messages if we have any
        {
            let subs = self.subscriptions.read().await;
            for (stream, enabled) in subs.iter() {
                if *enabled {
                    let combined_streams = vec![stream.clone()];
                    let _params = combined_streams.join("/"); // Using _ to indicate unused variable
                    let msg = serde_json::json!({
                        "method": "SUBSCRIBE",
                        "params": combined_streams,
                        "id": 1
                    });

                    let mut sink_lock = shared_sink.lock().await;
                    if let Err(e) = sink_lock.send(Message::Text(msg.to_string())).await {
                        error!("Failed to send subscription message: {}", e);
                    } else {
                        info!("Subscribed to stream: {}", stream);
                    }
                    drop(sink_lock);
                }
            }
        }

        // Send ping messages periodically
        let ping_handle = {
            let sink_clone = shared_sink.clone();
            let event_tx = self.event_tx.clone();
            let subscriptions = self.subscriptions.clone();
            
            tokio::spawn(async move {
                let mut ping_interval = tokio::time::interval(Duration::from_secs(30));
                
                loop {
                    ping_interval.tick().await;
                    
                    let mut sink_lock = sink_clone.lock().await;
                    if let Err(e) = sink_lock.send(Message::Ping(vec![])).await {
                        error!("Failed to send ping: {}", e);
                        let _ = event_tx.send(BinanceWsEvent::Error(format!("Ping failed: {}", e)));
                        drop(sink_lock);
                        break;
                    }
                    drop(sink_lock);
                    
                    // Check for subscription changes and send updates
                    {
                        let subs = subscriptions.read().await;
                        let active_streams: Vec<String> = subs
                            .iter()
                            .filter_map(|(stream, enabled)| if *enabled { Some(stream.clone()) } else { None })
                            .collect();
                        
                        if !active_streams.is_empty() {
                            // We can send subscription updates here if needed
                        }
                    }
                }
            })
        };

        // Read messages
        let event_tx_clone = self.event_tx.clone();
        let sink_for_pong = shared_sink.clone();
        let read_handle = tokio::spawn(async move {
            let mut read_stream = read_stream;
            while let Some(message) = read_stream.next().await {
                match message {
                    Ok(Message::Text(text)) => {
                        if let Ok(json_value) = serde_json::from_str::<Value>(&text) {
                            // Handle combined stream format
                            if let Some(stream_name) = json_value.get("stream") {
                                if let Some(data) = json_value.get("data") {
                                    if let Some(stream_str) = stream_name.as_str() {
                                        let event = BinanceWsEvent::Message {
                                            stream: stream_str.to_string(),
                                            data: data.clone(),
                                        };
                                        let _ = event_tx_clone.send(event);
                                    }
                                }
                            } else {
                                // Handle non-combined streams
                                debug!("Received non-combined stream message: {}", text);
                            }
                        } else {
                            warn!("Failed to parse JSON message: {}", text);
                        }
                    }
                    Ok(Message::Ping(_)) => {
                        // Respond to ping with pong
                        let mut sink_lock = sink_for_pong.lock().await;
                        if let Err(e) = sink_lock.send(Message::Pong(vec![])).await {
                            error!("Failed to send pong: {}", e);
                        }
                        drop(sink_lock);
                    }
                    Ok(Message::Pong(_)) => {
                        debug!("Received pong");
                    }
                    Ok(Message::Close(frame)) => {
                        info!("Received close frame: {:?}", frame);
                        let _ = event_tx_clone.send(BinanceWsEvent::Disconnected);
                        break;
                    }
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        let _ = event_tx_clone.send(BinanceWsEvent::Error(e.to_string()));
                        break;
                    }
                    _ => {}
                }
            }
        });

        // Wait for either task to finish
        tokio::select! {
            _ = ping_handle => {},
            _ = read_handle => {},
        }

        Ok(())
    }

    pub fn subscribe_to_events(&self) -> broadcast::Receiver<BinanceWsEvent> {
        self.event_tx.subscribe()
    }

    pub async fn health_check(&self) -> Result<bool> {
        // Check if we have any active subscriptions
        let subs = self.subscriptions.read().await;
        let has_subscriptions = subs.values().any(|&enabled| enabled);
        drop(subs);

        if !has_subscriptions {
            warn!("No active subscriptions in WebSocket client");
            return Ok(false);
        }

        // We could implement a more sophisticated health check here
        // For now, just return true if we have subscriptions
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_binance_ws_client_creation() {
        let client = BinanceWsClient::new("wss://stream.binance.com:9443");
        assert!(client.is_ok());
    }
}
