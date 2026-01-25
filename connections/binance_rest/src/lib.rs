use anyhow::Result;
use reqwest::Client;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

pub struct BinanceRestClient {
    pub client: Client,
    base_url: String,
    timeout: Duration,
    retries: u32,
    backoff_ms: u64,
    backoff_max_ms: u64,
}

impl BinanceRestClient {
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let timeout = Duration::from_secs(10);

        let client = Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(32)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .connect_timeout(Duration::from_secs(5))
            .build()?;

        Ok(Self {
            client,
            base_url,
            timeout,
            retries: 5,
            backoff_ms: 500,
            backoff_max_ms: 10000,
        })
    }

    pub fn with_timeouts(mut self, timeout: Duration, connect_timeout: Duration) -> Result<Self> {
        self.timeout = timeout;
        self.client = Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(32)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .connect_timeout(connect_timeout)
            .build()?;

        Ok(self)
    }

    pub fn with_retries(mut self, retries: u32, backoff_ms: u64, backoff_max_ms: u64) -> Self {
        self.retries = retries;
        self.backoff_ms = backoff_ms;
        self.backoff_max_ms = backoff_max_ms;
        self
    }

    fn url(&self, path: &str) -> String {
        let suffix = if path.starts_with('/') { path } else { &format!("/{}", path) };
        format!("{}{}", self.base_url.trim_end_matches('/'), suffix)
    }

    pub async fn health_check(&self) -> Result<bool> {
        let url = self.url("/api/v3/time");
        
        for attempt in 1..=self.retries {
            match self.client.get(&url).send().await {
                Ok(response) => {
                    if response.status().is_success() {
                        debug!("Binance REST health check successful");
                        return Ok(true);
                    } else {
                        warn!("Binance REST health check failed with status: {}", response.status());
                    }
                }
                Err(e) => {
                    warn!("Binance REST health check attempt {} failed: {}", attempt, e);
                    if attempt < self.retries {
                        let delay = std::cmp::min(self.backoff_ms * (2_u64.pow(attempt - 1)), self.backoff_max_ms);
                        sleep(Duration::from_millis(delay)).await;
                    }
                }
            }
        }
        
        error!("Binance REST health check failed after {} attempts", self.retries);
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_binance_rest_client_creation() {
        let client = BinanceRestClient::new("https://api.binance.com");
        assert!(client.is_ok());
    }
}
