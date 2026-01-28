use anyhow::Result;
use reqwest::{Client, ClientBuilder};
use serde::Deserialize;
use std::time::Duration;
use tracing::{debug, error};

#[derive(Debug, Clone)]
pub struct BinanceApi {
    client: Client,
    base_url: String,
}

#[derive(Deserialize, Debug)]
pub struct ExchangeInfo {
    pub symbols: Vec<Symbol>,
}

#[derive(Deserialize, Debug)]
pub struct Symbol {
    pub symbol: String,
    pub status: String,
    pub base_asset: String,
    pub quote_asset: String,
}

impl BinanceApi {
    pub fn new() -> Result<Self> {
        let client = ClientBuilder::new()
            .timeout(Duration::from_secs(30))
            .build()?;

        Ok(BinanceApi {
            client,
            base_url: "https://api.binance.com".to_string(),
        })
    }

    pub async fn ping(&self) -> Result<()> {
        let url = format!("{}/api/v3/ping", self.base_url);
        
        let response = self.client.get(&url).send().await?;
        
        if response.status().is_success() {
            debug!("Binance REST ping ok");
            Ok(())
        } else {
            error!("Binance API connection failed with status: {}", response.status());
            anyhow::bail!("Binance API connection failed");
        }
    }

    pub async fn get_exchange_info(&self) -> Result<ExchangeInfo> {
        let url = format!("{}/api/v3/exchangeInfo", self.base_url);
        
        let response = self.client.get(&url).send().await?;
        
        if response.status().is_success() {
            let exchange_info: ExchangeInfo = response.json().await?;
            debug!("Retrieved exchange info with {} symbols", exchange_info.symbols.len());
            Ok(exchange_info)
        } else {
            error!("Failed to get exchange info: {}", response.status());
            anyhow::bail!("Failed to get exchange info");
        }
    }

    pub async fn get_klines(&self, symbol: &str, interval: &str, limit: u32) -> Result<String> {
        let url = format!(
            "{}/api/v3/klines?symbol={}&interval={}&limit={}",
            self.base_url, symbol.to_uppercase(), interval, limit
        );
        
        let response = self.client.get(&url).send().await?;
        
        if response.status().is_success() {
            let klines_data = response.text().await?;
            debug!("Retrieved klines data for {} with interval {}", symbol, interval);
            Ok(klines_data)
        } else {
            error!("Failed to get klines: {}", response.status());
            anyhow::bail!("Failed to get klines");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_binance_api_connection() {
        let api = BinanceApi::new().unwrap();
        assert!(api.ping().await.is_ok());
    }
}