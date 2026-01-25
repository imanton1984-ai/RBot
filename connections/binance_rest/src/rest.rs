use crate::rate_limits::RateLimiter;
use anyhow::{anyhow, bail, Result};
use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer};
use std::time::Duration;
use tracing::{debug, warn};

#[derive(Clone)]
pub struct BinanceRestClient {
    base_url: String,
    http: reqwest::Client,
    limiter: Option<RateLimiter>,
    timeout: Duration,
    retries: u32,
    backoff_ms: u64,
    backoff_max_ms: u64,
}

impl BinanceRestClient {
    /// Дефолты под твой config/binance.toml:
    /// timeout=8000ms, retries=5, backoff=250ms..5000ms, limiter=8rps/16burst (подключается отдельно).
    pub fn new(base_url: impl Into<String>) -> Result<Self> {
        let base_url = base_url.into();
        let timeout = Duration::from_millis(8000);

        let http = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(32)
            .tcp_nodelay(true)
            .build()?;

        Ok(Self {
            base_url,
            http,
            limiter: None,
            timeout,
            retries: 5,
            backoff_ms: 250,
            backoff_max_ms: 5000,
        })
    }

    pub fn with_rate_limiter(mut self, limiter: RateLimiter) -> Self {
        self.limiter = Some(limiter);
        self
    }

    pub fn with_timeouts(mut self, timeout: Duration) -> Result<Self> {
        self.timeout = timeout;
        self.http = reqwest::Client::builder()
            .timeout(timeout)
            .pool_max_idle_per_host(32)
            .tcp_nodelay(true)
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
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    async fn get_json<T: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, String)],
        weight: u32,
    ) -> Result<T> {
        let url = self.url(path);

        let mut attempt: u32 = 0;
        let mut backoff = self.backoff_ms;

        loop {
            attempt += 1;

            // soft rate limit
            let _rate_guard = if let Some(lim) = &self.limiter {
                Some(lim.acquire(weight).await)
            } else {
                None
            };

            let resp = self.http.get(&url).query(query).send().await;

            match resp {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        return Ok(r.json::<T>().await?);
                    }

                    let body = r.text().await.unwrap_or_default();

                    // retry only on “obvious transient”
                    if attempt <= self.retries && should_retry(status) {
                        warn!("REST retry (attempt={attempt}/{}) {} -> {} body={}",
                              self.retries, url, status, trim_body(&body));
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        backoff = (backoff * 2).min(self.backoff_max_ms);
                        continue;
                    }

                    bail!("REST {} failed: status={} body={}", url, status, trim_body(&body));
                }
                Err(e) => {
                    if attempt <= self.retries {
                        warn!("REST error retry (attempt={attempt}/{}) {} err={e}",
                              self.retries, url);
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                        backoff = (backoff * 2).min(self.backoff_max_ms);
                        continue;
                    }
                    return Err(anyhow!(e));
                }
            }
        }
    }

    pub async fn exchange_info_futures(&self) -> Result<ExchangeInfoResp> {
        // futures exchangeInfo: /fapi/v1/exchangeInfo
        self.get_json("/fapi/v1/exchangeInfo", &[], 1).await
    }

    pub async fn ticker_24h_futures(&self) -> Result<Vec<Ticker24h>> {
        // futures ticker/24hr: /fapi/v1/ticker/24hr
        self.get_json("/fapi/v1/ticker/24hr", &[], 1).await
    }

    /// Исторические свечи (под backfill/gap-fill).
    /// limit max 1500 на Binance.
    pub async fn klines_futures(
        &self,
        symbol: &str,
        interval: &str,
        limit: u16,
        start_time_ms: Option<i64>,
        end_time_ms: Option<i64>,
    ) -> Result<Vec<Kline>> {
        let mut q: Vec<(&str, String)> = vec![
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.to_string()),
        ];
        if let Some(v) = start_time_ms {
            q.push(("startTime", v.to_string()));
        }
        if let Some(v) = end_time_ms {
            q.push(("endTime", v.to_string()));
        }

        debug!("GET klines {} {} limit={} start={:?} end={:?}", symbol, interval, limit, start_time_ms, end_time_ms);

        self.get_json("/fapi/v1/klines", &q, 1).await
    }
}

fn should_retry(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::BAD_GATEWAY
        || status == StatusCode::SERVICE_UNAVAILABLE
        || status == StatusCode::GATEWAY_TIMEOUT
        || status == StatusCode::REQUEST_TIMEOUT
}

fn trim_body(s: &str) -> String {
    let s = s.trim();
    if s.len() > 240 {
        format!("{}...", &s[..240])
    } else {
        s.to_string()
    }
}

#[derive(Debug, Deserialize)]
pub struct ExchangeInfoResp {
    pub symbols: Vec<ExSymbol>,
}

#[derive(Debug, Deserialize)]
pub struct ExSymbol {
    pub symbol: String,
    pub status: String,
    #[serde(rename = "baseAsset")]
    pub base_asset: String,
    #[serde(rename = "quoteAsset")]
    pub quote_asset: String,
    #[serde(rename = "contractType")]
    pub contract_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Ticker24h {
    pub symbol: String,
    #[serde(rename = "quoteVolume")]
    pub quote_volume: String,
    #[serde(rename = "lastPrice")]
    pub last_price: String,
}

/// REST klines row (array) -> struct
#[derive(Debug, Clone)]
pub struct Kline {
    pub open_time_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub close_time_ms: i64,
}

impl<'de> Deserialize<'de> for Kline {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;

        impl<'de> serde::de::Visitor<'de> for V {
            type Value = Kline;

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "binance kline array")
            }

            fn visit_seq<A>(self, mut seq: A) -> std::result::Result<Kline, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let open_time_ms: i64 = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing open_time"))?;

                let open_s: String = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing open"))?;
                let high_s: String = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing high"))?;
                let low_s: String = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing low"))?;
                let close_s: String = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing close"))?;
                let vol_s: String = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing volume"))?;

                let close_time_ms: i64 = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::custom("missing close_time"))?;

                // Остальные поля (quoteAssetVolume, trades, ...) — пропускаем
                // (до тех пор пока не понадобятся)
                // 7..11
                for _ in 0..5 {
                    let _ = seq.next_element::<serde_json::Value>()?;
                }

                let open = open_s.parse::<f64>().map_err(serde::de::Error::custom)?;
                let high = high_s.parse::<f64>().map_err(serde::de::Error::custom)?;
                let low = low_s.parse::<f64>().map_err(serde::de::Error::custom)?;
                let close = close_s.parse::<f64>().map_err(serde::de::Error::custom)?;
                let volume = vol_s.parse::<f64>().map_err(serde::de::Error::custom)?;

                Ok(Kline {
                    open_time_ms,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    close_time_ms,
                })
            }
        }

        deserializer.deserialize_seq(V)
    }
}
