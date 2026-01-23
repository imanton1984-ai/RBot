use anyhow::{Context, Result};
use common::timeframe::Timeframe;
use serde::Deserialize;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct KlineRow {
    pub open_time_ms: i64,
    pub close_time_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

// Binance klines: array of arrays (strings for prices/vol)
// [ openTime, open, high, low, close, volume, closeTime, ... ]
#[derive(Debug, Deserialize)]
struct RawKline(
    i64,    // open time
    String, // open
    String, // high
    String, // low
    String, // close
    String, // volume
    i64,    // close time
);

fn pf64(s: &str) -> Result<f64> {
    Ok(s.parse::<f64>().with_context(|| format!("parse f64: {}", s))?)
}

pub fn now_ms() -> i64 {
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    (dur.as_millis()) as i64
}

pub fn tf_ms(tf: Timeframe) -> i64 {
    tf.duration_ms() as i64
}

pub async fn fetch_klines(
    client: &reqwest::Client,
    rest_base: &str,
    symbol: &str,
    tf: Timeframe,
    limit: usize,
    start_time_ms: Option<i64>,
) -> Result<Vec<KlineRow>> {
    let url = format!("{}/fapi/v1/klines", rest_base.trim_end_matches('/'));
    let mut req = client
        .get(url)
        .query(&[
            ("symbol", symbol),
            ("interval", tf.as_binance_interval()),
            ("limit", &limit.to_string()),
        ]);

    if let Some(st) = start_time_ms {
        req = req.query(&[("startTime", &st.to_string())]);
    }

    // простой retry/backoff (на 429/5xx)
    let mut backoff_ms = 200u64;
    for attempt in 0..7u32 {
        let resp = req
            .try_clone()
            .context("reqwest::RequestBuilder::try_clone failed")?
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => {
                let bytes = r.bytes().await?;
                // быстрый parse через simd-json
                let mut buf = bytes.to_vec();
                let raw: Vec<RawKline> = simd_json::from_slice(&mut buf)?;
                let mut out = Vec::with_capacity(raw.len());
                for rk in raw {
                    out.push(KlineRow {
                        open_time_ms: rk.0,
                        close_time_ms: rk.6,
                        open: pf64(&rk.1)?,
                        high: pf64(&rk.2)?,
                        low: pf64(&rk.3)?,
                        close: pf64(&rk.4)?,
                        volume: pf64(&rk.5)?,
                    });
                }
                return Ok(out);
            }
            Ok(r) => {
                let code = r.status().as_u16();
                if code == 429 || code >= 500 {
                    tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                    backoff_ms = (backoff_ms * 2).min(5000);
                    continue;
                }
                let body = r.text().await.unwrap_or_default();
                anyhow::bail!("klines http {} body={}", code, body);
            }
            Err(e) => {
                if attempt >= 6 {
                    return Err(e.into());
                }
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
                backoff_ms = (backoff_ms * 2).min(5000);
            }
        }
    }
    anyhow::bail!("fetch_klines: exhausted retries")
}
