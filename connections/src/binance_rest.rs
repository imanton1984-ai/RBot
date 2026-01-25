use anyhow::{Context, Result};
use reqwest::StatusCode;
use serde::de::DeserializeOwned;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Semaphore};
use tracing::{warn};

use crate::now_ms;

/// Настройка rate-limit для REST.
/// Важно: это *сознательно консервативно*, чтобы не ловить 418.
/// Потом можно поднять, когда пайплайн будет стабилен.
#[derive(Debug, Clone)]
pub struct RestRateLimitCfg {
    /// Макс запросов/сек (мягкий лимит)
    pub rps: u32,
    /// Допустимый burst
    pub burst: u32,
    /// Макс параллельных HTTP запросов (самое важное против банов)
    pub max_in_flight: usize,
    /// Таймаут ожидания токена
    pub acquire_timeout: Duration,
}

impl Default for RestRateLimitCfg {
    fn default() -> Self {
        Self {
            rps: 6,
            burst: 12,
            max_in_flight: 8,
            acquire_timeout: Duration::from_secs(10),
        }
    }
}

/// Gate на случай 418/1003 (бан до timestamp).
#[derive(Debug)]
struct HttpGate {
    banned_until_ms: AtomicI64,
}

impl HttpGate {
    fn new() -> Self {
        Self { banned_until_ms: AtomicI64::new(0) }
    }

    fn is_banned(&self) -> bool {
        now_ms() < self.banned_until_ms.load(Ordering::Relaxed)
    }

    fn banned_for(&self) -> Duration {
        let until = self.banned_until_ms.load(Ordering::Relaxed);
        if until <= 0 {
            return Duration::from_millis(0);
        }
        let left = until - now_ms();
        if left <= 0 { Duration::from_millis(0) } else { Duration::from_millis(left as u64) }
    }

    fn set_banned_until(&self, until_ms: i64) {
        self.banned_until_ms.store(until_ms, Ordering::Relaxed);
    }
}

/// Простой token-bucket (мягкий) + semaphore для in-flight.
#[derive(Debug)]
struct RateLimiter {
    cfg: RestRateLimitCfg,
    tokens: Mutex<(f64, Instant)>,
    in_flight: Arc<Semaphore>, // Change to Arc<Semaphore> 
}

impl RateLimiter {
    fn new(cfg: RestRateLimitCfg) -> Self {
        Self {
            tokens: Mutex::new((cfg.burst as f64, Instant::now())),
            in_flight: Arc::new(Semaphore::new(cfg.max_in_flight)), // Wrap in Arc 
            cfg,
        }
    }

    async fn acquire(&self) -> Result<tokio::sync::OwnedSemaphorePermit> {
        // 1) ограничиваем параллелизм
        // Added explicit type Result<OwnedSemaphorePermit, Elapsed> for timeout [cite: 536]
        let permit = tokio::time::timeout(
            self.cfg.acquire_timeout,
            self.in_flight.clone().acquire_owned(), // Now works because Arc is Clone [cite: 379, 536]
        )
        .await
        .context("rate limiter acquire timeout (in-flight)")?
        .context("rate limiter acquire failed (in-flight)")?;

        // 2) мягкий token bucket (RPS)
        loop {
            let mut g = self.tokens.lock().await;
            let (refill_tokens, last) = *g;

            let now = Instant::now();
            let elapsed = (now - last).as_secs_f64();
            let new_tokens = (refill_tokens + elapsed * self.cfg.rps as f64)
                .min(self.cfg.burst as f64);

            if new_tokens >= 1.0 {
                *g = (new_tokens - 1.0, now);
                return Ok(permit);
            }

            // не хватает токенов — подождём чуть-чуть
            let need = 1.0 - new_tokens;
            let sleep_s = need / self.cfg.rps.max(1) as f64;
            let sleep_d = Duration::from_millis((sleep_s * 1000.0).clamp(5.0, 250.0) as u64);
            drop(g);
            tokio::time::sleep(sleep_d).await;
        }
    }
}

#[derive(Clone)]
pub struct BinanceRestClient {
    http: reqwest::Client,
    base_url: String,
    limiter: Arc<RateLimiter>,
    gate: Arc<HttpGate>,
}

impl BinanceRestClient {
    pub fn new(base_url: impl Into<String>, cfg: RestRateLimitCfg) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(15))
            .pool_max_idle_per_host(32)
            .build()
            .context("failed to build reqwest client")?;

        Ok(Self {
            http,
            base_url: base_url.into(),
            limiter: Arc::new(RateLimiter::new(cfg)),
            gate: Arc::new(HttpGate::new()),
        })
    }

    /// Главный “безопасный” GET.
    async fn get_json<T: DeserializeOwned>(&self, path: &str, query: &[(&str, String)]) -> Result<T> {
        // если бан — ждём (это лучше чем добивать IP)
        if self.gate.is_banned() {
            let d = self.gate.banned_for();
            warn!("REST gate banned, sleeping {:?}", d);
            tokio::time::sleep(d).await;
        }

        let _permit = self.limiter.acquire().await?;

        let url = format!("{}{}", self.base_url.trim_end_matches('/'), path);

        // retry с backoff (но без бешенства)
        let mut backoff = Duration::from_millis(250);
        for attempt in 1..=6 {
            let resp = self.http.get(&url).query(&query).send().await;

            match resp {
                Ok(r) => {
                    let status = r.status();
                    if status.is_success() {
                        let v = r.json::<T>().await.context("json decode failed")?;
                        return Ok(v);
                    }

                    // читаем body (важно для -1003 banned until)
                    let body = r.text().await.unwrap_or_default();

                    // 418/429: ставим gate и уходим в сон
                    if status == StatusCode::TOO_MANY_REQUESTS || status == StatusCode::IM_A_TEAPOT {
                        if let Some(until_ms) = parse_ban_until_ms(&body) {
                            self.gate.set_banned_until(until_ms);
                            let d = self.gate.banned_for();
                            warn!("REST banned detected, until_ms={until_ms}, sleeping {:?}", d);
                            tokio::time::sleep(d).await;
                            continue;
                        }

                        // если не смогли распарсить — просто backoff
                        warn!("REST {} (attempt {attempt}), body={}", status, body);
                        tokio::time::sleep(backoff).await;
                        backoff = (backoff * 2).min(Duration::from_secs(10));
                        continue;
                    }

                    // прочие ошибки — ограниченный retry
                    warn!("REST {} (attempt {attempt}), body={}", status, body);
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(5));
                }
                Err(e) => {
                    warn!("REST error (attempt {attempt}): {e}");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(5));
                }
            }
        }

        anyhow::bail!("REST request failed after retries: {path}");
    }

    /// Быстрый список пар (exchangeInfo) — редкий вызов (на старте / раз в N минут).
    pub async fn futures_exchange_info(&self) -> Result<serde_json::Value> {
        self.get_json("/fapi/v1/exchangeInfo", &[]).await
    }

    /// 24h тикеры — для фильтров вселенной (объём/волатильность)
    pub async fn futures_ticker_24h(&self) -> Result<Vec<serde_json::Value>> {
        self.get_json("/fapi/v1/ticker/24hr", &[]).await
    }

    /// Klines — основной backfill.
    pub async fn futures_klines(
        &self,
        symbol: &str,
        interval: &str,
        limit: u32,
        start_time: Option<i64>,
        end_time: Option<i64>,
    ) -> Result<Vec<Vec<serde_json::Value>>> {
        let mut q = vec![
            ("symbol", symbol.to_string()),
            ("interval", interval.to_string()),
            ("limit", limit.to_string()),
        ];
        if let Some(v) = start_time { q.push(("startTime", v.to_string())); }
        if let Some(v) = end_time { q.push(("endTime", v.to_string())); }

        self.get_json("/fapi/v1/klines", &q).await
    }
}

/// Парсим "banned until <digits>" из binance body, например:
/// {"code":-1003,"msg":"Way too many requests; IP(...) banned until 1769..."}
fn parse_ban_until_ms(body: &str) -> Option<i64> {
    let lower = body.to_lowercase();

    // ищем "banned until"
    let idx = lower.find("banned until")?;
    let tail = &lower[idx..];

    // вытаскиваем подряд идущие цифры (epoch ms)
    let mut digits = String::new();
    for c in tail.chars() {
        if c.is_ascii_digit() {
            digits.push(c);
        } else if !digits.is_empty() {
            break;
        }
    }
    if digits.len() < 8 {
        return None;
    }
    digits.parse::<i64>().ok()
}
