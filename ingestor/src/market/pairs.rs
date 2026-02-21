use anyhow::{bail, Context, Result};
use common::{load_config, AppConfig};
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tracing::{info, warn, error};

#[derive(Debug, Clone)]
pub struct UniversePairsResult {
    pub selected_cnt: i64,
    pub active_cnt: i64,
}

#[derive(Debug, Deserialize)]
struct FuturesExchangeInfo {
    symbols: Vec<FuturesSymbol>,
}

#[derive(Debug, Deserialize)]
struct FuturesSymbol {
    symbol: String,
    status: String,
    #[serde(rename = "baseAsset")]
    base_asset: String,
    #[serde(rename = "quoteAsset")]
    quote_asset: String,
    #[serde(rename = "contractType")]
    contract_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FuturesTicker24h {
    symbol: String,
    #[serde(rename = "quoteVolume")]
    quote_volume: String,
    #[serde(rename = "lastPrice")]
    last_price: String,
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

/// Fetches exchange info with retries and exponential backoff.
/// Uses a generous timeout (30s) since the response can be very large (~400 symbols).
async fn fetch_exchange_info(client: &Client, base: &str, retries: u32) -> Result<FuturesExchangeInfo> {
    let url = format!("{}/fapi/v1/exchangeInfo", base.trim_end_matches('/'));
    let max_retries = retries.max(1);
    let mut backoff_ms: u64 = 1_000;

    for attempt in 1..=max_retries {
        match client.get(&url).send().await {
            Ok(resp) => match resp.error_for_status() {
                Ok(ok) => match ok.json::<FuturesExchangeInfo>().await {
                    Ok(data) => return Ok(data),
                    Err(e) => {
                        warn!("exchangeInfo: attempt {}/{} JSON decode error: {}", attempt, max_retries, e);
                    }
                },
                Err(e) => {
                    warn!("exchangeInfo: attempt {}/{} HTTP error: {}", attempt, max_retries, e);
                }
            },
            Err(e) => {
                warn!("exchangeInfo: attempt {}/{} request error: {}", attempt, max_retries, e);
            }
        }

        if attempt < max_retries {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(10_000);
        }
    }

    bail!("fetch_exchange_info failed after {} retries", max_retries)
}

/// Fetches 24h ticker data with retries and exponential backoff.
/// Uses a generous timeout (30s) since the response returns data for all futures pairs.
async fn fetch_ticker_24h(client: &Client, base: &str, retries: u32) -> Result<Vec<FuturesTicker24h>> {
    let url = format!("{}/fapi/v1/ticker/24hr", base.trim_end_matches('/'));
    let max_retries = retries.max(1);
    let mut backoff_ms: u64 = 1_000;

    for attempt in 1..=max_retries {
        match client.get(&url).send().await {
            Ok(resp) => match resp.error_for_status() {
                Ok(ok) => match ok.json::<Vec<FuturesTicker24h>>().await {
                    Ok(data) => return Ok(data),
                    Err(e) => {
                        warn!("ticker24h: attempt {}/{} JSON decode error: {}", attempt, max_retries, e);
                    }
                },
                Err(e) => {
                    warn!("ticker24h: attempt {}/{} HTTP error: {}", attempt, max_retries, e);
                }
            },
            Err(e) => {
                warn!("ticker24h: attempt {}/{} request error: {}", attempt, max_retries, e);
            }
        }

        if attempt < max_retries {
            tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
            backoff_ms = (backoff_ms * 2).min(10_000);
        }
    }

    bail!("fetch_ticker_24h failed after {} retries", max_retries)
}

/// Не создаём DDL в ingestor.
/// Только проверяем, что core-схема применена.
async fn require_market_core(pool: &PgPool) -> Result<()> {
    let (reg,): (Option<String>,) =
        sqlx::query_as("SELECT to_regclass('market.pairs')::text")
            .fetch_one(pool)
            .await
            .context("failed to check market.pairs existence")?;

    if reg.is_none() {
        bail!(
            "DB schema is not initialized: relation market.pairs does not exist. \
Run your db init (svc_db_init) or apply ddl/010_market_core.sql to this DATABASE_URL."
        );
    }
    Ok(())
}

async fn refresh_pairs_inner() -> Result<UniversePairsResult> {
    let cfg: AppConfig = load_config().context("load_config() failed")?;

    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());
    let pool = PgPool::connect(&db_url).await.context("connect DB failed")?;

    // ✅ важное отличие: без DDL, только проверка
    require_market_core(&pool).await?;

    // Используем увеличенный таймаут для exchangeInfo/ticker24h (крупные ответы).
    // Стандартный http_timeout_ms (8с) недостаточен для десериализации ответа
    // с ~400 символами, особенно при медленном соединении.
    let large_timeout = Duration::from_millis(cfg.binance.http_timeout_ms.max(30_000));
    let client = Client::builder().timeout(large_timeout).build()?;
    let retries = cfg.binance.http_retries.max(3);

    let rest_base = cfg.binance.rest_base_url.clone();
    let uni = cfg.universe.clone();

    let exchange = fetch_exchange_info(&client, &rest_base, retries).await?;
    let tickers = fetch_ticker_24h(&client, &rest_base, retries).await?;

    let ticker_map: HashMap<String, FuturesTicker24h> =
        tickers.into_iter().map(|t| (t.symbol.clone(), t)).collect();

    let allow: HashSet<String> = uni.allowlist.iter().cloned().collect();
    let deny: HashSet<String> = uni.denylist.iter().cloned().collect();
    let excl_base: HashSet<String> = uni.exclude_base_assets.iter().cloned().collect();
    let excl_sym: HashSet<String> = uni.exclude_symbols.iter().cloned().collect();

    let mut selected = Vec::new();

    for s in exchange.symbols {
        let sym = s.symbol;

        // Filter out non-alphanumeric symbols to avoid issues with weird names from the exchange
        if !sym.chars().all(|c| c.is_ascii_alphanumeric()) {
            warn!("Skipping non-alphanumeric symbol: {}", sym);
            continue;
        }

        if deny.contains(&sym) || excl_sym.contains(&sym) {
            continue;
        }
        if s.quote_asset != uni.quote_asset {
            continue;
        }
        if excl_base.contains(&s.base_asset) {
            continue;
        }
        if uni.perpetual_only && s.contract_type.as_deref() != Some("PERPETUAL") {
            continue;
        }
        if s.status != "TRADING" {
            continue;
        }

        let (vol24h, last_price) = match ticker_map.get(&sym) {
            Some(t) => (parse_f64(&t.quote_volume), parse_f64(&t.last_price)),
            None => (0.0, 0.0),
        };

        if !allow.contains(&sym) && vol24h < uni.min_quote_volume_usdt_24h {
            continue;
        }

        let manual_allow = allow.contains(&sym);
        let manual_deny = false;

        selected.push((
            sym,
            s.base_asset,
            s.quote_asset,
            vol24h,
            last_price,
            manual_allow,
            manual_deny,
        ));
    }

    if selected.is_empty() {
        warn!("Universe selection is empty after filters — check universe.toml and Binance endpoints");
    }

    let selected_cnt = selected.len() as i64;

    let mut tx = pool.begin().await?;

    sqlx::query(
        r#"
        UPDATE market.pairs
        SET is_active = FALSE
        WHERE manual_allow = FALSE
        "#,
    )
    .execute(&mut *tx)
    .await?;

    // ✅ под твой DDL: symbol_id — BIGSERIAL PK, symbol UNIQUE
    //    значит вставляем без symbol_id, а конфликт — по symbol.
    for (symbol, base, quote, vol, last, manual_allow, manual_deny) in selected {
        sqlx::query(
            r#"
            INSERT INTO market.pairs
              (symbol, base_asset, quote_asset, is_active,
               futures_eligible, perpetual_only,
               volume_24h_usdt, last_price, last_refreshed_at,
               manual_allow, manual_deny, meta)
            VALUES
              ($1, $2, $3, TRUE,
               TRUE, TRUE,
               $4, $5, now(),
               $6, $7, '{}'::jsonb)
            ON CONFLICT (symbol) DO UPDATE
            SET
              base_asset        = EXCLUDED.base_asset,
              quote_asset       = EXCLUDED.quote_asset,
              is_active         = TRUE,
              futures_eligible  = TRUE,
              perpetual_only    = TRUE,
              volume_24h_usdt   = EXCLUDED.volume_24h_usdt,
              last_price        = EXCLUDED.last_price,
              last_refreshed_at = now(),
              manual_allow      = EXCLUDED.manual_allow,
              manual_deny       = EXCLUDED.manual_deny
            "#,
        )
        .bind(&symbol)
        .bind(&base)
        .bind(&quote)
        .bind(vol)
        .bind(last)
        .bind(manual_allow)
        .bind(manual_deny)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;

    let (active_cnt,): (i64,) =
        sqlx::query_as("SELECT COUNT(*) FROM market.pairs WHERE is_active = TRUE")
            .fetch_one(&pool)
            .await?;

    let res = UniversePairsResult { selected_cnt, active_cnt };

    // ✅ читаем поля тут — варнинга про "selected_cnt never read" не будет,
    // даже если bin не использует результат.
    info!(
        "Pairs refreshed: selected_cnt={}, active_cnt={}",
        res.selected_cnt, res.active_cnt
    );

    Ok(res)
}

pub async fn refresh_universe_pairs() -> Result<UniversePairsResult> {
    refresh_pairs_inner().await
}

/// Проверяет, есть ли уже активные пары в market.pairs.
/// Используется для fallback-логики: если refresh не удался, но пары уже есть — можно продолжить.
pub async fn has_active_pairs_in_db() -> Result<bool> {
    let cfg: AppConfig = load_config().context("load_config() failed")?;
    let db_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| cfg.database.url());
    let pool = PgPool::connect(&db_url).await.context("connect DB failed")?;

    let result: Option<(i64,)> =
        sqlx::query_as("SELECT COUNT(*) FROM market.pairs WHERE is_active = TRUE")
            .fetch_optional(&pool)
            .await
            .ok()
            .flatten();

    match result {
        Some((count,)) => Ok(count > 0),
        None => Ok(false),
    }
}

