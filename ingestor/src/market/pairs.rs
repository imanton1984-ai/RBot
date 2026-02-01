use anyhow::{bail, Context, Result};
use common::{load_config, AppConfig};
use reqwest::Client;
use serde::Deserialize;
use sqlx::PgPool;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tracing::{info, warn};

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

async fn fetch_exchange_info(client: &Client, base: &str) -> Result<FuturesExchangeInfo> {
    let url = format!("{}/fapi/v1/exchangeInfo", base.trim_end_matches('/'));
    let resp = client.get(url).send().await?.error_for_status()?;
    Ok(resp.json::<FuturesExchangeInfo>().await?)
}

async fn fetch_ticker_24h(client: &Client, base: &str) -> Result<Vec<FuturesTicker24h>> {
    let url = format!("{}/fapi/v1/ticker/24hr", base.trim_end_matches('/'));
    let resp = client.get(url).send().await?.error_for_status()?;
    Ok(resp.json::<Vec<FuturesTicker24h>>().await?)
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

    let timeout = Duration::from_millis(cfg.binance.http_timeout_ms);
    let client = Client::builder().timeout(timeout).build()?;

    let rest_base = cfg.binance.rest_base_url.clone();
    let uni = cfg.universe.clone();

    let exchange = fetch_exchange_info(&client, &rest_base).await?;
    let tickers = fetch_ticker_24h(&client, &rest_base).await?;

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



