use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tokio_postgres::NoTls;
use tracing::{info, warn};

use crate::AppState;

fn default_refresh_interval_sec() -> u64 {
    3600
}

#[derive(Debug, Deserialize)]
struct UniverseToml {
    universe: UniverseCfg,
}

#[derive(Debug, Clone, Deserialize)]
struct UniverseCfg {
    pub quote_asset: String,

    // оставляем поле для совместимости с твоими конфигами
    #[serde(rename = "futures_only", default)]
    pub futures_only: bool,

    #[serde(default)]
    pub perpetual_only: bool,

    #[serde(default)]
    pub min_quote_volume_usdt_24h: f64,

    #[serde(default = "default_refresh_interval_sec")]
    pub refresh_interval_sec: u64,

    #[serde(default)]
    pub exclude_base_assets: Vec<String>,
    #[serde(default)]
    pub exclude_symbols: Vec<String>,
    #[serde(default)]
    pub allowlist: Vec<String>,
    #[serde(default)]
    pub denylist: Vec<String>,

    // можно держать на будущее, если захочешь режимы применения
    #[serde(default)]
    pub apply_mode: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExchangeInfoResp {
    symbols: Vec<ExSymbol>,
}

#[derive(Debug, Deserialize)]
struct ExSymbol {
    symbol: String,
    status: String,

    #[serde(rename = "baseAsset")]
    base_asset: String,

    #[serde(rename = "quoteAsset")]
    quote_asset: String,

    // Binance futures exchangeInfo
    #[serde(rename = "contractType")]
    contract_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Ticker24h {
    pub symbol: String,

    #[serde(rename = "quoteVolume")]
    pub quote_volume: String,

    #[serde(rename = "lastPrice")]
    pub last_price: String,
}

/// Делает первичную загрузку/обновление market.pairs и возвращает количество активированных символов.
/// Стадии тут НЕ ставим (ставит main), чтобы не было гонок.
pub async fn universe_startup(st: &AppState) -> Result<usize> {
    let n = refresh_pairs(
        st.db_url.as_str(),
        st.rest_base.as_str(),
        st.universe_cfg_path.as_str(),
    )
    .await
    .context("refresh_pairs(startup) failed")?;

    Ok(n)
}

/// Фоновый loop: периодически обновляет пары.
/// Не лезет в stage (иначе будет ломать RUN во время работы).
pub async fn run_universe_loop(st: AppState) -> Result<()> {
    // читаем interval из конфига, чтобы не компилировать “3600” в код
    let u_cfg = load_universe_cfg(st.universe_cfg_path.as_str())
        .context("load_universe_cfg failed")?;

    let mut interval = tokio::time::interval(Duration::from_secs(u_cfg.refresh_interval_sec));
    loop {
        interval.tick().await;
        info!("Universe refresh tick...");
        match refresh_pairs(
            st.db_url.as_str(),
            st.rest_base.as_str(),
            st.universe_cfg_path.as_str(),
        )
        .await
        {
            Ok(n) => info!("Universe refresh OK. active pairs: {}", n),
            Err(e) => warn!("Universe refresh failed: {:#}", e),
        }
    }
}

fn load_universe_cfg(path: &str) -> Result<UniverseCfg> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("read universe config: {}", path))?;
    let parsed: UniverseToml =
        toml::from_str(&content).context("toml::from_str(universe.toml) failed")?;
    Ok(parsed.universe)
}

/// Основная логика:
/// - читает universe.toml
/// - тянет exchangeInfo + ticker/24hr
/// - фильтрует
/// - upsert в market.pairs + is_active
pub async fn refresh_pairs(db_url: &str, rest_base: &str, config_path: &str) -> Result<usize> {
    // 1) Load config (TOML)
    let u = load_universe_cfg(config_path)?;

    let denylist: HashSet<String> = u.denylist.iter().cloned().collect();
    let allowlist: HashSet<String> = u.allowlist.iter().cloned().collect();
    let exclude_symbols: HashSet<String> = u.exclude_symbols.iter().cloned().collect();
    let exclude_base: HashSet<String> = u.exclude_base_assets.iter().cloned().collect();

    // 2) REST fetch
    let http = Client::builder()
        .timeout(Duration::from_secs(15))
        .tcp_nodelay(true)
        .pool_max_idle_per_host(64)
        .build()
        .context("build reqwest client")?;

    // Futures endpoints (как у тебя в коде)
    let ex_url = format!("{rest_base}/fapi/v1/exchangeInfo");
    let ex: ExchangeInfoResp = http
        .get(&ex_url)
        .send()
        .await
        .context("GET exchangeInfo failed")?
        .error_for_status()
        .context("exchangeInfo non-200")?
        .json()
        .await
        .context("exchangeInfo json decode failed")?;

    let t_url = format!("{rest_base}/fapi/v1/ticker/24hr");
    let tickers: Vec<Ticker24h> = http
        .get(&t_url)
        .send()
        .await
        .context("GET ticker/24hr failed")?
        .error_for_status()
        .context("ticker/24hr non-200")?
        .json()
        .await
        .context("ticker/24hr json decode failed")?;

    // 3) Build maps
    let mut vol_map: HashMap<String, f64> = HashMap::with_capacity(tickers.len());
    let mut price_map: HashMap<String, f64> = HashMap::with_capacity(tickers.len());
    for t in tickers {
        if let Ok(v) = t.quote_volume.parse::<f64>() {
            vol_map.insert(t.symbol.clone(), v);
        }
        if let Ok(p) = t.last_price.parse::<f64>() {
            price_map.insert(t.symbol, p);
        }
    }

    // 4) Filter symbols
    let mut selected: Vec<(String, String, String, f64, f64)> = Vec::new();

    for s in ex.symbols {
        let sym = &s.symbol;

        if denylist.contains(sym) {
            continue;
        }
        if s.status != "TRADING" {
            continue;
        }
        if s.quote_asset != u.quote_asset {
            continue;
        }
        if exclude_symbols.contains(sym) {
            continue;
        }
        if exclude_base.contains(&s.base_asset) {
            continue;
        }

        // Perpetual-only фильтр (для futures exchangeInfo contractType)
        if u.perpetual_only {
            let is_perp = s.contract_type.as_deref() == Some("PERPETUAL");
            if !is_perp && !allowlist.contains(sym) {
                continue;
            }
        }

        let vol = *vol_map.get(sym).unwrap_or(&0.0);
        let price = *price_map.get(sym).unwrap_or(&0.0);

        // Объёмный фильтр (allowlist всегда проходит)
        if vol < u.min_quote_volume_usdt_24h && !allowlist.contains(sym) {
            continue;
        }

        selected.push((
            s.symbol,
            s.base_asset,
            s.quote_asset,
            vol,
            price,
        ));
    }

    // dedup/sort
    selected.sort_by(|a, b| a.0.cmp(&b.0));
    selected.dedup_by(|a, b| a.0 == b.0);

    // 5) Write to DB
    let (mut client_pg, connection) = tokio_postgres::connect(db_url, NoTls)
        .await
        .context("connect postgres failed")?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("postgres connection error: {e}");
        }
    });

    let tx = client_pg.transaction().await.context("begin tx failed")?;

    // deactivate all
    tx.execute("UPDATE market.pairs SET is_active = false", &[])
        .await
        .context("deactivate pairs failed")?;

    // activate selected
    let mut count: usize = 0;
    for (sym, base, quote, vol, price) in &selected {
        tx.execute(
            r#"
            INSERT INTO market.pairs(symbol, base_asset, quote_asset, volume_24h_usdt, last_price, is_active)
            VALUES ($1, $2, $3, $4, $5, true)
            ON CONFLICT (symbol) DO UPDATE
              SET is_active = EXCLUDED.is_active,
                  base_asset = EXCLUDED.base_asset,
                  quote_asset = EXCLUDED.quote_asset,
                  volume_24h_usdt = EXCLUDED.volume_24h_usdt,
                  last_price = EXCLUDED.last_price
            "#,
            &[sym, base, quote, vol, price],
        )
        .await
        .with_context(|| format!("upsert pair {}", sym))?;

        count += 1;
    }

    tx.commit().await.context("commit tx failed")?;
    Ok(count)
}





