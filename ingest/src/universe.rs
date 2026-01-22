use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::time::Duration;
use tokio_postgres::NoTls;

#[derive(Debug, Deserialize)]
struct UniverseToml {
    universe: UniverseCfg,
}

#[derive(Debug, Deserialize)]
struct UniverseCfg {
    quote_asset: String,
    #[serde(rename = "futures_only")]
    _futures_only: bool,
    
    perpetual_only: bool,

    min_quote_volume_usdt_24h: f64,

    #[serde(default)]
    exclude_base_assets: Vec<String>,
    #[serde(default)]
    exclude_symbols: Vec<String>,

    #[serde(default)]
    allowlist: Vec<String>,
    #[serde(default)]
    denylist: Vec<String>,
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
    #[serde(rename = "contractType")]
    contract_type: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Ticker24h {
    pub symbol: String,
    
    #[serde(rename = "quoteVolume")] // На всякий случай проверьте и это
    pub quote_volume: String,

    #[serde(rename = "lastPrice")]    // <--- ДОБАВЬТЕ ЭТО
    pub last_price: String,
}

pub async fn refresh_pairs(db_url: &str, rest_base: &str, universe_cfg_path: &str) -> Result<u64> {
    // 1) Load Config
    let cfg_txt = std::fs::read_to_string(universe_cfg_path)
        .with_context(|| format!("read universe config: {universe_cfg_path}"))?;
    let cfg: UniverseToml = toml::from_str(&cfg_txt).context("parse universe.toml")?;
    let u = cfg.universe;

    let exclude_base: HashSet<String> = u.exclude_base_assets.into_iter().collect();
    let exclude_symbols: HashSet<String> = u.exclude_symbols.into_iter().collect();
    let allowlist: HashSet<String> = u.allowlist.into_iter().collect();
    let denylist: HashSet<String> = u.denylist.into_iter().collect();

    // 2) REST calls
    let client = Client::builder()
        .timeout(Duration::from_secs(12))
        .build()?;

    let ex_url = format!("{rest_base}/fapi/v1/exchangeInfo");
    let ex: ExchangeInfoResp = client
        .get(&ex_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let t_url = format!("{rest_base}/fapi/v1/ticker/24hr");
    let tickers: Vec<Ticker24h> = client
        .get(&t_url)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let mut vol_map: HashMap<String, f64> = HashMap::new();
    let mut price_map: HashMap<String, f64> = HashMap::new();
    for t in tickers {
        if let Ok(v) = t.quote_volume.parse::<f64>() {
            vol_map.insert(t.symbol.clone(), v);
        }
        if let Ok(p) = t.last_price.parse::<f64>() { // Добавили этот блок
            price_map.insert(t.symbol, p);
        }
    }

    // 3) Filter
    // CHANGED: Store (symbol, base, quote) instead of just symbol
    let mut selected: Vec<(String, String, String, f64, f64)> = Vec::new();

    for s in ex.symbols {
        if denylist.contains(&s.symbol) { continue; }
        if s.status != "TRADING" { continue; }
        if s.quote_asset != u.quote_asset { continue; }
        if exclude_symbols.contains(&s.symbol) { continue; }
        if exclude_base.contains(&s.base_asset) { continue; }

        if u.perpetual_only {
            if let Some(ct) = &s.contract_type {
                if ct != "PERPETUAL" {
                    if !allowlist.contains(&s.symbol) { continue; }
                }
            } else if !allowlist.contains(&s.symbol) {
                continue;
            }
        }

        let vol = *vol_map.get(&s.symbol).unwrap_or(&0.0);
        let price = *price_map.get(&s.symbol).unwrap_or(&0.0);
        if vol < u.min_quote_volume_usdt_24h && !allowlist.contains(&s.symbol) {
            continue;
        }

        // Push tuple with all required info
       
        selected.push((s.symbol, s.base_asset, s.quote_asset, vol, price));
    }

    // Sort by symbol
    selected.sort_by(|a, b| a.0.cmp(&b.0));
    selected.dedup_by(|a, b| a.0 == b.0);

    // 4) Write to DB
    let (mut client_pg, connection) = tokio_postgres::connect(db_url, NoTls)
        .await
        .context("connect postgres")?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("postgres connection error: {e}");
        }
    });

    let tx = client_pg.transaction().await?;

    // Deactivate all
    tx.execute("UPDATE market.pairs SET is_active = false", &[])
        .await
        .context("deactivate pairs")?;

    // Activate selected
    let mut count: u64 = 0;
    for (sym, base, quote, vol, price) in &selected {
        tx.query_one(
            r#"
            INSERT INTO market.pairs(symbol, base_asset, quote_asset, volume_24h_usdt, last_price, is_active)
            VALUES ($1, $2, $3, $4, $5, true)
            ON CONFLICT (symbol) DO UPDATE
              SET is_active = EXCLUDED.is_active,
                  base_asset = EXCLUDED.base_asset,
                  quote_asset = EXCLUDED.quote_asset,
                  volume_24h_usdt = EXCLUDED.volume_24h_usdt,
                  last_price = EXCLUDED.last_price
            RETURNING symbol_id
            "#,
            &[sym, base, quote, vol, price], // Передаем 5 параметров
        )
        .await
        .with_context(|| format!("upsert pair {sym}"))?;
        count += 1;
    }

    tx.commit().await?;

    Ok(count)
}




