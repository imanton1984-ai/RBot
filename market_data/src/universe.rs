use anyhow::{Context, Result};
use connections::BinanceRestClient;
use serde_json::Value;
use tokio_postgres::NoTls;

use common::config::load_config;

/// Возвращает количество активированных пар.
pub async fn refresh_universe_once(db_url: &str, rest: &BinanceRestClient) -> Result<usize> {
    // 1) грузим конфиг (universe фильтры)
    let cfg = load_config().context("load_config failed")?;
    let u = cfg.universe.clone();

    // 2) exchangeInfo (futures)
    let info: Value = rest
        .futures_exchange_info()
        .await
        .context("futures_exchange_info failed")?;
    let symbols = info["symbols"].as_array().cloned().unwrap_or_default();

    // 3) ticker24h
    let tick_arr: Vec<Value> = rest.futures_ticker_24h().await?;
    let tickers = serde_json::Value::Array(tick_arr);
    let tick_arr = tickers.as_array().cloned().unwrap_or_default();

    // map symbol->(quoteVolume,lastPrice)
    use std::collections::HashMap;
    let mut tv: HashMap<String, (f64, f64)> = HashMap::new();
    for t in tick_arr {
        let sym = t["symbol"].as_str().unwrap_or("").to_string();
        if sym.is_empty() {
            continue;
        }
        let qv = t["quoteVolume"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        let lp = t["lastPrice"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        tv.insert(sym, (qv, lp));
    }

    // 4) отбор
    let mut picked: Vec<(String, String, String, f64, f64)> = Vec::new(); // (symbol, base, quote, vol, price)

    for s in symbols {
        let symbol = s["symbol"].as_str().unwrap_or("").to_string();
        if symbol.is_empty() {
            continue;
        }

        let quote = s["quoteAsset"].as_str().unwrap_or("").to_string();
        let base = s["baseAsset"].as_str().unwrap_or("").to_string();

        // только нужный quote
        if quote != u.quote_asset {
            continue;
        }

        // только perpetual (если надо)
        if u.perpetual_only {
            let ct = s["contractType"].as_str().unwrap_or("");
            if ct != "PERPETUAL" {
                continue;
            }
        }

        // статус
        let status = s["status"].as_str().unwrap_or("");
        if status != "TRADING" {
            continue;
        }

        // deny/exclude
        if u.exclude_symbols.iter().any(|x| x == &symbol) {
            continue;
        }
        if u.exclude_base_assets.iter().any(|x| x == &base) {
            continue;
        }
        if !u.denylist.is_empty() && u.denylist.iter().any(|x| x == &symbol) {
            continue;
        }
        if !u.allowlist.is_empty() && !u.allowlist.iter().any(|x| x == &symbol) {
            continue;
        }

        let (vol, price) = tv.get(&symbol).copied().unwrap_or((0.0, 0.0));
        if vol < u.min_quote_volume_usdt_24h {
            continue;
        }

        picked.push((symbol, base, quote, vol, price));
    }

    // 5) пишем в DB (deactivate all, затем upsert выбранных)
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // выключаем все, но manual_allow не трогаем
    client
        .execute(
            "UPDATE market.pairs SET is_active=false WHERE manual_allow=false",
            &[],
        )
        .await?;

    let stmt = client.prepare(
        "INSERT INTO market.pairs
         (symbol, base_asset, quote_asset, is_active, futures_eligible, perpetual_only, volume_24h_usdt, last_price, last_refreshed_at)
         VALUES ($1,$2,$3,true,true,$4,$5,$6,now())
         ON CONFLICT(symbol) DO UPDATE SET
           base_asset=EXCLUDED.base_asset,
           quote_asset=EXCLUDED.quote_asset,
           is_active=true,
           futures_eligible=true,
           perpetual_only=EXCLUDED.perpetual_only,
           volume_24h_usdt=EXCLUDED.volume_24h_usdt,
           last_price=EXCLUDED.last_price,
           last_refreshed_at=now()"
    ).await?;

    for (sym, base, quote, vol, price) in picked.iter() {
        client
            .execute(&stmt, &[sym, base, quote, &u.perpetual_only, vol, price])
            .await?;
    }

    Ok(picked.len())
}
