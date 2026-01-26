use anyhow::{Context, Result};
use connections::BinanceRestClient;
use serde_json::Value;
use tokio_postgres::{NoTls, Transaction};

use common::config::load_config;

use std::collections::{HashMap, HashSet};

fn v_f64(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse::<f64>().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Возвращает количество активированных пар.
pub async fn refresh_universe_once(db_url: &str, rest: &BinanceRestClient) -> Result<usize> {
    // 1) грузим конфиг (universe фильтры)
    let cfg = load_config().context("load_config failed")?;
    let u = cfg.universe.clone();

    let allow: HashSet<String> = u.allowlist.iter().cloned().collect();
    let deny: HashSet<String> = u.denylist.iter().cloned().collect();
    let excl_sym: HashSet<String> = u.exclude_symbols.iter().cloned().collect();
    let excl_base: HashSet<String> = u.exclude_base_assets.iter().cloned().collect();

    // 2) exchangeInfo (futures)
    let info: Value = rest
        .futures_exchange_info()
        .await
        .context("futures_exchange_info failed")?;
    let symbols = info["symbols"].as_array().cloned().unwrap_or_default();

    // 3) ticker24h
    let tick_arr: Vec<Value> = rest
        .futures_ticker_24h()
        .await
        .context("futures_ticker_24h failed")?;
    let tickers = serde_json::Value::Array(tick_arr);
    let tick_arr = tickers.as_array().cloned().unwrap_or_default();

    // map symbol->(quoteVolume,lastPrice)
    let mut tv: HashMap<String, (f64, f64)> = HashMap::new();
    for t in tick_arr {
        let sym = t["symbol"].as_str().unwrap_or("").to_string();
        if sym.is_empty() {
            continue;
        }
        let qv = v_f64(&t["quoteVolume"]);
        let lp = v_f64(&t["lastPrice"]);
        tv.insert(sym, (qv, lp));
    }

    // 4) отбор
    let mut picked: Vec<(String, String, String, f64, f64)> = Vec::new(); // (symbol, base, quote, vol, price)

    for s in symbols {
        let symbol = s["symbol"].as_str().unwrap_or("").to_string();
        if symbol.is_empty() {
            continue;
        }

        // deny — всегда выбрасываем
        if deny.contains(&symbol) {
            continue;
        }
        // exclude_symbols — тоже всегда выбрасываем
        if excl_sym.contains(&symbol) {
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

        // exclude base assets
        if excl_base.contains(&base) {
            continue;
        }

        let (vol, price) = tv.get(&symbol).copied().unwrap_or((0.0, 0.0));

        // allowlist = форс включение (даже если volume ниже)
        let forced = allow.contains(&symbol);

        if !forced && vol < u.min_quote_volume_usdt_24h {
            continue;
        }

        picked.push((symbol, base, quote, vol, price));
    }

    // 5) пишем в DB (deactivate all, затем upsert выбранных)
    let (mut client, conn) = tokio_postgres::connect(db_url, NoTls)
        .await
        .context("connect db failed")?;
    tokio::spawn(async move {
        let _ = conn.await;
    });

    // churn-guard: сравнить с текущим количеством активных
    let prev_active: i64 = client
        .query_one("SELECT COUNT(*) FROM market.pairs WHERE is_active=true", &[])
        .await
        .map(|r| r.get::<_, i64>(0))
        .unwrap_or(0);

    let new_active: i64 = picked.len() as i64;
    if prev_active > 0 && u.max_change_ratio > 0.0 {
        let diff = (new_active - prev_active).abs() as f64;
        let ratio = diff / (prev_active as f64);

        // background: если слишком резкое изменение — не применяем
        if ratio > u.max_change_ratio && u.apply_mode == "background" {
            tracing::warn!(
                "universe churn-guard: prev_active={}, new_active={}, ratio={:.3} > max_change_ratio={:.3}; SKIP apply (background mode)",
                prev_active,
                new_active,
                ratio,
                u.max_change_ratio
            );
            return Ok(prev_active as usize);
        }

        if ratio > u.max_change_ratio {
            tracing::warn!(
                "universe churn-guard: prev_active={}, new_active={}, ratio={:.3} > max_change_ratio={:.3}; APPLY anyway (apply_mode={})",
                prev_active,
                new_active,
                ratio,
                u.max_change_ratio,
                u.apply_mode
            );
        }
    }

    let tx = client.transaction().await.context("tx begin failed")?;
    apply_pairs(&tx, &picked, u.perpetual_only).await?;
    tx.commit().await.context("tx commit failed")?;

    Ok(picked.len())
}

async fn apply_pairs(
    tx: &Transaction<'_>,
    picked: &[(String, String, String, f64, f64)],
    perpetual_only: bool,
) -> Result<()> {
    // выключаем все, но manual_allow не трогаем
    tx.execute(
        "UPDATE market.pairs SET is_active=false WHERE manual_allow=false",
        &[],
    )
    .await
    .context("deactivate pairs failed")?;

    let stmt = tx
        .prepare(
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
               last_refreshed_at=now()",
        )
        .await
        .context("prepare upsert failed")?;

    for (sym, base, quote, vol, price) in picked.iter() {
        tx.execute(&stmt, &[sym, base, quote, &perpetual_only, vol, price])
            .await
            .with_context(|| format!("upsert failed for {sym}"))?;
    }

    Ok(())
}

/// Получает список активных торговых пар из БД.
pub async fn fetch_active_symbols(db_url: &str) -> Result<Vec<String>> {
    let (client, connection) = tokio_postgres::connect(db_url, NoTls)
        .await
        .context("connect db failed")?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            tracing::error!("DB connection error: {}", e);
        }
    });

    let rows = client
        .query(
            "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol",
            &[],
        )
        .await
        .context("query active symbols failed")?;

    Ok(rows.into_iter().map(|r| r.get::<_, String>(0)).collect())
}

