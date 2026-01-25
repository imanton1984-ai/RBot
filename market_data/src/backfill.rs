use anyhow::{Context, Result};
use futures::{stream::FuturesUnordered, StreamExt};
use rdkafka::producer::FutureProducer;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_postgres::NoTls;
use tracing::{info, warn};

use connections::BinanceRestClient;
use data_writer::messages::CandleCloseMsg;

use crate::candle_builder::fetch_klines;
use crate::config::IngestConfig;
use crate::health::now_ms;
use crate::gap_fill::send_close_mp;

pub type LastCloseMap = HashMap<(String, String), i64>;

async fn load_active_symbols(db_url: &str) -> Result<Vec<String>> {
    let (client, conn) = tokio_postgres::connect(db_url, NoTls).await?;
    tokio::spawn(async move { let _ = conn.await; });

    let rows = client
        .query("SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol", &[])
        .await
        .context("query market.pairs failed")?;

    Ok(rows.into_iter().map(|r| r.get::<_, String>(0)).collect())
}

pub async fn run_backfill(
    cfg: Arc<IngestConfig>,
    rest: BinanceRestClient,
    producer: FutureProducer,
) -> Result<(usize, LastCloseMap)> {
    info!("backfill: loading active symbols...");
    let symbols = load_active_symbols(&cfg.db_url).await?;
    info!("backfill: {} symbols", symbols.len());

    let sem = Arc::new(tokio::sync::Semaphore::new(cfg.http_concurrency.max(1)));
    let mut last_map: LastCloseMap = HashMap::new();

    let mut published = 0usize;

    for &tf in cfg.timeframes.iter() {
        info!("backfill: tf={} candles={}", tf.as_binance_interval(), cfg.backfill_candles);

        let mut futs = FuturesUnordered::new();

        for sym in symbols.iter().cloned() {
            let cfg = cfg.clone();
            let rest = rest.clone();
            let producer = producer.clone();
            let sem = sem.clone();

            futs.push(async move {
                let _permit = sem.acquire_owned().await.context("semaphore closed")?;

                // последние N (достаточно для warmup)
                let ks = fetch_klines(&rest, &sym, tf, cfg.backfill_candles, None).await?;
                if ks.is_empty() {
                    return Ok::<(usize, Option<((String, String), i64)>), anyhow::Error>((0, None));
                }

                let mut cnt = 0usize;

                for k in ks.iter() {
                    let evt = CandleCloseMsg {
                        symbol: sym.clone(),
                        tf: tf.as_binance_interval().to_string(),
                        close_time_ms: k.close_time_ms,
                        open: k.open,
                        high: k.high,
                        low: k.low,
                        close: k.close,
                        volume: k.volume,
                        source_event_time_ms: Some(now_ms()),
                    };
                    let key = format!("{}|{}", evt.symbol, evt.tf);
                    send_close_mp(&producer, &cfg.topic_candles_close, &key, &evt).await?;
                    cnt += 1;
                }

                let last = ks.last().unwrap().close_time_ms;
                Ok((cnt, Some(((sym, tf.as_binance_interval().to_string()), last))))
            });
        }

        while let Some(res) = futs.next().await {
            match res {
                Ok((cnt, Some((k, last)))) => {
                    published += cnt;
                    last_map.insert(k, last);
                }
                Ok((_cnt, None)) => {}
                Err(e) => warn!("backfill task error: {e:#}"),
            }
        }
    }

    info!("backfill: done. published={} last_map={}", published, last_map.len());
    Ok((published, last_map))
}

