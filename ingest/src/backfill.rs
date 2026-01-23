use anyhow::{Context, Result};
use common::timeframe::Timeframe;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::collections::HashMap;
use std::sync::Arc;
use futures::{StreamExt, stream}; // Важно: нужен futures
use rmp_serde;

use crate::candle_builder::{fetch_klines, now_ms};
use crate::config::IngestConfig;
use crate::producer::build_producer;
use db_writer::messages::CandleCloseMsg;

pub type LastCloseMap = HashMap<(String, String), i64>;

// load_active_symbols: теперь возвращает символы с их последней известной свечей
async fn load_active_symbols_with_latest_candle(db_url: &str, tf: Timeframe) -> Result<HashMap<String, i64>> {
    let (client, connection) = tokio_postgres::connect(db_url, tokio_postgres::NoTls).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("postgres connection error: {e:?}");
        }
    });
    
    let table_name = tf.candles_table();
    let query = format!(
        "SELECT p.symbol, MAX(c.time_ms) as latest_time
         FROM market.pairs p
         LEFT JOIN {} c ON p.symbol_id = c.symbol_id
         WHERE p.is_active = true
         GROUP BY p.symbol_id, p.symbol",
        table_name
    );
    
    let rows = client.query(&query, &[]).await?;
    let mut result = HashMap::new();
    
    for row in rows {
        let symbol: String = row.get(0);
        let latest_time: Option<i64> = row.get(1);
        result.insert(symbol, latest_time.unwrap_or(0)); // если нет данных, начинаем с 0
    }
    
    Ok(result)
}

// make_close оставляем без изменений...
fn make_close(symbol: &str, tf: Timeframe, k: &crate::candle_builder::KlineRow) -> CandleCloseMsg {
    CandleCloseMsg {
        symbol: symbol.to_string(),
        tf: tf.as_binance_interval().to_string(),
        close_time_ms: k.close_time_ms,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
        volume: k.volume,
        source_event_time_ms: Some(now_ms()),
    }
}

// send_close оставляем без изменений...
async fn send_close(
    producer: &FutureProducer,
    topic: &str,
    key: &str,
    msg: &CandleCloseMsg,
) -> Result<()> {
    let payload = rmp_serde::to_vec_named(msg)?;
    loop {
        match producer.send(
            FutureRecord::to(topic).key(key).payload(&payload),
            Timeout::Never,
        ) {
            Ok(_delivery) => return Ok(()),
            Err((e, _)) => {
                if e.to_string().contains("Queue full") {
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    continue;
                }
                return Err(e.into());
            }
        }
    }
}

pub async fn run_backfill(cfg: Arc<IngestConfig>) -> Result<(Vec<String>, LastCloseMap)> {
    let symbols = load_active_symbols(&cfg.db_url)
        .await
        .context("load_active_symbols failed")?;
    
    let producer = build_producer(&cfg)?;
    let http = reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_max_idle_per_host(32) // Увеличиваем пул соединений
        .timeout(std::time::Duration::from_millis(15_000))
        .build()?;

    let mut last_map: LastCloseMap = HashMap::new();
    
    // Генерируем список задач (tuple)
    let mut tasks = Vec::new();
    for sym in &symbols {
        for &tf in &cfg.timeframes {
            tasks.push((sym.clone(), tf));
        }
    }

    println!("Starting backfill for {} tasks with concurrency {}", tasks.len(), cfg.http_concurrency);

    // Используем stream::iter + buffer_unordered для "умного" параллелизма
    let mut stream = stream::iter(tasks)
        .map(|(sym, tf)| {
            let http = http.clone();
            let producer = producer.clone();
            let cfg = cfg.clone();
            
            async move {
                // Внутри async блока - логика для одной пары/тф
                let klines = fetch_klines(
                    &http,
                    &cfg.rest_base_url,
                    &sym,
                    tf,
                    cfg.backfill_candles,
                    None,
                ).await;

                match klines {
                    Ok(klines) => {
                        let mut local_last: Option<i64> = None;
                        for k in klines.iter() {
                            let evt = make_close(&sym, tf, k);
                            let key = format!("{}|{}", sym, evt.tf);
                            if let Err(e) = send_close(&producer, &cfg.topic_candles_close, &key, &evt).await {
                                return Err(anyhow::anyhow!("producer error: {}", e));
                            }
                            local_last = Some(k.close_time_ms);
                        }
                        Ok((sym, tf.as_binance_interval().to_string(), local_last))
                    },
                    Err(e) => {
                        // Логируем ошибку, но не роняем весь процесс, возвращаем None
                        eprintln!("Failed backfill {} {}: {:#}", sym, tf, e);
                        Ok((sym, tf.as_binance_interval().to_string(), None))
                    }
                }
            }
        })
        .buffer_unordered(cfg.http_concurrency); // Здесь вся магия скорости

    // Собираем результаты по мере поступления
    while let Some(result) = stream.next().await {
        match result {
            Ok((sym, tf_str, Some(last))) => {
                last_map.insert((sym, tf_str), last);
            },
            Ok((_, _, None)) => { /* skip failed or empty */ },
            Err(e) => return Err(e),
        }
    }

    println!("Backfill sent to Kafka. Flushing producer...");
    // Ждем, пока Redpanda подтвердит прием
    producer.flush(std::time::Duration::from_secs(60));
    
    Ok((symbols, last_map))
}
