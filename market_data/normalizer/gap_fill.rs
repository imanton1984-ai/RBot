use anyhow::{Context, Result};
use common::timeframe::Timeframe;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;
use std::time::Duration;

use crate::candle_builder::{fetch_klines, tf_ms, KlineRow};
use crate::config::IngestConfig;
use db_writer::messages::CandleCloseMsg;

/// Публикуем закрытую свечу в Redpanda/Kafka в **MessagePack**.
/// Ключ: "{symbol}|{tf}" — чтобы db_writer мог легко шардинговать/упорядочивать.
async fn send_close_mp(
    producer: &FutureProducer,
    topic: &str,
    key: &str,
    msg: &CandleCloseMsg,
) -> Result<()> {
    let payload = rmp_serde::to_vec_named(msg).context("rmp_serde::to_vec_named failed")?;

    loop {
        match producer
            .send(
                FutureRecord::to(topic).key(key).payload(&payload),
                Timeout::Never,
            )
            .await
        {
            Ok(_) => return Ok(()),
            Err((e, _)) => {
                // мягкий backpressure — Kafka очередь забилась
                if e.to_string().contains("Queue full") {
                    tokio::time::sleep(Duration::from_millis(5)).await;
                    continue;
                }
                return Err(e.into());
            }
        }
    }
}

fn make_close(symbol: &str, tf: Timeframe, k: &KlineRow, src_ms: i64) -> CandleCloseMsg {
    CandleCloseMsg {
        symbol: symbol.to_string(),
        tf: tf.as_binance_interval().to_string(),
        close_time_ms: k.close_time_ms,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
        volume: k.volume,
        source_event_time_ms: Some(src_ms),
    }
}

/// Gap-fill: дозаливает пропущенные закрытия свечей между `last_close_ms` и `new_close_ms`,
/// НЕ включая `new_close_ms` (чтобы не задвоить текущий бар, который уже пришёл из WS).
///
/// Возвращает `last_published_close_ms` (последний закрытый бар, который удалось опубликовать).
pub async fn gap_fill_between(
    cfg: &IngestConfig,
    http: &reqwest::Client,
    producer: &FutureProducer,
    symbol: &str,
    tf: Timeframe,
    last_close_ms: i64,
    new_close_ms: i64,
    source_event_time_ms: i64,
) -> Result<i64> {
    let step = tf_ms(tf);

    // нет зазора → нечего заполнять
    if new_close_ms <= last_close_ms + step {
        return Ok(last_close_ms);
    }

    // Binance klines startTime = openTime.
    // Следующий бар начинается сразу после предыдущего closeTime: open = last_close + 1
    let mut cursor_open = last_close_ms + 1;
    let mut last_published = last_close_ms;

    // защитный лимит, чтобы теоретически не зациклиться на странном ответе API
    let mut guard_iters: u32 = 0;

    while cursor_open + step < new_close_ms {
        guard_iters += 1;
        if guard_iters > 10_000 {
            // это уже “невозможно” при нормальных данных; лучше выйти, чем повиснуть
            break;
        }

        // Берём пачку (до 1000) начиная с cursor_open
        let klines = fetch_klines(
            http,
            &cfg.rest_base_url,
            symbol,
            tf,
            1000,
            Some(cursor_open),
        )
        .await
        .with_context(|| format!("fetch_klines gapfill {} {}", symbol, tf.as_binance_interval()))?;

        if klines.is_empty() {
            break;
        }

        let mut advanced = false;

        for k in klines {
            // не лезем на текущий бар из WS
            if k.close_time_ms >= new_close_ms {
                break;
            }

            // строго после последнего опубликованного
            if k.close_time_ms <= last_published {
                continue;
            }

            let evt = make_close(symbol, tf, &k, source_event_time_ms);
            let key = format!("{}|{}", symbol, evt.tf);
            send_close_mp(producer, &cfg.topic_candles_close, &key, &evt).await?;

            last_published = k.close_time_ms;
            cursor_open = k.close_time_ms + 1;
            advanced = true;
        }

        // если API вернуло что-то, но мы не продвинулись — выходим, чтобы не зациклиться
        if !advanced {
            break;
        }

        // уже дошли до нужной границы
        if last_published + step >= new_close_ms {
            break;
        }
    }

    Ok(last_published)
}


