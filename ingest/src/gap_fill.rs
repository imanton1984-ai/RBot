use anyhow::Result;
use common::timeframe::Timeframe;
use rdkafka::producer::{FutureProducer, FutureRecord};
use rdkafka::util::Timeout;

use crate::candle_builder::{fetch_klines, tf_ms};
use crate::config::IngestConfig;
use db_writer::messages::CandleCloseMsg;

async fn send_close(
    producer: &FutureProducer,
    topic: &str,
    key: &str,
    msg: &CandleCloseMsg,
) -> Result<()> {
    let payload = serde_json::to_vec(msg)?;
    loop {
        match producer.send(
            FutureRecord::to(topic).key(key).payload(&payload),
            Timeout::Never,
        ) {
            Ok(_) => return Ok(()),
            Err((e, _)) => {
                if e.to_string().contains("Queue full") {
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    continue;
                }
                return Err(e.into());
            }
        }
    }
}

fn make_close(symbol: &str, tf: Timeframe, k: &crate::candle_builder::KlineRow, src_ms: i64) -> CandleCloseMsg {
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
    if new_close_ms <= last_close_ms + step {
        return Ok(last_close_ms);
    }

    // startTime по Binance — это openTime, но мы двигаемся от next bar open:
    let mut cursor_open = last_close_ms + 1;
    let mut last_published = last_close_ms;

    while cursor_open + step < new_close_ms {
        // берём кусок, чтобы не выйти за лимиты (1000)
        let klines = fetch_klines(http, &cfg.rest_base_url, symbol, tf, 1000, Some(cursor_open)).await?;

        if klines.is_empty() {
            break;
        }

        for k in klines {
            // публикуем только то, что строго “до” new_close (чтобы не задвоить текущий бар из WS)
            if k.close_time_ms >= new_close_ms {
                break;
            }
            let evt = make_close(symbol, tf, &k, source_event_time_ms);
            let key = format!("{}|{}", symbol, evt.tf);
            send_close(producer, &cfg.topic_candles_close, &key, &evt).await?;
            last_published = k.close_time_ms;
            cursor_open = k.close_time_ms + 1;
        }

        if last_published + step >= new_close_ms {
            break;
        }
    }

    Ok(last_published)
}
