use anyhow::Result;
use common::timeframe::Timeframe;
use connections::BinanceRestClient;
use rdkafka::producer::FutureProducer;

use crate::candle_builder::{fetch_klines, tf_ms};
use crate::config::IngestConfig;
use crate::producer::send_close_mp;
use data_writer::messages::CandleCloseMsg;

/// Заполняем пропуски между prev_close_ms и new_close_ms.
/// Возвращает последний опубликованный close_time_ms (либо prev_close_ms, если ничего не публиковали).
pub async fn gap_fill_between_mp(
    cfg: &IngestConfig,
    rest: &BinanceRestClient,
    producer: &FutureProducer,
    symbol: &str,
    tf: Timeframe,
    prev_close_ms: i64,
    new_close_ms: i64,
    source_event_time_ms: i64,
) -> Result<i64> {
    let step = tf_ms(tf);

    // стартуем с ближайшего следующего бара после prev_close
    let mut cursor_open = prev_close_ms + 1;
    let mut last_published = prev_close_ms;

    loop {
        // тянем пачку, но не слишком жирно (чтобы не копить бан)
        let ks = fetch_klines(rest, symbol, tf, 500, Some(cursor_open)).await?;
        if ks.is_empty() {
            break;
        }

        let mut advanced = false;

        for k in ks {
            // Binance иногда возвращает лишнее — фильтруем
            if k.close_time_ms <= prev_close_ms {
                continue;
            }
            if k.close_time_ms >= new_close_ms {
                return Ok(last_published);
            }

            let evt = CandleCloseMsg {
                symbol: symbol.to_string(),
                tf: tf.as_binance_interval().to_string(),
                close_time_ms: k.close_time_ms,
                open: k.open,
                high: k.high,
                low: k.low,
                close: k.close,
                volume: k.volume,
                source_event_time_ms: Some(source_event_time_ms),
            };

            let key = format!("{}|{}", symbol, evt.tf);
            send_close_mp(producer, &cfg.topic_candles_close, &key, &evt).await?;

            last_published = k.close_time_ms;
            cursor_open = k.close_time_ms + 1;
            advanced = true;

            if last_published + step >= new_close_ms {
                return Ok(last_published);
            }
        }

        if !advanced {
            break;
        }
    }

    Ok(last_published)
}
