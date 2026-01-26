use anyhow::{Context, Result};
use common::timeframe::Timeframe;
use connections::BinanceRestClient;
use serde_json::Value;

/// Нормализованная свеча (kline) из Binance.
/// Времена в миллисекундах Unix epoch.
#[derive(Debug, Clone)]
pub struct KlineRow {
    pub open_time_ms: i64,
    pub close_time_ms: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[inline]
fn parse_f64(v: &Value) -> Option<f64> {
    match v {
        Value::String(s) => s.parse::<f64>().ok(),
        Value::Number(n) => n.as_f64(),
        _ => None,
    }
}

#[inline]
fn parse_i64(v: &Value) -> Option<i64> {
    match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.parse::<i64>().ok(),
        _ => None,
    }
}

/// Длительность таймфрейма в миллисекундах.
#[inline]
pub fn tf_ms(tf: Timeframe) -> i64 {
    tf.duration_secs() as i64 * 1000
}

/// Загружает klines через connections::BinanceRestClient и приводит к KlineRow.
///
/// Важно:
/// - Binance возвращает массив массивов:
///   [ openTime, open, high, low, close, volume, closeTime, ...]
/// - Мы НЕ “лечим” битые строки нулями — мы их пропускаем.
///   Для ingestion это правильнее: иначе потом ловишь тихие corrupted свечи.
///
/// start_time_ms — это startTime для запроса Binance (в ms).
pub async fn fetch_klines(
    rest: &BinanceRestClient,
    symbol: &str,
    tf: Timeframe,
    limit: usize,
    start_time_ms: Option<i64>,
) -> Result<Vec<KlineRow>> {
    // connections::BinanceRestClient возвращает сырые Vec<Vec<Value>>
    let raw: Vec<Vec<Value>> = rest
        .futures_klines(
            symbol,
            tf.as_binance_interval(),
            limit as u32,
            start_time_ms,
            None,
        )
        .await
        .with_context(|| {
            format!(
                "futures_klines {} {} limit={} start={:?}",
                symbol,
                tf.as_binance_interval(),
                limit,
                start_time_ms
            )
        })?;

    let mut out = Vec::with_capacity(raw.len());

    for row in raw {
        // Binance: [ openTime, open, high, low, close, volume, closeTime, ... ]
        if row.len() < 7 {
            continue;
        }

        let open_time_ms = match parse_i64(&row[0]) {
            Some(v) if v > 0 => v,
            _ => continue,
        };

        let close_time_ms = match parse_i64(&row[6]) {
            Some(v) if v > 0 => v,
            _ => continue,
        };

        // Базовая sanity-check логика
        if close_time_ms <= open_time_ms {
            continue;
        }

        let open = match parse_f64(&row[1]) {
            Some(v) => v,
            None => continue,
        };
        let high = match parse_f64(&row[2]) {
            Some(v) => v,
            None => continue,
        };
        let low = match parse_f64(&row[3]) {
            Some(v) => v,
            None => continue,
        };
        let close = match parse_f64(&row[4]) {
            Some(v) => v,
            None => continue,
        };
        let volume = match parse_f64(&row[5]) {
            Some(v) => v,
            None => continue,
        };

        out.push(KlineRow {
            open_time_ms,
            close_time_ms,
            open,
            high,
            low,
            close,
            volume,
        });
    }

    Ok(out)
}
