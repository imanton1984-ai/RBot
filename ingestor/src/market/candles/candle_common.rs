/*
 * FILE: candle_common.rs
 *
 * PURPOSE:
 * This module contains shared data structures, types, and utility functions
 * used across different candle ingestion components (REST, WebSocket, Writer).
 *
 * RESPONSIBILITIES:
 * - Defines core data structures like CandleRow, PairInfo, WriterMsg
 * - Provides common utility functions like prepare_copy_batch, str_f64
 * - Implements shared logic like WeightLimiter for rate limiting
 * - Contains helper functions for building table names and URLs
 *
 * WORKFLOW:
 * 1. Data structures are defined and used by other modules
 * 2. Utility functions are called during data processing
 * 3. Rate limiting logic is applied to API requests
 */
use common::timeframe::TimeFrame;
use serde::Deserialize;
use std::borrow::Cow;
use std::collections::HashMap;
use tokio::sync::Semaphore;

#[derive(Debug, Clone)]
pub struct PairInfo {
    pub symbol_id: i64,
    pub symbol: String, // "BTCUSDT"
}

#[derive(Debug, Clone)]
pub struct CandleRow {
    pub time_ms: i64, // close time ms
    pub symbol_id: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct WriterMsg {
    pub rows: Vec<CandleRow>,
}

// ---------- REST (zero-copy-ish) ----------
#[allow(dead_code)]
#[derive(Deserialize)]
pub struct RestKline<'a>(
    pub i64,                       // open_time
    #[serde(borrow)] pub Cow<'a, str>, // open
    #[serde(borrow)] pub Cow<'a, str>, // high
    #[serde(borrow)] pub Cow<'a, str>, // low
    #[serde(borrow)] pub Cow<'a, str>, // close
    #[serde(borrow)] pub Cow<'a, str>, // volume
    pub i64,                       // close_time
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
    serde_json::Value,
);

// ---------- WS (zero-copy-ish) ----------
#[derive(Deserialize)]
pub struct WsEnvelope<'a> {
    #[serde(borrow)]
    pub data: WsData<'a>,
}

#[derive(Deserialize)]
pub struct WsData<'a> {
    #[serde(rename = "k", borrow)]
    pub k: WsKline<'a>,
}

#[derive(Deserialize)]
pub struct WsKline<'a> {
    #[serde(rename = "s", borrow)]
    pub symbol: Cow<'a, str>, // "BTCUSDT"
    #[serde(rename = "i", borrow)]
    pub interval: Cow<'a, str>, // "1m"
    #[serde(rename = "T")]
    pub close_time: i64,
    #[serde(rename = "o", borrow)]
    pub open: Cow<'a, str>,
    #[serde(rename = "h", borrow)]
    pub high: Cow<'a, str>,
    #[serde(rename = "l", borrow)]
    pub low: Cow<'a, str>,
    #[serde(rename = "c", borrow)]
    pub close: Cow<'a, str>,
    #[serde(rename = "v", borrow)]
    pub volume: Cow<'a, str>,
    #[serde(rename = "x")]
    pub is_closed: bool,
}

#[derive(Clone)]
pub struct WeightLimiter {
    pub sem: std::sync::Arc<Semaphore>,
}

impl WeightLimiter {
    pub fn new(weight_per_sec: u32, burst: u32) -> Self {
        let wps = weight_per_sec.max(1);
        let cap = burst.max(wps).max(1) as usize;
        let sem = std::sync::Arc::new(Semaphore::new(cap));
        let sem2 = sem.clone();

        // Размазываем пополнение равномерно: 1 permit каждые (1/wps) секунды
        let period_ns = (1_000_000_000u64 / wps as u64).max(1);
        let period = tokio::time::Duration::from_nanos(period_ns);

        // Spawn a background task that continuously adds permits to the semaphore
        // at a rate of 1 permit per `period` nanoseconds, up to the capacity
        tokio::spawn(async move {
            let mut itv = tokio::time::interval(period);
            itv.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                itv.tick().await;
                if sem2.available_permits() < cap {
                    sem2.add_permits(1);
                }
            }
        });

        Self { sem }
    }

    pub async fn acquire(&self, weight: u32) {
        if weight == 0 {
            return;
        }
        // weight для klines максимум 10, так что acquire_many ок.
        match self.sem.clone().acquire_many_owned(weight).await {
            Ok(p) => p.forget(),
            Err(_) => {} // shutdown
        }
    }
}

pub fn klines_weight(limit: usize) -> u32 {
    match limit {
        0..=100 => 1,
        101..=500 => 2,
        501..=1000 => 5,
        _ => 10,
    }
}

pub fn parse_tf(s: &str) -> Option<TimeFrame> {
    match s {
        "1m" => Some(TimeFrame::M1),
        "5m" => Some(TimeFrame::M5),
        "15m" => Some(TimeFrame::M15),
        "1h" => Some(TimeFrame::H1),
        "4h" => Some(TimeFrame::H4),
        "1d" => Some(TimeFrame::D1),
        _ => None,
    }
}

pub fn build_table_name(tf: TimeFrame) -> String {
    format!("market.candles_{}", tf.as_str())
}

pub fn build_ws_base(ws_base: &str) -> String {
    let b = ws_base.trim_end_matches('/');
    if b.ends_with("/stream") {
        b.to_string()
    } else {
        format!("{}/stream", b)
    }
}

pub fn build_combined_ws_url(ws_base: &str, streams: &[String]) -> String {
    let base = build_ws_base(ws_base);
    let joined = streams.join("/");
    format!("{}?streams={}", base, joined)
}

pub fn str_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

pub fn prepare_copy_batch(buf: &mut Vec<CandleRow>, last_seen: &HashMap<i64, i64>) {
    buf.retain(|c| c.time_ms > *last_seen.get(&c.symbol_id).unwrap_or(&0));

    if buf.len() <= 1 {
        return;
    }

    buf.sort_unstable_by(|a, b| {
        match a.symbol_id.cmp(&b.symbol_id) {
            std::cmp::Ordering::Equal => a.time_ms.cmp(&b.time_ms),
            other => other,
        }
    });

    buf.dedup_by(|a, b| a.symbol_id == b.symbol_id && a.time_ms == b.time_ms);
}