use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleCloseMsg {
    pub symbol: String,        // "BTCUSDT"
    pub tf: String,            // "1m" etc
    pub close_time_ms: i64,    // epoch millis (close time)
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub source_event_time_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorsSnapshotMsg {
    pub symbol: String,
    pub tf: String,
    pub close_time_ms: i64,

    // core subset (match your SQL columns)
    pub ema20: Option<f32>,
    pub ema50: Option<f32>,
    pub ema200: Option<f32>,
    pub sma: Option<f32>,

    pub rsi: Option<f32>,
    pub macd: Option<f32>,
    pub macd_signal: Option<f32>,
    pub macd_hist: Option<f32>,

    pub atr: Option<f32>,
    pub adx: Option<f32>,

    pub bb_upper: Option<f32>,
    pub bb_mid: Option<f32>,
    pub bb_lower: Option<f32>,

    pub stoch_k: Option<f32>,
    pub stoch_d: Option<f32>,

    pub vwap: Option<f32>,
    pub obv: Option<f32>,
    pub cci: Option<f32>,
    pub williams: Option<f32>,

    pub alli_jaw: Option<f32>,
    pub alli_teeth: Option<f32>,
    pub alli_lips: Option<f32>,

    pub sr_levels: Option<serde_json::Value>, // jsonb

    pub features_version: Option<String>, // default v1
}
