use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::timeframe::TimeFrame;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub time_ms: i64,           // Close time in milliseconds (for database)
    pub symbol_id: i64,         // Pair ID from database
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandleWithTime {
    pub timestamp: DateTime<Utc>,  // Open time (derived from time_ms)
    pub symbol_id: i64,
    pub timeframe: TimeFrame,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawCandleData {
    pub open_time: i64,
    pub open: String,
    pub high: String,
    pub low: String,
    pub close: String,
    pub volume: String,
    pub close_time: i64,
    pub quote_asset_volume: String,
    pub number_of_trades: i64,
    pub taker_buy_base_asset_volume: String,
    pub taker_buy_quote_asset_volume: String,
}

impl From<RawCandleData> for Candle {
    fn from(raw: RawCandleData) -> Self {
        Candle {
            time_ms: raw.close_time,
            symbol_id: 0, // Will be set later when we know the symbol_id
            open: raw.open.parse().unwrap_or(0.0),
            high: raw.high.parse().unwrap_or(0.0),
            low: raw.low.parse().unwrap_or(0.0),
            close: raw.close.parse().unwrap_or(0.0),
            volume: raw.volume.parse().unwrap_or(0.0),
        }
    }
}

impl Candle {
    pub fn get_table_name(&self, timeframe: &TimeFrame) -> String {
        format!("market.candles_{}", timeframe.as_str().replace("m", "").replace("h", "").replace("d", "").replace("w", ""))
    }

    pub fn get_table_name_by_tf(timeframe: &TimeFrame) -> String {
        format!("market.candles_{}", timeframe.as_str().replace("m", "").replace("h", "").replace("d", "").replace("w", ""))
    }
}
