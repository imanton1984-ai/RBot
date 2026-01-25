pub struct Candle {
    pub time: chrono::DateTime<chrono::Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

pub struct SymbolId(pub String);

pub struct Timeframe(pub u32);
