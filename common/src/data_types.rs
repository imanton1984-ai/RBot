use serde::{Deserialize, Serialize};
use std::fmt;

// Using the same TimeFrame from the existing module
pub use crate::timeframe::TimeFrame as Timeframe;

// Define Symbol as a simple string wrapper
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl Symbol {
    pub fn new(s: impl Into<String>) -> Self {
        Symbol(s.into())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for Symbol {
    fn from(s: String) -> Self {
        Symbol(s)
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol(s.to_string())
    }
}

// Define Candle structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: i64,
    pub symbol: Symbol,
    pub timeframe: Timeframe,
}

impl Candle {
    pub fn new(
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        timestamp: i64,
        symbol: Symbol,
        timeframe: Timeframe,
    ) -> Self {
        Self {
            open,
            high,
            low,
            close,
            volume,
            timestamp,
            symbol,
            timeframe,
        }
    }
}

/// Kafka event: a candle has closed on a given timeframe
/// Shared across all entry points (main.rs, compute_history, compute_realtime)
#[derive(Debug, Clone, Deserialize)]
pub struct CandleCloseEvent {
    pub symbol: String,
    pub timeframe: String,
    pub close_time: i64,
}