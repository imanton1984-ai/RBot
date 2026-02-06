use crate::{Candle, Symbol, Timeframe};
use anyhow::Result;
use std::{collections::HashMap, collections::VecDeque, hash::Hash};
use tokio::sync::RwLock;

#[derive(Clone, Debug)]
pub struct ProcessorConfig {
    pub max_candles_per_series: usize,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self { max_candles_per_series: 1500 }
    }
}

#[derive(Clone, Debug)]
pub struct CandleBatch {
    pub time_ms: Vec<i64>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct CandleLite {
    time_ms: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Clone, Debug)]
struct SeriesState {
    buf: VecDeque<CandleLite>,
    last_time_ms: Option<i64>,
}

#[derive(Clone, Debug, Eq)]
struct SeriesKey {
    symbol: Symbol,
    timeframe: Timeframe,
}

impl PartialEq for SeriesKey {
    fn eq(&self, other: &Self) -> bool {
        self.symbol == other.symbol && self.timeframe.to_minutes() == other.timeframe.to_minutes()
    }
}

impl Hash for SeriesKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.symbol.hash(state);
        self.timeframe.to_minutes().hash(state);
    }
}

pub struct CandleProcessor {
    config: ProcessorConfig,
    state: RwLock<HashMap<SeriesKey, SeriesState>>,
}

impl CandleProcessor {
    pub fn new(config: ProcessorConfig) -> Self {
        Self {
            config,
            state: RwLock::new(HashMap::new()),
        }
    }

    /// Добавляет свечу в серию (symbol+tf). Дубликаты по time_ms игнорируются.
    pub async fn push(&self, candle: Candle) -> Result<()> {
        let key = SeriesKey { symbol: candle.symbol.clone(), timeframe: candle.timeframe };
        let mut state = self.state.write().await;

        let s = state.entry(key).or_insert_with(|| SeriesState {
            buf: VecDeque::with_capacity(self.config.max_candles_per_series),
            last_time_ms: None,
        });

        if s.last_time_ms == Some(candle.timestamp) {
            return Ok(());
        }

        s.buf.push_back(CandleLite {
            time_ms: candle.timestamp,
            open: candle.open,
            high: candle.high,
            low: candle.low,
            close: candle.close,
            volume: candle.volume,
        });

        s.last_time_ms = Some(candle.timestamp);

        while s.buf.len() > self.config.max_candles_per_series {
            s.buf.pop_front();
        }

        Ok(())
    }

    pub async fn len(&self, symbol: &Symbol, timeframe: Timeframe) -> usize {
        let key = SeriesKey { symbol: symbol.clone(), timeframe };
        let state = self.state.read().await;
        state.get(&key).map(|s| s.buf.len()).unwrap_or(0)
    }

    pub async fn is_ready(&self, symbol: &Symbol, timeframe: Timeframe, min_bars: usize) -> bool {
        self.len(symbol, timeframe).await >= min_bars
    }

    /// Снимок последних N баров в columnar виде.
    pub async fn snapshot_last(&self, symbol: &Symbol, timeframe: Timeframe, last_n: usize) -> Option<CandleBatch> {
        let key = SeriesKey { symbol: symbol.clone(), timeframe };
        let state = self.state.read().await;
        let s = state.get(&key)?;

        let n = last_n.min(s.buf.len());
        let start = s.buf.len().saturating_sub(n);

        let mut time_ms = Vec::with_capacity(n);
        let mut open = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        let mut low = Vec::with_capacity(n);
        let mut close = Vec::with_capacity(n);
        let mut volume = Vec::with_capacity(n);

        for c in s.buf.iter().skip(start) {
            time_ms.push(c.time_ms);
            open.push(c.open);
            high.push(c.high);
            low.push(c.low);
            close.push(c.close);
            volume.push(c.volume);
        }

        Some(CandleBatch { time_ms, open, high, low, close, volume })
    }
}
