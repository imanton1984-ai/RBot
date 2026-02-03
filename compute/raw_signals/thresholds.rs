use common::{Symbol, Timeframe};
use serde_json::Value as JsonValue;

/// Signal strength thresholds
pub const MIN_INTERESTING_THRESHOLD: f64 = 0.80;  // Raw signals keeps everything above 0.80
pub const MAX_INTERESTING_THRESHOLD: f64 = 1.0;
pub const MIN_USELESS_THRESHOLD: f64 = 0.80;      // Below this gets filtered out

/// Raw signal types
#[derive(Debug, Clone, PartialEq, Copy)]
pub enum RawSignalType {
    Atr = 1,
    Adx = 2,
    Rsi = 3,
    Macd = 4,
    BollingerBands = 5,
    Cci = 6,
    Stochastic = 7,
    Williams = 8,
    VolumeSpike = 9,
    Trend = 10,
    SrLevels = 11,
    Ema = 12,
    Sma = 13,
    Obv = 14,
    Poc = 15,
    Vwap = 16,
}

impl RawSignalType {
    pub fn to_indicator_id(&self) -> i16 {
        *self as i16
    }
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub enum SignalKind {
    PriceRelation = 1,
    Breakout = 2,
    Convergence = 3,
    Volatility = 4,
    Crossover = 5,
}

impl SignalKind {
    pub fn to_i16(&self) -> i16 {
        *self as i16
    }
}

/// Configuration for signal processing
#[derive(Debug, Clone)]
pub struct SignalConfig {
    pub min_interesting_score: f64,
    pub max_interesting_score: f64,
    pub min_useless_score: f64,
    pub enable_filtering: bool,
    pub batch_size: usize,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            min_interesting_score: MIN_INTERESTING_THRESHOLD,  // 0.80 for raw signals
            max_interesting_score: MAX_INTERESTING_THRESHOLD,
            min_useless_score: MIN_USELESS_THRESHOLD,
            enable_filtering: true,
            batch_size: 1000,
        }
    }
}

/// Signal result with normalized score
#[derive(Debug, Clone)]
pub struct RawSignal {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,
    pub indicator_id: i16,
    pub signal_kind: i16,
    pub side: i16,
    pub score: f32,
    pub value: f32,
    pub details: Option<JsonValue>,
}

impl RawSignal {
    pub fn new(
        symbol: Symbol,
        timeframe: Timeframe,
        timestamp: i64,
        indicator_id: i16,
        signal_kind: i16,
        side: i16,
        score: f32,
        value: f32,
        details: Option<JsonValue>,
    ) -> Self {
        Self {
            symbol,
            timeframe,
            timestamp,
            indicator_id,
            signal_kind,
            side,
            score,
            value,
            details,
        }
    }

    /// Check if this signal passes raw signal filtering (above 0.80)
    /// Note: The 0.96 threshold for final scoring will be applied after predictor
    pub fn is_above_threshold(&self, config: &SignalConfig) -> bool {
        if !config.enable_filtering {
            return true;
        }

        self.score as f64 >= config.min_interesting_score
    }
}