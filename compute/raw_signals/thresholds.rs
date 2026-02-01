/// Signal strength thresholds
pub const MIN_INTERESTING_THRESHOLD: f64 = 0.80;  // Raw signals keeps everything above 0.80
pub const MAX_INTERESTING_THRESHOLD: f64 = 1.0;
pub const MIN_USELESS_THRESHOLD: f64 = 0.80;      // Below this gets filtered out

/// Raw signal types
#[derive(Debug, Clone, PartialEq)]
pub enum RawSignalType {
    Atr,
    Adx,
    Rsi,
    Macd,
    BollingerBands,
    Cci,
    Stochastic,
    Williams,
    VolumeSpike,
    Trend,
    SrLevels,
    Ema,
    Sma,
    Obv,
    Poc,
    Vwap,
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
    pub signal_type: RawSignalType,
    pub score: f64,           // Normalized score [0, 1]
    pub raw_value: f64,       // Original raw value
    pub is_interesting: bool, // Whether score >= min_interesting_score
    pub timestamp: i64,
    pub symbol: String,
    pub timeframe: String,
}

impl RawSignal {
    pub fn new(
        signal_type: RawSignalType,
        raw_value: f64,
        score: f64,
        timestamp: i64,
        symbol: String,
        timeframe: String,
    ) -> Self {
        let is_interesting = score >= MIN_INTERESTING_THRESHOLD;
        
        Self {
            signal_type,
            score,
            raw_value,
            is_interesting,
            timestamp,
            symbol,
            timeframe,
        }
    }

    /// Check if this signal passes raw signal filtering (above 0.80)
    /// Note: The 0.96 threshold for final scoring will be applied after predictor
    pub fn is_above_threshold(&self, config: &SignalConfig) -> bool {
        if !config.enable_filtering {
            return true;
        }

        self.score >= config.min_interesting_score  // Currently 0.80 for raw signals
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_signal_creation() {
        let signal = RawSignal::new(
            RawSignalType::Rsi,
            85.0,
            0.97,
            1634567890000,
            "BTCUSDT".to_string(),
            "1h".to_string(),
        );

        assert_eq!(signal.signal_type, RawSignalType::Rsi);
        assert_eq!(signal.raw_value, 85.0);
        assert_eq!(signal.score, 0.97);
        assert!(signal.is_interesting);
        assert_eq!(signal.timestamp, 1634567890000);
    }

    #[test]
    fn test_signal_threshold() {
        let config = SignalConfig::default();
        
        let interesting_signal = RawSignal::new(
            RawSignalType::Rsi,
            85.0,
            0.97,
            1634567890000,
            "BTCUSDT".to_string(),
            "1h".to_string(),
        );
        
        let weak_signal = RawSignal::new(
            RawSignalType::Rsi,
            60.0,
            0.75,
            1634567890000,
            "BTCUSDT".to_string(),
            "1h".to_string(),
        );
        
        assert!(interesting_signal.is_above_threshold(&config));
        assert!(!weak_signal.is_above_threshold(&config));
    }
}