// strategies/ml_pump_dump/src/signal_generator.rs
//
// Signal type and generation logic for Pump/Dump strategy.
//
// Converts model predictions into trade signals:
//   - PUMP (pred >= 0.65) → LONG signal with 15% target, 10 bar max hold
//   - DUMP (pred >= 0.65) → SHORT signal with 15% target, 10 bar max hold

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::pump_dump::EventType;

// ═════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═════════════════════════════════════════════════════════════════════════════

/// Pump/Dump signal generation config (loaded from env).
#[derive(Debug, Clone)]
pub struct PumpDumpSignalConfig {
    /// Minimum prediction probability to generate a signal.
    /// Env: PD_MIN_PRED (default: 0.65)
    pub min_pred: f32,

    /// Target move percentage for TP.
    /// Env: PD_TARGET_PCT (default: 15.0)
    pub target_pct: f64,

    /// Stop-loss as a fraction of target (0.5 = SL is 50% of TP distance).
    /// Env: PD_SL_FRACTION (default: 0.65)
    pub sl_fraction: f64,

    /// Maximum bars to hold before force-close.
    /// Env: PD_MAX_HOLD_BARS (default: 10)
    pub max_hold_bars: i16,

    /// Allowed finest TF range: only generate signals when finest_tf is
    /// within [min_finest_tf, max_finest_tf].
    /// Env: PD_MIN_FINEST_TF (default: 1 = 1m)
    /// Env: PD_MAX_FINEST_TF (default: 60 = 1h)
    pub min_finest_tf: i32,
    pub max_finest_tf: i32,
}

impl Default for PumpDumpSignalConfig {
    fn default() -> Self {
        Self {
            min_pred: 0.65,
            target_pct: 15.0,
            sl_fraction: 0.65,
            max_hold_bars: 10,
            min_finest_tf: 1,
            max_finest_tf: 60,
        }
    }
}

impl PumpDumpSignalConfig {
    /// Load from environment variables with defaults.
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("PD_MIN_PRED") {
            if let Ok(n) = v.parse::<f32>() { cfg.min_pred = n.clamp(0.5, 0.99); }
        }
        if let Ok(v) = std::env::var("PD_TARGET_PCT") {
            if let Ok(n) = v.parse::<f64>() { cfg.target_pct = n.clamp(1.0, 50.0); }
        }
        if let Ok(v) = std::env::var("PD_SL_FRACTION") {
            if let Ok(n) = v.parse::<f64>() { cfg.sl_fraction = n.clamp(0.1, 1.5); }
        }
        if let Ok(v) = std::env::var("PD_MAX_HOLD_BARS") {
            if let Ok(n) = v.parse::<i16>() { cfg.max_hold_bars = n.clamp(3, 100); }
        }
        if let Ok(v) = std::env::var("PD_MIN_FINEST_TF") {
            if let Ok(n) = v.parse::<i32>() { cfg.min_finest_tf = n; }
        }
        if let Ok(v) = std::env::var("PD_MAX_FINEST_TF") {
            if let Ok(n) = v.parse::<i32>() { cfg.max_finest_tf = n; }
        }

        cfg
    }

    /// Check if a given finest_tf passes the filter.
    pub fn is_finest_tf_allowed(&self, finest_tf: i32) -> bool {
        finest_tf >= self.min_finest_tf && finest_tf <= self.max_finest_tf
    }

    /// Log config summary.
    pub fn log_summary(&self) {
        tracing::info!("PumpDump Signal Config:");
        tracing::info!("  min_pred: {:.2}", self.min_pred);
        tracing::info!("  target_pct: {:.1}%", self.target_pct);
        tracing::info!("  sl_fraction: {:.2} (SL = {:.1}%)", self.sl_fraction, self.target_pct * self.sl_fraction);
        tracing::info!("  max_hold_bars: {}", self.max_hold_bars);
        tracing::info!("  finest_tf range: {}m - {}m", self.min_finest_tf, self.max_finest_tf);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// SIGNAL TYPE
// ═════════════════════════════════════════════════════════════════════════════

/// A generated pump/dump signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PumpDumpSignal {
    pub symbol: String,
    pub symbol_id: i64,
    /// Finest TF used as the signal timeframe.
    pub tf_minutes: i32,
    pub time: DateTime<Utc>,
    pub time_ms: i64,
    /// 1 = LONG (pump), -1 = SHORT (dump)
    pub side: i16,
    pub event_type: String,
    pub entry_price: f64,
    pub sl_price: f64,
    pub tp_price: f64,
    /// Model prediction probability [0..1]
    pub pred: f32,
    /// Finest TF reached during multi-TF feature extraction
    pub finest_tf: i32,
    /// Target move percentage
    pub move_pct: f64,
    /// Max bars to hold
    pub max_hold_bars: i16,
    pub reason: serde_json::Value,
}

/// Generate a pump/dump signal from model prediction.
///
/// # Arguments
/// * `config` — signal generation config
/// * `event_type` — Pump or Dump
/// * `pred` — model prediction probability
/// * `finest_tf` — finest TF reached in feature extraction
/// * `symbol` — trading pair
/// * `symbol_id` — DB symbol ID
/// * `time` — signal timestamp (latest candle time)
/// * `close_price` — current close price (entry)
///
/// # Returns
/// `Some(PumpDumpSignal)` if prediction passes threshold and finest_tf filter,
/// `None` otherwise.
pub fn generate_signal(
    config: &PumpDumpSignalConfig,
    event_type: EventType,
    pred: f32,
    finest_tf: i32,
    symbol: &str,
    symbol_id: i64,
    time: DateTime<Utc>,
    close_price: f64,
) -> Option<PumpDumpSignal> {
    // Gate 1: prediction threshold
    if pred < config.min_pred {
        return None;
    }

    // Gate 2: finest TF filter (1m - 1h)
    if !config.is_finest_tf_allowed(finest_tf) {
        return None;
    }

    let side: i16 = match event_type {
        EventType::Dump => 1,
        EventType::Pump => -1,
    };

    let target_pct = config.target_pct;
    let sl_pct = target_pct * config.sl_fraction;

    let (tp_price, sl_price) = if side == 1 {
        // LONG (pump)
        let tp = close_price * (1.0 + target_pct / 100.0);
        let sl = close_price * (1.0 - sl_pct / 100.0);
        (tp, sl)
    } else {
        // SHORT (dump)
        let tp = close_price * (1.0 - target_pct / 100.0);
        let sl = close_price * (1.0 + sl_pct / 100.0);
        (tp, sl)
    };

    let reason = json!({
        "strategy": "pump_dump_v1",
        "event_type": event_type.to_string(),
        "pred": pred,
        "finest_tf": finest_tf,
        "target_pct": target_pct,
        "sl_pct": sl_pct,
        "max_hold_bars": config.max_hold_bars,
        "min_pred": config.min_pred,
    });

    Some(PumpDumpSignal {
        symbol: symbol.to_string(),
        symbol_id,
        tf_minutes: finest_tf,
        time,
        time_ms: time.timestamp_millis(),
        side,
        event_type: event_type.to_string(),
        entry_price: close_price,
        sl_price,
        tp_price,
        pred,
        finest_tf,
        move_pct: target_pct,
        max_hold_bars: config.max_hold_bars,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_pump_signal() {
        let config = PumpDumpSignalConfig::default();
        let signal = generate_signal(
            &config, EventType::Pump, 0.75, 15,
            "BTCUSDT", 1, Utc::now(), 50000.0,
        );
        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert_eq!(sig.side, 1);
        assert_eq!(sig.event_type, "PUMP");
        assert!(sig.tp_price > sig.entry_price);
        assert!(sig.sl_price < sig.entry_price);
        assert_eq!(sig.max_hold_bars, 10);
    }

    #[test]
    fn test_generate_dump_signal() {
        let config = PumpDumpSignalConfig::default();
        let signal = generate_signal(
            &config, EventType::Dump, 0.80, 60,
            "ETHUSDT", 2, Utc::now(), 3000.0,
        );
        assert!(signal.is_some());
        let sig = signal.unwrap();
        assert_eq!(sig.side, -1);
        assert_eq!(sig.event_type, "DUMP");
        assert!(sig.tp_price < sig.entry_price);
        assert!(sig.sl_price > sig.entry_price);
    }

    #[test]
    fn test_below_threshold_no_signal() {
        let config = PumpDumpSignalConfig::default();
        let signal = generate_signal(
            &config, EventType::Pump, 0.50, 15,
            "BTCUSDT", 1, Utc::now(), 50000.0,
        );
        assert!(signal.is_none());
    }

    #[test]
    fn test_finest_tf_filter() {
        let config = PumpDumpSignalConfig::default();
        // finest_tf = 1440 (daily) — should be filtered out
        let signal = generate_signal(
            &config, EventType::Pump, 0.90, 1440,
            "BTCUSDT", 1, Utc::now(), 50000.0,
        );
        assert!(signal.is_none());

        // finest_tf = 240 (4h) — should be filtered out
        let signal = generate_signal(
            &config, EventType::Pump, 0.90, 240,
            "BTCUSDT", 1, Utc::now(), 50000.0,
        );
        assert!(signal.is_none());

        // finest_tf = 60 (1h) — should pass
        let signal = generate_signal(
            &config, EventType::Pump, 0.90, 60,
            "BTCUSDT", 1, Utc::now(), 50000.0,
        );
        assert!(signal.is_some());

        // finest_tf = 1 (1m) — should pass
        let signal = generate_signal(
            &config, EventType::Pump, 0.90, 1,
            "BTCUSDT", 1, Utc::now(), 50000.0,
        );
        assert!(signal.is_some());
    }
}
