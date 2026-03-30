// strategies/imbalance_strategy/src/imbalance.rs
//
// Imbalance Candle Detector v4
//
// v4 CHANGES:
//   1. Config loaded from TOML file (config/imbalance.toml) for rapid iteration
//   2. TF pairs are configurable (enabled/disabled per pair)
//   3. Reversals disabled by default (WR 7.8% in backtest — useless)
//   4. Stricter scoring: emphasizes trend + volume + ADX alignment
//   5. Pullback entry mode (in confirmation.rs) for better entry points
//   6. min_signal_score raised to 0.80 for higher quality signals
//
// DETECTION CRITERIA:
//   1. Body move (|close - open| / open) >= threshold for the TF
//   2. OR wick move (high - low) / open >= threshold * 1.2 with body confirming direction
//   3. Volume >= vol_avg_20 * min_volume_ratio (default 1.5x)
//
// SCORING (0.0 - 1.0):
//   Each imbalance gets a quality score based on:
//   - Trend alignment (30 pts) — most important for continuation
//   - Volume strength (20 pts)
//   - Candle structure (20 pts) — body ratio + wick quality
//   - Momentum context (15 pts) — RSI/Stoch/MACD/CMF agreement
//   - ADX strength (15 pts)

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ═════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═════════════════════════════════════════════════════════════════════════════

/// Timeframe pair: parent (detection) → child (trading).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TfPair {
    /// Parent TF in minutes (where we detect the imbalance candle).
    pub parent_tf: i32,
    /// Child TF in minutes (where we enter and manage the trade).
    pub child_tf: i32,
    /// Minimum body move % to qualify as an imbalance on the parent TF.
    pub min_move_pct: f64,
    /// Whether this pair is active.
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool { true }

/// TOML configuration file structure.
#[derive(Debug, Clone, Deserialize)]
pub struct TomlConfig {
    pub strategy: Option<StrategySection>,
    pub tp_sl: Option<TpSlSection>,
    pub exit: Option<ExitSection>,
    pub confirmation: Option<ConfirmationSection>,
    pub detection: Option<DetectionSection>,
    pub tf_pairs: Option<Vec<TfPair>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct StrategySection {
    pub min_signal_score: Option<f64>,
    pub allow_reversals: Option<bool>,
    pub allow_short_continuation: Option<bool>,
    pub min_volume_ratio: Option<f64>,
    pub max_hold_candles: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TpSlSection {
    pub tp_atr_mult: Option<f64>,
    pub sl_atr_mult: Option<f64>,
    pub tp_body_fraction: Option<f64>,
    pub sl_range_fraction: Option<f64>,
    pub max_sl_pct: Option<f64>,
    pub max_tp_pct: Option<f64>,
    pub min_tp_pct: Option<f64>,
    pub min_sl_pct: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExitSection {
    pub profit_exit_after_candles: Option<usize>,
    pub trailing_start_candle: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConfirmationSection {
    pub min_confirm_body_ratio: Option<f64>,
    pub continuation_pullback_pct: Option<f64>,
    pub use_pullback_entry: Option<bool>,
    pub max_entry_wait_candles: Option<usize>,
    pub min_pullback_atr: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DetectionSection {
    pub continuation_min_body_ratio: Option<f64>,
    pub reversal_max_body_ratio: Option<f64>,
    pub rsi_oversold: Option<f64>,
    pub rsi_overbought: Option<f64>,
}

/// Full strategy configuration (v4 — TOML-driven).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImbalanceConfig {
    /// Minimum volume relative to 20-period average.
    pub min_volume_ratio: f64,
    /// Maximum candles to hold on the child TF.
    pub max_hold_candles: usize,
    /// TP multiplier of child ATR.
    pub tp_atr_mult: f64,
    /// SL multiplier of child ATR.
    pub sl_atr_mult: f64,
    /// LEGACY: tp_body_fraction (fallback).
    pub tp_body_fraction: f64,
    /// LEGACY: sl_range_fraction (fallback).
    pub sl_range_fraction: f64,
    /// Minimum body ratio for CONTINUATION.
    pub continuation_min_body_ratio: f64,
    /// Maximum body ratio for REVERSAL.
    pub reversal_max_body_ratio: f64,
    /// RSI oversold threshold.
    pub rsi_oversold: f64,
    /// RSI overbought threshold.
    pub rsi_overbought: f64,
    /// Minimum score to accept a signal.
    pub min_signal_score: f64,
    /// Allow reversal signals.
    pub allow_reversals: bool,
    /// Allow SHORT continuation trades.
    pub allow_short_continuation: bool,
    // ── TP/SL caps ──
    /// Maximum SL distance as % of entry price. Prevents catastrophic losses.
    pub max_sl_pct: f64,
    /// Maximum TP distance as % of entry price.
    pub max_tp_pct: f64,
    /// Minimum TP distance as % of entry price.
    pub min_tp_pct: f64,
    /// Minimum SL distance as % of entry price.
    pub min_sl_pct: f64,
    // ── Exit rules ──
    /// Exit at close after N candles if in profit (0 = disabled).
    pub profit_exit_after_candles: usize,
    /// Start trailing stop after N candles in profit.
    pub trailing_start_candle: usize,
    // ── Confirmation settings ──
    /// Minimum body ratio on confirmation candle.
    pub min_confirm_body_ratio: f64,
    /// Maximum pullback as fraction of parent body.
    pub continuation_pullback_pct: f64,
    /// Use pullback entry mode.
    pub use_pullback_entry: bool,
    /// Max child candles to wait for entry in pullback mode.
    pub max_entry_wait_candles: usize,
    /// Minimum pullback depth in ATR units.
    pub min_pullback_atr: f64,
    // ── TF pairs (loaded from TOML) ──
    #[serde(skip)]
    pub tf_pairs: Vec<TfPair>,
}

impl Default for ImbalanceConfig {
    fn default() -> Self {
        Self {
            min_volume_ratio: 1.5,
            max_hold_candles: 3,
            tp_atr_mult: 1.5,
            sl_atr_mult: 1.0,
            tp_body_fraction: 0.30,
            sl_range_fraction: 0.20,
            continuation_min_body_ratio: 0.55,
            reversal_max_body_ratio: 0.40,
            rsi_oversold: 25.0,
            rsi_overbought: 75.0,
            min_signal_score: 0.80,
            allow_reversals: false,
            allow_short_continuation: false,
            max_sl_pct: 5.0,
            max_tp_pct: 15.0,
            min_tp_pct: 0.5,
            min_sl_pct: 0.3,
            profit_exit_after_candles: 0,
            trailing_start_candle: 2,
            min_confirm_body_ratio: 0.15,
            continuation_pullback_pct: 0.35,
            use_pullback_entry: true,
            max_entry_wait_candles: 4,
            min_pullback_atr: 0.3,
            tf_pairs: vec![
                TfPair { parent_tf: 1440, child_tf: 240, min_move_pct: 15.0, enabled: true },
                TfPair { parent_tf: 240, child_tf: 60, min_move_pct: 15.0, enabled: true },
            ],
        }
    }
}

impl ImbalanceConfig {
    /// Load from TOML file, falling back to defaults for missing fields.
    pub fn from_toml(path: &str) -> Self {
        let mut cfg = Self::default();

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Cannot read config {}: {}. Using defaults.", path, e);
                return cfg;
            }
        };

        let toml: TomlConfig = match toml::from_str(&content) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("Cannot parse config {}: {}. Using defaults.", path, e);
                return cfg;
            }
        };

        // ── Strategy section ──
        if let Some(s) = &toml.strategy {
            if let Some(v) = s.min_signal_score { cfg.min_signal_score = v.clamp(0.1, 0.99); }
            if let Some(v) = s.allow_reversals { cfg.allow_reversals = v; }
            if let Some(v) = s.allow_short_continuation { cfg.allow_short_continuation = v; }
            if let Some(v) = s.min_volume_ratio { cfg.min_volume_ratio = v.max(1.0); }
            if let Some(v) = s.max_hold_candles { cfg.max_hold_candles = v.clamp(1, 10); }
        }

        // ── TP/SL section ──
        if let Some(ts) = &toml.tp_sl {
            if let Some(v) = ts.tp_atr_mult { cfg.tp_atr_mult = v.clamp(0.3, 5.0); }
            if let Some(v) = ts.sl_atr_mult { cfg.sl_atr_mult = v.clamp(0.3, 5.0); }
            if let Some(v) = ts.tp_body_fraction { cfg.tp_body_fraction = v; }
            if let Some(v) = ts.sl_range_fraction { cfg.sl_range_fraction = v; }
            if let Some(v) = ts.max_sl_pct { cfg.max_sl_pct = v.clamp(0.5, 50.0); }
            if let Some(v) = ts.max_tp_pct { cfg.max_tp_pct = v.clamp(0.5, 100.0); }
            if let Some(v) = ts.min_tp_pct { cfg.min_tp_pct = v.clamp(0.1, 10.0); }
            if let Some(v) = ts.min_sl_pct { cfg.min_sl_pct = v.clamp(0.1, 10.0); }
        }

        // ── Exit section ──
        if let Some(e) = &toml.exit {
            if let Some(v) = e.profit_exit_after_candles { cfg.profit_exit_after_candles = v; }
            if let Some(v) = e.trailing_start_candle { cfg.trailing_start_candle = v.max(1); }
        }

        // ── Confirmation section ──
        if let Some(c) = &toml.confirmation {
            if let Some(v) = c.min_confirm_body_ratio { cfg.min_confirm_body_ratio = v; }
            if let Some(v) = c.continuation_pullback_pct { cfg.continuation_pullback_pct = v; }
            if let Some(v) = c.use_pullback_entry { cfg.use_pullback_entry = v; }
            if let Some(v) = c.max_entry_wait_candles { cfg.max_entry_wait_candles = v.clamp(1, 10); }
            if let Some(v) = c.min_pullback_atr { cfg.min_pullback_atr = v; }
        }

        // ── Detection section ──
        if let Some(d) = &toml.detection {
            if let Some(v) = d.continuation_min_body_ratio { cfg.continuation_min_body_ratio = v; }
            if let Some(v) = d.reversal_max_body_ratio { cfg.reversal_max_body_ratio = v; }
            if let Some(v) = d.rsi_oversold { cfg.rsi_oversold = v; }
            if let Some(v) = d.rsi_overbought { cfg.rsi_overbought = v; }
        }

        // ── TF Pairs ──
        if let Some(pairs) = toml.tf_pairs {
            cfg.tf_pairs = pairs.into_iter().filter(|p| p.enabled).collect();
        }

        tracing::info!("Config loaded from {}", path);
        cfg
    }

    /// Load from environment variables (legacy fallback).
    pub fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("IMB_MIN_VOL_RATIO") {
            if let Ok(n) = v.parse::<f64>() { cfg.min_volume_ratio = n.max(1.0); }
        }
        if let Ok(v) = std::env::var("IMB_MAX_HOLD") {
            if let Ok(n) = v.parse::<usize>() { cfg.max_hold_candles = n.clamp(1, 10); }
        }
        if let Ok(v) = std::env::var("IMB_TP_ATR") {
            if let Ok(n) = v.parse::<f64>() { cfg.tp_atr_mult = n.clamp(0.5, 5.0); }
        }
        if let Ok(v) = std::env::var("IMB_SL_ATR") {
            if let Ok(n) = v.parse::<f64>() { cfg.sl_atr_mult = n.clamp(0.3, 3.0); }
        }
        if let Ok(v) = std::env::var("IMB_MIN_SCORE") {
            if let Ok(n) = v.parse::<f64>() { cfg.min_signal_score = n.clamp(0.1, 0.9); }
        }
        if let Ok(v) = std::env::var("IMB_ALLOW_SHORT") {
            cfg.allow_short_continuation = v == "1" || v.to_lowercase() == "true";
        }
        cfg
    }

    /// Print config summary.
    pub fn log_summary(&self) {
        tracing::info!("ImbalanceConfig:");
        tracing::info!("  min_volume_ratio: {:.1}x", self.min_volume_ratio);
        tracing::info!("  max_hold_candles: {}", self.max_hold_candles);
        tracing::info!("  tp_atr_mult: {:.1}× child ATR", self.tp_atr_mult);
        tracing::info!("  sl_atr_mult: {:.1}× child ATR (R:R={:.1}:1)",
                        self.sl_atr_mult, self.tp_atr_mult / self.sl_atr_mult);
        tracing::info!("  max_sl_pct: {:.1}%, max_tp_pct: {:.1}%", self.max_sl_pct, self.max_tp_pct);
        tracing::info!("  min_sl_pct: {:.1}%, min_tp_pct: {:.1}%", self.min_sl_pct, self.min_tp_pct);
        tracing::info!("  min_signal_score: {:.2}", self.min_signal_score);
        tracing::info!("  allow_reversals: {}", self.allow_reversals);
        tracing::info!("  allow_short_continuation: {}", self.allow_short_continuation);
        tracing::info!("  profit_exit_after: {} candles", self.profit_exit_after_candles);
        tracing::info!("  trailing_start: candle {}", self.trailing_start_candle);
        tracing::info!("  use_pullback_entry: {}", self.use_pullback_entry);
        tracing::info!("  max_entry_wait_candles: {}", self.max_entry_wait_candles);
        tracing::info!("  min_pullback_atr: {:.2}", self.min_pullback_atr);
        tracing::info!("  continuation body ratio: ≥{:.2}", self.continuation_min_body_ratio);
        tracing::info!("  TF pairs ({} active):", self.tf_pairs.len());
        for p in &self.tf_pairs {
            tracing::info!("    {}m (≥{:.0}%) → trade on {}m", p.parent_tf, p.min_move_pct, p.child_tf);
        }
    }

    /// Get enabled TF pairs.
    pub fn active_tf_pairs(&self) -> &[TfPair] {
        &self.tf_pairs
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// DATA TYPES
// ═════════════════════════════════════════════════════════════════════════════

/// Direction of the imbalance candle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Bullish,
    Bearish,
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Direction::Bullish => write!(f, "BULL"),
            Direction::Bearish => write!(f, "BEAR"),
        }
    }
}

/// Trade direction based on how we interpret the imbalance candle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    Long,
    Short,
}

impl std::fmt::Display for TradeDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradeDirection::Long => write!(f, "LONG"),
            TradeDirection::Short => write!(f, "SHORT"),
        }
    }
}

/// Signal type: continuation or reversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalType {
    Continuation,
    Reversal,
}

impl std::fmt::Display for SignalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalType::Continuation => write!(f, "CONT"),
            SignalType::Reversal => write!(f, "REV"),
        }
    }
}

/// A single candle with indicator data.
#[derive(Debug, Clone)]
pub struct Candle {
    pub time: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub rsi: f64,
    pub stoch_k: f64,
    pub stoch_d: f64,
    pub williams: f64,
    pub adx: f64,
    pub atr: f64,
    pub bb_upper: f64,
    pub bb_lower: f64,
    pub ema_20: f64,
    pub ema_50: f64,
    pub volume_spike: f64,
    pub trend: f64,
    pub trend_short: f64,
    pub supertrend_dir: f64,
    pub macd_hist: f64,
    pub cmf: f64,
    pub mfi: f64,
    pub obv: f64,
}

/// A detected imbalance candle with all analysis metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImbalanceCandle {
    pub symbol: String,
    pub time: DateTime<Utc>,
    pub parent_tf: i32,
    pub child_tf: i32,
    pub direction: Direction,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub body_ratio: f64,
    pub upper_wick_ratio: f64,
    pub lower_wick_ratio: f64,
    pub body_move_pct: f64,
    pub range_pct: f64,
    pub volume_ratio: f64,
    pub rsi: f64,
    pub stoch_k: f64,
    pub adx: f64,
    pub trend: f64,
    pub supertrend_dir: f64,
    pub macd_hist: f64,
    pub cmf: f64,
    pub atr: f64,
    pub signal_type: SignalType,
    pub trade_direction: TradeDirection,
    pub score: f64,
    pub tp_distance: f64,
    pub sl_distance: f64,
}

// ═════════════════════════════════════════════════════════════════════════════
// DETECTION
// ═════════════════════════════════════════════════════════════════════════════

/// Detect if a candle qualifies as an imbalance candle.
pub fn detect_imbalance(
    candle: &Candle,
    recent_candles: &[Candle],
    tf_pair: &TfPair,
    config: &ImbalanceConfig,
) -> Option<ImbalanceCandle> {
    if candle.open.abs() < 1e-12 { return None; }

    let body = candle.close - candle.open;
    let body_abs = body.abs();
    let range = candle.high - candle.low;
    if range < 1e-12 { return None; }

    let body_move_pct = body_abs / candle.open * 100.0;
    let range_pct = range / candle.open * 100.0;

    // ── Check minimum move threshold ──
    let body_qualifies = body_move_pct >= tf_pair.min_move_pct;
    let range_qualifies = range_pct >= tf_pair.min_move_pct * 1.2
        && body_abs / range > 0.30;

    if !body_qualifies && !range_qualifies {
        return None;
    }

    // ── Direction ──
    let direction = if body > 0.0 { Direction::Bullish } else { Direction::Bearish };

    // ── Candle structure ──
    let body_ratio = body_abs / range;
    let upper_wick = candle.high - candle.close.max(candle.open);
    let lower_wick = candle.close.min(candle.open) - candle.low;
    let upper_wick_ratio = upper_wick / range;
    let lower_wick_ratio = lower_wick / range;

    // ── Volume context ──
    let vol_avg = if recent_candles.len() >= 5 {
        let n = recent_candles.len().min(20);
        let sum: f64 = recent_candles[recent_candles.len() - n..].iter()
            .map(|c| c.volume)
            .sum();
        sum / n as f64
    } else {
        candle.volume
    };
    let volume_ratio = if vol_avg > 1e-12 { candle.volume / vol_avg } else { 1.0 };

    if volume_ratio < config.min_volume_ratio {
        return None;
    }

    // ── Determine signal type ──
    let (signal_type, trade_direction) = determine_signal_type(
        candle, direction, body_ratio, upper_wick_ratio, lower_wick_ratio, config,
    )?;

    // ── Score the imbalance quality (v4: rebalanced weights) ──
    let score = compute_score(
        candle, direction, body_ratio, volume_ratio,
        upper_wick_ratio, lower_wick_ratio, signal_type,
    );

    // ── Apply score threshold ──
    if score < config.min_signal_score {
        return None;
    }

    // ── Filter: SHORT continuation disabled ──
    if trade_direction == TradeDirection::Short
        && signal_type == SignalType::Continuation
        && !config.allow_short_continuation
    {
        return None;
    }

    // ── Compute dynamic TP/SL (legacy, overridden by child ATR in confirmation) ──
    let (tp_distance, sl_distance) = compute_tp_sl(
        candle, body_abs, range, signal_type, config,
    );

    Some(ImbalanceCandle {
        symbol: candle.symbol.clone(),
        time: candle.time,
        parent_tf: tf_pair.parent_tf,
        child_tf: tf_pair.child_tf,
        direction,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
        volume: candle.volume,
        body_ratio,
        upper_wick_ratio,
        lower_wick_ratio,
        body_move_pct,
        range_pct,
        volume_ratio,
        rsi: candle.rsi,
        stoch_k: candle.stoch_k,
        adx: candle.adx,
        trend: candle.trend,
        supertrend_dir: candle.supertrend_dir,
        macd_hist: candle.macd_hist,
        cmf: candle.cmf,
        atr: candle.atr,
        signal_type,
        trade_direction,
        score,
        tp_distance,
        sl_distance,
    })
}

/// Determine signal type: continuation or reversal.
///
/// v4: Reversals can be completely disabled via config.
fn determine_signal_type(
    candle: &Candle,
    direction: Direction,
    body_ratio: f64,
    upper_wick_ratio: f64,
    lower_wick_ratio: f64,
    config: &ImbalanceConfig,
) -> Option<(SignalType, TradeDirection)> {
    // ── CONTINUATION ──
    if body_ratio >= config.continuation_min_body_ratio {
        match direction {
            Direction::Bullish => {
                if upper_wick_ratio < 0.30 {
                    return Some((SignalType::Continuation, TradeDirection::Long));
                }
            }
            Direction::Bearish => {
                if lower_wick_ratio < 0.30 {
                    return Some((SignalType::Continuation, TradeDirection::Short));
                }
            }
        }
    }

    // ── REVERSAL (only if enabled) ──
    if config.allow_reversals && body_ratio <= config.reversal_max_body_ratio {
        match direction {
            Direction::Bullish => {
                if upper_wick_ratio > 0.35 && candle.rsi > config.rsi_overbought {
                    return Some((SignalType::Reversal, TradeDirection::Short));
                }
            }
            Direction::Bearish => {
                if lower_wick_ratio > 0.35 && candle.rsi < config.rsi_oversold {
                    return Some((SignalType::Reversal, TradeDirection::Long));
                }
            }
        }
    }

    None
}

/// Compute quality score (v4 — rebalanced for higher discrimination).
///
/// Weight distribution:
///   Trend alignment: 30 pts (was 25) — THE key predictor
///   Volume: 20 pts (unchanged)
///   Candle structure: 20 pts (unchanged)
///   Momentum: 15 pts (was 20) — less noise
///   ADX: 15 pts (unchanged)
fn compute_score(
    candle: &Candle,
    direction: Direction,
    body_ratio: f64,
    volume_ratio: f64,
    upper_wick_ratio: f64,
    lower_wick_ratio: f64,
    signal_type: SignalType,
) -> f64 {
    let mut score = 0.0f64;
    let mut max_score = 0.0f64;

    // 1. Trend alignment (30 points) — MOST IMPORTANT
    max_score += 30.0;
    let trend_dir = if candle.trend > 0.5 { 1.0 }
                    else if candle.trend < -0.5 { -1.0 }
                    else { 0.0 };
    let trend_short_dir = if candle.trend_short > 0.5 { 1.0 }
                          else if candle.trend_short < -0.5 { -1.0 }
                          else { 0.0 };
    let candle_dir = match direction {
        Direction::Bullish => 1.0,
        Direction::Bearish => -1.0,
    };

    match signal_type {
        SignalType::Continuation => {
            let long_aligned = trend_dir * candle_dir > 0.0;
            let short_aligned = trend_short_dir * candle_dir > 0.0;
            let supertrend_aligned = (candle.supertrend_dir > 0.0 && candle_dir > 0.0)
                || (candle.supertrend_dir < 0.0 && candle_dir < 0.0);

            if long_aligned && short_aligned && supertrend_aligned {
                score += 30.0; // all three agree — strongest signal
            } else if long_aligned && short_aligned {
                score += 24.0; // both trends agree
            } else if long_aligned && supertrend_aligned {
                score += 20.0;
            } else if long_aligned {
                score += 15.0; // long trend agrees only
            } else if short_aligned {
                score += 10.0; // only short trend
            } else if trend_dir == 0.0 && trend_short_dir == 0.0 {
                score += 8.0; // neutral — meh
            }
            // Against both trends → 0 points
        }
        SignalType::Reversal => {
            if trend_dir * candle_dir > 0.0 {
                score += 25.0;
            } else {
                score += 8.0;
            }
        }
    }

    // 2. Volume confirmation (20 points)
    max_score += 20.0;
    let vol_score = ((volume_ratio - 1.0) / 2.0).clamp(0.0, 1.0) * 20.0;
    score += vol_score;

    // 3. Candle structure quality (20 points)
    max_score += 20.0;
    match signal_type {
        SignalType::Continuation => {
            let body_quality = body_ratio.clamp(0.0, 1.0);
            let wick_quality = match direction {
                Direction::Bullish => 1.0 - upper_wick_ratio,
                Direction::Bearish => 1.0 - lower_wick_ratio,
            };
            // v4: bonus for very clean candles (body > 0.75)
            let clean_bonus = if body_ratio > 0.75 { 0.15 } else { 0.0 };
            score += ((body_quality * 0.5 + wick_quality * 0.35 + clean_bonus) * 20.0).min(20.0);
        }
        SignalType::Reversal => {
            let body_quality = (1.0 - body_ratio).clamp(0.0, 1.0);
            let wick_quality = match direction {
                Direction::Bullish => upper_wick_ratio,
                Direction::Bearish => lower_wick_ratio,
            };
            score += (body_quality * 0.4 + wick_quality * 0.6) * 20.0;
        }
    }

    // 4. Momentum context (15 points)
    max_score += 15.0;
    let mut momentum_hits = 0u32;
    let mut momentum_total = 0u32;

    // RSI in momentum zone (not oversold/overbought for continuation)
    momentum_total += 1;
    match signal_type {
        SignalType::Continuation => {
            match direction {
                Direction::Bullish if candle.rsi > 45.0 && candle.rsi < 80.0 => momentum_hits += 1,
                Direction::Bearish if candle.rsi < 55.0 && candle.rsi > 20.0 => momentum_hits += 1,
                _ => {}
            }
        }
        SignalType::Reversal => {
            match direction {
                Direction::Bullish if candle.rsi > 70.0 => momentum_hits += 1,
                Direction::Bearish if candle.rsi < 30.0 => momentum_hits += 1,
                _ => {}
            }
        }
    }

    // MACD histogram
    momentum_total += 1;
    match (signal_type, direction) {
        (SignalType::Continuation, Direction::Bullish) if candle.macd_hist > 0.0 => momentum_hits += 1,
        (SignalType::Continuation, Direction::Bearish) if candle.macd_hist < 0.0 => momentum_hits += 1,
        (SignalType::Reversal, _) => momentum_hits += 1,
        _ => {}
    }

    // CMF (money flow)
    momentum_total += 1;
    match (signal_type, direction) {
        (SignalType::Continuation, Direction::Bullish) if candle.cmf > 0.05 => momentum_hits += 1,
        (SignalType::Continuation, Direction::Bearish) if candle.cmf < -0.05 => momentum_hits += 1,
        (SignalType::Reversal, _) if candle.cmf.abs() < 0.1 => momentum_hits += 1,
        _ => {}
    }

    // MFI (money flow index)
    momentum_total += 1;
    match (signal_type, direction) {
        (SignalType::Continuation, Direction::Bullish) if candle.mfi > 50.0 => momentum_hits += 1,
        (SignalType::Continuation, Direction::Bearish) if candle.mfi < 50.0 => momentum_hits += 1,
        _ => {}
    }

    if momentum_total > 0 {
        score += momentum_hits as f64 / momentum_total as f64 * 15.0;
    }

    // 5. ADX strength (15 points)
    max_score += 15.0;
    match signal_type {
        SignalType::Continuation => {
            if candle.adx > 40.0 {
                score += 15.0;
            } else if candle.adx > 30.0 {
                score += 12.0;
            } else if candle.adx > 20.0 {
                score += 8.0;
            } else if candle.adx > 15.0 {
                score += 4.0;
            }
        }
        SignalType::Reversal => {
            if candle.adx > 20.0 && candle.adx < 40.0 {
                score += 15.0;
            } else if candle.adx >= 40.0 {
                score += 8.0;
            } else {
                score += 5.0;
            }
        }
    }

    (score / max_score).clamp(0.0, 1.0)
}

/// Compute dynamic TP and SL distances (legacy, overridden by child ATR in confirmation).
fn compute_tp_sl(
    candle: &Candle,
    body_abs: f64,
    range: f64,
    signal_type: SignalType,
    config: &ImbalanceConfig,
) -> (f64, f64) {
    let atr_floor = candle.atr.max(range * 0.02);

    let tp = match signal_type {
        SignalType::Continuation => {
            (body_abs * config.tp_body_fraction).max(atr_floor)
        }
        SignalType::Reversal => {
            (range * config.tp_body_fraction * 0.6).max(atr_floor)
        }
    };

    let sl_raw = match signal_type {
        SignalType::Continuation => {
            (range * config.sl_range_fraction).max(atr_floor * 0.5)
        }
        SignalType::Reversal => {
            (range * config.sl_range_fraction * 0.7).max(atr_floor * 0.5)
        }
    };

    let sl = sl_raw.min(tp * 0.80).max(atr_floor * 0.3);
    (tp, sl)
}

/// Scan parent TF candles for imbalance signals.
pub fn scan_for_imbalances(
    candles: &[Candle],
    tf_pair: &TfPair,
    config: &ImbalanceConfig,
) -> Vec<ImbalanceCandle> {
    let mut results = Vec::new();
    let context_window = 20;

    for i in context_window..candles.len() {
        let recent = &candles[(i - context_window)..i];
        if let Some(imb) = detect_imbalance(&candles[i], recent, tf_pair, config) {
            results.push(imb);
        }
    }

    results
}

// ═════════════════════════════════════════════════════════════════════════════
// TESTS
// ═════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candle(open: f64, high: f64, low: f64, close: f64, vol: f64) -> Candle {
        Candle {
            time: Utc::now(), symbol: "TESTUSDT".to_string(),
            open, high, low, close, volume: vol,
            rsi: 50.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            adx: 30.0, atr: (high - low) * 0.5,
            bb_upper: close * 1.02, bb_lower: close * 0.98,
            ema_20: close, ema_50: close,
            volume_spike: 1.0, trend: 0.5, trend_short: 0.5,
            supertrend_dir: 1.0, macd_hist: 0.1, cmf: 0.1,
            mfi: 50.0, obv: 0.0,
        }
    }

    fn make_context(n: usize, vol: f64) -> Vec<Candle> {
        (0..n).map(|_| make_candle(100.0, 101.0, 99.0, 100.5, vol)).collect()
    }

    #[test]
    fn test_detect_bullish_imbalance() {
        let config = ImbalanceConfig::default();
        let tf_pair = TfPair { parent_tf: 240, child_tf: 60, min_move_pct: 8.0, enabled: true };
        let context = make_context(20, 1000.0);

        let mut candle = make_candle(100.0, 111.0, 99.5, 110.0, 2000.0);
        candle.trend = 1.0;
        candle.trend_short = 1.0;
        candle.adx = 35.0;
        candle.macd_hist = 0.5;
        candle.cmf = 0.2;
        candle.mfi = 65.0;
        let result = detect_imbalance(&candle, &context, &tf_pair, &config);

        assert!(result.is_some(), "Should detect 10% bullish candle with good trend");
        let imb = result.unwrap();
        assert_eq!(imb.direction, Direction::Bullish);
        assert_eq!(imb.signal_type, SignalType::Continuation);
        assert_eq!(imb.trade_direction, TradeDirection::Long);
        assert!(imb.score >= config.min_signal_score, "Score {} < min {}", imb.score, config.min_signal_score);
    }

    #[test]
    fn test_reject_ambiguous_signal() {
        let config = ImbalanceConfig::default();
        let tf_pair = TfPair { parent_tf: 240, child_tf: 60, min_move_pct: 8.0, enabled: true };
        let context = make_context(20, 1000.0);

        let candle = make_candle(100.0, 115.0, 99.5, 105.0, 2000.0);
        let result = detect_imbalance(&candle, &context, &tf_pair, &config);
        assert!(result.is_none(), "Ambiguous signal should be rejected");
    }

    #[test]
    fn test_reject_low_volume() {
        let config = ImbalanceConfig::default();
        let tf_pair = TfPair { parent_tf: 240, child_tf: 60, min_move_pct: 8.0, enabled: true };
        let context = make_context(20, 2000.0);

        let candle = make_candle(100.0, 111.0, 99.5, 110.0, 2000.0);
        let result = detect_imbalance(&candle, &context, &tf_pair, &config);
        assert!(result.is_none(), "Should reject — volume not above 1.5x average");
    }

    #[test]
    fn test_reversals_disabled_by_default() {
        let config = ImbalanceConfig::default();
        assert!(!config.allow_reversals);

        let tf_pair = TfPair { parent_tf: 240, child_tf: 60, min_move_pct: 8.0, enabled: true };
        let context = make_context(20, 1000.0);

        // Reversal candidate: big wick, small body, RSI extreme
        let mut candle = make_candle(100.0, 115.0, 99.5, 102.0, 3000.0);
        candle.rsi = 80.0;
        let result = detect_imbalance(&candle, &context, &tf_pair, &config);
        assert!(result.is_none(), "Reversals should be disabled by default");
    }
}
