// calculates final signal based on all previous steps - indicators, raw_signals, ml predictors, heruistic predictors, market_parameter (volatility, trend)
//signal must contain Symbol (traiding pair) time created, timeframe (1m, 5m, 15m, 1h, 4h, 1d), stop_loss, take_profits_1,2,3, side (short/long), combined final score, scores from prevous steps like raw_score, ind_score, ml_score, heruistic_score,

// compute/scoring/trade_signal_calculator.rs

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use crate::predictors::types::{PredictionRow, PredictionAspect};

use super::final_score::{FinalScorer, FinalScoreBreakdown, SetupKind};
use super::market_params_calculator::MarketParams;

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL TP/SL PARAMETERS - loaded from config/signal_params.toml
// ═══════════════════════════════════════════════════════════════════════════
// Edit config/signal_params.toml to calibrate TP/SL values

#[derive(Debug, Clone, Copy)]
struct TfTargets {
    min_tp1_pct: f64,
    min_tp2_pct: f64,
    min_tp3_pct: f64,
    min_sl_pct:  f64,
    /// Maximum allowed SL distance as % of entry — caps catastrophic losses
    max_sl_pct:  f64,
    /// Minimum acceptable Risk:Reward ratio (TP1 / SL). Below this, signal is rejected.
    min_rr: f64,
}

fn load_signal_params_config() -> config::Config {
    config::Config::builder()
        .add_source(config::File::with_name("config/signal_params").required(false))
        .build()
        .unwrap_or_else(|_| config::Config::default())
}

fn get_tf_from_config(tf_minutes: i16, key: &str, default: f64) -> f64 {
    let tf_section = match tf_minutes {
        1 => "timeframe_1m",
        5 => "timeframe_5m",
        15 => "timeframe_15m",
        60 => "timeframe_1h",
        240 => "timeframe_4h",
        1440 => "timeframe_1d",
        _ => "timeframe_1h",
    };
    
    load_signal_params_config()
        .get::<f64>(&format!("{}.{}", tf_section, key))
        .unwrap_or(default)
}

fn tf_targets(tf_minutes: i16) -> TfTargets {
    // Loads TP/SL values from config/signal_params.toml
    // Values in TOML are in percent (0.3 = 0.3%), converted to decimal here
    let mut t = TfTargets {
        min_tp1_pct: get_tf_from_config(tf_minutes, "tp1_pct", 1.5) / 100.0,
        min_tp2_pct: get_tf_from_config(tf_minutes, "tp2_pct", 2.8) / 100.0,
        min_tp3_pct: get_tf_from_config(tf_minutes, "tp3_pct", 4.0) / 100.0,
        min_sl_pct:  get_tf_from_config(tf_minutes, "sl_min_pct", 1.0) / 100.0,
        max_sl_pct:  get_tf_from_config(tf_minutes, "sl_pct", 3.0) / 100.0,
        min_rr:      0.0,
    };

    // enforce monotonic targets (strict)
    if t.min_tp2_pct <= t.min_tp1_pct { t.min_tp2_pct = t.min_tp1_pct * 1.6; }
    if t.min_tp3_pct <= t.min_tp2_pct { t.min_tp3_pct = t.min_tp2_pct * 1.45; }

    t
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSide {
    Long,
    Short,
}

impl TradeSide {
    pub fn as_i8(&self) -> i8 {
        match self {
            TradeSide::Long => 1,
            TradeSide::Short => -1,
        }
    }
    pub fn from_i16(v: i16) -> Option<Self> {
        match v.signum() {
            1 => Some(TradeSide::Long),
            -1 => Some(TradeSide::Short),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub time: DateTime<Utc>,
    pub time_ms: i64,

    pub symbol_id: i64,
    pub symbol: String,
    pub tf_minutes: i16,

    pub side: i8, // +1 long / -1 short

    pub entry: f64,
    pub stop_loss: f64,
    pub tp1: f64,
    pub tp2: f64,
    pub tp3: f64,

    pub leverage: i16,

    pub final_score: f64,
    pub breakdown_json: Value,
}

pub struct TradeSignalCalculator {
    pub min_final_score: f64,

    pub base_leverage: i16,

    /// Risk parameters - BOUNCE (short TP1 for high winrate)
    pub sl_atr_mult_bounce: f64,      // 0.55 (SL behind level)
    pub tp1_atr_mult_bounce: f64,     // 0.6-0.9 (short TP for easy hit)
    pub tp2_atr_mult_bounce: f64,     // 1.4
    pub tp3_atr_mult_bounce: f64,     // 2.2

    /// Risk parameters - BREAKOUT (larger targets)
    pub sl_atr_mult_breakout: f64,    // 0.75
    pub tp1_atr_mult_breakout: f64,   // 1.1
    pub tp2_atr_mult_breakout: f64,   // 1.8
    pub tp3_atr_mult_breakout: f64,   // 2.8

    pub tp2_ratio_min: f64,
    pub tp3_ratio_min: f64,
    pub min_tp_gap_pct: f64,

    /// Fallback if ATR missing: ATR = entry * fallback_atr_pct
    pub fallback_atr_pct: f64,
}

// Helper to load ATR parameters from config
fn get_atr_param(section: &str, key: &str, default: f64) -> f64 {
    load_signal_params_config()
        .get::<f64>(&format!("{}.{}", section, key))
        .unwrap_or(default)
}

fn get_other_param(key: &str, default: f64) -> f64 {
    load_signal_params_config()
        .get::<f64>(&format!("other.{}", key))
        .unwrap_or(default)
}

impl Default for TradeSignalCalculator {
    fn default() -> Self {
        // Loads ATR parameters from config/signal_params.toml
        // Defaults used if file not found or key missing
        Self {
            min_final_score: 0.96,
            base_leverage: 5,
            // Bounce parameters from [atr_bounce] section
            sl_atr_mult_bounce:  get_atr_param("atr_bounce", "sl_mult", 0.55),
            tp1_atr_mult_bounce: get_atr_param("atr_bounce", "tp1_mult", 0.75),
            tp2_atr_mult_bounce: get_atr_param("atr_bounce", "tp2_mult", 1.4),
            tp3_atr_mult_bounce: get_atr_param("atr_bounce", "tp3_mult", 2.2),
            // Breakout parameters from [atr_breakout] section
            sl_atr_mult_breakout:  get_atr_param("atr_breakout", "sl_mult", 0.75),
            tp1_atr_mult_breakout: get_atr_param("atr_breakout", "tp1_mult", 1.1),
            tp2_atr_mult_breakout: get_atr_param("atr_breakout", "tp2_mult", 1.8),
            tp3_atr_mult_breakout: get_atr_param("atr_breakout", "tp3_mult", 2.8),
            // Other parameters from [other] section
            tp2_ratio_min:    get_other_param("tp2_ratio_min", 1.6),
            tp3_ratio_min:    get_other_param("tp3_ratio_min", 1.45),
            min_tp_gap_pct:   get_other_param("min_tp_gap_pct", 0.002),
            fallback_atr_pct: get_other_param("fallback_atr_pct", 0.008),
        }
    }
}

impl TradeSignalCalculator {
    pub fn new(min_final_score: f64) -> Self {
        Self { min_final_score, ..Default::default() }
    }

    /// Level-aware TP/SL calculation with setup kind (bounce vs breakout).
    ///
    /// For **BOUNCE** setups:
    /// - SL  = nearest support/resistance behind entry − buffer (0.55 ATR)
    /// - TP1 = short target (0.6-0.9 ATR) for high winrate
    /// - TP2/TP3 = next levels or ATR multiples
    ///
    /// For **BREAKOUT** setups:
    /// - SL  = entry − 0.75 ATR (tighter, behind retest)
    /// - TP1 = larger target (1.1+ ATR)
    /// - TP2/TP3 = next levels or larger ATR multiples
    ///
    /// Falls back to fully ATR-based targets when insufficient levels exist.
    /// Minimum distances from `tf_targets` are always enforced.
    #[allow(clippy::too_many_arguments)]
    fn level_aware_targets(
        &self,
        entry_price: f64,
        side: i8,       // +1 long, -1 short
        atr: f64,
        support_levels: &[f64],    // sorted ascending
        resistance_levels: &[f64], // sorted ascending
        tf_targets: &TfTargets,
        setup_kind: SetupKind,
    ) -> (f64, f64, f64, f64) {
        let side_f = side as f64;

        // Select ATR multipliers based on setup kind
        let (sl_mult, tp1_mult, tp2_mult, tp3_mult) = match setup_kind {
            SetupKind::Bounce => (
                self.sl_atr_mult_bounce,
                self.tp1_atr_mult_bounce,
                self.tp2_atr_mult_bounce,
                self.tp3_atr_mult_bounce,
            ),
            SetupKind::Breakout => (
                self.sl_atr_mult_breakout,
                self.tp1_atr_mult_breakout,
                self.tp2_atr_mult_breakout,
                self.tp3_atr_mult_breakout,
            ),
        };

        // ATR-based fallbacks
        let fb_sl  = entry_price - side_f * sl_mult * atr;
        let fb_tp1 = entry_price + side_f * tp1_mult * atr;
        let fb_tp2 = entry_price + side_f * tp2_mult * atr;
        let fb_tp3 = entry_price + side_f * tp3_mult * atr;

        // Minimum TP1 distance = 0.5 ATR to avoid picking too-close levels
        // (was 0.8 — too aggressive, filtered too many valid levels)
        let min_tp1_atr_dist = 0.5 * atr;

        let (stop_loss, tp1, tp2, tp3) = if side > 0 {
            // ── LONG ───────────────────────────────────────────────────
            // SL: nearest support *below* entry − buffer (reduced buffer)
            let sl_buffer = match setup_kind {
                SetupKind::Bounce => 0.3 * atr,   // Behind level (original)
                SetupKind::Breakout => 0.0,        // Tight, at entry level
            };
            let sl = support_levels
                .iter()
                .rev() // descending
                .find(|&&p| p < entry_price)
                .map(|&p| p - sl_buffer)
                .unwrap_or(fb_sl);

            // TP1: nearest resistance *above* entry, but SKIP levels too close (< 0.8 ATR)
            let mut res_iter = resistance_levels.iter()
                .filter(|&&p| p > entry_price && (p - entry_price) >= min_tp1_atr_dist);
            let t1 = res_iter.next().copied().unwrap_or(fb_tp1);
            let t2 = res_iter.next().copied().unwrap_or(fb_tp2);
            let t3_level = res_iter.next().copied();
            let t3 = match t3_level {
                Some(lv) => lv.max(fb_tp3),
                None => fb_tp3,
            };

            (sl, t1, t2, t3)
        } else {
            // ── SHORT ──────────────────────────────────────────────────
            // SL: nearest resistance *above* entry + buffer (reduced buffer)
            let sl_buffer = match setup_kind {
                SetupKind::Bounce => 0.3 * atr,   // Behind level (original)
                SetupKind::Breakout => 0.0,        // Tight
            };
            let sl = resistance_levels
                .iter()
                .find(|&&p| p > entry_price)
                .map(|&p| p + sl_buffer)
                .unwrap_or(fb_sl);

            // TP1: nearest support *below* entry, SKIP levels too close (< 0.8 ATR)
            let mut sup_iter = support_levels.iter().rev()
                .filter(|&&p| p < entry_price && (entry_price - p) >= min_tp1_atr_dist);
            let t1 = sup_iter.next().copied().unwrap_or(fb_tp1);
            let t2 = sup_iter.next().copied().unwrap_or(fb_tp2);
            let t3_level = sup_iter.next().copied();
            let t3 = match t3_level {
                Some(lv) => lv.min(fb_tp3),
                None => fb_tp3,
            };

            (sl, t1, t2, t3)
        };

        // ── Enforce minimum AND maximum distances from TfTargets ────────
        let d_sl_min  = entry_price * tf_targets.min_sl_pct;
        let d_sl_max  = entry_price * tf_targets.max_sl_pct;  // NEW: cap SL distance
        let d_tp1_min = entry_price * tf_targets.min_tp1_pct;
        let d_tp2_min = entry_price * tf_targets.min_tp2_pct;
        let d_tp3_min = entry_price * tf_targets.min_tp3_pct;

        let enforce_min = |raw: f64, min_delta: f64, is_tp: bool| -> f64 {
            let actual_delta = (raw - entry_price).abs();
            if actual_delta >= min_delta {
                raw
            } else if is_tp {
                entry_price + side_f * min_delta
            } else {
                // SL is on the opposite side
                entry_price - side_f * min_delta
            }
        };

        let mut stop_loss = enforce_min(stop_loss, d_sl_min, false).max(0.0);
        // Cap SL distance to prevent catastrophic losses
        let actual_sl_dist = (stop_loss - entry_price).abs();
        if actual_sl_dist > d_sl_max {
            stop_loss = if side_f > 0.0 {
                entry_price - d_sl_max
            } else {
                entry_price + d_sl_max
            };
        }
        let tp1 = enforce_min(tp1, d_tp1_min, true).max(0.0);
        let mut tp2 = enforce_min(tp2, d_tp2_min, true).max(0.0);
        let mut tp3 = enforce_min(tp3, d_tp3_min, true).max(0.0);

        // Enforce strict monotonic ordering for TPs
        let gap = entry_price * self.min_tp_gap_pct;
        if side > 0 {
            if tp2 <= tp1 { tp2 = tp1 + gap; }
            if tp3 <= tp2 { tp3 = tp2 + gap; }
        } else {
            if tp2 >= tp1 { tp2 = tp1 - gap; }
            if tp3 >= tp2 { tp3 = tp2 - gap; }
        }

        (stop_loss, tp1, tp2, tp3)
    }

    pub async fn build_trade_signal(
        &self,
        pool: &PgPool,
        scorer: &FinalScorer,
        market_params: Option<&MarketParams>,

        time: DateTime<Utc>,
        time_ms: i64,
        symbol_id: i64,
        symbol: &Symbol,
        tf_minutes: i16,

        entry_price: f64,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<TradeSignal>> {
        let side = infer_side(raw_signals_summary, predictors)
            .or_else(|| infer_side_from_summary_fields(raw_signals_summary))
            .unwrap_or(0);

        if side == 0 {
            // no direction -> no trade
            // Diagnostic: sample how often side=0 rejects
            static SIDE0_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let cnt = SIDE0_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if cnt % 10000 == 0 {
                tracing::debug!(
                    target: "trade_signal_calculator",
                    "side=0 rejection #{} for {} tf={}",
                    cnt, symbol.0, tf_minutes
                );
            }
            return Ok(None);
        }

        // Final scoring (now with side + market)
        let breakdown_opt: Option<FinalScoreBreakdown> = scorer
            .score_signal_verbose(
                symbol.0.as_str(),
                tf_minutes,
                time,
                side as i16,
                raw_signals_summary,
                predictors,
                market_params,
            )
            .await?;

        let breakdown = match breakdown_opt {
            Some(b) => b,
            None => return Ok(None),
        };

        if breakdown.final_score < self.min_final_score {
            return Ok(None);
        }

        let targets = tf_targets(tf_minutes);

        // Entry filter based on price prediction — now a soft check (log + skip)
        // instead of a hard reject.
        // Previously this would reject ALL signals where predicted move was too small,
        // effectively blocking most signals since heuristic predictors often predict conservative moves.
        let price_target_pred = predictors.iter().find(|p| p.aspect == PredictionAspect::PriceTarget);
        if let Some(pred) = price_target_pred {
            let predicted_move_pct = (pred.value - entry_price) / entry_price * side as f64;
            if predicted_move_pct < targets.min_tp1_pct {
                tracing::debug!(
                    target: "trade_signal_calculator",
                    "Price target filter: {} tf={} predicted_move={:.5} < min_tp1={:.5} — proceeding anyway (soft filter)",
                    symbol.0, tf_minutes, predicted_move_pct, targets.min_tp1_pct
                );
                // NOTE: We no longer reject here. The signal still has a valid score.
                // The predicted move is just one factor; TP/SL are calculated from ATR/levels anyway.
            }
        }

        // ATR
        let atr = match extract_atr_from_summary(raw_signals_summary) {
            Some(v) => v,
            None => match fetch_last_atr(pool, symbol_id, tf_minutes).await {
                Ok(Some(v)) => v,
                _ => entry_price * self.fallback_atr_pct,
            },
        };

        let side_f = side as f64;

        // ── Try level-aware TP/SL first ────────────────────────────────
        let (mut support_prices, mut resistance_prices) =
            extract_sr_levels_from_summary(raw_signals_summary);

        // Also extract level prices from predictors (LevelBounce / LevelBreakout rows)
        for p in predictors {
            if let Some(lp) = p.level_price {
                if lp <= 0.0 { continue; }
                match p.level_kind {
                    Some(1) => support_prices.push(lp),     // Support
                    Some(2) => resistance_prices.push(lp),   // Resistance
                    _ => {}
                }
            }
        }

        support_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        support_prices.dedup();
        resistance_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        resistance_prices.dedup();

        let has_levels = !support_prices.is_empty() || !resistance_prices.is_empty();

        // Get setup kind from breakdown (default to Bounce if not available)
        let setup_kind = breakdown.setup_kind;

        let (stop_loss, tp1, tp2, tp3) = if has_levels {
            // ── Level-aware path ───────────────────────────────────────
            self.level_aware_targets(
                entry_price,
                side,
                atr,
                &support_prices,
                &resistance_prices,
                &targets,
                setup_kind,
            )
        } else {
            // ── Setup-aware ATR-only path ──────────────────────────────
            let (sl_mult, tp1_mult, tp2_mult, tp3_mult) = match setup_kind {
                SetupKind::Bounce => (
                    self.sl_atr_mult_bounce,
                    self.tp1_atr_mult_bounce,
                    self.tp2_atr_mult_bounce,
                    self.tp3_atr_mult_bounce,
                ),
                SetupKind::Breakout => (
                    self.sl_atr_mult_breakout,
                    self.tp1_atr_mult_breakout,
                    self.tp2_atr_mult_breakout,
                    self.tp3_atr_mult_breakout,
                ),
            };

            let d_sl_atr  = sl_mult * atr;
            let d_tp1_atr = tp1_mult * atr;
            let d_tp2_atr = tp2_mult * atr;
            let d_tp3_atr = tp3_mult * atr;

            let d_sl_min  = entry_price * targets.min_sl_pct;
            let d_sl_max  = entry_price * targets.max_sl_pct;  // NEW: cap SL
            let d_tp1_min = entry_price * targets.min_tp1_pct;
            let d_tp2_min = entry_price * targets.min_tp2_pct;
            let d_tp3_min = entry_price * targets.min_tp3_pct;

            let s = breakdown.final_score.clamp(0.0, 1.0);
            let boost_tp1 = (0.98 + 0.10 * s).clamp(0.98, 1.08);
            let boost_tp2 = (0.98 + 0.22 * s).clamp(1.00, 1.20);
            let boost_tp3 = (0.98 + 0.38 * s).clamp(1.05, 1.35);

            // Apply SL cap: min(ATR-based, max_sl_pct)
            let d_sl  = d_sl_atr.max(d_sl_min).min(d_sl_max);
            let d_tp1 = (d_tp1_atr.max(d_tp1_min)) * boost_tp1;
            let mut d_tp2 = (d_tp2_atr.max(d_tp2_min)) * boost_tp2;
            let mut d_tp3 = (d_tp3_atr.max(d_tp3_min)) * boost_tp3;

            let gap = entry_price * self.min_tp_gap_pct;
            d_tp2 = d_tp2.max(d_tp1 * self.tp2_ratio_min).max(d_tp1 + gap);
            d_tp3 = d_tp3.max(d_tp2 * self.tp3_ratio_min).max(d_tp2 + gap);

            let stop_loss = (entry_price - side_f * d_sl).max(0.0);
            let tp1 = (entry_price + side_f * d_tp1).max(0.0);
            let tp2 = (entry_price + side_f * d_tp2).max(0.0);
            let tp3 = (entry_price + side_f * d_tp3).max(0.0);

            (stop_loss, tp1, tp2, tp3)
        };

        // ────────────────────────────────────────────────────────────────
        // CRITICAL: Enforce minimum Risk:Reward ratio (R:R ≥ 1.5:1)
        // Reject signals where TP1 distance < SL distance × min_rr
        // This is THE most important filter for positive PnL.
        // ────────────────────────────────────────────────────────────────
        let tp1_dist = (tp1 - entry_price).abs();
        let sl_dist = (stop_loss - entry_price).abs();
        let risk_reward = if sl_dist > 0.0 { tp1_dist / sl_dist } else { 0.0 };

        if risk_reward < targets.min_rr {
            // Diagnostic logging
            static RR_REJECT_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let cnt = RR_REJECT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if cnt % 5000 == 0 {
                tracing::info!(
                    target: "trade_signal_calculator",
                    "R:R rejection #{}: {} tf={} R:R={:.2} < min={:.2} (tp1_dist={:.6} sl_dist={:.6})",
                    cnt, symbol.0, tf_minutes, risk_reward, targets.min_rr, tp1_dist, sl_dist
                );
            }
            return Ok(None);
        }

        // Leverage = base * market_factor * score_factor
        let market_factor = market_params
            .map(|m| m.leverage_factor(side))
            .unwrap_or(0.6); // if no market params -> conservative

        let score_factor = (0.55 + 0.45 * breakdown.final_score).clamp(0.55, 1.0);

        let lev = ((self.base_leverage as f64) * market_factor * score_factor)
            .round()
            .clamp(1.0, self.base_leverage as f64) as i16;

        let atr_pct = if entry_price > 0.0 { atr / entry_price } else { 0.0 };

        let breakdown_json = json!({
            "final": breakdown.final_score,
            "base_score": breakdown.base_score,
            "coverage": breakdown.coverage_score,
            "consensus": breakdown.consensus_score,
            "predictors_score": breakdown.predictors_score,
            "raw_signals_score": breakdown.raw_signals_score,
            "indicators_score": breakdown.indicators_score,
            "market_score": breakdown.market_score,
            "market_factor": market_factor,
            "score_factor": score_factor,
            "atr": atr,
            "atr_pct": atr_pct,
            "risk_reward": risk_reward,
            "level_aware": has_levels,
            "support_levels_used": support_prices,
            "resistance_levels_used": resistance_prices,
            "setup_kind": format!("{:?}", breakdown.setup_kind),
            "setup_confidence": breakdown.setup_confidence,
            "bounce_prob": breakdown.bounce_prob,
            "breakout_prob": breakdown.breakout_prob,
            "market_params": market_params.map(|m| m.details_json.clone()),
            "debug": breakdown.debug,
        });

        Ok(Some(TradeSignal {
            time,
            time_ms,
            symbol_id,
            symbol: symbol.0.clone(),
            tf_minutes,
            side,
            entry: entry_price,
            stop_loss,
            tp1,
            tp2,
            tp3,
            leverage: lev,
            final_score: breakdown.final_score,
            breakdown_json,
        }))
    }
}

fn infer_side(raw_signals_summary: &Value, predictors: &[PredictionRow]) -> Option<i8> {
    // 1) raw summary explicit
    if let Some(s) = raw_signals_summary.get("side").and_then(|v| v.as_i64()) {
        let ss = (s as i8).signum();
        if ss != 0 { return Some(ss); }
    }

    // 2) Majority voting with score weighting
    // Count weighted votes for long vs short
    let mut long_score: f32 = 0.0;
    let mut short_score: f32 = 0.0;
    let mut vote_count: usize = 0;

    for p in predictors {
        let sc = p.score_norm;
        let sd = p.side.unwrap_or(0);
        if sd == 0 { continue; }

        vote_count += 1;
        if sd > 0 {
            long_score += sc;
        } else {
            short_score += sc;
        }
    }

    // Need at least 2 predictors voting
    if vote_count < 2 {
        return None;
    }

    // Return direction with higher weighted score
    // Require at least 60% agreement to avoid ambiguous signals
    let total = long_score + short_score;
    if total <= 0.0 {
        return None;
    }

    let long_ratio = long_score / total;
    let short_ratio = short_score / total;

    if long_ratio >= 0.6 {
        Some(1)
    } else if short_ratio >= 0.6 {
        Some(-1)
    } else {
        // Ambiguous - use simple majority
        if long_score > short_score {
            Some(1)
        } else if short_score > long_score {
            Some(-1)
        } else {
            None
        }
    }
}

fn infer_side_from_summary_fields(raw_signals_summary: &Value) -> Option<i8> {
    // optional: if you store dominant_side or similar
    raw_signals_summary
        .get("dominant_side")
        .and_then(|v| v.as_i64())
        .map(|x| (x as i8).signum())
}

fn extract_atr_from_summary(raw_signals_summary: &Value) -> Option<f64> {
    raw_signals_summary
        .get("atr")
        .and_then(|v| v.as_f64())
        .or_else(|| raw_signals_summary.get("atr_value").and_then(|v| v.as_f64()))
}

async fn fetch_last_atr(pool: &PgPool, symbol_id: i64, tf_minutes: i16) -> Result<Option<f64>> {
    let row = sqlx::query(
        r#"
        SELECT atr
        FROM market.indicators_wide
        WHERE symbol_id = $1 AND tf_minutes = $2
        ORDER BY time DESC
        LIMIT 1
        "#,
    )
    .bind(symbol_id)
    .bind(tf_minutes)
    .fetch_optional(pool)
    .await
    .with_context(|| format!("fetch_last_atr sym_id={} tf={}", symbol_id, tf_minutes))?;

    Ok(row.and_then(|r| r.try_get::<Option<f64>, _>("atr").ok()).flatten())
}

/// Extracts support and resistance level prices from `raw_signals_summary` JSON.
///
/// Looks for the SRLLevels-style keys (`strong_support`, `mid_support`, …)
/// and returns two sorted-ascending vectors: (supports, resistances).
fn extract_sr_levels_from_summary(raw_signals_summary: &Value) -> (Vec<f64>, Vec<f64>) {
    let mut supports = Vec::new();
    let mut resistances = Vec::new();

    // Try nested "sr_levels" object first
    let sr = raw_signals_summary
        .get("sr_levels")
        .unwrap_or(raw_signals_summary);

    for key in ["strong_support", "mid_support", "light_support"] {
        if let Some(p) = sr.get(key).and_then(|v| v.as_f64()) {
            if p.is_finite() && p > 0.0 {
                supports.push(p);
            }
        }
    }
    for key in ["strong_resistance", "mid_resistance", "light_resistance"] {
        if let Some(p) = sr.get(key).and_then(|v| v.as_f64()) {
            if p.is_finite() && p > 0.0 {
                resistances.push(p);
            }
        }
    }

    supports.sort_by(|a, b| a.partial_cmp(b).unwrap());
    resistances.sort_by(|a, b| a.partial_cmp(b).unwrap());

    (supports, resistances)
}
