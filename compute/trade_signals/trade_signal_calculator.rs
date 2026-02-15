// calculates final signal based on all previous steps - indicators, raw_signals, ml predictors, heruistic predictors, market_parameter (volatility, trend)
//signal must contain Symbol (traiding pair) time created, timeframe (1m, 5m, 15m, 1h, 4h, 1d), stop_loss, take_profits_1,2,3, side (short/long), combined final score, scores from prevous steps like raw_score, ind_score, ml_score, heruistic_score, 

// compute/scoring/trade_signal_calculator.rs

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use crate::predictors::types::PredictionRow;

use super::final_score::{FinalScorer, FinalScoreBreakdown};
use super::market_params_calculator::MarketParams;

#[derive(Debug, Clone, Copy)]
struct TfTargets {
    min_tp1_pct: f64,
    min_tp2_pct: f64,
    min_tp3_pct: f64,
    min_sl_pct:  f64,
}

fn tf_targets(tf_minutes: i16) -> TfTargets {
    let mut t = match tf_minutes {
        1 => TfTargets {  // 1m
            min_tp1_pct: 0.010, min_tp2_pct: 0.018, min_tp3_pct: 0.026, min_sl_pct: 0.007
        },
        5 => TfTargets {  // 5m
            min_tp1_pct: 0.012, min_tp2_pct: 0.022, min_tp3_pct: 0.032, min_sl_pct: 0.008
        },
        15 => TfTargets { // 15m
            min_tp1_pct: 0.015, min_tp2_pct: 0.028, min_tp3_pct: 0.040, min_sl_pct: 0.010
        },
        60 => TfTargets { // 1h
            min_tp1_pct: 0.020, min_tp2_pct: 0.038, min_tp3_pct: 0.055, min_sl_pct: 0.013
        },
        240 => TfTargets { // 4h
            min_tp1_pct: 0.030, min_tp2_pct: 0.055, min_tp3_pct: 0.080, min_sl_pct: 0.018
        },
        1440 => TfTargets { // 1d
            min_tp1_pct: 0.050, min_tp2_pct: 0.090, min_tp3_pct: 0.130, min_sl_pct: 0.028
        },
        _ => TfTargets { // default conservative
            min_tp1_pct: 0.015, min_tp2_pct: 0.028, min_tp3_pct: 0.040, min_sl_pct: 0.010
        },
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

    /// Risk parameters
    pub sl_atr_mult: f64,
    pub tp1_atr_mult: f64,
    pub tp2_atr_mult: f64,
    pub tp3_atr_mult: f64,

    pub tp2_ratio_min: f64,
    pub tp3_ratio_min: f64,
    pub min_tp_gap_pct: f64,

    /// Fallback if ATR missing: ATR = entry * fallback_atr_pct
    pub fallback_atr_pct: f64,
}

impl Default for TradeSignalCalculator {
    fn default() -> Self {
        Self {
            min_final_score: 0.95,
            base_leverage: 5,
            sl_atr_mult: 1.6,
            tp1_atr_mult: 1.2,
            tp2_atr_mult: 2.2,
            tp3_atr_mult: 3.4,
            tp2_ratio_min: 1.6,
            tp3_ratio_min: 1.45,
            min_tp_gap_pct: 0.002,
            fallback_atr_pct: 0.008, // 0.8%
        }
    }
}

impl TradeSignalCalculator {
    pub fn new(min_final_score: f64) -> Self {
        Self { min_final_score, ..Default::default() }
    }

    /// Level-aware TP/SL calculation.
    ///
    /// For **LONG** (`side = +1`):
    /// - SL  = nearest support below entry − 0.3 × ATR
    /// - TP1 = nearest resistance above entry
    /// - TP2 = next resistance above TP1
    /// - TP3 = max(ATR-based, next resistance above TP2)
    ///
    /// For **SHORT** (`side = -1`):
    /// - SL  = nearest resistance above entry + 0.3 × ATR
    /// - TP1 = nearest support below entry
    /// - TP2 = next support below TP1
    /// - TP3 = min(ATR-based, next support below TP2)
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
    ) -> (f64, f64, f64, f64) {
        let side_f = side as f64;

        // ATR-based fallbacks
        let fb_sl  = entry_price - side_f * self.sl_atr_mult  * atr;
        let fb_tp1 = entry_price + side_f * self.tp1_atr_mult * atr;
        let fb_tp2 = entry_price + side_f * self.tp2_atr_mult * atr;
        let fb_tp3 = entry_price + side_f * self.tp3_atr_mult * atr;

        let (stop_loss, tp1, tp2, tp3) = if side > 0 {
            // ── LONG ───────────────────────────────────────────────────
            // SL: nearest support *below* entry − 0.3 ATR
            let sl = support_levels
                .iter()
                .rev() // descending
                .find(|&&p| p < entry_price)
                .map(|&p| p - 0.3 * atr)
                .unwrap_or(fb_sl);

            // TP1: nearest resistance *above* entry
            let mut res_iter = resistance_levels.iter().filter(|&&p| p > entry_price);
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
            // SL: nearest resistance *above* entry + 0.3 ATR
            let sl = resistance_levels
                .iter()
                .find(|&&p| p > entry_price)
                .map(|&p| p + 0.3 * atr)
                .unwrap_or(fb_sl);

            // TP1: nearest support *below* entry (descending order)
            let mut sup_iter = support_levels.iter().rev().filter(|&&p| p < entry_price);
            let t1 = sup_iter.next().copied().unwrap_or(fb_tp1);
            let t2 = sup_iter.next().copied().unwrap_or(fb_tp2);
            let t3_level = sup_iter.next().copied();
            let t3 = match t3_level {
                Some(lv) => lv.min(fb_tp3),
                None => fb_tp3,
            };

            (sl, t1, t2, t3)
        };

        // ── Enforce minimum distances from TfTargets ───────────────────
        let d_sl_min  = entry_price * tf_targets.min_sl_pct;
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

        let stop_loss = enforce_min(stop_loss, d_sl_min, false).max(0.0);
        let mut tp1 = enforce_min(tp1, d_tp1_min, true).max(0.0);
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

        // Entry filter based on price prediction
        let price_target_pred = predictors.iter().find(|p| p.aspect == crate::predictors::types::PredictionAspect::PriceTarget);
        if let Some(pred) = price_target_pred {
            let predicted_move_pct = (pred.value - entry_price) / entry_price * side as f64;
            if predicted_move_pct < targets.min_tp1_pct {
                return Ok(None); // Not enough expected profit
            }
        }

        // ATR
        let atr = extract_atr_from_summary(raw_signals_summary)
            .or_else(|| futures::executor::block_on(fetch_last_atr(pool, symbol_id, tf_minutes)).ok().flatten())
            .unwrap_or(entry_price * self.fallback_atr_pct);

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

        let (stop_loss, tp1, tp2, tp3) = if has_levels {
            // ── Level-aware path ───────────────────────────────────────
            self.level_aware_targets(
                entry_price,
                side,
                atr,
                &support_prices,
                &resistance_prices,
                &targets,
            )
        } else {
            // ── Original ATR-only path ─────────────────────────────────
            let d_sl_atr  = self.sl_atr_mult  * atr;
            let d_tp1_atr = self.tp1_atr_mult * atr;
            let d_tp2_atr = self.tp2_atr_mult * atr;
            let d_tp3_atr = self.tp3_atr_mult * atr;

            let d_sl_min  = entry_price * targets.min_sl_pct;
            let d_tp1_min = entry_price * targets.min_tp1_pct;
            let d_tp2_min = entry_price * targets.min_tp2_pct;
            let d_tp3_min = entry_price * targets.min_tp3_pct;

            let s = breakdown.final_score.clamp(0.0, 1.0);
            let boost_tp1 = (0.98 + 0.10 * s).clamp(0.98, 1.08);
            let boost_tp2 = (0.98 + 0.22 * s).clamp(1.00, 1.20);
            let boost_tp3 = (0.98 + 0.38 * s).clamp(1.05, 1.35);

            let d_sl  = d_sl_atr.max(d_sl_min);
            let mut d_tp1 = (d_tp1_atr.max(d_tp1_min)) * boost_tp1;
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

        // Leverage = base * market_factor * score_factor
        let market_factor = market_params
            .map(|m| m.leverage_factor(side))
            .unwrap_or(0.6); // if no market params -> conservative

        let score_factor = (0.55 + 0.45 * breakdown.final_score).clamp(0.55, 1.0);

        let lev = ((self.base_leverage as f64) * market_factor * score_factor)
            .round()
            .clamp(1.0, self.base_leverage as f64) as i16;

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
            "level_aware": has_levels,
            "support_levels_used": support_prices,
            "resistance_levels_used": resistance_prices,
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

    // 2) best prediction by score_norm
    let mut best: Option<(f32, i16)> = None;
    for p in predictors {
        let sc = p.score_norm;
        let sd = p.side;
        if sd == 0 { continue; }
        match best {
            None => best = Some((sc, sd)),
            Some((bsc, _)) if sc > bsc => best = Some((sc, sd)),
            _ => {}
        }
    }

    best.map(|(_, sd)| (sd.signum() as i8))
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
