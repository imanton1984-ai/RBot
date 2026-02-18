// strategies/level_heuristic_strategy/src/trade_signal_calculator.rs

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use predictors::types::{PredictionRow, PredictionAspect, CalcSource};

use crate::scorer::{LevelHeuristicScorer, LevelHeuristicScoreBreakdown, SetupKind};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeSignal {
    pub time: DateTime<Utc>,
    pub time_ms: i64,
    pub symbol_id: i64,
    pub symbol: String,
    pub tf_minutes: i16,
    pub side: i8,
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
    pub fallback_atr_pct: f64,
    pub sl_atr_mult: f64,
    pub tp1_atr_mult: f64,
    pub tp2_atr_mult: f64,
    pub tp3_atr_mult: f64,
}

impl Default for TradeSignalCalculator {
    fn default() -> Self {
        Self {
            min_final_score: 0.55,
            base_leverage: 5,
            fallback_atr_pct: 0.008,
            sl_atr_mult: 0.75,
            tp1_atr_mult: 1.1,
            tp2_atr_mult: 1.8,
            tp3_atr_mult: 2.8,
        }
    }
}

impl TradeSignalCalculator {
    pub fn new(min_final_score: f64) -> Self {
        Self { min_final_score, ..Default::default() }
    }

    pub async fn build_trade_signal(
        &self,
        pool: &PgPool,
        scorer: &LevelHeuristicScorer,
        time: DateTime<Utc>,
        time_ms: i64,
        symbol_id: i64,
        symbol: &Symbol,
        tf_minutes: i16,
        entry_price: f64,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<TradeSignal>> {
        let side = infer_side_heuristic(predictors)
            .or_else(|| infer_side_from_summary(raw_signals_summary))
            .unwrap_or(0);

        if side == 0 {
            return Ok(None);
        }

        let breakdown_opt: Option<LevelHeuristicScoreBreakdown> = scorer
            .score_signal_verbose(
                symbol.0.as_str(),
                tf_minutes,
                time,
                side as i16,
                raw_signals_summary,
                predictors,
            )
            .await?;

        let breakdown = match breakdown_opt {
            Some(b) => b,
            None => return Ok(None),
        };

        if breakdown.final_score < self.min_final_score {
            return Ok(None);
        }

        // Heuristic price target
        let heuristic_price_target = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::PriceTarget && p.calc_source == CalcSource::Hard)
            .map(|p| p.value);

        let atr = match extract_atr_from_summary(raw_signals_summary) {
            Some(v) => v,
            None => match fetch_last_atr(pool, symbol_id, tf_minutes).await {
                Ok(Some(v)) => v,
                _ => entry_price * self.fallback_atr_pct,
            },
        };

        let (stop_loss, tp1, tp2, tp3) = self.calculate_tp_sl(
            entry_price,
            side,
            atr,
            heuristic_price_target,
            &breakdown,
        );

        let tp1_dist = (tp1 - entry_price).abs();
        let sl_dist = (stop_loss - entry_price).abs();
        let risk_reward = if sl_dist > 0.0 { tp1_dist / sl_dist } else { 0.0 };

        if risk_reward < 1.5 {
            return Ok(None);
        }

        let score_factor = (0.55 + 0.45 * breakdown.final_score).clamp(0.55, 1.0);
        let lev = ((self.base_leverage as f64) * score_factor)
            .round()
            .clamp(1.0, self.base_leverage as f64) as i16;

        let atr_pct = if entry_price > 0.0 { atr / entry_price } else { 0.0 };

        let breakdown_json = json!({
            "final": breakdown.final_score,
            "heuristic_predictors_score": breakdown.heuristic_predictors_score,
            "raw_signals_score": breakdown.raw_signals_score,
            "indicators_score": breakdown.indicators_score,
            "setup_kind": format!("{:?}", breakdown.setup_kind),
            "setup_confidence": breakdown.setup_confidence,
            "atr": atr,
            "atr_pct": atr_pct,
            "risk_reward": risk_reward,
            "heuristic_price_target": heuristic_price_target,
            "strategy_id": 5,
            "strategy_name": "level_heuristic",
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

    fn calculate_tp_sl(
        &self,
        entry_price: f64,
        side: i8,
        atr: f64,
        heuristic_price_target: Option<f64>,
        _breakdown: &LevelHeuristicScoreBreakdown,
    ) -> (f64, f64, f64, f64) {
        let side_f = side as f64;

        let target_tp1 = if let Some(target) = heuristic_price_target {
            let predicted_move = (target - entry_price) / entry_price;
            if predicted_move * side_f > 0.0 {
                target
            } else {
                entry_price + side_f * atr * self.tp1_atr_mult
            }
        } else {
            entry_price + side_f * atr * self.tp1_atr_mult
        };

        let min_tp1 = entry_price + side_f * entry_price * 0.015;
        let tp1 = if side > 0 {
            target_tp1.max(min_tp1)
        } else {
            target_tp1.min(min_tp1)
        };

        let tp1_dist = (tp1 - entry_price).abs();
        let tp2 = entry_price + side_f * tp1_dist * 1.6;
        let tp3 = entry_price + side_f * tp1_dist * 2.5;

        let sl_atr = entry_price - side_f * atr * self.sl_atr_mult;
        let sl_min = entry_price - side_f * entry_price * 0.01;
        let sl_max = entry_price - side_f * entry_price * 0.03;

        let mut stop_loss = if side > 0 {
            sl_atr.min(sl_min)
        } else {
            sl_atr.max(sl_min)
        };

        if side > 0 {
            stop_loss = stop_loss.max(sl_max);
        } else {
            stop_loss = stop_loss.min(sl_max);
        }

        let d_sl_min = entry_price * 0.01;
        let d_tp1_min = entry_price * 0.015;

        let enforce_min = |raw: f64, min_delta: f64, is_tp: bool| -> f64 {
            let actual_delta = (raw - entry_price).abs();
            if actual_delta >= min_delta {
                raw
            } else if is_tp {
                entry_price + side_f * min_delta
            } else {
                entry_price - side_f * min_delta
            }
        };

        stop_loss = enforce_min(stop_loss, d_sl_min, false).max(0.0);
        let tp1 = enforce_min(tp1, d_tp1_min, true).max(0.0);

        let gap = entry_price * 0.002;
        let tp2 = tp2.max(if side > 0 { tp1 + gap } else { tp1 - gap });
        let tp3 = tp3.max(if side > 0 { tp2 + gap } else { tp2 - gap });

        (stop_loss, tp1, tp2, tp3)
    }
}

fn infer_side_heuristic(predictors: &[PredictionRow]) -> Option<i8> {
    let heuristic_predictors: Vec<&PredictionRow> = predictors.iter()
        .filter(|p| p.calc_source == CalcSource::Hard)
        .collect();

    if heuristic_predictors.is_empty() {
        return None;
    }

    let mut long_score: f32 = 0.0;
    let mut short_score: f32 = 0.0;

    for p in &heuristic_predictors {
        let sc = p.score_norm;
        let sd = p.side.unwrap_or(0);
        if sd == 0 { continue; }

        if sd > 0 { long_score += sc; } else { short_score += sc; }
    }

    let total = long_score + short_score;
    if total <= 0.0 { return None; }

    let long_ratio = long_score / total;
    if long_ratio >= 0.6 { Some(1) }
    else if long_ratio <= 0.4 { Some(-1) }
    else { None }
}

fn infer_side_from_summary(raw_signals_summary: &Value) -> Option<i8> {
    raw_signals_summary.get("side").and_then(|v| v.as_i64()).map(|x| (x as i8).signum())
}

fn extract_atr_from_summary(raw_signals_summary: &Value) -> Option<f64> {
    raw_signals_summary.get("atr").and_then(|v| v.as_f64())
}

async fn fetch_last_atr(pool: &PgPool, symbol_id: i64, tf_minutes: i16) -> Result<Option<f64>> {
    let row = sqlx::query(
        r#"SELECT atr FROM market.indicators_wide WHERE symbol_id = $1 AND tf_minutes = $2 AND atr IS NOT NULL ORDER BY time DESC LIMIT 1"#,
    ).bind(symbol_id).bind(tf_minutes as i32).fetch_optional(pool).await?;
    Ok(row.and_then(|r| r.try_get::<f64, _>(0).ok()))
}
