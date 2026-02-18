// strategies/indicator_consensus_strategy/src/trade_signal_calculator.rs

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use predictors::types::{PredictionRow, PredictionAspect, CalcSource};

use crate::scorer::{IndicatorConsensusScorer, IndicatorConsensusScoreBreakdown};

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
}

impl Default for TradeSignalCalculator {
    fn default() -> Self {
        Self {
            min_final_score: 0.55,
            base_leverage: 5,
            fallback_atr_pct: 0.008,
        }
    }
}

impl TradeSignalCalculator {
    pub fn new(min_final_score: f64) -> Self {
        Self { min_final_score, ..Default::default() }
    }

    fn calculate_tp_sl(
        &self,
        entry_price: f64,
        side: i8,
        atr: f64,
        consensus_price_target: Option<f64>,
        min_tp1_pct: f64,
        min_sl_pct: f64,
        max_sl_pct: f64,
    ) -> (f64, f64, f64, f64) {
        let side_f = side as f64;
        
        // Consensus TP = average of ML and heuristic if both exist
        // SANITY CHECK: cap target at reasonable distance (10 ATR max)
        let consensus_tp1 = if let Some(target) = consensus_price_target {
            let max_ml_dist = atr * 10.0;
            let target_dist = (target - entry_price).abs();
            let predicted_move = (target - entry_price) / entry_price;
            
            if predicted_move * side_f > 0.0 && target_dist <= max_ml_dist {
                target
            } else if predicted_move * side_f > 0.0 && target_dist > max_ml_dist {
                // Target unreasonably far → cap
                entry_price + side_f * max_ml_dist
            } else {
                // Wrong direction → ATR fallback
                entry_price + side_f * atr * 1.0
            }
        } else {
            entry_price + side_f * atr * 1.0
        };
        
        let min_tp1 = entry_price + side_f * entry_price * min_tp1_pct;
        
        let tp1 = if side > 0 {
            consensus_tp1.max(min_tp1)
        } else {
            consensus_tp1.min(min_tp1)
        };
        
        let tp1_dist = (tp1 - entry_price).abs();
        let tp2 = entry_price + side_f * tp1_dist * 1.6;
        let tp3 = entry_price + side_f * tp1_dist * 2.5;
        
        let sl_atr = entry_price - side_f * atr * 0.75;
        let sl_min = entry_price - side_f * entry_price * min_sl_pct;
        let sl_max = entry_price - side_f * entry_price * max_sl_pct;
        
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
        
        let d_sl_min = entry_price * min_sl_pct;
        let d_tp1_min = entry_price * min_tp1_pct;
        let d_tp2_min = entry_price * min_tp1_pct * 1.6;
        let d_tp3_min = entry_price * min_tp1_pct * 2.5;
        
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
        let mut tp2 = enforce_min(tp2, d_tp2_min, true).max(0.0);
        let mut tp3 = enforce_min(tp3, d_tp3_min, true).max(0.0);
        
        let gap = entry_price * 0.002;
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
        scorer: &IndicatorConsensusScorer,
        time: DateTime<Utc>,
        time_ms: i64,
        symbol_id: i64,
        symbol: &Symbol,
        tf_minutes: i16,
        entry_price: f64,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<TradeSignal>> {
        let side = infer_side_consensus(predictors)
            .or_else(|| infer_side_from_summary_fields(raw_signals_summary))
            .unwrap_or(0);

        if side == 0 {
            return Ok(None);
        }

        let breakdown_opt: Option<IndicatorConsensusScoreBreakdown> = scorer
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

        // Consensus price target = average of ML and heuristic
        // SANITY CHECK: validate ML target before averaging (ML may return garbage)
        let ml_target_raw = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::PriceTarget && p.calc_source == CalcSource::Ml)
            .map(|p| p.value);
        let heuristic_target = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::PriceTarget && p.calc_source == CalcSource::Hard)
            .map(|p| p.value);
        
        // Filter ML target: reject if deviation from entry exceeds 20%
        let ml_target = ml_target_raw.filter(|&v| {
            let deviation = (v - entry_price).abs() / entry_price;
            deviation <= 0.20 && v > 0.0
        });
        
        let consensus_price_target = match (ml_target, heuristic_target) {
            (Some(ml), Some(hc)) => Some((ml + hc) / 2.0),
            (Some(ml), None) => Some(ml),
            (None, Some(hc)) => Some(hc),
            (None, None) => None,
        };

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
            consensus_price_target,
            0.015,
            0.01,
            0.03,
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
            "consensus_prediction_score": breakdown.consensus_prediction_score,
            "ml_score": breakdown.ml_score,
            "heuristic_score": breakdown.heuristic_score,
            "consensus_agreement": breakdown.consensus_agreement,
            "indicator_score": breakdown.indicator_score,
            "rsi_score": breakdown.rsi_score,
            "macd_score": breakdown.macd_score,
            "stoch_score": breakdown.stoch_score,
            "adx_score": breakdown.adx_score,
            "atr_score": breakdown.atr_score,
            "volume_score": breakdown.volume_score,
            "trend_score": breakdown.trend_score,
            "setup_kind": format!("{:?}", breakdown.setup_kind),
            "setup_confidence": breakdown.setup_confidence,
            "atr": atr,
            "atr_pct": atr_pct,
            "risk_reward": risk_reward,
            "consensus_price_target": consensus_price_target,
            "strategy_id": 3,
            "strategy_name": "indicator_consensus",
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

fn infer_side_consensus(predictors: &[PredictionRow]) -> Option<i8> {
    let mut long_ml: f32 = 0.0;
    let mut short_ml: f32 = 0.0;
    let mut long_hc: f32 = 0.0;
    let mut short_hc: f32 = 0.0;
    
    for p in predictors {
        let sc = p.score_norm;
        let sd = p.side.unwrap_or(0);
        if sd == 0 { continue; }

        match p.calc_source {
            CalcSource::Ml => {
                if sd > 0 { long_ml += sc; } else { short_ml += sc; }
            }
            CalcSource::Hard => {
                if sd > 0 { long_hc += sc; } else { short_hc += sc; }
            }
        }
    }
    
    // Consensus: both ML and heuristic must agree
    let ml_side = if long_ml > short_ml { 1 } else if short_ml > long_ml { -1 } else { 0 };
    let hc_side = if long_hc > short_hc { 1 } else if short_hc > long_hc { -1 } else { 0 };
    
    // Return side only if both agree
    if ml_side == hc_side && ml_side != 0 {
        Some(ml_side)
    } else if ml_side != 0 {
        Some(ml_side)
    } else if hc_side != 0 {
        Some(hc_side)
    } else {
        None
    }
}

fn infer_side_from_summary_fields(raw_signals_summary: &Value) -> Option<i8> {
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
