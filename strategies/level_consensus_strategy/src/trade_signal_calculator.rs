// strategies/level_consensus_strategy/src/trade_signal_calculator.rs

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use predictors::types::{PredictionRow, PredictionAspect};

use crate::scorer::{LevelConsensusScorer, LevelConsensusScoreBreakdown, SetupKind};

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
    
    // Bounce parameters
    pub sl_atr_mult_bounce: f64,
    pub tp1_atr_mult_bounce: f64,
    pub tp2_atr_mult_bounce: f64,
    pub tp3_atr_mult_bounce: f64,
    
    // Breakout parameters
    pub sl_atr_mult_breakout: f64,
    pub tp1_atr_mult_breakout: f64,
    pub tp2_atr_mult_breakout: f64,
    pub tp3_atr_mult_breakout: f64,
}

impl Default for TradeSignalCalculator {
    fn default() -> Self {
        Self {
            min_final_score: 0.55,
            base_leverage: 5,
            fallback_atr_pct: 0.008,
            sl_atr_mult_bounce: 0.55,
            tp1_atr_mult_bounce: 0.75,
            tp2_atr_mult_bounce: 1.4,
            tp3_atr_mult_bounce: 2.2,
            sl_atr_mult_breakout: 0.75,
            tp1_atr_mult_breakout: 1.1,
            tp2_atr_mult_breakout: 1.8,
            tp3_atr_mult_breakout: 2.8,
        }
    }
}

impl TradeSignalCalculator {
    pub fn new(min_final_score: f64) -> Self {
        Self { min_final_score, ..Default::default() }
    }

    fn level_aware_targets(
        &self,
        entry_price: f64,
        side: i8,
        atr: f64,
        support_levels: &[f64],
        resistance_levels: &[f64],
        setup_kind: SetupKind,
    ) -> (f64, f64, f64, f64) {
        let side_f = side as f64;

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

        let fb_sl = entry_price - side_f * sl_mult * atr;
        let fb_tp1 = entry_price + side_f * tp1_mult * atr;
        let fb_tp2 = entry_price + side_f * tp2_mult * atr;
        let fb_tp3 = entry_price + side_f * tp3_mult * atr;

        let min_tp1_atr_dist = 0.5 * atr;

        if side > 0 {
            let sl_buffer = match setup_kind {
                SetupKind::Bounce => 0.3 * atr,
                SetupKind::Breakout => 0.0,
            };
            let sl = support_levels.iter().rev()
                .find(|&&p| p < entry_price)
                .map(|&p| p - sl_buffer)
                .unwrap_or(fb_sl);

            let mut res_iter = resistance_levels.iter()
                .filter(|&&p| p > entry_price && (p - entry_price) >= min_tp1_atr_dist);
            let t1 = res_iter.next().copied().unwrap_or(fb_tp1);
            let t2 = res_iter.next().copied().unwrap_or(fb_tp2);
            let t3 = res_iter.next().copied().unwrap_or(fb_tp3);

            (sl, t1, t2, t3)
        } else {
            let sl_buffer = match setup_kind {
                SetupKind::Bounce => 0.3 * atr,
                SetupKind::Breakout => 0.0,
            };
            let sl = resistance_levels.iter()
                .find(|&&p| p > entry_price)
                .map(|&p| p + sl_buffer)
                .unwrap_or(fb_sl);

            let mut sup_iter = support_levels.iter().rev()
                .filter(|&&p| p < entry_price && (entry_price - p) >= min_tp1_atr_dist);
            let t1 = sup_iter.next().copied().unwrap_or(fb_tp1);
            let t2 = sup_iter.next().copied().unwrap_or(fb_tp2);
            let t3 = sup_iter.next().copied().unwrap_or(fb_tp3);

            (sl, t1, t2, t3)
        }
    }

    pub async fn build_trade_signal(
        &self,
        pool: &PgPool,
        scorer: &LevelConsensusScorer,
        time: DateTime<Utc>,
        time_ms: i64,
        symbol_id: i64,
        symbol: &Symbol,
        tf_minutes: i16,
        entry_price: f64,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<TradeSignal>> {
        let side = infer_side(predictors)
            .or_else(|| infer_side_from_summary(raw_signals_summary))
            .unwrap_or(0);

        if side == 0 {
            return Ok(None);
        }

        let breakdown_opt: Option<LevelConsensusScoreBreakdown> = scorer
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

        let atr = match extract_atr_from_summary(raw_signals_summary) {
            Some(v) => v,
            None => match fetch_last_atr(pool, symbol_id, tf_minutes).await {
                Ok(Some(v)) => v,
                _ => entry_price * self.fallback_atr_pct,
            },
        };

        let (mut support_prices, mut resistance_prices) = extract_sr_levels(raw_signals_summary);
        
        for p in predictors {
            if let Some(lp) = p.level_price {
                if lp <= 0.0 { continue; }
                match p.level_kind {
                    Some(1) => support_prices.push(lp),
                    Some(2) => resistance_prices.push(lp),
                    _ => {}
                }
            }
        }

        support_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        support_prices.dedup();
        resistance_prices.sort_by(|a, b| a.partial_cmp(b).unwrap());
        resistance_prices.dedup();

        let has_levels = !support_prices.is_empty() || !resistance_prices.is_empty();
        let setup_kind = breakdown.setup_kind;

        let (stop_loss, tp1, tp2, tp3) = if has_levels {
            self.level_aware_targets(
                entry_price,
                side,
                atr,
                &support_prices,
                &resistance_prices,
                setup_kind,
            )
        } else {
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

            let d_sl = sl_mult * atr;
            let d_tp1 = tp1_mult * atr;
            let d_tp2 = tp2_mult * atr;
            let d_tp3 = tp3_mult * atr;

            let side_f = side as f64;
            let stop_loss = (entry_price - side_f * d_sl).max(0.0);
            let tp1 = (entry_price + side_f * d_tp1).max(0.0);
            let tp2 = (entry_price + side_f * d_tp2).max(0.0);
            let tp3 = (entry_price + side_f * d_tp3).max(0.0);

            (stop_loss, tp1, tp2, tp3)
        };

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
            "base_score": breakdown.base_score,
            "predictors_score": breakdown.predictors_score,
            "raw_signals_score": breakdown.raw_signals_score,
            "indicators_score": breakdown.indicators_score,
            "coverage_score": breakdown.coverage_score,
            "consensus_score": breakdown.consensus_score,
            "setup_kind": format!("{:?}", breakdown.setup_kind),
            "setup_confidence": breakdown.setup_confidence,
            "atr": atr,
            "atr_pct": atr_pct,
            "risk_reward": risk_reward,
            "level_aware": has_levels,
            "support_levels_used": support_prices,
            "resistance_levels_used": resistance_prices,
            "strategy_id": 6,
            "strategy_name": "level_consensus",
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

fn infer_side(predictors: &[PredictionRow]) -> Option<i8> {
    let mut long_score: f32 = 0.0;
    let mut short_score: f32 = 0.0;
    let mut vote_count: usize = 0;

    for p in predictors {
        let sc = p.score_norm;
        let sd = p.side.unwrap_or(0);
        if sd == 0 { continue; }

        vote_count += 1;
        if sd > 0 { long_score += sc; } else { short_score += sc; }
    }

    if vote_count < 2 { return None; }

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

fn extract_sr_levels(raw_signals_summary: &Value) -> (Vec<f64>, Vec<f64>) {
    let mut supports = Vec::new();
    let mut resistances = Vec::new();

    // sr_levels is an OBJECT with named keys (strong_support, mid_support, etc.),
    // NOT an array. Parse accordingly.
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

async fn fetch_last_atr(pool: &PgPool, symbol_id: i64, tf_minutes: i16) -> Result<Option<f64>> {
    let row = sqlx::query(
        r#"SELECT atr FROM market.indicators_wide WHERE symbol_id = $1 AND tf_minutes = $2 AND atr IS NOT NULL ORDER BY time DESC LIMIT 1"#,
    ).bind(symbol_id).bind(tf_minutes as i32).fetch_optional(pool).await?;
    Ok(row.and_then(|r| r.try_get::<f64, _>(0).ok()))
}
