// calculates final signal based on all previous steps - indicators, raw_signals, ml predictors, heruistic predictors, market_parameter (volatility, trend)
//signal must contain Symbol (traiding pair) time created, timeframe (1m, 5m, 15m, 1h, 4h, 1d), stop_loss, take_profits_1,2,3, side (short/long), combined final score, scores from prevous steps like raw_score, ind_score, ml_score, heruistic_score, 

// compute/scoring/trade_signal_calculator.rs

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use crate::predictions::types::PredictionRow;

use super::final_score::{FinalScorer, FinalScoreBreakdown};
use super::market_params_calculator::MarketParams;

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
            fallback_atr_pct: 0.008, // 0.8%
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
        scorer: &FinalScorer,
        market_params: Option<&MarketParams>,

        time: DateTime<Utc>,
        time_ms: i64,
        symbol_id: i64,
        symbol: &Symbol,
        tf_minutes: i16,

        entry_price: f64,
        raw_signals_summary: &Value,
        predictions: &[PredictionRow],
    ) -> Result<Option<TradeSignal>> {
        let side = infer_side(raw_signals_summary, predictions)
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
                predictions,
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

        // ATR
        let atr = extract_atr_from_summary(raw_signals_summary)
            .or_else(|| futures::executor::block_on(fetch_last_atr(pool, symbol_id, tf_minutes)).ok().flatten())
            .unwrap_or(entry_price * self.fallback_atr_pct);

        let side_f = side as f64;

        let stop_loss = (entry_price - side_f * self.sl_atr_mult * atr).max(0.0);
        let tp1 = (entry_price + side_f * self.tp1_atr_mult * atr).max(0.0);
        let tp2 = (entry_price + side_f * self.tp2_atr_mult * atr).max(0.0);
        let tp3 = (entry_price + side_f * self.tp3_atr_mult * atr).max(0.0);

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
            "predictions_score": breakdown.predictions_score,
            "raw_signals_score": breakdown.raw_signals_score,
            "indicators_score": breakdown.indicators_score,
            "market_score": breakdown.market_score,
            "market_factor": market_factor,
            "score_factor": score_factor,
            "atr": atr,
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

fn infer_side(raw_signals_summary: &Value, predictions: &[PredictionRow]) -> Option<i8> {
    // 1) raw summary explicit
    if let Some(s) = raw_signals_summary.get("side").and_then(|v| v.as_i64()) {
        let ss = (s as i8).signum();
        if ss != 0 { return Some(ss); }
    }

    // 2) best prediction by score_norm
    let mut best: Option<(f32, i16)> = None;
    for p in predictions {
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
