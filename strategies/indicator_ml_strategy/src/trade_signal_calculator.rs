// strategies/indicator_ml_strategy/src/trade_signal_calculator.rs
//
// Trade Signal Calculator для Indicator ML Strategy
//
// TP/SL расчет:
// - Для ML: берем цену из ML предсказания (price10_target)
// - Для консенсуса: считаем через ATR
// - Следим чтобы TP не сильно отклонялся от предсказаний

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{PgPool, Row};

use common::Symbol;
use predictors::types::{PredictionRow, PredictionAspect, CalcSource};

use crate::scorer::{IndicatorMlScorer, IndicatorMlScoreBreakdown, SetupKind};

// ═══════════════════════════════════════════════════════════════════════════
// TP/SL параметры для Indicator ML стратегии
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
struct TfTargets {
    min_tp1_pct: f64,
    min_tp2_pct: f64,
    min_tp3_pct: f64,
    min_sl_pct:  f64,
    max_sl_pct:  f64,
    min_rr: f64,
    
    /// Максимальное отклонение TP от предсказания (в %)
    max_pred_deviation_pct: f64,
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
    let mut t = TfTargets {
        min_tp1_pct: get_tf_from_config(tf_minutes, "tp1_pct", 1.5) / 100.0,
        min_tp2_pct: get_tf_from_config(tf_minutes, "tp2_pct", 2.8) / 100.0,
        min_tp3_pct: get_tf_from_config(tf_minutes, "tp3_pct", 4.0) / 100.0,
        min_sl_pct:  get_tf_from_config(tf_minutes, "sl_min_pct", 1.0) / 100.0,
        max_sl_pct:  get_tf_from_config(tf_minutes, "sl_pct", 3.0) / 100.0,
        min_rr:      1.5,
        max_pred_deviation_pct: get_tf_from_config(tf_minutes, "max_pred_deviation_pct", 15.0) / 100.0,
    };

    // enforce monotonic targets
    if t.min_tp2_pct <= t.min_tp1_pct { t.min_tp2_pct = t.min_tp1_pct * 1.6; }
    if t.min_tp3_pct <= t.min_tp2_pct { t.min_tp3_pct = t.min_tp2_pct * 1.45; }

    t
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
}

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

    /// TP/SL calculation для Indicator ML стратегии
    /// - Берем price10_target из ML предсказания как базу для TP
    /// - SL считаем через ATR
    /// - Проверяем чтобы TP не отклонялся сильно от предсказания
    fn calculate_tp_sl(
        &self,
        entry_price: f64,
        side: i8,
        atr: f64,
        ml_price_target: Option<f64>,
        targets: &TfTargets,
    ) -> (f64, f64, f64, f64) {
        let side_f = side as f64;
        
        // ═══════════════════════════════════════════════════════════════
        // 1. Определяем базовый TP из ML предсказания
        // ═══════════════════════════════════════════════════════════════
        let ml_tp1 = if let Some(target) = ml_price_target {
            let predicted_move = (target - entry_price) / entry_price;
            
            // SANITY CHECK: cap ML target at reasonable distance (10 ATR max)
            let max_ml_dist = atr * 10.0;
            let ml_dist = (target - entry_price).abs();
            
            if predicted_move * side_f > 0.0 && ml_dist <= max_ml_dist {
                // ML agrees with direction AND is within reasonable range
                target
            } else if predicted_move * side_f > 0.0 && ml_dist > max_ml_dist {
                // ML agrees with direction but target is unreasonably far → cap it
                entry_price + side_f * max_ml_dist
            } else {
                // ML disagrees with direction → ATR fallback
                entry_price + side_f * atr * 1.0
            }
        } else {
            // No ML prediction → ATR fallback
            entry_price + side_f * atr * 1.0
        };
        
        // ═══════════════════════════════════════════════════════════════
        // 2. Проверяем отклонение от минимальных требований
        // ═══════════════════════════════════════════════════════════════
        let min_tp1 = entry_price + side_f * entry_price * targets.min_tp1_pct;
        
        let tp1 = if side > 0 {
            ml_tp1.max(min_tp1)
        } else {
            ml_tp1.min(min_tp1)
        };
        
        // ═══════════════════════════════════════════════════════════════
        // 3. TP2 и TP3 как расширения TP1
        // ═══════════════════════════════════════════════════════════════
        let tp1_dist = (tp1 - entry_price).abs();
        let tp2 = entry_price + side_f * tp1_dist * 1.6;
        let tp3 = entry_price + side_f * tp1_dist * 2.5;
        
        // ═══════════════════════════════════════════════════════════════
        // 4. SL через ATR с защитой от слишком большого
        // ═══════════════════════════════════════════════════════════════
        let sl_atr = entry_price - side_f * atr * 0.75;
        let sl_min = entry_price - side_f * entry_price * targets.min_sl_pct;
        let sl_max = entry_price - side_f * entry_price * targets.max_sl_pct;
        
        let mut stop_loss = if side > 0 {
            sl_atr.min(sl_min)
        } else {
            sl_atr.max(sl_min)
        };
        
        // Cap SL
        if side > 0 {
            stop_loss = stop_loss.max(sl_max);
        } else {
            stop_loss = stop_loss.min(sl_max);
        }
        
        // ═══════════════════════════════════════════════════════════════
        // 5. Enforce minimum distances
        // ═══════════════════════════════════════════════════════════════
        let d_sl_min = entry_price * targets.min_sl_pct;
        let d_tp1_min = entry_price * targets.min_tp1_pct;
        let d_tp2_min = entry_price * targets.min_tp2_pct;
        let d_tp3_min = entry_price * targets.min_tp3_pct;
        
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
        
        // Monotonic ordering
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
        scorer: &IndicatorMlScorer,

        time: DateTime<Utc>,
        time_ms: i64,
        symbol_id: i64,
        symbol: &Symbol,
        tf_minutes: i16,

        entry_price: f64,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<TradeSignal>> {
        let side = infer_side_ml(predictors)
            .or_else(|| infer_side_from_summary_fields(raw_signals_summary))
            .unwrap_or(0);

        if side == 0 {
            return Ok(None);
        }

        // Scoring
        let breakdown_opt: Option<IndicatorMlScoreBreakdown> = scorer
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

        let targets = tf_targets(tf_minutes);

        // ═══════════════════════════════════════════════════════════════
        // Extract ML price target
        // ═══════════════════════════════════════════════════════════════
        let ml_price_target = predictors.iter()
            .find(|p| p.aspect == PredictionAspect::PriceTarget && p.calc_source == CalcSource::Ml)
            .map(|p| p.value);

        // ATR
        let atr = match extract_atr_from_summary(raw_signals_summary) {
            Some(v) => v,
            None => match fetch_last_atr(pool, symbol_id, tf_minutes).await {
                Ok(Some(v)) => v,
                _ => entry_price * self.fallback_atr_pct,
            },
        };

        // TP/SL calculation
        let (stop_loss, tp1, tp2, tp3) = self.calculate_tp_sl(
            entry_price,
            side,
            atr,
            ml_price_target,
            &targets,
        );

        // ═══════════════════════════════════════════════════════════════
        // Risk:Reward check
        // ═══════════════════════════════════════════════════════════════
        let tp1_dist = (tp1 - entry_price).abs();
        let sl_dist = (stop_loss - entry_price).abs();
        let risk_reward = if sl_dist > 0.0 { tp1_dist / sl_dist } else { 0.0 };

        if risk_reward < targets.min_rr {
            return Ok(None);
        }

        // Leverage
        let score_factor = (0.55 + 0.45 * breakdown.final_score).clamp(0.55, 1.0);
        let lev = ((self.base_leverage as f64) * score_factor)
            .round()
            .clamp(1.0, self.base_leverage as f64) as i16;

        let atr_pct = if entry_price > 0.0 { atr / entry_price } else { 0.0 };

        let breakdown_json = json!({
            "final": breakdown.final_score,
            "base_score": breakdown.indicator_score,
            "ml_prediction_score": breakdown.ml_prediction_score,
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
            "ml_price_target": ml_price_target,
            "strategy_id": 1,
            "strategy_name": "indicator_ml",
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

fn infer_side_ml(predictors: &[PredictionRow]) -> Option<i8> {
    // Используем ТОЛЬКО ML предсказания для определения направления
    let ml_predictors: Vec<&PredictionRow> = predictors.iter()
        .filter(|p| p.calc_source == CalcSource::Ml)
        .collect();
    
    if ml_predictors.is_empty() {
        return None;
    }
    
    let mut long_score: f32 = 0.0;
    let mut short_score: f32 = 0.0;
    
    for p in &ml_predictors {
        let sc = p.score_norm;
        let sd = p.side.unwrap_or(0);
        if sd == 0 { continue; }

        if sd > 0 {
            long_score += sc;
        } else {
            short_score += sc;
        }
    }
    
    let total = long_score + short_score;
    if total <= 0.0 {
        return None;
    }
    
    let long_ratio = long_score / total;
    
    if long_ratio >= 0.6 {
        Some(1)
    } else if long_ratio <= 0.4 {
        Some(-1)
    } else {
        None
    }
}

fn infer_side_from_summary_fields(raw_signals_summary: &Value) -> Option<i8> {
    raw_signals_summary
        .get("side")
        .and_then(|v| v.as_i64())
        .map(|x| (x as i8).signum())
}

fn extract_atr_from_summary(raw_signals_summary: &Value) -> Option<f64> {
    raw_signals_summary.get("atr").and_then(|v| v.as_f64())
}

async fn fetch_last_atr(
    pool: &PgPool,
    symbol_id: i64,
    tf_minutes: i16,
) -> Result<Option<f64>> {
    let row = sqlx::query(
        r#"
        SELECT atr
        FROM market.indicators_wide
        WHERE symbol_id = $1 AND tf_minutes = $2 AND atr IS NOT NULL
        ORDER BY time DESC
        LIMIT 1
        "#,
    )
    .bind(symbol_id)
    .bind(tf_minutes as i32)
    .fetch_optional(pool)
    .await?;

    Ok(row.and_then(|r| r.try_get::<f64, _>(0).ok()))
}
