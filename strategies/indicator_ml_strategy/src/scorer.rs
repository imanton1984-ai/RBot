// strategies/indicator_ml_strategy/src/scorer.rs
//
// Indicator ML Strategy Scorer
//
// Эта стратегия полагается исключительно на:
// 1. ML предсказания (PriceTarget от ML)
// 2. Индикаторы с равными весами (RSI, MACD, Stochastic, ADX, ATR, Volume, Trend)
//
// Веса индикаторов:
// - RSI: 0.20
// - MACD: 0.20
// - Stochastic: 0.15
// - ADX: 0.15
// - ATR (volatility): 0.10
// - Volume: 0.10
// - Trend (EMA): 0.10

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use predictors::types::{PredictionAspect, PredictionRow, CalcSource};

/// Setup kind: для indicator стратегий используем только направление тренда
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetupKind {
    Momentum,    // Сильный тренд
    Reversal,    // Разворот
}

/// Final scorer configuration для Indicator ML стратегии
pub struct IndicatorMlScorer {
    min_final_score: f64,
    
    /// Веса индикаторов (должны суммироваться в 1.0)
    indicator_weights: IndicatorWeights,
    
    /// Вес ML предсказания
    ml_prediction_weight: f64,
    
    /// Порог ADX для фильтрации слабых трендов
    adx_filter_threshold: f64,
    
    /// Порог RSI для overbought/oversold
    rsi_overbought: f64,
    rsi_oversold: f64,
}

/// Веса для индикаторов
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorWeights {
    pub rsi_weight: f64,        // 0.20
    pub macd_weight: f64,       // 0.20
    pub stoch_weight: f64,      // 0.15
    pub adx_weight: f64,        // 0.15
    pub atr_weight: f64,        // 0.10
    pub volume_weight: f64,     // 0.10
    pub trend_weight: f64,      // 0.10
}

impl Default for IndicatorWeights {
    fn default() -> Self {
        Self {
            rsi_weight: 0.20,
            macd_weight: 0.20,
            stoch_weight: 0.15,
            adx_weight: 0.15,
            atr_weight: 0.10,
            volume_weight: 0.10,
            trend_weight: 0.10,
        }
    }
}

impl IndicatorWeights {
    pub fn normalized(mut self) -> Self {
        let sum = self.rsi_weight + self.macd_weight + self.stoch_weight 
                + self.adx_weight + self.atr_weight + self.volume_weight + self.trend_weight;
        if sum > 0.0 {
            self.rsi_weight /= sum;
            self.macd_weight /= sum;
            self.stoch_weight /= sum;
            self.adx_weight /= sum;
            self.atr_weight /= sum;
            self.volume_weight /= sum;
            self.trend_weight /= sum;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorMlScoreBreakdown {
    pub final_score: f64,
    pub indicator_score: f64,
    pub ml_prediction_score: f64,
    
    // Детали по индикаторам
    pub rsi_score: f64,
    pub macd_score: f64,
    pub stoch_score: f64,
    pub adx_score: f64,
    pub atr_score: f64,
    pub volume_score: f64,
    pub trend_score: f64,
    
    // Setup inference
    pub setup_kind: SetupKind,
    pub setup_confidence: f64,
    
    // Direction alignment
    pub trend_align: f64,
    pub momentum_ok: f64,
    
    pub debug: Value,
}

impl IndicatorMlScorer {
    pub fn new(min_final_score: f64) -> Self {
        Self {
            min_final_score,
            indicator_weights: IndicatorWeights::default().normalized(),
            ml_prediction_weight: 0.5, // 50% ML prediction, 50% indicators
            adx_filter_threshold: 25.0,
            rsi_overbought: 70.0,
            rsi_oversold: 30.0,
        }
    }
    
    pub fn with_ml_weight(mut self, ml_weight: f64) -> Self {
        self.ml_prediction_weight = ml_weight.clamp(0.0, 1.0);
        self
    }
    
    /// Returns only final number (compat)
    pub async fn score_signal(
        &self,
        symbol: &str,
        tf_minutes: i16,
        timestamp: DateTime<Utc>,
        side: i16,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<f64>> {
        Ok(self
            .score_signal_verbose(symbol, tf_minutes, timestamp, side, raw_signals_summary, predictors)
            .await?
            .map(|b| b.final_score))
    }
    
    /// Verbose score with breakdown
    pub async fn score_signal_verbose(
        &self,
        symbol: &str,
        tf_minutes: i16,
        timestamp: DateTime<Utc>,
        side: i16,
        raw_signals_summary: &Value,
        predictors: &[PredictionRow],
    ) -> Result<Option<IndicatorMlScoreBreakdown>> {
        let side_i8 = side.signum() as i8;
        
        // ═══════════════════════════════════════════════════════════════
        // 1. ML Prediction Score - используем ТОЛЬКО ML предсказания
        // ═══════════════════════════════════════════════════════════════
        let ml_prediction_score = self.calculate_ml_prediction_score(predictors);
        
        // ═══════════════════════════════════════════════════════════════
        // 2. Indicator Score - все индикаторы с равными весами
        // ═══════════════════════════════════════════════════════════════
        let (indicator_score, rsi_score, macd_score, stoch_score, adx_score, 
             atr_score, volume_score, trend_score) = 
            self.calculate_indicator_score(raw_signals_summary, side_i8);
        
        // ═══════════════════════════════════════════════════════════════
        // 3. ADX Filter - слабые тренды фильтруем
        // ═══════════════════════════════════════════════════════════════
        let adx = raw_signals_summary.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        if adx > 0.0 && adx < self.adx_filter_threshold {
            return Ok(None);
        }
        
        // ═══════════════════════════════════════════════════════════════
        // 4. Final Score - комбинация ML + Indicators
        // ═══════════════════════════════════════════════════════════════
        let base_score = ml_prediction_score * self.ml_prediction_weight 
                       + indicator_score * (1.0 - self.ml_prediction_weight);
        
        // ═══════════════════════════════════════════════════════════════
        // 5. Setup Inference - Momentum vs Reversal
        // ═══════════════════════════════════════════════════════════════
        let (setup_kind, setup_confidence) = self.infer_setup(raw_signals_summary, side_i8);
        
        // ═══════════════════════════════════════════════════════════════
        // 6. Regime alignment
        // ═══════════════════════════════════════════════════════════════
        let trend_align = raw_signals_summary.get("trend_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        
        let momentum_strength = raw_signals_summary.get("momentum_strength")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        
        let momentum_ok = match setup_kind {
            SetupKind::Momentum => momentum_strength,
            SetupKind::Reversal => (1.0 - momentum_strength).clamp(0.0, 1.0),
        };
        
        // ═══════════════════════════════════════════════════════════════
        // 7. Final calculation with dampening
        // ═══════════════════════════════════════════════════════════════
        let setup_damp = 0.8 + 0.2 * setup_confidence;
        let final_score = (base_score * setup_damp).clamp(0.0, 1.0);
        
        let debug = json!({
            "symbol": symbol,
            "tf_minutes": tf_minutes,
            "timestamp": timestamp.to_rfc3339(),
            "side": side_i8,
            "ml_prediction_weight": self.ml_prediction_weight,
            "base_score": base_score,
            "ml_prediction_score": ml_prediction_score,
            "indicator_score": indicator_score,
            "rsi_score": rsi_score,
            "macd_score": macd_score,
            "stoch_score": stoch_score,
            "adx_score": adx_score,
            "atr_score": atr_score,
            "volume_score": volume_score,
            "trend_score": trend_score,
            "final_score": final_score,
            "setup_kind": format!("{:?}", setup_kind),
            "setup_confidence": setup_confidence,
            "adx": adx,
            "adx_filter_passed": adx == 0.0 || adx >= self.adx_filter_threshold,
        });
        
        if final_score >= self.min_final_score {
            Ok(Some(IndicatorMlScoreBreakdown {
                final_score,
                indicator_score,
                ml_prediction_score,
                rsi_score,
                macd_score,
                stoch_score,
                adx_score,
                atr_score,
                volume_score,
                trend_score,
                setup_kind,
                setup_confidence,
                trend_align,
                momentum_ok,
                debug,
            }))
        } else {
            Ok(None)
        }
    }
    
    /// Calculate ML prediction score - uses ONLY ML predictions
    fn calculate_ml_prediction_score(&self, predictors: &[PredictionRow]) -> f64 {
        // Фильтруем только ML предсказания
        let ml_predictors: Vec<&PredictionRow> = predictors.iter()
            .filter(|p| p.calc_source == CalcSource::Ml)
            .collect();
        
        if ml_predictors.is_empty() {
            return 0.0;
        }
        
        // Aspect weights: price target is king
        let aspect_weight = |a: i16| -> f64 {
            match a {
                x if x == PredictionAspect::PriceTarget.as_int() => 1.0,
                x if x == PredictionAspect::LevelBounce.as_int() => 0.7,
                x if x == PredictionAspect::LevelBreakout.as_int() => 0.7,
                _ => 0.5,
            }
        };
        
        let mut total_ws = 0.0;
        let mut total_w = 0.0;
        
        for p in &ml_predictors {
            let aspect = p.aspect.as_int();
            let score = (p.score_norm as f64).clamp(0.0, 1.0);
            let w = aspect_weight(aspect);
            
            total_ws += score * w;
            total_w += w;
        }
        
        if total_w > 0.0 {
            (total_ws / total_w).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }
    
    /// Calculate indicator score with equal weights
    fn calculate_indicator_score(
        &self,
        raw_signals_summary: &Value,
        side: i8,
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        // Extract indicator values
        let rsi = raw_signals_summary.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let macd_hist = raw_signals_summary.get("macd_hist").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let stoch_k = raw_signals_summary.get("stoch_k").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let stoch_d = raw_signals_summary.get("stoch_d").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let adx = raw_signals_summary.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let atr = raw_signals_summary.get("atr").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let close = raw_signals_summary.get("close").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let volume = raw_signals_summary.get("volume").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let volume_sma = raw_signals_summary.get("volume_sma").and_then(|v| v.as_f64()).unwrap_or(1.0);
        let trend_short = raw_signals_summary.get("trend_short").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ema_20 = raw_signals_summary.get("ema_20").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let ema_50 = raw_signals_summary.get("ema_50").and_then(|v| v.as_f64()).unwrap_or(0.0);
        
        // RSI Score - normalized to 0-1
        // RSI > 50 is bullish, < 50 is bearish
        let rsi_score = if side > 0 {
            // Long: prefer RSI > 50 but not overbought
            if rsi >= self.rsi_oversold && rsi <= self.rsi_overbought {
                ((rsi - 50.0) / 20.0).clamp(0.0, 1.0)
            } else if rsi > self.rsi_overbought {
                0.3 // Overbought penalty
            } else {
                0.2 // Oversold penalty
            }
        } else {
            // Short: prefer RSI < 50 but not oversold
            if rsi >= self.rsi_oversold && rsi <= self.rsi_overbought {
                ((50.0 - rsi) / 20.0).clamp(0.0, 1.0)
            } else if rsi < self.rsi_oversold {
                0.3 // Oversold penalty
            } else {
                0.2 // Overbought penalty
            }
        };
        
        // MACD Score - histogram direction
        let macd_score = if side > 0 {
            if macd_hist > 0.0 {
                (macd_hist.abs() / 0.001).clamp(0.5, 1.0)
            } else {
                0.2
            }
        } else {
            if macd_hist < 0.0 {
                (macd_hist.abs() / 0.001).clamp(0.5, 1.0)
            } else {
                0.2
            }
        };
        
        // Stochastic Score
        let stoch_mid = (stoch_k + stoch_d) / 2.0;
        let stoch_score = if side > 0 {
            if stoch_mid >= 20.0 && stoch_mid <= 80.0 {
                ((stoch_mid - 50.0) / 30.0).clamp(0.0, 1.0)
            } else if stoch_mid > 80.0 {
                0.3
            } else {
                0.2
            }
        } else {
            if stoch_mid >= 20.0 && stoch_mid <= 80.0 {
                ((50.0 - stoch_mid) / 30.0).clamp(0.0, 1.0)
            } else if stoch_mid < 20.0 {
                0.3
            } else {
                0.2
            }
        };
        
        // ADX Score - trend strength
        let adx_score = (adx / 50.0).clamp(0.0, 1.0);
        
        // ATR Score - volatility (prefer moderate)
        let atr_pct = if close > 0.0 { atr / close } else { 0.0 };
        let atr_score = if atr_pct >= 0.005 && atr_pct <= 0.03 {
            1.0
        } else if atr_pct < 0.005 {
            0.5
        } else {
            0.3
        };
        
        // Volume Score
        let volume_ratio = if volume_sma > 0.0 { volume / volume_sma } else { 1.0 };
        let volume_score = if volume_ratio >= 1.0 && volume_ratio <= 3.0 {
            1.0
        } else if volume_ratio > 3.0 {
            0.7 // Too high might be exhaustion
        } else {
            0.5
        };
        
        // Trend Score - EMA alignment
        let trend_score = if ema_20 > 0.0 && ema_50 > 0.0 {
            if side > 0 {
                if ema_20 > ema_50 { 1.0 } else { 0.3 }
            } else {
                if ema_20 < ema_50 { 1.0 } else { 0.3 }
            }
        } else {
            0.5
        };
        
        // Weighted average
        let indicator_score = rsi_score * self.indicator_weights.rsi_weight
            + macd_score * self.indicator_weights.macd_weight
            + stoch_score * self.indicator_weights.stoch_weight
            + adx_score * self.indicator_weights.adx_weight
            + atr_score * self.indicator_weights.atr_weight
            + volume_score * self.indicator_weights.volume_weight
            + trend_score * self.indicator_weights.trend_weight;
        
        (indicator_score, rsi_score, macd_score, stoch_score, adx_score, 
         atr_score, volume_score, trend_score)
    }
    
    /// Infer setup kind: Momentum vs Reversal
    fn infer_setup(&self, raw_signals_summary: &Value, _side: i8) -> (SetupKind, f64) {
        let rsi = raw_signals_summary.get("rsi").and_then(|v| v.as_f64()).unwrap_or(50.0);
        let adx = raw_signals_summary.get("adx").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let _trend_short = raw_signals_summary.get("trend_short").and_then(|v| v.as_f64()).unwrap_or(0.0);

        // Momentum: strong trend (ADX > 25) + RSI in trend zone
        let is_momentum = adx >= 25.0
            && ((adx > 0.0 && adx < 50.0));

        // Reversal: RSI at extremes + weak trend
        let is_reversal = adx < 25.0;

        if is_momentum {
            (SetupKind::Momentum, (adx / 50.0).clamp(0.5, 1.0))
        } else if is_reversal {
            (SetupKind::Reversal, 0.5)
        } else {
            (SetupKind::Momentum, 0.5)
        }
    }
}
