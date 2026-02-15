// compute/predictors/feature_view.rs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::level_view::LevelView;

/// Represents a normalized feature vector for both hardcode and ML predictors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub values: Vec<f32>,
    pub schema_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct IndicatorsWideRow {
    pub close: f32,
    pub high: f32,
    pub low: f32,
    pub open: f32,
    pub volume: f32,
    pub rsi: f32,
    pub macd_line: f32,
    pub macd_signal: f32,
    pub macd_histogram: f32,
    pub ema_20: f32,
    pub ema_50: f32,
    pub ema_200: f32,
    pub sma: f32,
    pub bb_upper: f32,
    pub bb_lower: f32,
    pub bb_middle: f32,
    pub atr: f32,
    pub adx: f32,
    pub vwap: f32,
    pub obv: f32,
    pub cci: f32,
    pub stoch_k: f32,
    pub stoch_d: f32,
    pub williams_r: f32,
    pub trend_short: f32,
    pub trend_medium: f32,
    pub trend_long: f32,
    pub volume_sma: f32,
    pub volume_spike: f32,
}

/// View for extracting features from indicators_wide and raw_signals
pub struct FeatureView {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
    pub is_realtime: bool,

    pub indicators: IndicatorsWideRow,

    // Support/Resistance levels
    pub sr_levels: Option<serde_json::Value>, // JSON representation of levels

    // Raw signals summary
    pub raw_signals_summary: Option<serde_json::Value>,

    // Custom computed features
    pub custom_features: HashMap<String, f64>,
}

impl FeatureView {
    /// Creates a new feature view from raw indicator data
    pub fn new(
        timestamp: chrono::DateTime<chrono::Utc>,
        symbol: String,
        timeframe: String,
        is_realtime: bool,
        indicators: IndicatorsWideRow,
        raw_signals_data: Option<&serde_json::Value>,
        sr_levels: Option<serde_json::Value>,
    ) -> Result<Self> {
        let mut feature_view = FeatureView {
            timestamp,
            symbol,
            timeframe,
            is_realtime,
            indicators,
            sr_levels,
            raw_signals_summary: raw_signals_data.cloned(),
            custom_features: HashMap::new(),
        };

        // Compute derived features
        feature_view.compute_derived_features();

        Ok(feature_view)
    }

    /// Returns true if the feature view is for a real-time candle
    pub fn is_realtime(&self) -> bool {
        self.is_realtime
    }
    
    /// Computes derived features based on the raw data
    fn compute_derived_features(&mut self) {
        let close = self.indicators.close as f64;
        // Calculate candle position relative to EMAs
        let ema_20 = self.indicators.ema_20 as f64;
        if ema_20 != 0.0 {
            let pos_to_ema20 = (close - ema_20) / ema_20;
            self.custom_features.insert("position_to_ema20".to_string(), pos_to_ema20);
        }
        
        let ema_50 = self.indicators.ema_50 as f64;
        if ema_50 != 0.0 {
            let pos_to_ema50 = (close - ema_50) / ema_50;
            self.custom_features.insert("position_to_ema50".to_string(), pos_to_ema50);
        }
        
        let ema_200 = self.indicators.ema_200 as f64;
        if ema_200 != 0.0 {
            let pos_to_ema200 = (close - ema_200) / ema_200;
            self.custom_features.insert("position_to_ema200".to_string(), pos_to_ema200);
        }
        
        // Calculate volatility state based on Bollinger Bands
        let bb_upper = self.indicators.bb_upper as f64;
        let bb_lower = self.indicators.bb_lower as f64;
        let bb_middle = self.indicators.bb_middle as f64;

        if bb_middle != 0.0 && (bb_upper - bb_lower) != 0.0 {
            let bb_width = (bb_upper - bb_lower) / bb_middle; // Normalized BB width
            self.custom_features.insert("bb_normalized_width".to_string(), bb_width);
            
            let bb_position = (close - bb_lower) / (bb_upper - bb_lower); // Position within BB
            self.custom_features.insert("bb_position".to_string(), bb_position);
        }
        
        // Calculate ATR-based volatility
        let atr = self.indicators.atr as f64;
        if close != 0.0 {
            let atr_ratio = atr / close;
            self.custom_features.insert("atr_ratio".to_string(), atr_ratio);
        }
        
        // Calculate volume spike ratio if SMA is available
        let volume_sma = self.indicators.volume_sma as f64;
        if volume_sma > 0.0 {
            let volume_ratio = self.indicators.volume as f64 / volume_sma;
            self.custom_features.insert("volume_ratio".to_string(), volume_ratio);
        }
    }

    /// Computes level-related features from the SR levels embedded in this view.
    ///
    /// Returns a map with keys such as:
    /// - `distance_to_nearest_support_in_atr`
    /// - `distance_to_nearest_resistance_in_atr`
    /// - `ema_stack_direction` (1.0 bullish / -1.0 bearish / 0.0 mixed)
    /// - `nearest_level_touch_count`
    /// - `nearest_level_strength`
    pub fn compute_level_features(&self) -> HashMap<String, f64> {
        let mut out = HashMap::new();

        let close = self.indicators.close as f64;
        let atr = self.indicators.atr as f64;

        // ── EMA stack direction ────────────────────────────────────────
        let ema20 = self.indicators.ema_20 as f64;
        let ema50 = self.indicators.ema_50 as f64;
        let ema200 = self.indicators.ema_200 as f64;

        let ema_stack = if ema20 > ema50 && ema50 > ema200 {
            1.0
        } else if ema20 < ema50 && ema50 < ema200 {
            -1.0
        } else {
            0.0
        };
        out.insert("ema_stack_direction".to_string(), ema_stack);

        // ── Level features ─────────────────────────────────────────────
        let sr_levels = match self.get_sr_levels() {
            Ok(l) => l,
            Err(_) => return out,
        };
        if sr_levels.is_empty() || atr <= 0.0 {
            return out;
        }

        let level_view = match LevelView::new(
            sr_levels,
            close,
            atr,
            self.timestamp,
            self.symbol.clone(),
            self.timeframe.clone(),
            5, // generous limit
            None,
        ) {
            Ok(lv) => lv,
            Err(_) => return out,
        };

        // Nearest support (below current price)
        if let Some(sup) = level_view
            .get_levels_by_kind(SrLevelKind::Support)
            .into_iter()
            .filter(|l| l.level_price <= close)
            .min_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap())
        {
            out.insert(
                "distance_to_nearest_support_in_atr".to_string(),
                sup.distance_atr as f64,
            );
        }

        // Nearest resistance (above current price)
        if let Some(res) = level_view
            .get_levels_by_kind(SrLevelKind::Resistance)
            .into_iter()
            .filter(|l| l.level_price >= close)
            .min_by(|a, b| a.distance_atr.partial_cmp(&b.distance_atr).unwrap())
        {
            out.insert(
                "distance_to_nearest_resistance_in_atr".to_string(),
                res.distance_atr as f64,
            );
        }

        // Overall nearest level (any kind) for touch_count & strength
        if let Some(nearest) = level_view.levels.first() {
            out.insert(
                "nearest_level_touch_count".to_string(),
                nearest.touch_count as f64,
            );
            out.insert(
                "nearest_level_strength".to_string(),
                nearest.level_strength as f64,
            );
        }

        out
    }

    /// Converts the feature view to a normalized feature vector suitable for ML models
    pub fn to_feature_vector(&self) -> FeatureVector {
        let mut features = Vec::new();
        
        // Add basic price features
        features.push(self.indicators.close);
        features.push(self.indicators.high);
        features.push(self.indicators.low);
        features.push(self.indicators.open);
        
        // Add technical indicators (with default 0.0 if None)
        features.push(self.indicators.rsi);
        features.push(self.indicators.macd_line);
        features.push(self.indicators.macd_signal);
        features.push(self.indicators.macd_histogram);
        features.push(self.indicators.ema_20);
        features.push(self.indicators.ema_50);
        features.push(self.indicators.ema_200);
        features.push(self.indicators.sma);
        features.push(self.indicators.bb_upper);
        features.push(self.indicators.bb_lower);
        features.push(self.indicators.bb_middle);
        features.push(self.indicators.atr);
        features.push(self.indicators.adx);
        features.push(self.indicators.vwap);
        features.push(self.indicators.obv);
        features.push(self.indicators.cci);
        features.push(self.indicators.stoch_k);
        features.push(self.indicators.stoch_d);
        features.push(self.indicators.williams_r);
        
        // Add trend features
        features.push(self.indicators.trend_short);
        features.push(self.indicators.trend_medium);
        features.push(self.indicators.trend_long);
        
        // Add volume features
        features.push(self.indicators.volume);
        features.push(self.indicators.volume_sma);
        features.push(self.indicators.volume_spike);
        
        // Add derived features
        for (_, &value) in &self.custom_features {
            features.push(value as f32);
        }
        
        // Generate a schema ID based on the feature vector length and timestamp
        let schema_id = format!("fv_{}_{}", features.len(), self.timestamp.timestamp());
        
        FeatureVector {
            values: features,
            schema_id,
            timestamp: self.timestamp,
            symbol: self.symbol.clone(),
            timeframe: self.timeframe.clone(),
        }
    }
    
    /// Gets the most recent SR levels from the JSON data
    pub fn get_sr_levels(&self) -> Result<Vec<SrLevel>> {
        match &self.sr_levels {
            Some(json_val) => {
                // Check if the JSON value is an object (the SRLLevels struct format)
                if let serde_json::Value::Object(ref obj) = json_val {
                    // Handle the SRLLevels struct format by converting to individual SrLevel objects
                    let levels = self.convert_srl_levels_to_sr_levels(obj)?;
                    Ok(levels)
                } else {
                    // If it's already an array of SrLevel objects, parse directly
                    let levels: Vec<SrLevel> = serde_json::from_value(json_val.clone())
                        .map_err(|e| anyhow::anyhow!("Failed to parse SR levels: {}", e))?;
                    Ok(levels)
                }
            }
            None => Ok(Vec::new()),
        }
    }

    /// Converts SRLLevels struct format to individual SrLevel objects
    fn convert_srl_levels_to_sr_levels(&self, obj: &serde_json::Map<String, serde_json::Value>) -> Result<Vec<SrLevel>> {
        let mut levels = Vec::new();

        // Extract individual level values from the SRLLevels struct
        if let Some(strong_support_val) = obj.get("strong_support") {
            if let Some(price) = strong_support_val.as_f64() {
                if !price.is_nan() {
                    levels.push(SrLevel {
                        price,
                        kind: SrLevelKind::Support,
                        strength: 0.9, // Strong support
                        hash: format!("support_strong_{:.6}", price),
                    });
                }
            }
        }

        if let Some(mid_support_val) = obj.get("mid_support") {
            if let Some(price) = mid_support_val.as_f64() {
                if !price.is_nan() {
                    levels.push(SrLevel {
                        price,
                        kind: SrLevelKind::Support,
                        strength: 0.6, // Mid support
                        hash: format!("support_mid_{:.6}", price),
                    });
                }
            }
        }

        if let Some(light_support_val) = obj.get("light_support") {
            if let Some(price) = light_support_val.as_f64() {
                if !price.is_nan() {
                    levels.push(SrLevel {
                        price,
                        kind: SrLevelKind::Support,
                        strength: 0.3, // Light support
                        hash: format!("support_light_{:.6}", price),
                    });
                }
            }
        }

        if let Some(strong_resistance_val) = obj.get("strong_resistance") {
            if let Some(price) = strong_resistance_val.as_f64() {
                if !price.is_nan() {
                    levels.push(SrLevel {
                        price,
                        kind: SrLevelKind::Resistance,
                        strength: 0.9, // Strong resistance
                        hash: format!("resistance_strong_{:.6}", price),
                    });
                }
            }
        }

        if let Some(mid_resistance_val) = obj.get("mid_resistance") {
            if let Some(price) = mid_resistance_val.as_f64() {
                if !price.is_nan() {
                    levels.push(SrLevel {
                        price,
                        kind: SrLevelKind::Resistance,
                        strength: 0.6, // Mid resistance
                        hash: format!("resistance_mid_{:.6}", price),
                    });
                }
            }
        }

        if let Some(light_resistance_val) = obj.get("light_resistance") {
            if let Some(price) = light_resistance_val.as_f64() {
                if !price.is_nan() {
                    levels.push(SrLevel {
                        price,
                        kind: SrLevelKind::Resistance,
                        strength: 0.3, // Light resistance
                        hash: format!("resistance_light_{:.6}", price),
                    });
                }
            }
        }

        Ok(levels)
    }
    pub fn get_value(&self, name: &str) -> f32 {
        if let Some(val) = self.custom_features.get(name) {
            return *val as f32;
        }

        match name {
            "close" => self.indicators.close,
            "high" => self.indicators.high,
            "low" => self.indicators.low,
            "open" => self.indicators.open,
            "volume" => self.indicators.volume,
            "rsi" => self.indicators.rsi,
            "macd" => self.indicators.macd_line, // Assuming macd is macd_line
            "macd_signal" => self.indicators.macd_signal,
            "macd_hist" => self.indicators.macd_histogram,
            "ema_20" => self.indicators.ema_20,
            "ema_50" => self.indicators.ema_50,
            "ema_200" => self.indicators.ema_200,
            "sma" => self.indicators.sma,
            "bb_upper" => self.indicators.bb_upper,
            "bb_lower" => self.indicators.bb_lower,
            "bb_mid" => self.indicators.bb_middle,
            "atr" => self.indicators.atr,
            "adx" => self.indicators.adx,
            "vwap" => self.indicators.vwap,
            "obv" => self.indicators.obv,
            "cci" => self.indicators.cci,
            "stoch_k" => self.indicators.stoch_k,
            "stoch_d" => self.indicators.stoch_d,
            "williams" => self.indicators.williams_r,
            "trend" => self.indicators.trend_medium, // Assuming trend is medium
            "trend_short" => self.indicators.trend_short,
            _ => 0.0,
        }
    }
}

/// Represents a support/resistance level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SrLevel {
    pub price: f64,
    pub kind: SrLevelKind, // 1 = support, 2 = resistance
    pub strength: f32,
    pub hash: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[repr(i16)]
pub enum SrLevelKind {
    Support = 1,
    Resistance = 2,
}

impl SrLevelKind {
    pub fn as_i16(&self) -> i16 {
        *self as i16
    }
}


