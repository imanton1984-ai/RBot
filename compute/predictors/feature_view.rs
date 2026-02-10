// compute/predictions/feature_view.rs

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents a normalized feature vector for both hardcode and ML predictors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    pub values: Vec<f32>,
    pub schema_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
}

/// View for extracting features from indicators_wide and raw_signals
pub struct FeatureView {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub symbol: String,
    pub timeframe: String,
    
    // Price-based features
    pub close: f64,
    pub high: f64,
    pub low: f64,
    pub open: f64,
    
    // Technical indicators
    pub rsi: Option<f64>,
    pub macd_line: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_histogram: Option<f64>,
    pub ema_20: Option<f64>,
    pub ema_50: Option<f64>,
    pub ema_200: Option<f64>,
    pub sma: Option<f64>,
    pub bb_upper: Option<f64>,
    pub bb_lower: Option<f64>,
    pub bb_middle: Option<f64>,
    pub atr: Option<f64>,
    pub adx: Option<f64>,
    pub vwap: Option<f64>,
    pub obv: Option<f64>,
    pub cci: Option<f64>,
    pub stoch_k: Option<f64>,
    pub stoch_d: Option<f64>,
    pub williams_r: Option<f64>,
    
    // Trend features
    pub trend_short: Option<f64>,
    pub trend_medium: Option<f64>,
    pub trend_long: Option<f64>,
    
    // Volume features
    pub volume: f64,
    pub volume_sma: Option<f64>,
    pub volume_spike: Option<f64>,
    
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
        indicators_data: &serde_json::Value,
        raw_signals_data: Option<&serde_json::Value>,
    ) -> Result<Self> {
        let mut feature_view = FeatureView {
            timestamp,
            symbol,
            timeframe,
            
            // Extract basic price data
            close: indicators_data.get("close").and_then(|v| v.as_f64()).unwrap_or(0.0),
            high: indicators_data.get("high").and_then(|v| v.as_f64()).unwrap_or(0.0),
            low: indicators_data.get("low").and_then(|v| v.as_f64()).unwrap_or(0.0),
            open: indicators_data.get("open").and_then(|v| v.as_f64()).unwrap_or(0.0),
            
            // Extract technical indicators
            rsi: indicators_data.get("rsi").and_then(|v| v.as_f64()),
            macd_line: indicators_data.get("macd_line").and_then(|v| v.as_f64()),
            macd_signal: indicators_data.get("macd_signal").and_then(|v| v.as_f64()),
            macd_histogram: indicators_data.get("macd_histogram").and_then(|v| v.as_f64()),
            ema_20: indicators_data.get("ema_20").and_then(|v| v.as_f64()),
            ema_50: indicators_data.get("ema_50").and_then(|v| v.as_f64()),
            ema_200: indicators_data.get("ema_200").and_then(|v| v.as_f64()),
            sma: indicators_data.get("sma").and_then(|v| v.as_f64()),
            bb_upper: indicators_data.get("bb_upper").and_then(|v| v.as_f64()),
            bb_lower: indicators_data.get("bb_lower").and_then(|v| v.as_f64()),
            bb_middle: indicators_data.get("bb_middle").and_then(|v| v.as_f64()),
            atr: indicators_data.get("atr").and_then(|v| v.as_f64()),
            adx: indicators_data.get("adx").and_then(|v| v.as_f64()),
            vwap: indicators_data.get("vwap").and_then(|v| v.as_f64()),
            obv: indicators_data.get("obv").and_then(|v| v.as_f64()),
            cci: indicators_data.get("cci").and_then(|v| v.as_f64()),
            stoch_k: indicators_data.get("stoch_k").and_then(|v| v.as_f64()),
            stoch_d: indicators_data.get("stoch_d").and_then(|v| v.as_f64()),
            williams_r: indicators_data.get("williams_r").and_then(|v| v.as_f64()),
            
            // Extract trend data
            trend_short: indicators_data.get("trend_short").and_then(|v| v.as_f64()),
            trend_medium: indicators_data.get("trend_medium").and_then(|v| v.as_f64()),
            trend_long: indicators_data.get("trend_long").and_then(|v| v.as_f64()),
            
            // Extract volume data
            volume: indicators_data.get("volume").and_then(|v| v.as_f64()).unwrap_or(0.0),
            volume_sma: indicators_data.get("volume_sma").and_then(|v| v.as_f64()),
            volume_spike: indicators_data.get("volume_spike").and_then(|v| v.as_f64()),
            
            // Extract SR levels
            sr_levels: indicators_data.get("sr_levels").cloned(),
            
            // Extract raw signals
            raw_signals_summary: raw_signals_data.cloned(),
            
            // Initialize empty custom features map
            custom_features: HashMap::new(),
        };
        
        // Compute derived features
        feature_view.compute_derived_features();
        
        Ok(feature_view)
    }
    
    /// Computes derived features based on the raw data
    fn compute_derived_features(&mut self) {
        // Calculate candle position relative to EMAs
        if let Some(ema_20) = self.ema_20 {
            let pos_to_ema20 = (self.close - ema_20) / ema_20;
            self.custom_features.insert("position_to_ema20".to_string(), pos_to_ema20);
        }
        
        if let Some(ema_50) = self.ema_50 {
            let pos_to_ema50 = (self.close - ema_50) / ema_50;
            self.custom_features.insert("position_to_ema50".to_string(), pos_to_ema50);
        }
        
        if let Some(ema_200) = self.ema_200 {
            let pos_to_ema200 = (self.close - ema_200) / ema_200;
            self.custom_features.insert("position_to_ema200".to_string(), pos_to_ema200);
        }
        
        // Calculate volatility state based on Bollinger Bands
        if let (Some(bb_upper), Some(bb_lower), Some(bb_middle)) = (self.bb_upper, self.bb_lower, self.bb_middle) {
            let bb_width = (bb_upper - bb_lower) / bb_middle; // Normalized BB width
            self.custom_features.insert("bb_normalized_width".to_string(), bb_width);
            
            let bb_position = (self.close - bb_lower) / (bb_upper - bb_lower); // Position within BB
            self.custom_features.insert("bb_position".to_string(), bb_position);
        }
        
        // Calculate ATR-based volatility
        if let Some(atr) = self.atr {
            let atr_ratio = atr / self.close;
            self.custom_features.insert("atr_ratio".to_string(), atr_ratio);
        }
        
        // Calculate volume spike ratio if SMA is available
        if let Some(volume_sma) = self.volume_sma {
            if volume_sma > 0.0 {
                let volume_ratio = self.volume / volume_sma;
                self.custom_features.insert("volume_ratio".to_string(), volume_ratio);
            }
        }
    }
    
    /// Converts the feature view to a normalized feature vector suitable for ML models
    pub fn to_feature_vector(&self) -> FeatureVector {
        let mut features = Vec::new();
        
        // Add basic price features
        features.push(self.close as f32);
        features.push(self.high as f32);
        features.push(self.low as f32);
        features.push(self.open as f32);
        
        // Add technical indicators (with default 0.0 if None)
        features.push(self.rsi.unwrap_or(0.0) as f32);
        features.push(self.macd_line.unwrap_or(0.0) as f32);
        features.push(self.macd_signal.unwrap_or(0.0) as f32);
        features.push(self.macd_histogram.unwrap_or(0.0) as f32);
        features.push(self.ema_20.unwrap_or(0.0) as f32);
        features.push(self.ema_50.unwrap_or(0.0) as f32);
        features.push(self.ema_200.unwrap_or(0.0) as f32);
        features.push(self.sma.unwrap_or(0.0) as f32);
        features.push(self.bb_upper.unwrap_or(0.0) as f32);
        features.push(self.bb_lower.unwrap_or(0.0) as f32);
        features.push(self.bb_middle.unwrap_or(0.0) as f32);
        features.push(self.atr.unwrap_or(0.0) as f32);
        features.push(self.adx.unwrap_or(0.0) as f32);
        features.push(self.vwap.unwrap_or(0.0) as f32);
        features.push(self.obv.unwrap_or(0.0) as f32);
        features.push(self.cci.unwrap_or(0.0) as f32);
        features.push(self.stoch_k.unwrap_or(0.0) as f32);
        features.push(self.stoch_d.unwrap_or(0.0) as f32);
        features.push(self.williams_r.unwrap_or(0.0) as f32);
        
        // Add trend features
        features.push(self.trend_short.unwrap_or(0.0) as f32);
        features.push(self.trend_medium.unwrap_or(0.0) as f32);
        features.push(self.trend_long.unwrap_or(0.0) as f32);
        
        // Add volume features
        features.push(self.volume as f32);
        features.push(self.volume_sma.unwrap_or(0.0) as f32);
        features.push(self.volume_spike.unwrap_or(0.0) as f32);
        
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
                // Parse the JSON value into a vector of SR levels
                let levels: Vec<SrLevel> = serde_json::from_value(json_val.clone())
                    .map_err(|e| anyhow::anyhow!("Failed to parse SR levels: {}", e))?;
                Ok(levels)
            }
            None => Ok(Vec::new()),
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SrLevelKind {
    Support = 1,
    Resistance = 2,
}

impl SrLevelKind {
    pub fn as_i16(&self) -> i16 {
        *self as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_view_creation() {
        let indicators_json = serde_json::json!({
            "close": 100.0,
            "high": 102.0,
            "low": 98.0,
            "open": 99.0,
            "rsi": 60.0,
            "macd_line": 1.5,
            "macd_signal": 1.2,
            "macd_histogram": 0.3,
            "ema_20": 99.5,
            "ema_50": 98.0,
            "volume": 1000.0,
            "atr": 2.0
        });
        
        let raw_signals_json = Some(serde_json::json!({
            "signal_strength": 0.8,
            "confidence": 0.9
        }));
        
        let timestamp = chrono::Utc::now();
        let feature_view = FeatureView::new(
            timestamp,
            "BTCUSDT".to_string(),
            "5m".to_string(),
            &indicators_json,
            raw_signals_json.as_ref(),
        ).expect("Failed to create feature view");
        
        assert_eq!(feature_view.close, 100.0);
        assert_eq!(feature_view.high, 102.0);
        assert_eq!(feature_view.low, 98.0);
        assert_eq!(feature_view.open, 99.0);
        assert_eq!(feature_view.rsi, Some(60.0));
        assert_eq!(feature_view.macd_line, Some(1.5));
        assert_eq!(feature_view.volume, 1000.0);
        assert_eq!(feature_view.atr, Some(2.0));
        
        // Check that derived features were computed
        assert!(feature_view.custom_features.contains_key("position_to_ema20"));
        assert!(feature_view.custom_features.contains_key("position_to_ema50"));
        assert!(feature_view.custom_features.contains_key("atr_ratio"));
    }
    
    #[test]
    fn test_feature_vector_conversion() {
        let indicators_json = serde_json::json!({
            "close": 100.0,
            "high": 102.0,
            "low": 98.0,
            "open": 99.0,
            "rsi": 60.0,
            "volume": 1000.0
        });
        
        let timestamp = chrono::Utc::now();
        let feature_view = FeatureView::new(
            timestamp,
            "BTCUSDT".to_string(),
            "5m".to_string(),
            &indicators_json,
            None,
        ).expect("Failed to create feature view");
        
        let feature_vector = feature_view.to_feature_vector();
        
        // Basic checks
        assert!(!feature_vector.values.is_empty());
        assert!(!feature_vector.schema_id.is_empty());
        assert_eq!(feature_vector.symbol, "BTCUSDT");
        assert_eq!(feature_vector.timeframe, "5m");
    }
}