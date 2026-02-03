use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{
    ComputeBackend, ComputeJob, ComputeResult, FeatureWindow, FeatureValue
};
use crate::compute_indicators; // Используем модуль напрямую, без wildcard

pub struct CpuBackend {
    // Configuration could go here
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }

    pub fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
        compute_indicators::calculate_rsi(prices, period)
    }

    pub fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
        compute_indicators::calculate_ema(prices, period)
    }

    pub fn calculate_sma(prices: &[f64], period: usize) -> Vec<f64> {
        compute_indicators::calculate_sma(prices, period)
    }

    pub fn calculate_macd(
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        compute_indicators::calculate_macd(prices, fast_period, slow_period, signal_period)
    }

    pub fn calculate_adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        compute_indicators::calculate_adx(high, low, close, period)
    }

    pub fn calculate_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        compute_indicators::calculate_atr(high, low, close, period)
    }

    pub fn calculate_bollinger_bands(
        prices: &[f64],
        period: usize,
        num_std_dev: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        compute_indicators::calculate_bollinger_bands(prices, period, num_std_dev)
    }

    pub fn calculate_cci(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        compute_indicators::calculate_cci(high, low, close, period)
    }

    pub fn calculate_obv(close_prices: &[f64], volumes: &[f64]) -> Vec<f64> {
        compute_indicators::calculate_obv(close_prices, volumes)
    }

    pub fn calculate_stochastic(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k_period: usize,
        d_period: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        compute_indicators::calculate_stochastic(high, low, close, k_period, d_period)
    }

    pub fn calculate_vwap(high: &[f64], low: &[f64], close: &[f64], volume: &[f64]) -> Vec<f64> {
        compute_indicators::calculate_vwap(high, low, close, volume)
    }

    pub fn calculate_williams_r(
        high: &[f64],
        low: &[f64],
        close: &[f64],
        period: usize,
    ) -> Vec<f64> {
        compute_indicators::calculate_williams_r(high, low, close, period)
    }

    pub fn calculate_alligator(
        source: &[f64],
        jaw_period: usize,
        teeth_period: usize,
        lips_period: usize,
        jaw_offset: usize,
        teeth_offset: usize,
        lips_offset: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        compute_indicators::calculate_alligator(source, jaw_period, teeth_period, lips_period, jaw_offset, teeth_offset, lips_offset)
    }
}

#[async_trait::async_trait]
impl ComputeBackend for CpuBackend {
    async fn compute_indicators(
        &self,
        jobs: Vec<ComputeJob>,
    ) -> Result<Vec<Arc<FeatureWindow>>, Box<dyn std::error::Error + Send + Sync>> {
        let mut results = Vec::new();

        for job in jobs {
            let candle_window = match &job.candle_window {
                Some(cw) => cw,
                None => continue,
            };

            let mut features = Vec::new();
            
            for indicator in &job.indicators {
                match indicator.as_str() {
                    "adx" => {
                        let adx_values = Self::calculate_adx(&candle_window.high, &candle_window.low, &candle_window.close, 14);
                        for (i, &adx_val) in adx_values.iter().enumerate() {
                            if !adx_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("adx".to_string(), FeatureValue::Float(adx_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "atr" => {
                        let atr_values = Self::calculate_atr(&candle_window.high, &candle_window.low, &candle_window.close, 14);
                        for (i, &atr_val) in atr_values.iter().enumerate() {
                            if !atr_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("atr".to_string(), FeatureValue::Float(atr_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "bb" => {
                        let (upper, middle, lower) = Self::calculate_bollinger_bands(&candle_window.close, 20, 2.0);
                        for i in 0..upper.len() {
                            if !upper[i].is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("bb_upper".to_string(), FeatureValue::Float(upper[i]));
                                feature_map.insert("bb_mid".to_string(), FeatureValue::Float(middle[i]));
                                feature_map.insert("bb_lower".to_string(), FeatureValue::Float(lower[i]));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "cci" => {
                        let cci_values = Self::calculate_cci(&candle_window.high, &candle_window.low, &candle_window.close, 20);
                        for (i, &cci_val) in cci_values.iter().enumerate() {
                            if !cci_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("cci".to_string(), FeatureValue::Float(cci_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "ema" => {
                        let ema20 = Self::calculate_ema(&candle_window.close, 20);
                        let ema50 = Self::calculate_ema(&candle_window.close, 50);
                        let ema200 = Self::calculate_ema(&candle_window.close, 200);
                        for i in 0..ema20.len() {
                            if !ema20[i].is_nan() || !ema50[i].is_nan() || !ema200[i].is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                if !ema20[i].is_nan() {
                                    feature_map.insert("ema20".to_string(), FeatureValue::Float(ema20[i]));
                                }
                                if !ema50[i].is_nan() {
                                    feature_map.insert("ema50".to_string(), FeatureValue::Float(ema50[i]));
                                }
                                if !ema200[i].is_nan() {
                                    feature_map.insert("ema200".to_string(), FeatureValue::Float(ema200[i]));
                                }
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "macd" => {
                        let (macd_line, signal_line, histogram) = Self::calculate_macd(&candle_window.close, 12, 26, 9);
                        
                        for i in 0..macd_line.len() {
                            if !macd_line[i].is_nan() && !signal_line[i].is_nan() && !histogram[i].is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("macd".to_string(), FeatureValue::Float(macd_line[i]));
                                feature_map.insert("macd_signal".to_string(), FeatureValue::Float(signal_line[i]));
                                feature_map.insert("macd_hist".to_string(), FeatureValue::Float(histogram[i]));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "obv" => {
                        let obv_values = Self::calculate_obv(&candle_window.close, &candle_window.volume);
                        for (i, &obv_val) in obv_values.iter().enumerate() {
                            if !obv_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("obv".to_string(), FeatureValue::Float(obv_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "rsi" => {
                        let rsi_values = Self::calculate_rsi(&candle_window.close, 14);
                        for (i, &rsi_val) in rsi_values.iter().enumerate() {
                            if !rsi_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("rsi".to_string(), FeatureValue::Float(rsi_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "sma" => {
                        let sma_values = Self::calculate_sma(&candle_window.close, 20);
                        for (i, &sma_val) in sma_values.iter().enumerate() {
                            if !sma_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("sma".to_string(), FeatureValue::Float(sma_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "stoch" => {
                        let (k, d) = Self::calculate_stochastic(&candle_window.high, &candle_window.low, &candle_window.close, 14, 3);
                        for i in 0..k.len() {
                            if !k[i].is_nan() && !d[i].is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("stoch_k".to_string(), FeatureValue::Float(k[i]));
                                feature_map.insert("stoch_d".to_string(), FeatureValue::Float(d[i]));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "vwap" => {
                        let vwap_values = Self::calculate_vwap(&candle_window.high, &candle_window.low, &candle_window.close, &candle_window.volume);
                        for (i, &vwap_val) in vwap_values.iter().enumerate() {
                            if !vwap_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("vwap".to_string(), FeatureValue::Float(vwap_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "williams" => {
                        let williams_values = Self::calculate_williams_r(&candle_window.high, &candle_window.low, &candle_window.close, 14);
                        for (i, &williams_val) in williams_values.iter().enumerate() {
                            if !williams_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("williams".to_string(), FeatureValue::Float(williams_val));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "alligator" => {
                        let (jaw, teeth, lips) = Self::calculate_alligator(&candle_window.close, 13, 8, 5, 8, 5, 3);
                        for i in 0..jaw.len() {
                            if !jaw[i].is_nan() && !teeth[i].is_nan() && !lips[i].is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("alli_jaw".to_string(), FeatureValue::Float(jaw[i]));
                                feature_map.insert("alli_teeth".to_string(), FeatureValue::Float(teeth[i]));
                                feature_map.insert("alli_lips".to_string(), FeatureValue::Float(lips[i]));
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: candle_window.timestamps[i],
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "sr_levels" => {
                        let levels = compute_indicators::calculate_sr_levels(&candle_window.high, &candle_window.low, &candle_window.close, 0.5);
                        if let Ok(json_levels) = serde_json::to_value(&levels) {
                            let mut feature_map = std::collections::HashMap::new();
                            feature_map.insert("sr_levels".to_string(), FeatureValue::Json(json_levels));
                            
                            features.push(ComputeResult {
                                symbol: job.symbol.clone(),
                                timeframe: job.timeframe,
                                timestamp: candle_window.timestamps.last().cloned().unwrap_or(job.window_end),
                                features: feature_map,
                            });
                        }
                    }
                    _ => {
                        // For other indicators, we could add more cases
                    }
                }
            }

            let feature_window = Arc::new(FeatureWindow {
                features,
                start_time: job.window_start,
                end_time: job.window_end,
                candle_window: job.candle_window.clone(),
            });

            results.push(feature_window);
        }

        Ok(results)
    }

    async fn compute_single_indicator(
        &self,
        _symbol: Symbol,
        _timeframe: Timeframe,
        prices: &[f64],
        indicator_name: &str,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        match indicator_name {
            "rsi" => Ok(Self::calculate_rsi(prices, 14)),
            "ema" => Ok(Self::calculate_ema(prices, 20)),
            "sma" => Ok(Self::calculate_sma(prices, 20)),
            _ => Err(format!("Unknown indicator: {}", indicator_name).into()),
        }
    }
}