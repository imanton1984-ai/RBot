use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{
    ComputeBackend, ComputeJob, ComputeResult, FeatureWindow
};

pub struct CpuBackend {
    // Configuration could go here
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {}
    }

    pub fn calculate_rsi(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(prices.len());
        
        if prices.len() < period + 1 {
            for _ in 0..prices.len() {
                result.push(f64::NAN);
            }
            return result;
        }

        // Calculate initial average gain and loss
        let mut avg_gain = 0.0;
        let mut avg_loss = 0.0;
        
        for i in 1..=period {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                avg_gain += change;
            } else {
                avg_loss += change.abs();
            }
        }
        
        avg_gain /= period as f64;
        avg_loss /= period as f64;
        
        let mut rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 0.0 };
        let rsi = 100.0 - (100.0 / (1.0 + rs));
        result.push(rsi);

        // Calculate subsequent values
        for i in period + 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { change.abs() } else { 0.0 };

            avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
            avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;

            rs = if avg_loss != 0.0 { avg_gain / avg_loss } else { 100.0 };
            let rsi = 100.0 - (100.0 / (1.0 + rs));
            result.push(rsi);
        }

        // Pad beginning with NaN
        for _ in 0..period {
            result.insert(0, f64::NAN);
        }

        result
    }

    pub fn calculate_ema(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(prices.len());
        let multiplier = 2.0 / (period as f64 + 1.0);
        
        // Calculate SMA for the first EMA value
        if prices.len() < period {
            for _ in 0..prices.len() {
                result.push(f64::NAN);
            }
            return result;
        }

        let mut sma_sum = 0.0;
        for i in 0..period {
            sma_sum += prices[i];
        }
        let mut ema = sma_sum / period as f64;
        result.push(ema);

        // Calculate subsequent EMA values
        for i in period..prices.len() {
            ema = (prices[i] - ema) * multiplier + ema;
            result.push(ema);
        }

        // Pad beginning with NaN
        for _ in 0..(period - 1) {
            result.insert(0, f64::NAN);
        }

        result
    }

    fn calculate_sma(prices: &[f64], period: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(prices.len());
        
        for i in 0..prices.len() {
            if i < period - 1 {
                result.push(f64::NAN);
            } else {
                let sum: f64 = prices[(i - period + 1)..=i].iter().sum();
                result.push(sum / period as f64);
            }
        }
        
        result
    }

    pub fn calculate_macd(
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ema_fast = Self::calculate_ema(prices, fast_period);
        let ema_slow = Self::calculate_ema(prices, slow_period);
        
        let mut macd_line = Vec::with_capacity(prices.len());
        for i in 0..prices.len() {
            if ema_fast[i].is_nan() || ema_slow[i].is_nan() {
                macd_line.push(f64::NAN);
            } else {
                macd_line.push(ema_fast[i] - ema_slow[i]);
            }
        }
        
        let signal_line = Self::calculate_ema(&macd_line, signal_period);
        
        let mut histogram = Vec::with_capacity(prices.len());
        for i in 0..prices.len() {
            if macd_line[i].is_nan() || signal_line[i].is_nan() {
                histogram.push(f64::NAN);
            } else {
                histogram.push(macd_line[i] - signal_line[i]);
            }
        }
        
        (macd_line, signal_line, histogram)
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
            // In a real implementation, we would fetch the price data from DB based on job.window_start and job.window_end
            // For now, we'll simulate with dummy data
            let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1)).collect(); // Simulated price data
            
            let mut features = Vec::new();
            
            for indicator in &job.indicators {
                match indicator.as_str() {
                    "rsi" => {
                        let rsi_values = Self::calculate_rsi(&prices, 14);
                        for (i, &rsi_val) in rsi_values.iter().enumerate() {
                            if !rsi_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("rsi_14".to_string(), rsi_val);
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: (i as i64) * 60000, // Assuming 1-minute intervals
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "ema" => {
                        let ema_values = Self::calculate_ema(&prices, 20);
                        for (i, &ema_val) in ema_values.iter().enumerate() {
                            if !ema_val.is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("ema_20".to_string(), ema_val);
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: (i as i64) * 60000,
                                    features: feature_map,
                                });
                            }
                        }
                    }
                    "macd" => {
                        let (macd_line, signal_line, histogram) = Self::calculate_macd(&prices, 12, 26, 9);
                        
                        for i in 0..macd_line.len() {
                            if !macd_line[i].is_nan() && !signal_line[i].is_nan() && !histogram[i].is_nan() {
                                let mut feature_map = std::collections::HashMap::new();
                                feature_map.insert("macd_line".to_string(), macd_line[i]);
                                feature_map.insert("macd_signal".to_string(), signal_line[i]);
                                feature_map.insert("macd_histogram".to_string(), histogram[i]);
                                
                                features.push(ComputeResult {
                                    symbol: job.symbol.clone(),
                                    timeframe: job.timeframe,
                                    timestamp: (i as i64) * 60000,
                                    features: feature_map,
                                });
                            }
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