use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{
    ComputeBackend, ComputeJob, ComputeResult, FeatureWindow,
    BatchTensor, FeatureValue
};

pub struct CudaBackend {
    #[allow(dead_code)]
    device_id: usize,
    initialized: bool,
}

impl CudaBackend {
    pub fn new() -> Self {
        Self {
            device_id: 0, // Default to first device
            initialized: false,
        }
    }

    pub fn initialize(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if self.is_cuda_available() {
            self.initialized = true;
            Ok(())
        } else {
            self.initialized = true; 
            Ok(())
        }
    }

    fn is_cuda_available(&self) -> bool {
        cfg!(feature = "cuda")
    }

    fn run_rsi_kernel(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_rsi(input, period));
        }

        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_rsi(input, period)
    }

    fn run_ema_kernel(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_ema(input, period));
        }

        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_ema(input, period)
    }

    fn run_macd_kernel(
        &self,
        input: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_macd(input, fast_period, slow_period, signal_period));
        }

        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_macd(input, fast_period, slow_period, signal_period)
    }

    fn run_adx_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_adx(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_adx(high, low, close, period)
    }

    fn run_atr_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_atr(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_atr(high, low, close, period)
    }

    fn run_bollinger_bands_kernel(
        &self,
        prices: &[f64],
        period: usize,
        num_std_dev: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_bollinger_bands(prices, period, num_std_dev));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_bollinger_bands(prices, period, num_std_dev)
    }
    
    fn run_cci_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_cci(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_cci(high, low, close, period)
    }

    fn run_obv_kernel(
        &self,
        close: &[f64], volume: &[f64]
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_obv(close, volume));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_obv(close, volume)
    }

    fn run_stochastic_kernel(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k_period: usize,
        d_period: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_stochastic(high, low, close, k_period, d_period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_stochastic(high, low, close, k_period, d_period)
    }

    fn run_vwap_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], volume: &[f64]
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_vwap(high, low, close, volume));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_vwap(high, low, close, volume)
    }

    fn run_williams_r_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_williams_r(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_williams_r(high, low, close, period)
    }
    
    fn run_alligator_kernel(
        &self,
        source: &[f64],
        jaw_period: usize,
        teeth_period: usize,
        lips_period: usize,
        jaw_offset: usize,
        teeth_offset: usize,
        lips_offset: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized || !self.is_cuda_available() {
            return Ok(super::cpu_backend::CpuBackend::calculate_alligator(source, jaw_period, teeth_period, lips_period, jaw_offset, teeth_offset, lips_offset));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        runner.calculate_alligator(source, jaw_period, teeth_period, lips_period, jaw_offset, teeth_offset, lips_offset)
    }


    #[allow(dead_code)]
    fn run_batch_kernel(
        &self,
        _batch_tensor: &BatchTensor,
    ) -> Result<Vec<Arc<FeatureWindow>>, Box<dyn std::error::Error + Send + Sync>> {
        Ok(vec![])
    }
}

#[async_trait::async_trait]
impl ComputeBackend for CudaBackend {
    async fn compute_indicators(
        &self,
        jobs: Vec<ComputeJob>,
    ) -> Result<Vec<Arc<FeatureWindow>>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            let cpu_backend = super::cpu_backend::CpuBackend::new();
            return cpu_backend.compute_indicators(jobs).await;
        }

        let mut results = Vec::new();

        for job in jobs {
            let candle_window = match &job.candle_window {
                Some(cw) => cw,
                None => continue,
            };

            // Validate that the candle window has sufficient data for calculations
            if candle_window.close.is_empty() || candle_window.timestamps.is_empty() {
                continue; // Skip if no data
            }

            // Ensure all arrays have the same length
            let expected_len = candle_window.close.len();
            if candle_window.open.len() != expected_len ||
               candle_window.high.len() != expected_len ||
               candle_window.low.len() != expected_len ||
               candle_window.volume.len() != expected_len ||
               candle_window.timestamps.len() != expected_len {
                continue; // Skip if data arrays have inconsistent lengths
            }

            // Create a map to group features by timestamp
            let mut timestamp_features: std::collections::HashMap<i64, std::collections::HashMap<String, FeatureValue>> = std::collections::HashMap::new();

            for indicator in &job.indicators {
                match indicator.as_str() {
                    "adx" => {
                        let adx_values = self.run_adx_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14)?;
                        for (i, &adx_val) in adx_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !adx_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("adx".to_string(), FeatureValue::Float(adx_val));
                            }
                        }
                    }
                    "atr" => {
                        let atr_values = self.run_atr_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14)?;
                        for (i, &atr_val) in atr_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !atr_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("atr".to_string(), FeatureValue::Float(atr_val));
                            }
                        }
                    }
                    "bb" => {
                        let (upper, middle, lower) = self.run_bollinger_bands_kernel(&candle_window.close, 20, 2.0)?;
                        for i in 0..std::cmp::min(upper.len(), candle_window.timestamps.len()) {
                            if !upper[i].is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                let features_map = timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new);
                                features_map.insert("bb_upper".to_string(), FeatureValue::Float(upper[i]));
                                features_map.insert("bb_mid".to_string(), FeatureValue::Float(middle[i]));
                                features_map.insert("bb_lower".to_string(), FeatureValue::Float(lower[i]));
                            }
                        }
                    }
                    "cci" => {
                        let cci_values = self.run_cci_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 20)?;
                        for (i, &cci_val) in cci_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !cci_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("cci".to_string(), FeatureValue::Float(cci_val));
                            }
                        }
                    }
                    "ema" => {
                        let ema20 = self.run_ema_kernel(&candle_window.close, 20)?;
                        let ema50 = self.run_ema_kernel(&candle_window.close, 50)?;
                        let ema200 = self.run_ema_kernel(&candle_window.close, 200)?;
                        for i in 0..std::cmp::min(ema20.len(), candle_window.timestamps.len()) {
                            if !ema20[i].is_nan() || !ema50[i].is_nan() || !ema200[i].is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                let features_map = timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new);
                                if !ema20[i].is_nan() {
                                    features_map.insert("ema20".to_string(), FeatureValue::Float(ema20[i]));
                                }
                                if !ema50[i].is_nan() {
                                    features_map.insert("ema50".to_string(), FeatureValue::Float(ema50[i]));
                                }
                                if !ema200[i].is_nan() {
                                    features_map.insert("ema200".to_string(), FeatureValue::Float(ema200[i]));
                                }
                            }
                        }
                    }
                    "macd" => {
                        let (macd_line, signal_line, histogram) = self.run_macd_kernel(&candle_window.close, 12, 26, 9)?;

                        for i in 0..std::cmp::min(macd_line.len(), candle_window.timestamps.len()) {
                            if !macd_line[i].is_nan() && !signal_line[i].is_nan() && !histogram[i].is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                let features_map = timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new);
                                features_map.insert("macd".to_string(), FeatureValue::Float(macd_line[i]));
                                features_map.insert("macd_signal".to_string(), FeatureValue::Float(signal_line[i]));
                                features_map.insert("macd_hist".to_string(), FeatureValue::Float(histogram[i]));
                            }
                        }
                    }
                    "obv" => {
                        let obv_values = self.run_obv_kernel(&candle_window.close, &candle_window.volume)?;
                        for (i, &obv_val) in obv_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !obv_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("obv".to_string(), FeatureValue::Float(obv_val));
                            }
                        }
                    }
                    "rsi" => {
                        let rsi_values = self.run_rsi_kernel(&candle_window.close, 14)?;
                        for (i, &rsi_val) in rsi_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !rsi_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("rsi".to_string(), FeatureValue::Float(rsi_val));
                            }
                        }
                    }
                    "stoch" => {
                        let (k, d) = self.run_stochastic_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14, 3)?;
                        for i in 0..std::cmp::min(k.len(), candle_window.timestamps.len()) {
                            if !k[i].is_nan() && !d[i].is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                let features_map = timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new);
                                features_map.insert("stoch_k".to_string(), FeatureValue::Float(k[i]));
                                features_map.insert("stoch_d".to_string(), FeatureValue::Float(d[i]));
                            }
                        }
                    }
                    "vwap" => {
                        let vwap_values = self.run_vwap_kernel(&candle_window.high, &candle_window.low, &candle_window.close, &candle_window.volume)?;
                        for (i, &vwap_val) in vwap_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !vwap_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("vwap".to_string(), FeatureValue::Float(vwap_val));
                            }
                        }
                    }
                    "williams" => {
                        let williams_values = self.run_williams_r_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14)?;
                        for (i, &williams_val) in williams_values.iter().enumerate() {
                            if i < candle_window.timestamps.len() && !williams_val.is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new)
                                    .insert("williams".to_string(), FeatureValue::Float(williams_val));
                            }
                        }
                    }
                    "alligator" => {
                        let (jaw, teeth, lips) = self.run_alligator_kernel(&candle_window.close, 13, 8, 5, 8, 5, 3)?;
                        for i in 0..std::cmp::min(jaw.len(), candle_window.timestamps.len()) {
                            if !jaw[i].is_nan() && !teeth[i].is_nan() && !lips[i].is_nan() {
                                let timestamp = candle_window.timestamps[i];
                                let features_map = timestamp_features.entry(timestamp)
                                    .or_insert_with(std::collections::HashMap::new);
                                features_map.insert("alli_jaw".to_string(), FeatureValue::Float(jaw[i]));
                                features_map.insert("alli_teeth".to_string(), FeatureValue::Float(teeth[i]));
                                features_map.insert("alli_lips".to_string(), FeatureValue::Float(lips[i]));
                            }
                        }
                    }
                    "sr_levels" => {
                        let levels = compute_indicators::calculate_sr_levels(&candle_window.high, &candle_window.low, &candle_window.close, 0.5);
                        if let Ok(json_levels) = serde_json::to_value(&levels) {
                            // Use the last timestamp or the job's window end for SR levels
                            let timestamp = candle_window.timestamps.last().cloned().unwrap_or(job.window_end);
                            timestamp_features.entry(timestamp)
                                .or_insert_with(std::collections::HashMap::new)
                                .insert("sr_levels".to_string(), FeatureValue::Json(json_levels));
                        }
                    }
                    _ => {
                    }
                }
            }

            // Convert the grouped features into ComputeResult objects
            let mut features = Vec::new();
            for (timestamp, features_map) in timestamp_features {
                features.push(ComputeResult {
                    symbol: job.symbol.clone(),
                    timeframe: job.timeframe,
                    timestamp,
                    features: features_map,
                });
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
        symbol: Symbol,
        timeframe: Timeframe,
        prices: &[f64],
        indicator_name: &str,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            let cpu_backend = super::cpu_backend::CpuBackend::new();
            return cpu_backend.compute_single_indicator(symbol, timeframe, prices, indicator_name).await;
        }

        match indicator_name {
            "rsi" => self.run_rsi_kernel(prices, 14),
            "ema" => self.run_ema_kernel(prices, 20),
            _ => {
                let cpu_backend = super::cpu_backend::CpuBackend::new();
                cpu_backend.compute_single_indicator(symbol, timeframe, prices, indicator_name).await
            }
        }
    }
}