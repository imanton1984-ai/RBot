use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{
    ComputeBackend, ComputeJob, FeatureWindow,
    BatchTensor
};
use tracing;

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
        if cuda::get_cuda_device().is_some() {
            self.initialized = true;
            tracing::info!("CudaBackend initialized successfully.");
        } else {
            self.initialized = false;
            tracing::warn!("CudaBackend initialization failed. Falling back to CPU where applicable.");
        }
        Ok(())
    }

    fn run_rsi_kernel(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_rsi(input, period));
        }

        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_rsi(input, period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA RSI failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_rsi(input, period))
            }
        }
    }

    fn run_ema_kernel(
        &self,
        input: &[f64],
        period: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_ema(input, period));
        }

        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_ema(input, period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA EMA failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_ema(input, period))
            }
        }
    }

    fn run_macd_kernel(
        &self,
        input: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized{
            return Ok(super::cpu_backend::CpuBackend::calculate_macd(input, fast_period, slow_period, signal_period));
        }

        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_macd(input, fast_period, slow_period, signal_period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA MACD failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_macd(input, fast_period, slow_period, signal_period))
            }
        }
    }

    fn run_adx_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_adx(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_adx(high, low, close, period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA ADX failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_adx(high, low, close, period))
            }
        }
    }

    fn run_atr_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_atr(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_atr(high, low, close, period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA ATR failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_atr(high, low, close, period))
            }
        }
    }

    fn run_bollinger_bands_kernel(
        &self,
        prices: &[f64],
        period: usize,
        num_std_dev: f64,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_bollinger_bands(prices, period, num_std_dev));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_bollinger_bands(prices, period, num_std_dev) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA Bollinger Bands failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_bollinger_bands(prices, period, num_std_dev))
            }
        }
    }
    
    fn run_cci_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_cci(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_cci(high, low, close, period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA CCI failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_cci(high, low, close, period))
            }
        }
    }

    fn run_obv_kernel(
        &self,
        close: &[f64], volume: &[f64]
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_obv(close, volume));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_obv(close, volume) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA OBV failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_obv(close, volume))
            }
        }
    }

    fn run_stochastic_kernel(
        &self,
        high: &[f64],
        low: &[f64],
        close: &[f64],
        k_period: usize,
        d_period: usize,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_stochastic(high, low, close, k_period, d_period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_stochastic(high, low, close, k_period, d_period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA Stochastic failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_stochastic(high, low, close, k_period, d_period))
            }
        }
    }

    fn run_vwap_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], volume: &[f64]
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_vwap(high, low, close, volume));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_vwap(high, low, close, volume) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA VWAP failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_vwap(high, low, close, volume))
            }
        }
    }

    fn run_williams_r_kernel(
        &self,
        high: &[f64], low: &[f64], close: &[f64], period: usize
    ) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_williams_r(high, low, close, period));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_williams_r(high, low, close, period) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA Williams %R failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_williams_r(high, low, close, period))
            }
        }
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
        if !self.initialized {
            return Ok(super::cpu_backend::CpuBackend::calculate_alligator(source, jaw_period, teeth_period, lips_period, jaw_offset, teeth_offset, lips_offset));
        }
        let runner = cuda::IndicatorKernelRunner::new();
        match runner.calculate_alligator(source, jaw_period, teeth_period, lips_period, jaw_offset, teeth_offset, lips_offset) {
            Ok(res) => Ok(res),
            Err(e) => {
                tracing::error!("CUDA Alligator failed: {}. Falling back to CPU", e);
                Ok(super::cpu_backend::CpuBackend::calculate_alligator(source, jaw_period, teeth_period, lips_period, jaw_offset, teeth_offset, lips_offset))
            }
        }
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

            if candle_window.close.is_empty() || candle_window.timestamps.is_empty() {
                continue;
            }

            let expected_len = candle_window.close.len();
            if candle_window.open.len() != expected_len ||
               candle_window.high.len() != expected_len ||
               candle_window.low.len() != expected_len ||
               candle_window.volume.len() != expected_len ||
               candle_window.timestamps.len() != expected_len {
                continue;
            }

            let timestamps = candle_window.timestamps.clone();
            let n = timestamps.len();
            let mut batch = crate::FeatureBatch::new(timestamps);

            let mut bb_cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
            let mut macd_cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
            let mut stoch_cache: Option<(Vec<f64>, Vec<f64>)> = None;
            let mut alligator_cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;

            for indicator in &job.indicators {
                match indicator.as_str() {
                    "adx" => {
                        let v = self.run_adx_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14)?;
                        batch.push_f64("adx", v);
                    }
                    "atr" => {
                        let v = self.run_atr_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14)?;
                        batch.push_f64("atr", v);
                    }
                    "cci" => {
                        let v = self.run_cci_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 20)?;
                        batch.push_f64("cci", v);
                    }
                    "ema" => { // Assuming "ema" implies multiple EMAs
                        batch.push_f64("ema_20", self.run_ema_kernel(&candle_window.close, 20)?);
                        batch.push_f64("ema_50", self.run_ema_kernel(&candle_window.close, 50)?);
                        batch.push_f64("ema_200", self.run_ema_kernel(&candle_window.close, 200)?);
                    }
                    "rsi" => batch.push_f64("rsi", self.run_rsi_kernel(&candle_window.close, 14)?),
                    "obv" => batch.push_f64("obv", self.run_obv_kernel(&candle_window.close, &candle_window.volume)?),
                    "vwap" => batch.push_f64("vwap", self.run_vwap_kernel(&candle_window.high, &candle_window.low, &candle_window.close, &candle_window.volume)?),
                    "williams" => batch.push_f64("williams", self.run_williams_r_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14)?),

                    "bb" | "bb_upper" | "bb_mid" | "bb_lower" => {
                        if bb_cache.is_none() {
                            bb_cache = Some(self.run_bollinger_bands_kernel(&candle_window.close, 20, 2.0)?);
                        }
                        let (upper, mid, lower) = bb_cache.as_ref().unwrap();
                        if batch.get_f64("bb_upper").is_none() { batch.push_f64("bb_upper", upper.clone()); }
                        if batch.get_f64("bb_mid").is_none() { batch.push_f64("bb_mid", mid.clone()); }
                        if batch.get_f64("bb_lower").is_none() { batch.push_f64("bb_lower", lower.clone()); }
                    }

                    "macd" | "macd_signal" | "macd_hist" => {
                        if macd_cache.is_none() {
                            macd_cache = Some(self.run_macd_kernel(&candle_window.close, 12, 26, 9)?);
                        }
                        let (macd, signal, hist) = macd_cache.as_ref().unwrap();
                        if batch.get_f64("macd").is_none() { batch.push_f64("macd", macd.clone()); }
                        if batch.get_f64("macd_signal").is_none() { batch.push_f64("macd_signal", signal.clone()); }
                        if batch.get_f64("macd_hist").is_none() { batch.push_f64("macd_hist", hist.clone()); }
                    }

                    "stoch" | "stoch_k" | "stoch_d" => {
                        if stoch_cache.is_none() {
                            stoch_cache = Some(self.run_stochastic_kernel(&candle_window.high, &candle_window.low, &candle_window.close, 14, 3)?);
                        }
                        let (k, d) = stoch_cache.as_ref().unwrap();
                        if batch.get_f64("stoch_k").is_none() { batch.push_f64("stoch_k", k.clone()); }
                        if batch.get_f64("stoch_d").is_none() { batch.push_f64("stoch_d", d.clone()); }
                    }

                    "alligator" | "alligator_jaw" | "alligator_teeth" | "alligator_lips" => {
                        if alligator_cache.is_none() {
                            alligator_cache = Some(self.run_alligator_kernel(&candle_window.close, 13, 8, 5, 8, 5, 3)?);
                        }
                        let (jaw, teeth, lips) = alligator_cache.as_ref().unwrap();
                        if batch.get_f64("alligator_jaw").is_none() { batch.push_f64("alligator_jaw", jaw.clone()); }
                        if batch.get_f64("alligator_teeth").is_none() { batch.push_f64("alligator_teeth", teeth.clone()); }
                        if batch.get_f64("alligator_lips").is_none() { batch.push_f64("alligator_lips", lips.clone()); }
                    }

                    "volume_spike" => {
                        // Fall back to CPU implementation for volume_spike
                        let ratios = compute_indicators::calculate_volume_spike_ratio(&candle_window.volume, 20);
                        batch.push_f64("volume_spike", ratios);
                    }

                    "trend" => {
                        // Fall back to CPU implementation for trend
                        let trends = compute_indicators::calculate_long_term_trend(&candle_window.close, 20, 50);
                        // Convert i8 vector to f64 vector for storage
                        let trend_values: Vec<f64> = trends.iter().map(|&t| t as f64).collect();
                        batch.push_f64("trend", trend_values);
                    }

                    "trend_short" => {
                        // Fall back to CPU implementation for trend_short
                        let trends = compute_indicators::calculate_short_term_trend(&candle_window.close, 5, 10);
                        // Convert i8 vector to f64 vector for storage
                        let trend_values: Vec<f64> = trends.iter().map(|&t| t as f64).collect();
                        batch.push_f64("trend_short", trend_values);
                    }

                    "poc" => {
                        // Calculate POC (Point of Control) based on volume-weighted price
                        // This is a simplified approach using the close price of the candle with highest volume
                        let mut poc_values = Vec::with_capacity(n);
                        
                        for i in 0..n {
                            // Use a rolling window to calculate POC
                            let lookback = std::cmp::min(i + 1, 50); // Look at most recent 50 candles
                            let start_idx = i + 1 - lookback;
                            
                            let window_volumes = &candle_window.volume[start_idx..=i];
                            let window_closes = &candle_window.close[start_idx..=i];
                            
                            // Find the index of the candle with the highest volume in the window
                            let max_vol_idx = window_volumes
                                .iter()
                                .enumerate()
                                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                                .map(|(idx, _)| idx)
                                .unwrap_or(0);
                                
                            // Use the close price of the candle with highest volume as POC
                            poc_values.push(window_closes[max_vol_idx]);
                        }
                        
                        batch.push_f64("poc", poc_values);
                    }

                    "sr_levels" => {
                        // Calculate sr_levels for each candle using rolling window approach
                        let mut v = Vec::with_capacity(n);
                        for i in 0..n {
                            // Use a lookback window for calculating sr_levels up to current candle
                            let lookback = std::cmp::min(i + 1, 100); // Use up to 100 candles for calculation
                            let start_idx = i + 1 - lookback;
                            
                            let high_slice = &candle_window.high[start_idx..=i];
                            let low_slice = &candle_window.low[start_idx..=i];
                            let close_slice = &candle_window.close[start_idx..=i];
                            
                            let levels = compute_indicators::calculate_sr_levels(high_slice, low_slice, close_slice, 0.5);
                            v.push(serde_json::to_value(levels).unwrap_or(serde_json::Value::Null));
                        }
                        batch.push_json("sr_levels", v);
                    }
                    _ => {}
                }
            }

            let feature_window = Arc::new(FeatureWindow {
                symbol: job.symbol.clone(),
                timeframe: job.timeframe,
                start_time: job.window_start,
                end_time: job.window_end,
                is_realtime: job.is_realtime,
                candle_window: job.candle_window.clone(),
                batch,
                legacy_features: None,
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
            "ema_20" => self.run_ema_kernel(prices, 20),
            "ema_50" => self.run_ema_kernel(prices, 50),
            "ema_200" => self.run_ema_kernel(prices, 200),
            _ => {
                let cpu_backend = super::cpu_backend::CpuBackend::new();
                cpu_backend.compute_single_indicator(symbol, timeframe, prices, indicator_name).await
            }
        }
    }
}