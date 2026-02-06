use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{
    ComputeBackend, ComputeJob, FeatureWindow,
    BatchTensor
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

                    "sr_levels" => {
                        let levels = compute_indicators::calculate_sr_levels(&candle_window.high, &candle_window.low, &candle_window.close, 0.5);
                        let mut v = vec![serde_json::Value::Null; n];
                        if n > 0 {
                            v[n - 1] = serde_json::to_value(levels).unwrap_or(serde_json::Value::Null);
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
            _ => {
                let cpu_backend = super::cpu_backend::CpuBackend::new();
                cpu_backend.compute_single_indicator(symbol, timeframe, prices, indicator_name).await
            }
        }
    }
}