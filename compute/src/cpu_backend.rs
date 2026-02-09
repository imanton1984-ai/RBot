use std::sync::Arc;
use common::{Symbol, Timeframe};
use crate::{ComputeBackend, ComputeJob, FeatureWindow, FeatureBatch};
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
                None => {
                    continue;
                }
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

            // === NEW: build columnar FeatureBatch instead of per-bar hashmaps ===

            let timestamps = candle_window.timestamps.clone();
            let n = timestamps.len();
            let mut batch = FeatureBatch::new(timestamps);

            // Кеши, чтобы не пересчитывать одно и то же 3 раза (bb/macd/stoch/alligator)
            let mut bb_cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
            let mut macd_cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;
            let mut stoch_cache: Option<(Vec<f64>, Vec<f64>)> = None;
            let mut alligator_cache: Option<(Vec<f64>, Vec<f64>, Vec<f64>)> = None;

            // Важно: не создаём никаких HashMap на каждый бар.
            // Добавляем колонки (строка-имя 1 раз на колонку).
            for indicator in &job.indicators {
                match indicator.as_str() {
                    // ALIASES for grouped indicators
                    "ema" => {
                        if batch.get_f64("ema_20").is_none() { batch.push_f64("ema_20", Self::calculate_ema(&candle_window.close, 20)); }
                        if batch.get_f64("ema_50").is_none() { batch.push_f64("ema_50", Self::calculate_ema(&candle_window.close, 50)); }
                        if batch.get_f64("ema_200").is_none() { batch.push_f64("ema_200", Self::calculate_ema(&candle_window.close, 200)); }
                    }
                    "bb" | "bollinger" => {
                        if bb_cache.is_none() {
                            bb_cache = Some(Self::calculate_bollinger_bands(&candle_window.close, 20, 2.0));
                        }
                        let (upper, mid, lower) = bb_cache.as_ref().unwrap();
                        if batch.get_f64("bb_upper").is_none() { batch.push_f64("bb_upper", upper.clone()); }
                        if batch.get_f64("bb_mid").is_none() { batch.push_f64("bb_mid", mid.clone()); }
                        if batch.get_f64("bb_lower").is_none() { batch.push_f64("bb_lower", lower.clone()); }
                    }
                    "stoch" | "stochastic" => {
                        if stoch_cache.is_none() {
                            stoch_cache = Some(Self::calculate_stochastic(&candle_window.high, &candle_window.low, &candle_window.close, 14, 3));
                        }
                        let (k, d) = stoch_cache.as_ref().unwrap();
                        if batch.get_f64("stoch_k").is_none() { batch.push_f64("stoch_k", k.clone()); }
                        if batch.get_f64("stoch_d").is_none() { batch.push_f64("stoch_d", d.clone()); }
                    }
                    "alligator" => {
                        if alligator_cache.is_none() {
                            alligator_cache = Some(Self::calculate_alligator(&candle_window.close, 13, 8, 5, 8, 5, 3));
                        }
                        let (jaw, teeth, lips) = alligator_cache.as_ref().unwrap();
                        if batch.get_f64("alligator_jaw").is_none() { batch.push_f64("alligator_jaw", jaw.clone()); }
                        if batch.get_f64("alligator_teeth").is_none() { batch.push_f64("alligator_teeth", teeth.clone()); }
                        if batch.get_f64("alligator_lips").is_none() { batch.push_f64("alligator_lips", lips.clone()); }
                    }

                    "adx" => {
                        let v = Self::calculate_adx(&candle_window.high, &candle_window.low, &candle_window.close, 14);
                        batch.push_f64("adx", v);
                    }
                    "atr" => {
                        let v = Self::calculate_atr(&candle_window.high, &candle_window.low, &candle_window.close, 14);
                        batch.push_f64("atr", v);
                    }
                    "cci" => {
                        let v = Self::calculate_cci(&candle_window.high, &candle_window.low, &candle_window.close, 20);
                        batch.push_f64("cci", v);
                    }
                    "ema_20" => batch.push_f64("ema_20", Self::calculate_ema(&candle_window.close, 20)),
                    "ema_50" => batch.push_f64("ema_50", Self::calculate_ema(&candle_window.close, 50)),
                    "ema_200" => batch.push_f64("ema_200", Self::calculate_ema(&candle_window.close, 200)),
                    "sma" => batch.push_f64("sma", Self::calculate_sma(&candle_window.close, 20)),
                    "rsi" => batch.push_f64("rsi", Self::calculate_rsi(&candle_window.close, 14)),
                    "obv" => batch.push_f64("obv", Self::calculate_obv(&candle_window.close, &candle_window.volume)),
                    "vwap" => batch.push_f64("vwap", Self::calculate_vwap(&candle_window.high, &candle_window.low, &candle_window.close, &candle_window.volume)),
                    "williams" => batch.push_f64("williams", Self::calculate_williams_r(&candle_window.high, &candle_window.low, &candle_window.close, 14)),

                    "bb_upper" | "bb_mid" | "bb_lower" => {
                        if bb_cache.is_none() {
                            bb_cache = Some(Self::calculate_bollinger_bands(&candle_window.close, 20, 2.0));
                        }
                        let (upper, mid, lower) = bb_cache.as_ref().unwrap();
                        // пушим все 3 колонки один раз (не важно, на каком из 3 индикаторов мы сюда попали)
                        if batch.get_f64("bb_upper").is_none() { batch.push_f64("bb_upper", upper.clone()); }
                        if batch.get_f64("bb_mid").is_none() { batch.push_f64("bb_mid", mid.clone()); }
                        if batch.get_f64("bb_lower").is_none() { batch.push_f64("bb_lower", lower.clone()); }
                    }

                    "macd" | "macd_signal" | "macd_hist" => {
                        if macd_cache.is_none() {
                            macd_cache = Some(Self::calculate_macd(&candle_window.close, 12, 26, 9));
                        }
                        let (macd, signal, hist) = macd_cache.as_ref().unwrap();
                        if batch.get_f64("macd").is_none() { batch.push_f64("macd", macd.clone()); }
                        if batch.get_f64("macd_signal").is_none() { batch.push_f64("macd_signal", signal.clone()); }
                        if batch.get_f64("macd_hist").is_none() { batch.push_f64("macd_hist", hist.clone()); }
                    }

                    "stoch_k" | "stoch_d" => {
                        if stoch_cache.is_none() {
                            stoch_cache = Some(Self::calculate_stochastic(&candle_window.high, &candle_window.low, &candle_window.close, 14, 3));
                        }
                        let (k, d) = stoch_cache.as_ref().unwrap();
                        if batch.get_f64("stoch_k").is_none() { batch.push_f64("stoch_k", k.clone()); }
                        if batch.get_f64("stoch_d").is_none() { batch.push_f64("stoch_d", d.clone()); }
                    }

                    "alligator_jaw" | "alligator_teeth" | "alligator_lips" => {
                        if alligator_cache.is_none() {
                            alligator_cache = Some(Self::calculate_alligator(&candle_window.close, 13, 8, 5, 8, 5, 3));
                        }
                        let (jaw, teeth, lips) = alligator_cache.as_ref().unwrap();
                        if batch.get_f64("alligator_jaw").is_none() { batch.push_f64("alligator_jaw", jaw.clone()); }
                        if batch.get_f64("alligator_teeth").is_none() { batch.push_f64("alligator_teeth", teeth.clone()); }
                        if batch.get_f64("alligator_lips").is_none() { batch.push_f64("alligator_lips", lips.clone()); }
                    }

                    "volume_spike" => {
                        // Use the new ratio function instead of binary
                        let ratios = compute_indicators::calculate_volume_spike_ratio(&candle_window.volume, 20);
                        batch.push_f64("volume_spike", ratios);
                    }

                    "trend" => {
                        let trends = compute_indicators::calculate_long_term_trend(&candle_window.close, 20, 50);
                        // Convert i8 vector to f64 vector for storage
                        let trend_values: Vec<f64> = trends.iter().map(|&t| t as f64).collect();
                        batch.push_f64("trend", trend_values);
                    }

                    "trend_short" => {
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

                    _ => { /* unknown indicator — игнор */ }
                }
            }

            let feature_window = Arc::new(FeatureWindow {
                symbol: job.symbol.clone(),
                timeframe: job.timeframe,
                start_time: job.window_start,
                end_time: job.window_end,
                is_realtime: job.is_realtime,
                candle_window: Some(candle_window.clone()),
                batch,
                legacy_features: None,
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