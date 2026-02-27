// strategies/ewmac_strategy/src/ewmac.rs
//
// Core EWMAC (Exponentially Weighted Moving Average Crossover) calculation.
//
// Implements:
//   1. EMA computation (incrementally from price series)
//   2. Per-pair EWMAC signal: (EMA_fast - EMA_slow) / ATR
//   3. Forecast scaling: raw_signal * scalar, capped at [-max, +max]
//   4. Aggregate forecast: weighted average of all pairs * FDM
//
// Hot-path optimized:
//   - No .clone() in inner loops
//   - Pre-allocated output buffers
//   - Inline EMA update (single multiply + add)

use crate::config::EwmacConfig;

/// Result of EWMAC computation for a single candle
#[derive(Debug, Clone, Copy)]
pub struct EwmacResult {
    /// Aggregate forecast (scaled, capped to [-max_forecast, +max_forecast])
    pub forecast: f64,
    /// Raw aggregate signal (unnormalized: mean of per-pair raw values)
    pub raw_signal: f64,
    /// Normalized aggregate signal (mean of per-pair ATR-normalized values)
    pub norm_signal: f64,
    /// Signal strength: |forecast| / max_forecast (0..1)
    pub signal_strength: f64,
    /// Per-pair normalized values (in order of config.pairs)
    pub per_pair: [f64; 6],
    /// ATR at this candle
    pub atr: f64,
    /// ATR as percentage of close
    pub atr_pct: f64,
    /// Number of pairs agreeing on the forecast direction (same sign)
    pub pairs_agree: usize,
}

impl Default for EwmacResult {
    fn default() -> Self {
        Self {
            forecast: 0.0,
            raw_signal: 0.0,
            norm_signal: 0.0,
            signal_strength: 0.0,
            per_pair: [0.0; 6],
            atr: 0.0,
            atr_pct: 0.0,
            pairs_agree: 0,
        }
    }
}

/// EMA state for a single span
#[derive(Debug, Clone, Copy)]
struct EmaState {
    span: usize,
    alpha: f64,
    value: f64,
    initialized: bool,
    count: usize,
}

impl EmaState {
    fn new(span: usize) -> Self {
        let alpha = 2.0 / (span as f64 + 1.0);
        Self {
            span,
            alpha,
            value: 0.0,
            initialized: false,
            count: 0,
        }
    }

    /// Update EMA with a new price value.
    /// Uses SMA for the first `span` values as seed.
    #[inline(always)]
    fn update(&mut self, price: f64) {
        self.count += 1;
        if !self.initialized {
            if self.count == 1 {
                self.value = price;
            } else {
                // Running SMA until we have `span` values
                self.value += (price - self.value) / self.count as f64;
            }
            if self.count >= self.span {
                self.initialized = true;
            }
        } else {
            // Standard EMA update: new = alpha * price + (1 - alpha) * old
            self.value = self.alpha * price + (1.0 - self.alpha) * self.value;
        }
    }
}

/// ATR (Average True Range) state
#[derive(Debug, Clone, Copy)]
struct AtrState {
    period: usize,
    value: f64,
    count: usize,
    prev_close: f64,
    initialized: bool,
}

impl AtrState {
    fn new(period: usize) -> Self {
        Self {
            period,
            value: 0.0,
            count: 0,
            prev_close: 0.0,
            initialized: false,
        }
    }

    /// Update ATR with new OHLC data.
    /// True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    #[inline(always)]
    fn update(&mut self, high: f64, low: f64, close: f64) {
        let tr = if self.count == 0 {
            high - low
        } else {
            let hl = high - low;
            let hpc = (high - self.prev_close).abs();
            let lpc = (low - self.prev_close).abs();
            hl.max(hpc).max(lpc)
        };

        self.count += 1;

        if !self.initialized {
            if self.count == 1 {
                self.value = tr;
            } else {
                // Running average until period
                self.value += (tr - self.value) / self.count as f64;
            }
            if self.count >= self.period {
                self.initialized = true;
            }
        } else {
            // Wilder's smoothing: ATR = ((period-1) * prev_atr + TR) / period
            self.value = ((self.period as f64 - 1.0) * self.value + tr) / self.period as f64;
        }

        self.prev_close = close;
    }
}

/// EWMAC Calculator — maintains EMAs and ATR state for incremental computation
pub struct EwmacCalculator {
    config: EwmacConfig,
    /// EMA states: two per pair (fast + slow)
    ema_states: Vec<(EmaState, EmaState)>,
    /// ATR state (period = 14)
    atr_state: AtrState,
    /// Number of candles processed
    candle_count: usize,
}

impl EwmacCalculator {
    /// Create a new EWMAC calculator
    pub fn new(config: &EwmacConfig) -> Self {
        let ema_states: Vec<(EmaState, EmaState)> = config.pairs
            .iter()
            .map(|p| (EmaState::new(p.fast), EmaState::new(p.slow)))
            .collect();

        Self {
            config: config.clone(),
            ema_states,
            atr_state: AtrState::new(14),
            candle_count: 0,
        }
    }

    /// Reset calculator state (for processing a new symbol)
    pub fn reset(&mut self) {
        self.ema_states = self.config.pairs
            .iter()
            .map(|p| (EmaState::new(p.fast), EmaState::new(p.slow)))
            .collect();
        self.atr_state = AtrState::new(14);
        self.candle_count = 0;
    }

    /// Process a single candle (OHLC) and return the EWMAC result.
    ///
    /// Must be called sequentially for correct EMA/ATR state.
    /// Returns None during warmup period.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<EwmacResult> {
        self.candle_count += 1;

        // Update ATR
        self.atr_state.update(high, low, close);

        // Update all EMAs with close price
        for (fast_ema, slow_ema) in &mut self.ema_states {
            fast_ema.update(close);
            slow_ema.update(close);
        }

        // Need warmup before producing signals
        // warmup_bars = N means we skip the first N candles
        if self.candle_count <= self.config.warmup_bars {
            return None;
        }

        let atr = self.atr_state.value;
        if atr < 1e-12 {
            return Some(EwmacResult::default());
        }

        let atr_pct = if close > 0.0 { atr / close * 100.0 } else { 0.0 };

        // Compute per-pair signals
        let mut per_pair = [0.0f64; 6];
        let mut weighted_forecast_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut raw_sum = 0.0;
        let mut norm_sum = 0.0;
        let n_pairs = self.config.pairs.len().min(6);

        for i in 0..n_pairs {
            let (ref fast_ema, ref slow_ema) = self.ema_states[i];
            let pair = &self.config.pairs[i];

            // Raw crossover = EMA_fast - EMA_slow
            let raw_cross = fast_ema.value - slow_ema.value;
            raw_sum += raw_cross;

            // Normalize by ATR
            let norm_cross = raw_cross / atr;
            per_pair[i] = norm_cross;
            norm_sum += norm_cross;

            // Scale to forecast
            let scalar = self.config.scalar_for_pair(pair.fast, pair.slow);
            let forecast = (norm_cross * scalar).clamp(-self.config.max_forecast, self.config.max_forecast);

            weighted_forecast_sum += forecast * pair.weight;
            weight_sum += pair.weight;
        }

        // Aggregate forecast = weighted average * FDM, capped
        let aggregate_forecast = if weight_sum > 0.0 {
            let avg = weighted_forecast_sum / weight_sum;
            (avg * self.config.fdm).clamp(-self.config.max_forecast, self.config.max_forecast)
        } else {
            0.0
        };

        // Count pairs that agree with the aggregate direction
        let forecast_sign = if aggregate_forecast >= 0.0 { 1 } else { -1 };
        let pairs_agree = (0..n_pairs)
            .filter(|&i| {
                let pair_sign = if per_pair[i] >= 0.0 { 1 } else { -1 };
                pair_sign == forecast_sign
            })
            .count();

        let signal_strength = aggregate_forecast.abs() / self.config.max_forecast;

        Some(EwmacResult {
            forecast: aggregate_forecast,
            raw_signal: raw_sum / n_pairs as f64,
            norm_signal: norm_sum / n_pairs as f64,
            signal_strength,
            per_pair,
            atr,
            atr_pct,
            pairs_agree,
        })
    }

    /// Process a series of OHLC candles and return results for each.
    ///
    /// Returns Vec of (candle_index, EwmacResult) for candles after warmup.
    pub fn process_series(
        &mut self,
        candles: &[(f64, f64, f64, f64)], // (open, high, low, close)
    ) -> Vec<(usize, EwmacResult)> {
        let mut results = Vec::with_capacity(
            candles.len().saturating_sub(self.config.warmup_bars)
        );

        for (i, &(_open, high, low, close)) in candles.iter().enumerate() {
            if let Some(result) = self.update(high, low, close) {
                results.push((i, result));
            }
        }

        results
    }
}

/// Batch EWMAC computation for a Vec of CandleData (from DB).
/// Resets internal state before processing.
pub fn compute_ewmac_for_candles(
    config: &EwmacConfig,
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
) -> Vec<(usize, EwmacResult)> {
    assert_eq!(highs.len(), lows.len());
    assert_eq!(highs.len(), closes.len());

    let n = highs.len();
    let mut calc = EwmacCalculator::new(config);
    let mut results = Vec::with_capacity(n.saturating_sub(config.warmup_bars));

    for i in 0..n {
        if let Some(result) = calc.update(highs[i], lows[i], closes[i]) {
            results.push((i, result));
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> EwmacConfig {
        let mut cfg = EwmacConfig::default();
        cfg.warmup_bars = 10; // small warmup for tests
        cfg
    }

    #[test]
    fn test_ema_state_basic() {
        let mut ema = EmaState::new(3);
        // Feed 10 values
        for i in 1..=10 {
            ema.update(i as f64);
        }
        assert!(ema.initialized);
        // After 10 updates of 1,2,...10, EMA(3) should be near the later values
        assert!(ema.value > 7.0); // biased towards recent
    }

    #[test]
    fn test_atr_state_basic() {
        let mut atr = AtrState::new(14);
        // Constant range of 1.0
        for i in 0..20 {
            atr.update(101.0 + i as f64, 100.0 + i as f64, 100.5 + i as f64);
        }
        assert!(atr.initialized);
        // ATR should be approximately 1.0 for constant range
        assert!((atr.value - 1.0).abs() < 0.5);
    }

    #[test]
    fn test_ewmac_calculator_warmup() {
        let config = make_config();
        let mut calc = EwmacCalculator::new(&config);

        // During warmup (first 10 candles), no results
        for i in 0..10 {
            let result = calc.update(101.0, 99.0, 100.0);
            assert!(result.is_none(), "Expected None during warmup at candle {}", i);
        }

        // After warmup (11th candle), should produce results
        let result = calc.update(101.0, 99.0, 100.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_ewmac_trending_up() {
        let config = make_config();
        let mut calc = EwmacCalculator::new(&config);

        // Feed uptrending prices: 100, 101, 102, ...
        for i in 0..50 {
            let price = 100.0 + i as f64;
            let _ = calc.update(price + 0.5, price - 0.5, price);
        }

        // Last result should show positive forecast (uptrend)
        let result = calc.update(150.5, 149.5, 150.0);
        let r = result.expect("Should have result after warmup");
        assert!(r.forecast > 0.0, "Forecast should be positive in uptrend: {}", r.forecast);
        assert!(r.signal_strength > 0.0);
    }

    #[test]
    fn test_ewmac_trending_down() {
        let config = make_config();
        let mut calc = EwmacCalculator::new(&config);

        // Feed downtrending prices: 200, 199, 198, ...
        for i in 0..50 {
            let price = 200.0 - i as f64;
            let _ = calc.update(price + 0.5, price - 0.5, price);
        }

        let result = calc.update(150.5, 149.5, 150.0);
        let r = result.expect("Should have result");
        assert!(r.forecast < 0.0, "Forecast should be negative in downtrend: {}", r.forecast);
    }

    #[test]
    fn test_ewmac_reset() {
        let config = make_config();
        let mut calc = EwmacCalculator::new(&config);

        // Process some candles
        for _ in 0..20 {
            let _ = calc.update(101.0, 99.0, 100.0);
        }
        assert_eq!(calc.candle_count, 20);

        // Reset
        calc.reset();
        assert_eq!(calc.candle_count, 0);

        // Should return None again during warmup
        let result = calc.update(101.0, 99.0, 100.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_forecast_capping() {
        let config = make_config();
        let mut calc = EwmacCalculator::new(&config);

        // Create extreme trend to trigger capping
        for i in 0..100 {
            let price = 100.0 + (i as f64).powi(2) * 0.1; // exponential rise
            calc.update(price + 0.1, price - 0.1, price);
        }

        // Should not exceed max_forecast
        let result = calc.update(1100.0, 1099.5, 1099.8);
        if let Some(r) = result {
            assert!(r.forecast.abs() <= config.max_forecast + 1e-10,
                "Forecast {} exceeds max {}", r.forecast, config.max_forecast);
        }
    }

    #[test]
    fn test_batch_compute() {
        let config = make_config();
        let n = 50;
        let highs: Vec<f64> = (0..n).map(|i| 101.0 + i as f64).collect();
        let lows: Vec<f64> = (0..n).map(|i| 99.0 + i as f64).collect();
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();

        let results = compute_ewmac_for_candles(&config, &highs, &lows, &closes);

        // Should have n - warmup_bars results
        assert_eq!(results.len(), n - config.warmup_bars);

        // First result index should be warmup_bars
        assert_eq!(results[0].0, config.warmup_bars);

        // All results should have positive forecast (uptrend)
        for (_, r) in &results {
            assert!(r.forecast > 0.0, "Expected positive forecast in uptrend");
        }
    }
}
