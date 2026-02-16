
/// Normalizes a signal value to the range [0, 1] based on min/max values
pub fn normalize_signal(value: f64, min_val: f64, max_val: f64) -> f64 {
    if max_val == min_val {
        return 0.5; // Return middle value if min and max are the same
    }
    
    let normalized = (value - min_val) / (max_val - min_val);
    
    // Clamp to [0, 1] range
    normalized.clamp(0.0, 1.0)
}

/// Applies sigmoid normalization to emphasize strong signals
pub fn sigmoid_normalize(value: f64, center: f64, steepness: f64) -> f64 {
    let x = (value - center) * steepness;
    1.0 / (1.0 + (-x).exp())
}

/// Converts raw indicator values to normalized scores in [0, 1] range.
/// Backward-compatible version — uses fixed normalization constants.
/// For price-dependent indicators (ATR, MACD) prefer
/// [`normalize_indicator_to_score_with_price`] which normalizes relative to `close_price`.
pub fn normalize_indicator_to_score(raw_value: f64, indicator_type: &str) -> f64 {
    match indicator_type {
        "rsi" => normalize_rsi_score(raw_value),
        "macd" => normalize_macd_score(raw_value),
        "cci" => normalize_cci_score(raw_value),
        "stoch" => normalize_stoch_score(raw_value),
        "williams" => normalize_williams_score(raw_value),
        "atr" => normalize_atr_score(raw_value),
        "adx" => normalize_adx_score(raw_value),
        _ => normalize_generic_score(raw_value),
    }
}

/// Price-aware normalization for indicators whose absolute magnitude depends on asset price.
///
/// ATR and MACD values scale with the underlying price (BTC ATR ≈ 1500 USD vs DOGE ATR ≈ 0.002 USD).
/// This function normalizes them as a **fraction of `close_price`**, producing comparable scores
/// across all trading pairs.
///
/// For indicators that are already dimensionless (RSI, CCI, Stochastic, Williams, ADX) this
/// delegates to [`normalize_indicator_to_score`] — passing `close_price` is harmless for those.
pub fn normalize_indicator_to_score_with_price(
    raw_value: f64,
    indicator_type: &str,
    close_price: f64,
) -> f64 {
    match indicator_type {
        "atr" => normalize_atr_score_with_price(raw_value, close_price),
        "macd" => normalize_macd_score_with_price(raw_value, close_price),
        // Remaining indicators are dimensionless — close_price is irrelevant
        _ => normalize_indicator_to_score(raw_value, indicator_type),
    }
}

// ---------------------------------------------------------------------------
// Per-indicator normalization helpers
// ---------------------------------------------------------------------------

/// Normalize RSI values to score (RSI typically ranges 0-100)
fn normalize_rsi_score(rsi_value: f64) -> f64 {
    if rsi_value.is_nan() {
        return 0.0;
    }
    
    // RSI: values near 0 or 100 are strong signals
    // RSI near 50 is neutral
    let distance_from_extremes = ((rsi_value - 0.0).abs().min((rsi_value - 100.0).abs())).min(50.0);
    let normalized = 1.0 - (distance_from_extremes / 50.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.5, 8.0)
}

/// MACD: backward-compatible fixed normalization (kept for callers that lack close_price).
fn normalize_macd_score(macd_value: f64) -> f64 {
    if macd_value.is_nan() {
        return 0.0;
    }
    
    let abs_value = macd_value.abs();
    // Legacy fixed constant — inaccurate for multi-asset.
    // Prefer normalize_macd_score_with_price() instead.
    let normalized = (abs_value / 10.0).min(1.0);
    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// MACD: price-aware normalization.
/// Expresses MACD as permille (‰) of `close_price`, capped at 1.0.
/// For BTC @ 60 000 a MACD of 60 → 60/60000*1000 = 1.0 (max).
/// For DOGE @ 0.1 a MACD of 0.0001 → 0.0001/0.1*1000 = 1.0.
fn normalize_macd_score_with_price(macd_value: f64, close_price: f64) -> f64 {
    if macd_value.is_nan() {
        return 0.0;
    }

    let abs_value = macd_value.abs();
    let normalized = if close_price > 0.0 {
        (abs_value / close_price * 1000.0).min(1.0)
    } else {
        0.0
    };

    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// Normalize CCI values to score
fn normalize_cci_score(cci_value: f64) -> f64 {
    if cci_value.is_nan() {
        return 0.0;
    }
    
    // CCI: values beyond ±100 are considered strong
    let abs_value = cci_value.abs();
    let normalized = (abs_value / 200.0).min(1.0); // Assuming max meaningful value is 200
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// Normalize Stochastic values to score
fn normalize_stoch_score(stoch_value: f64) -> f64 {
    if stoch_value.is_nan() {
        return 0.0;
    }
    
    // Stochastic: values near 0 or 100 are strong signals
    let distance_from_extremes = ((stoch_value - 0.0).abs().min((stoch_value - 100.0).abs())).min(50.0);
    let normalized = 1.0 - (distance_from_extremes / 50.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.5, 8.0)
}

/// Normalize Williams %R values to score
fn normalize_williams_score(williams_value: f64) -> f64 {
    if williams_value.is_nan() {
        return 0.0;
    }
    
    // Williams %R: values near -0 or -100 are strong signals
    let abs_value = williams_value.abs();
    let distance_from_extremes = ((abs_value - 0.0).abs().min((abs_value - 100.0).abs())).min(50.0);
    let normalized = 1.0 - (distance_from_extremes / 50.0);
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.5, 8.0)
}

/// ATR: backward-compatible fixed normalization (kept for callers that lack close_price).
fn normalize_atr_score(atr_value: f64) -> f64 {
    if atr_value.is_nan() || atr_value <= 0.0 {
        return 0.0;
    }
    
    // Legacy fixed constant — inaccurate for multi-asset.
    // Prefer normalize_atr_score_with_price() instead.
    let normalized = (atr_value / 0.1).min(1.0);
    sigmoid_normalize(normalized, 0.2, 5.0)
}

/// ATR: price-aware normalization.
/// Expresses ATR as a percentage of `close_price`, capped at 5%.
/// For BTC @ 60 000 an ATR of 1500 → 1500/60000*100 = 2.5% → 2.5/5.0 = 0.5.
/// For DOGE @ 0.1 an ATR of 0.002 → 0.002/0.1*100 = 2.0% → 2.0/5.0 = 0.4.
fn normalize_atr_score_with_price(atr_value: f64, close_price: f64) -> f64 {
    if atr_value.is_nan() || atr_value <= 0.0 {
        return 0.0;
    }

    let normalized = if close_price > 0.0 {
        // ATR as % of price, cap at 5%
        (atr_value / close_price * 100.0).min(5.0) / 5.0
    } else {
        0.0
    };

    sigmoid_normalize(normalized, 0.2, 5.0)
}

/// Normalize ADX values to score (measures trend strength)
fn normalize_adx_score(adx_value: f64) -> f64 {
    if adx_value.is_nan() || adx_value <= 0.0 {
        return 0.0;
    }
    
    // ADX: values above 25 indicate trend, above 50 strong trend
    let normalized = (adx_value / 100.0).min(1.0); // ADX typically ranges 0-100
    
    // Apply sigmoid to emphasize strong signals
    sigmoid_normalize(normalized, 0.3, 6.0)
}

/// Generic normalization for unknown indicator types
fn normalize_generic_score(value: f64) -> f64 {
    if value.is_nan() {
        return 0.0;
    }
    
    // Use absolute value and normalize assuming max value of 100
    let abs_value = value.abs();
    let normalized = (abs_value / 100.0).min(1.0);
    
    sigmoid_normalize(normalized, 0.3, 5.0)
}

/// Applies threshold filtering to remove only weak signals (below 0.55)
/// Stronger signals (0.55+) are kept for further processing by predictor
/// NOTE: Lowered from 0.80 to 0.55 to allow more signals through
/// (0.80 was too aggressive and blocked most signals)
pub fn filter_weak_signals(score: f64) -> bool {
    score >= 0.55  // Allow signals >= 0.55 (was 0.80)
}

// ---------------------------------------------------------------------------
// Adaptive thresholds
// ---------------------------------------------------------------------------

/// Computes the value at the given `percentile` (0.0–1.0) from a sorted copy of `values`.
///
/// Use-case: derive adaptive RSI oversold/overbought thresholds from a recent window of
/// RSI readings so the levels react to the current market regime instead of being hard-coded.
///
/// # Examples
/// ```
/// use raw_signals::scoring::adaptive_threshold;
/// let rsi_window = vec![25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0];
/// let oversold  = adaptive_threshold(&rsi_window, 0.10); // ≈ 26.35
/// let overbought = adaptive_threshold(&rsi_window, 0.90); // ≈ 68.65
/// ```
///
/// Returns `f64::NAN` when `values` is empty.
pub fn adaptive_threshold(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }

    // Filter out NaN values and collect into a sorted vec
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p = percentile.clamp(0.0, 1.0);
    let float_idx = p * (sorted.len() as f64 - 1.0);
    let lo = float_idx.floor() as usize;
    let hi = float_idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = float_idx - lo as f64;

    // Linear interpolation between the two nearest values
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_signal() {
        assert_eq!(normalize_signal(50.0, 0.0, 100.0), 0.5);
        assert_eq!(normalize_signal(0.0, 0.0, 100.0), 0.0);
        assert_eq!(normalize_signal(100.0, 0.0, 100.0), 1.0);
    }

    #[test]
    fn test_normalize_rsi_score() {
        assert!(normalize_rsi_score(0.0) > 0.8); // Strong oversold signal
        assert!(normalize_rsi_score(100.0) > 0.8); // Strong overbought signal
        assert!(normalize_rsi_score(50.0) < 0.3); // Neutral signal
    }

    #[test]
    fn test_filter_weak_signals() {
        assert!(filter_weak_signals(0.85));  // Above threshold
        assert!(filter_weak_signals(0.55));  // At threshold (changed from 0.75)
        assert!(!filter_weak_signals(0.50)); // Below threshold
    }

    // ---- Price-aware ATR tests ----

    #[test]
    fn test_atr_price_aware_btc() {
        // BTC: ATR=1500, price=60000 → 2.5% → 0.5 normalized
        let score = normalize_atr_score_with_price(1500.0, 60_000.0);
        assert!(score > 0.0 && score < 1.0, "BTC ATR score = {score}");
    }

    #[test]
    fn test_atr_price_aware_doge() {
        // DOGE: ATR=0.002, price=0.1 → 2.0% → 0.4 normalized
        let score = normalize_atr_score_with_price(0.002, 0.1);
        assert!(score > 0.0 && score < 1.0, "DOGE ATR score = {score}");
    }

    #[test]
    fn test_atr_price_aware_comparable() {
        // Same relative volatility should produce same score regardless of absolute price
        let btc = normalize_atr_score_with_price(1200.0, 60_000.0);   // 2%
        let doge = normalize_atr_score_with_price(0.002, 0.1);        // 2%
        assert!((btc - doge).abs() < 0.01, "BTC={btc}, DOGE={doge}");
    }

    #[test]
    fn test_atr_price_aware_zero_price() {
        let score = normalize_atr_score_with_price(1500.0, 0.0);
        // Should fall back to sigmoid(0) ≈ 0.27
        assert!(score < 0.5, "zero-price ATR score = {score}");
    }

    // ---- Price-aware MACD tests ----

    #[test]
    fn test_macd_price_aware_btc() {
        let score = normalize_macd_score_with_price(500.0, 60_000.0);
        assert!(score > 0.0 && score < 1.0, "BTC MACD score = {score}");
    }

    #[test]
    fn test_macd_price_aware_doge() {
        let score = normalize_macd_score_with_price(0.0001, 0.1);
        assert!(score > 0.0 && score < 1.0, "DOGE MACD score = {score}");
    }

    #[test]
    fn test_macd_price_aware_comparable() {
        // Same relative MACD (1‰ of price) should produce same score
        let btc = normalize_macd_score_with_price(60.0, 60_000.0);  // 1‰
        let doge = normalize_macd_score_with_price(0.0001, 0.1);    // 1‰
        assert!((btc - doge).abs() < 0.01, "BTC={btc}, DOGE={doge}");
    }

    // ---- Adaptive threshold tests ----

    #[test]
    fn test_adaptive_threshold_basic() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let p10 = adaptive_threshold(&values, 0.10);
        let p50 = adaptive_threshold(&values, 0.50);
        let p90 = adaptive_threshold(&values, 0.90);

        assert!((p50 - 55.0).abs() < 1.0, "median = {p50}");
        assert!(p10 < p50);
        assert!(p90 > p50);
    }

    #[test]
    fn test_adaptive_threshold_empty() {
        assert!(adaptive_threshold(&[], 0.5).is_nan());
    }

    #[test]
    fn test_adaptive_threshold_single() {
        assert_eq!(adaptive_threshold(&[42.0], 0.5), 42.0);
    }

    #[test]
    fn test_adaptive_threshold_with_nan() {
        let values = vec![f64::NAN, 10.0, 20.0, f64::NAN, 30.0];
        let median = adaptive_threshold(&values, 0.5);
        assert!((median - 20.0).abs() < 0.001, "median = {median}");
    }

    // ---- normalize_indicator_to_score_with_price delegates correctly ----

    #[test]
    fn test_with_price_delegates_rsi() {
        let without = normalize_indicator_to_score(65.0, "rsi");
        let with = normalize_indicator_to_score_with_price(65.0, "rsi", 60_000.0);
        assert_eq!(without, with);
    }
}
