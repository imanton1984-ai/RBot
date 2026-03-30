// strategies/ml_entry_strategy/src/direction/dataset.rs
//
// Direction Model v4 — Pattern Dataset Builder
//
// Builds a CSV dataset with:
//   - CNN-like sliding window features (pure OHLCV, no indicators)
//   - Labels: UP (1) / FLAT (0) / DOWN (-1) based on future return
//
// DATA SOURCE:
//   Only needs OHLCV candles from market.candles_Xm tables.
//   Does NOT need indicators_wide — this is the key advantage.
//   We query candles directly with a lightweight SQL.
//
// STATIONARITY:
//   All prices normalized relative to window[0].open.
//   The model sees patterns like "2% dip then consolidation" not "price=65432.50"

use anyhow::Result;
use chrono::{DateTime, Utc};
use futures::stream::{self, StreamExt};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::io::Write;
use tracing::info;

use super::DirectionConfig;
use super::features::{compute_pattern_features, compute_label, direction_v4_feature_names};
use crate::dataset::{CandleWithIndicators, fetch_active_symbols};

/// Lightweight candle — only OHLCV, no indicators.
/// Used for dataset building where we don't need 33 indicator columns.
#[derive(Debug, Clone)]
pub struct RawCandle {
    pub time: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl RawCandle {
    /// Convert to CandleWithIndicators (indicators set to defaults).
    /// Required because compute_pattern_features works with CandleWithIndicators.
    pub fn to_candle_with_indicators(&self, symbol_id: i64) -> CandleWithIndicators {
        let c = self.close;
        CandleWithIndicators {
            time: self.time,
            symbol: self.symbol.clone(),
            symbol_id,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            // Indicators — defaults (not used by v4 pattern model)
            rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
            adx: 25.0, sma: c, ema_20: c, ema_50: c, ema_200: c,
            bb_upper: c * 1.02, bb_mid: c, bb_lower: c * 0.98,
            atr: c * 0.01, obv: 0.0, vwap: c, volume_spike: 1.0,
            trend: 0.0, trend_short: 0.0, poc: c,
            alligator_jaw: c, alligator_teeth: c, alligator_lips: c,
            mfi: 50.0, fibo_pivot: c, fibo_r1: c * 1.01, fibo_s1: c * 0.99,
            supertrend: c, supertrend_dir: 0.0, cmf: 0.0,
        }
    }
}

/// A single training example for Direction v4 pattern model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionPatternExample {
    pub symbol: String,
    pub tf_minutes: i32,
    pub timestamp: String, // ISO8601

    /// Pattern features (flattened sliding window)
    pub features: Vec<f64>,

    // ── Labels ──
    /// Predicted class: 1=UP, 0=FLAT, -1=DOWN
    pub label: i8,
    /// Future return at prediction_horizon bars (%)
    pub future_return_pct: f64,
    /// Max upward excursion within horizon (%)
    pub max_up_pct: f64,
    /// Max downward excursion within horizon (%)
    pub max_down_pct: f64,
}

// ── Legacy re-export for backward compat with bin/direction_dataset.rs ──
pub type DirectionExample = DirectionPatternExample;

/// Build pattern dataset for a slice of candles (one symbol).
///
/// For each valid position t where we have enough history (window_size)
/// and enough future (prediction_horizon), compute features and labels.
///
/// # Arguments
/// * `candles` — main TF candle data
/// * `config` — direction model configuration
/// * `tf_minutes` — timeframe in minutes
/// * `htf_candles` — optional higher-timeframe candle data for cross-TF context
pub fn build_pattern_labels(
    candles: &[CandleWithIndicators],
    config: &DirectionConfig,
    tf_minutes: i32,
    htf_candles: Option<&[CandleWithIndicators]>,
) -> Vec<DirectionPatternExample> {
    let n = candles.len();
    let min_required = config.min_candles_required();

    if n < min_required {
        return Vec::new();
    }

    // Start from the first position where we have a full window
    let start_idx = config.window_size - 1;
    // End before the prediction horizon boundary
    let end_idx = n - config.prediction_horizon;

    if start_idx >= end_idx {
        return Vec::new();
    }

    let mut examples = Vec::with_capacity(end_idx - start_idx);

    for t in start_idx..end_idx {
        // Compute features (with summary + HTF context)
        let features = match compute_pattern_features(candles, t, config, htf_candles) {
            Some(f) => f,
            None => continue,
        };

        // Compute label
        let (label, future_return_pct, max_up_pct, max_down_pct) = match compute_label(candles, t, config) {
            Some(l) => l,
            None => continue,
        };

        examples.push(DirectionPatternExample {
            symbol: candles[t].symbol.clone(),
            tf_minutes,
            timestamp: candles[t].time.to_rfc3339(),
            features,
            label,
            future_return_pct,
            max_up_pct,
            max_down_pct,
        });
    }

    examples
}

/// Export pattern dataset to CSV.
///
/// CSV columns:
///   symbol, tf_minutes, timestamp,
///   [pattern feature columns: w0_open_rel, w0_high_rel, ...],
///   label, future_return_pct, max_up_pct, max_down_pct
pub fn export_direction_csv(
    examples: &[DirectionPatternExample],
    output_path: &str,
    config: &DirectionConfig,
) -> Result<()> {
    let file = std::fs::File::create(output_path)?;
    let mut buf = std::io::BufWriter::with_capacity(1 << 20, file); // 1MB buffer

    // Header
    let feature_names = direction_v4_feature_names(config);
    let mut header = String::from("symbol,tf_minutes,timestamp");
    for name in &feature_names {
        header.push(',');
        header.push_str(name);
    }
    header.push_str(",label,future_return_pct,max_up_pct,max_down_pct");
    writeln!(buf, "{}", header)?;

    // Rows
    for ex in examples {
        let mut line = format!("{},{},{}", ex.symbol, ex.tf_minutes, ex.timestamp);
        for &val in &ex.features {
            line.push_str(&format!(",{:.6}", val));
        }
        line.push_str(&format!(
            ",{},{:.6},{:.6},{:.6}",
            ex.label,
            ex.future_return_pct,
            ex.max_up_pct,
            ex.max_down_pct,
        ));
        writeln!(buf, "{}", line)?;
    }

    buf.flush()?;
    Ok(())
}

/// Fetch raw candles (OHLCV only, no indicators) for one symbol.
/// Much faster than fetch_candles_with_indicators since no JOIN needed.
pub async fn fetch_raw_candles(
    pool: &PgPool,
    symbol: &str,
    tf_minutes: i32,
    limit: usize,
) -> Result<Vec<RawCandle>> {
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    };

    let sql = format!(
        "SELECT time, symbol, open, high, low, close, volume \
         FROM {candle_table} \
         WHERE symbol = $1 \
         ORDER BY time ASC \
         LIMIT $2"
    );

    #[derive(sqlx::FromRow)]
    struct Row {
        time: DateTime<Utc>,
        symbol: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    }

    let rows = sqlx::query_as::<_, Row>(&sql)
        .bind(symbol)
        .bind(limit as i64)
        .fetch_all(pool)
        .await?;

    Ok(rows.into_iter().map(|r| RawCandle {
        time: r.time, symbol: r.symbol,
        open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
    }).collect())
}

/// Fetch raw candles for ALL active symbols in one TF.
/// Returns data grouped by symbol. Uses a single query with window function.
pub async fn fetch_all_raw_candles_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    limit_per_symbol: usize,
) -> Result<std::collections::HashMap<String, Vec<RawCandle>>> {
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => anyhow::bail!("Unsupported timeframe: {}", tf_minutes),
    };

    let sql = format!(
        r#"
        WITH ranked AS (
            SELECT c.time, c.symbol, c.open, c.high, c.low, c.close, c.volume,
                   ROW_NUMBER() OVER (PARTITION BY c.symbol ORDER BY c.time DESC) as rn
            FROM {candle_table} c
            JOIN market.pairs p ON p.symbol = c.symbol AND p.is_active = true
        )
        SELECT time, symbol, open, high, low, close, volume
        FROM ranked
        WHERE rn <= $1
        ORDER BY symbol, time ASC
        "#
    );

    #[derive(sqlx::FromRow)]
    struct Row {
        time: DateTime<Utc>,
        symbol: String,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    }

    let rows = sqlx::query_as::<_, Row>(&sql)
        .bind(limit_per_symbol as i64)
        .fetch_all(pool)
        .await?;

    let mut grouped: std::collections::HashMap<String, Vec<RawCandle>> =
        std::collections::HashMap::new();

    for r in rows {
        grouped.entry(r.symbol.clone()).or_default().push(RawCandle {
            time: r.time, symbol: r.symbol,
            open: r.open, high: r.high, low: r.low, close: r.close, volume: r.volume,
        });
    }

    Ok(grouped)
}

/// Build the complete direction v4+ pattern dataset for one TF.
///
/// Steps:
///   1. Fetch active symbols
///   2. Fetch HTF candles (for cross-TF context features)
///   3. For each symbol: fetch raw candles, convert, compute features + labels
///   4. Aggregate all examples
pub async fn build_direction_dataset_for_tf(
    pool: &PgPool,
    tf_minutes: i32,
    config: &DirectionConfig,
    limit_per_symbol: usize,
) -> Result<Vec<DirectionPatternExample>> {
    use super::features::get_htf_minutes;

    info!("Building direction v4+ pattern dataset for TF {}m (window={}, horizon={}, features={}, summary={}, htf={})",
          tf_minutes, config.window_size, config.prediction_horizon, config.total_features(),
          super::SUMMARY_FEATURE_COUNT, super::HTF_FEATURE_COUNT);

    // 1. Load active symbols
    let symbols = fetch_active_symbols(pool).await?;
    info!("  {} active symbols", symbols.len());

    // 2. Pre-load HTF candles for cross-TF context (per symbol)
    let htf_minutes = get_htf_minutes(tf_minutes);
    let htf_data: Option<std::collections::HashMap<String, Vec<CandleWithIndicators>>> =
        if let Some(htf_tf) = htf_minutes {
            info!("  Loading HTF candles (TF {}m) for cross-TF context...", htf_tf);
            // HTF candles: fewer bars needed (50 bars history is enough for HTF features)
            let htf_limit = (limit_per_symbol / 4).max(200);
            match fetch_all_raw_candles_for_tf(pool, htf_tf, htf_limit).await {
                Ok(raw_htf) => {
                    let htf_converted: std::collections::HashMap<String, Vec<CandleWithIndicators>> = raw_htf
                        .into_iter()
                        .map(|(sym, raws)| {
                            let cwi: Vec<CandleWithIndicators> = raws.iter()
                                .map(|r| r.to_candle_with_indicators(0))
                                .collect();
                            (sym, cwi)
                        })
                        .collect();
                    let n_htf_symbols = htf_converted.len();
                    let n_htf_candles: usize = htf_converted.values().map(|v| v.len()).sum();
                    info!("  Loaded HTF: {} symbols, {} candles", n_htf_symbols, n_htf_candles);
                    Some(htf_converted)
                }
                Err(e) => {
                    tracing::warn!("  Failed to load HTF candles: {}. Proceeding without HTF context.", e);
                    None
                }
            }
        } else {
            info!("  No HTF available for TF {}m (highest TF)", tf_minutes);
            None
        };

    // Use Arc to share HTF data across async tasks without cloning
    let htf_data_arc: std::sync::Arc<Option<std::collections::HashMap<String, Vec<CandleWithIndicators>>>> =
        std::sync::Arc::new(htf_data);

    // 3. Process symbols concurrently
    let concurrency = std::env::var("DATASET_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(16);

    info!("  Processing with concurrency={}", concurrency);

    let dir_config = config.clone();

    let results: Vec<Vec<DirectionPatternExample>> = stream::iter(symbols)
        .map(|symbol| {
            let pool = pool.clone();
            let cfg = dir_config.clone();
            let min_required = cfg.min_candles_required();
            let htf_ref = htf_data_arc.clone();

            async move {
                // Fetch raw candles (no indicators needed!)
                let raw_candles = match fetch_raw_candles(&pool, &symbol, tf_minutes, limit_per_symbol).await {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::warn!("  Failed to fetch {}: {}", symbol, e);
                        return Vec::new();
                    }
                };

                if raw_candles.len() < min_required {
                    return Vec::new();
                }

                // Convert to CandleWithIndicators (indicators will be defaults)
                let candles: Vec<CandleWithIndicators> = raw_candles
                    .iter()
                    .map(|r| r.to_candle_with_indicators(0))
                    .collect();

                // Get HTF candles for this symbol (if available)
                let htf_candles: Option<&[CandleWithIndicators]> = htf_ref
                    .as_ref()
                    .as_ref()
                    .and_then(|m| m.get(&symbol))
                    .map(|v| v.as_slice());

                // Build pattern labels with HTF context
                build_pattern_labels(&candles, &cfg, tf_minutes, htf_candles)
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Aggregate
    let mut all_examples: Vec<DirectionPatternExample> = Vec::new();
    let mut n_symbols_ok = 0u32;
    let mut n_up = 0usize;
    let mut n_down = 0usize;
    let mut n_flat = 0usize;

    for examples in results {
        if !examples.is_empty() {
            n_symbols_ok += 1;
        }
        for ex in &examples {
            match ex.label {
                1 => n_up += 1,
                -1 => n_down += 1,
                _ => n_flat += 1,
            }
        }
        all_examples.extend(examples);
    }

    let total = all_examples.len();
    info!(
        "  TF {}m: {} examples from {} symbols",
        tf_minutes, total, n_symbols_ok
    );
    info!(
        "  Label distribution: UP={} ({:.1}%), FLAT={} ({:.1}%), DOWN={} ({:.1}%)",
        n_up, if total > 0 { n_up as f64 / total as f64 * 100.0 } else { 0.0 },
        n_flat, if total > 0 { n_flat as f64 / total as f64 * 100.0 } else { 0.0 },
        n_down, if total > 0 { n_down as f64 / total as f64 * 100.0 } else { 0.0 },
    );

    Ok(all_examples)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candles(n: usize, base_price: f64) -> Vec<CandleWithIndicators> {
        (0..n).map(|i| {
            let p = base_price + (i as f64 * 0.1).sin() * 2.0;
            CandleWithIndicators {
                time: chrono::Utc::now(),
                symbol: "TESTUSDT".to_string(),
                symbol_id: 1,
                open: p,
                high: p + 1.0,
                low: p - 0.5,
                close: p + 0.3,
                volume: 1000.0,
                rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
                macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
                adx: 25.0, sma: p, ema_20: p, ema_50: p, ema_200: p,
                bb_upper: p * 1.02, bb_mid: p, bb_lower: p * 0.98,
                atr: p * 0.01, obv: 0.0, vwap: p, volume_spike: 1.0,
                trend: 0.0, trend_short: 0.0, poc: p,
                alligator_jaw: p, alligator_teeth: p, alligator_lips: p,
                mfi: 50.0, fibo_pivot: p, fibo_r1: p * 1.01, fibo_s1: p * 0.99,
                supertrend: p, supertrend_dir: 1.0, cmf: 0.0,
            }
        }).collect()
    }

    #[test]
    fn test_build_pattern_labels() {
        let config = DirectionConfig {
            window_size: 10,
            prediction_horizon: 5,
            up_threshold_pct: 0.0, // any positive = UP
            down_threshold_pct: 0.0,
            ..Default::default()
        };

        let candles = make_test_candles(50, 100.0);
        let examples = build_pattern_labels(&candles, &config, 15, None);

        // Should produce examples from t=9 to t=44 (50 - 5 - 1)
        assert!(!examples.is_empty());

        // Check feature count
        for ex in &examples {
            assert_eq!(ex.features.len(), config.total_features());
        }

        // Check labels are valid
        for ex in &examples {
            assert!(ex.label == 1 || ex.label == 0 || ex.label == -1,
                "Invalid label: {}", ex.label);
        }
    }

    #[test]
    fn test_build_pattern_labels_too_few_candles() {
        let config = DirectionConfig {
            window_size: 30,
            prediction_horizon: 10,
            ..Default::default()
        };

        let candles = make_test_candles(35, 100.0); // Need 40, have 35
        let examples = build_pattern_labels(&candles, &config, 15, None);
        // 35 - 10 = 25 is end_idx, start_idx = 29 → 29 >= 25, no examples
        assert!(examples.is_empty());
    }
}
