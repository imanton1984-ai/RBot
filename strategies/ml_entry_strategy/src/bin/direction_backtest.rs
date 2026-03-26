// strategies/ml_entry_strategy/src/bin/direction_backtest.rs
//
// Direction v3 Backtester — FAST BATCH VERSION
//
// Uses batch data loading (single SQL query per TF, all symbols at once)
// AND batch model inference (all super points predicted in one GPU call).
//
// Compares direction predictions:
//   - Baseline v1: existing super_dir model (128 features, ~0.50 AUC)
//   - Direction v3: regression model (32 features → sign = direction, abs = confidence)
//   - With confidence gate analysis at different thresholds
//
// USAGE:
//   cargo build --release -p ml_entry_strategy --bin direction_backtest
//   ./target/release/direction_backtest
//   DIRECTION_GPU=1 ./target/release/direction_backtest
//
// ENV VARS:
//   DATABASE_URL            — postgres connection string
//   WFO_MIN_DATE=2026-01-13 — only count trades after this date
//   DIRECTION_TF=15,60,240  — timeframes to test (default: 15,60,240)
//   DIRECTION_GPU=1         — use GPU for model inference (default: CPU)

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use tracing::info;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{
    CandleWithIndicators, compute_dynamic_features_with_htf,
    fetch_all_candles_for_tf,
};
use ml_entry_strategy::direction::features::{
    compute_direction_v3_features, resolve_btc_context, resolve_htf_context,
    DIRECTION_V3_FEATURE_COUNT,
};
use ml_entry_strategy::heuristic::get_higher_tf;

use predictors::ml::xgb_runtime::{Booster, Device, ModelKind};

// ─────────────────────────────────────────────────────────────────────
// Trade simulation
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum Outcome { Win, Loss, Expired }

fn simulate_trade(
    candles: &[CandleWithIndicators],
    entry_idx: usize,
    direction: i8,
    tp_pct: f64,
    sl_pct: f64,
    max_hold: usize,
) -> (Outcome, f64) {
    let entry = candles[entry_idx].close;
    let (tp, sl) = if direction == 1 {
        (entry * (1.0 + tp_pct / 100.0), entry * (1.0 - sl_pct / 100.0))
    } else {
        (entry * (1.0 - tp_pct / 100.0), entry * (1.0 + sl_pct / 100.0))
    };

    let end = (entry_idx + max_hold).min(candles.len() - 1);
    for k in (entry_idx + 1)..=end {
        let (h, l) = (candles[k].high, candles[k].low);
        if direction == 1 {
            if l <= sl { return (Outcome::Loss, (sl - entry) / entry * 100.0); }
            if h >= tp { return (Outcome::Win, (tp - entry) / entry * 100.0); }
        } else {
            if h >= sl { return (Outcome::Loss, (entry - sl) / entry * 100.0); }
            if l <= tp { return (Outcome::Win, (entry - tp) / entry * 100.0); }
        }
    }
    let last = candles[end].close;
    let pnl = if direction == 1 { (last - entry) / entry * 100.0 }
              else { (entry - last) / entry * 100.0 };
    (Outcome::Expired, pnl)
}

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct Metrics {
    n: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    total_pnl: f64,
}

impl Metrics {
    fn add(&mut self, outcome: Outcome, pnl: f64) {
        self.n += 1;
        self.total_pnl += pnl;
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Loss => self.losses += 1,
            Outcome::Expired => self.expired += 1,
        }
    }
    fn wr(&self) -> f64 { if self.n > 0 { self.wins as f64 / self.n as f64 * 100.0 } else { 0.0 } }
    fn avg_pnl(&self) -> f64 { if self.n > 0 { self.total_pnl / self.n as f64 } else { 0.0 } }
}

/// A super point that passed P(super) threshold — ready for direction prediction
struct SuperPoint {
    symbol_idx: usize,      // index into symbols vec
    candle_idx: usize,       // index into that symbol's candle vec
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // Logging
    let log_path = "logs/direction_backtest.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true).append(true).open(log_path)?;

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(fmt::layer().with_target(false).with_ansi(false)
            .with_writer(std::sync::Mutex::new(log_file)))
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
        ))
        .init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let tf_str = std::env::var("DIRECTION_TF")
        .unwrap_or_else(|_| "15,60,240".to_string());
    let timeframes: Vec<i32> = tf_str.split(',')
        .filter_map(|s| s.trim().parse().ok()).collect();

    let use_gpu = std::env::var("DIRECTION_GPU")
        .map_or(false, |v| v == "1" || v.to_lowercase() == "true");

    let config = SuperEntryConfig::from_env();

    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE").ok()
        .and_then(|s| NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d").ok()
            .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc()));

    let device = if use_gpu { Device::Cuda } else { Device::Cpu };

    info!("╔═══════════════════════════════════════════════════════╗");
    info!("║  Direction v3 Backtester — BATCH GPU                 ║");
    info!("╚═══════════════════════════════════════════════════════╝");
    info!("Timeframes: {:?}", timeframes);
    info!("Device: {:?} (DIRECTION_GPU={})", device, if use_gpu { "1" } else { "0" });
    if let Some(d) = &wfo_min_date {
        info!("WFO OOS filter: trades after {}", d.format("%Y-%m-%d"));
    }

    let pool = PgPool::connect(&db_url).await?;
    let total_start = std::time::Instant::now();

    for &tf in &timeframes {
        let tp_pct = config.target_pct_for_tf(tf);
        let sl_pct = config.sl_pct_for_tf(tf);
        let limit = match tf { 1=>5000, 5=>12000, 15=>12000, 60=>12000, 240=>12000, 1440=>3700, _=>5000 };

        let sep = "═".repeat(60);
        info!("");
        info!("{}", sep);
        info!("  TF {}m  TP={:.2}%  SL={:.2}%", tf, tp_pct, sl_pct);
        info!("{}", sep);

        // ── Load models ──
        let super_path = config.model_path(tf);
        let super_model = match Booster::load(&super_path, device) {
            Ok(b) => { info!("  ✅ super model: {} ({:?})", super_path, device); b }
            Err(e) => {
                if use_gpu {
                    match Booster::load(&super_path, Device::Cpu) {
                        Ok(b) => { info!("  ✅ super model: {} (CPU fallback)", super_path); b }
                        Err(_) => { info!("  ❌ super model not found: {}", e); continue; }
                    }
                } else {
                    info!("  ❌ super model not found: {}", e);
                    continue;
                }
            }
        };

        // Legacy baseline direction model
        let base_dir_path = config.direction_model_path(tf);
        let base_dir_model = Booster::load(&base_dir_path, device)
            .or_else(|_| Booster::load(&base_dir_path, Device::Cpu)).ok();
        info!("  {} baseline direction: {}",
              if base_dir_model.is_some() { "✅" } else { "⚠ " }, base_dir_path);

        // Direction v3 model
        let v3_path = format!("models/direction_v3_tf{}.ubj", tf);
        let v3_model = Booster::load(&v3_path, device)
            .or_else(|_| Booster::load(&v3_path, Device::Cpu)).ok();
        info!("  {} direction v3: {} (32 features, regression)",
              if v3_model.is_some() { "✅" } else { "⚠ " }, v3_path);

        if v3_model.is_none() && base_dir_model.is_none() {
            info!("  No direction models for TF {}m. Skipping.", tf);
            continue;
        }

        // ── BATCH LOAD all candles for this TF ──
        let t_load = std::time::Instant::now();
        let all_candles = fetch_all_candles_for_tf(&pool, tf, limit).await?;
        let n_symbols = all_candles.len();
        let n_candles: usize = all_candles.values().map(|v| v.len()).sum();
        info!("  Loaded {} symbols, {} candles in {:.1}s",
              n_symbols, n_candles, t_load.elapsed().as_secs_f64());

        // ── Load BTC candles ──
        let btc_candles = all_candles.get("BTCUSDT").cloned().unwrap_or_default();
        let btc_ref = if btc_candles.len() >= 50 { Some(btc_candles.as_slice()) } else { None };
        info!("  BTC candles: {}", btc_candles.len());

        // ── Load HTF candles ──
        let htf_tf = get_higher_tf(tf);
        let htf_all = if let Some(htf) = htf_tf {
            let htf_data = fetch_all_candles_for_tf(&pool, htf, limit).await?;
            info!("  HTF {}m: {} symbols loaded", htf, htf_data.len());
            Some(htf_data)
        } else {
            None
        };

        let total_features = ml_entry_strategy::config::total_feature_count();

        // ══════════════════════════════════════════════════════════════
        // PHASE 1: Scan all candles, compute 128 features, find super points
        // ══════════════════════════════════════════════════════════════
        let t_phase1 = std::time::Instant::now();

        // Collect symbols into ordered vec for indexing
        let symbols: Vec<(&String, &Vec<CandleWithIndicators>)> = all_candles.iter()
            .filter(|(s, _)| s.as_str() != "BTCUSDT")
            .collect();

        // Pre-compute all 128-feature vectors and find super points
        // Accumulate feature batches for batch prediction
        let mut all_128_features: Vec<f32> = Vec::new();
        let mut super_point_indices: Vec<SuperPoint> = Vec::new();
        let mut point_candle_refs: Vec<(usize, usize)> = Vec::new(); // (symbol_idx, candle_idx)

        for (sym_idx, (symbol, candles)) in symbols.iter().enumerate() {
            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            let htf_candles_ref: Option<&[CandleWithIndicators]> = htf_all.as_ref()
                .and_then(|htf_map| htf_map.get(symbol.as_str()))
                .filter(|v| v.len() >= 50)
                .map(|v| v.as_slice());

            let start = config.warmup_bars;
            let end = candles.len() - config.lookahead_bars;

            for t in start..end {
                if candles[t].close <= 0.0 { continue; }

                // WFO date filter
                if let Some(min_d) = wfo_min_date {
                    if candles[t].time < min_d { continue; }
                }

                // Compute 128 features inline — fast, no allocation
                let c = &candles[t];
                let close = c.close;
                let safe_div = |a: f64, b: f64| -> f32 {
                    if b.abs() > 1e-12 { (a / b) as f32 } else { 0.0f32 }
                };

                // Raw indicators (33)
                all_128_features.push(c.rsi as f32);
                all_128_features.push(c.cci as f32);
                all_128_features.push(c.stoch_k as f32);
                all_128_features.push(c.stoch_d as f32);
                all_128_features.push(c.williams as f32);
                all_128_features.push(c.macd as f32);
                all_128_features.push(c.macd_signal as f32);
                all_128_features.push(c.macd_hist as f32);
                all_128_features.push(c.adx as f32);
                all_128_features.push(c.sma as f32);
                all_128_features.push(c.ema_20 as f32);
                all_128_features.push(c.ema_50 as f32);
                all_128_features.push(c.ema_200 as f32);
                all_128_features.push(c.bb_upper as f32);
                all_128_features.push(c.bb_mid as f32);
                all_128_features.push(c.bb_lower as f32);
                all_128_features.push(c.atr as f32);
                all_128_features.push(c.obv as f32);
                all_128_features.push(c.vwap as f32);
                all_128_features.push(c.volume_spike as f32);
                all_128_features.push(c.trend as f32);
                all_128_features.push(c.trend_short as f32);
                all_128_features.push(c.poc as f32);
                all_128_features.push(c.alligator_jaw as f32);
                all_128_features.push(c.alligator_teeth as f32);
                all_128_features.push(c.alligator_lips as f32);
                all_128_features.push(c.mfi as f32);
                all_128_features.push(c.fibo_pivot as f32);
                all_128_features.push(c.fibo_r1 as f32);
                all_128_features.push(c.fibo_s1 as f32);
                all_128_features.push(c.supertrend as f32);
                all_128_features.push(c.supertrend_dir as f32);
                all_128_features.push(c.cmf as f32);

                // Derived (19)
                let bb_range = c.bb_upper - c.bb_lower;
                all_128_features.push((c.rsi / 100.0) as f32);
                all_128_features.push((c.cci / 200.0) as f32);
                all_128_features.push((c.stoch_k / 100.0) as f32);
                all_128_features.push(((c.williams + 100.0) / 100.0) as f32);
                all_128_features.push(if bb_range.abs() > 1e-12 {
                    ((close - c.bb_lower) / bb_range) as f32
                } else { 0.5f32 });
                all_128_features.push(safe_div(bb_range, close) * 100.0);
                all_128_features.push(safe_div(c.atr, close) * 100.0);
                all_128_features.push(safe_div(close - c.sma, close) * 100.0);
                all_128_features.push(safe_div(close - c.ema_20, close) * 100.0);
                all_128_features.push(safe_div(close - c.ema_50, close) * 100.0);
                all_128_features.push(safe_div(close - c.ema_200, close) * 100.0);
                all_128_features.push(safe_div(close - c.vwap, close) * 100.0);
                all_128_features.push(safe_div(c.macd_hist, close) * 1000.0);
                all_128_features.push(0.0f32); // obv_change_pct
                all_128_features.push(if c.volume_spike > 2.0 { 1.0f32 } else { 0.0f32 });
                all_128_features.push((c.mfi / 100.0) as f32);
                all_128_features.push(safe_div(close - c.fibo_pivot, close) * 100.0);
                all_128_features.push(safe_div(close - c.supertrend, close) * 100.0);
                all_128_features.push(safe_div(c.alligator_jaw - c.alligator_lips, close) * 100.0);

                // Dynamic (76)
                let htf_candle = htf_candles_ref.and_then(|htf| {
                    let target_time = candles[t].time;
                    let idx = htf.partition_point(|c| c.time <= target_time);
                    if idx > 0 { Some(&htf[idx - 1]) } else { None }
                });
                let dyn_feats = compute_dynamic_features_with_htf(candles, t, htf_candle);
                for v in &dyn_feats {
                    all_128_features.push(*v as f32);
                }

                point_candle_refs.push((sym_idx, t));
            }
        }

        let n_total_points = point_candle_refs.len();
        info!("  Phase 1: {} candidate points, features computed in {:.1}s",
              n_total_points, t_phase1.elapsed().as_secs_f64());

        if n_total_points == 0 {
            info!("  No candidate points. Skipping TF {}m.", tf);
            continue;
        }

        // ══════════════════════════════════════════════════════════════
        // PHASE 2: BATCH P(super) prediction — one GPU call for ALL points
        // ══════════════════════════════════════════════════════════════
        let t_phase2 = std::time::Instant::now();

        let p_super_vec = super_model.predict_dense_cpu(
            &all_128_features, n_total_points, total_features, ModelKind::Regressor1,
        )?;

        // Filter super points
        let p_threshold = config.p_threshold as f32;
        for (i, &p_super) in p_super_vec.iter().enumerate() {
            if p_super.clamp(0.0, 1.0) >= p_threshold {
                let (sym_idx, candle_idx) = point_candle_refs[i];
                super_point_indices.push(SuperPoint { symbol_idx: sym_idx, candle_idx });
            }
        }

        let n_super = super_point_indices.len();
        info!("  Phase 2: {} super points ({:.1}%), batch predict in {:.1}s",
              n_super,
              if n_total_points > 0 { n_super as f64 / n_total_points as f64 * 100.0 } else { 0.0 },
              t_phase2.elapsed().as_secs_f64());

        if n_super == 0 {
            info!("  No super points. Skipping TF {}m.", tf);
            continue;
        }

        // ══════════════════════════════════════════════════════════════
        // PHASE 3: BATCH direction predictions on super points only
        // ══════════════════════════════════════════════════════════════
        let t_phase3 = std::time::Instant::now();

        // Build 128-feature batch for super points (for baseline)
        let mut super_128_batch: Vec<f32> = Vec::with_capacity(n_super * total_features);
        for sp in &super_point_indices {
            // Find the original feature slice
            let point_global_idx = point_candle_refs.iter().position(|&(s, c)| {
                s == sp.symbol_idx && c == sp.candle_idx
            }).unwrap();
            let feat_start = point_global_idx * total_features;
            let feat_end = feat_start + total_features;
            super_128_batch.extend_from_slice(&all_128_features[feat_start..feat_end]);
        }

        // Build 32-feature batch for direction v3
        let mut super_v3_batch: Vec<f32> = if v3_model.is_some() {
            Vec::with_capacity(n_super * DIRECTION_V3_FEATURE_COUNT)
        } else {
            Vec::new()
        };

        if v3_model.is_some() {
            for sp in &super_point_indices {
                let (_, candles) = &symbols[sp.symbol_idx];

                let htf_candles_ref: Option<&[CandleWithIndicators]> = htf_all.as_ref()
                    .and_then(|htf_map| htf_map.get(symbols[sp.symbol_idx].0.as_str()))
                    .filter(|v| v.len() >= 50)
                    .map(|v| v.as_slice());

                let btc_ctx = btc_ref.and_then(|btc|
                    resolve_btc_context(btc, candles[sp.candle_idx].time));
                let htf_ctx = htf_candles_ref.and_then(|htf|
                    resolve_htf_context(htf, candles[sp.candle_idx].time));

                let v3_feats = compute_direction_v3_features(
                    candles, sp.candle_idx,
                    btc_ctx.as_ref(), htf_ctx.as_ref(),
                );
                for v in &v3_feats {
                    super_v3_batch.push(*v as f32);
                }
            }
        }

        // BATCH predict — baseline direction (128 features)
        let base_preds = if let Some(ref bm) = base_dir_model {
            bm.predict_dense_cpu(&super_128_batch, n_super, total_features, ModelKind::Regressor1)?
        } else {
            vec![0.5f32; n_super]
        };

        // BATCH predict — direction v3 (32 features)
        let v3_preds = if let Some(ref vm) = v3_model {
            vm.predict_dense_cpu(
                &super_v3_batch, n_super, DIRECTION_V3_FEATURE_COUNT, ModelKind::Regressor1,
            )?
        } else {
            vec![0.0f32; n_super]
        };

        info!("  Phase 3: batch direction predict in {:.1}s", t_phase3.elapsed().as_secs_f64());

        // ══════════════════════════════════════════════════════════════
        // PHASE 4: Simulate trades and collect metrics
        // ══════════════════════════════════════════════════════════════
        let t_phase4 = std::time::Instant::now();

        let mut metrics_base = Metrics::default();
        let mut metrics_v3 = Metrics::default();
        // Confidence gates for v3
        let gates = [0.01f32, 0.03, 0.05, 0.10, 0.15, 0.20];
        let mut metrics_gated: Vec<Metrics> = gates.iter().map(|_| Metrics::default()).collect();
        let mut agree_count = 0usize;

        for (i, sp) in super_point_indices.iter().enumerate() {
            let (_, candles) = &symbols[sp.symbol_idx];
            let t = sp.candle_idx;

            // Baseline direction
            let base_p_long = base_preds[i].clamp(0.0, 1.0);
            let base_dir: i8 = if base_p_long >= 0.5 { 1 } else { -1 };

            // V3 direction
            let v3_raw = v3_preds[i];
            let v3_dir: i8 = if v3_raw >= 0.0 { 1 } else { -1 };
            let v3_confidence = v3_raw.abs();

            if base_dir == v3_dir { agree_count += 1; }

            // Simulate trades
            let (out_base, pnl_base) = simulate_trade(
                candles, t, base_dir, tp_pct, sl_pct, config.effective_max_hold());
            metrics_base.add(out_base, pnl_base);

            if v3_model.is_some() {
                let (out_v3, pnl_v3) = simulate_trade(
                    candles, t, v3_dir, tp_pct, sl_pct, config.effective_max_hold());
                metrics_v3.add(out_v3, pnl_v3);

                // Confidence-gated trades
                for (gi, &gate) in gates.iter().enumerate() {
                    if v3_confidence >= gate {
                        metrics_gated[gi].add(out_v3, pnl_v3);
                    }
                }
            }
        }

        info!("  Phase 4: trade simulation in {:.1}s", t_phase4.elapsed().as_secs_f64());

        // ── Print results ──
        info!("");
        info!("  TF {}m — {} super points", tf, n_super);
        info!("");
        info!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
              "Variant", "Trades", "Wins", "Losses", "Exprd", "WR%", "AvgPnL%");
        info!("  {}", "-".repeat(76));

        let print_row = |name: &str, m: &Metrics| {
            let marker = if m.wr() >= 65.0 { "✅" } else if m.wr() >= 55.0 { "⚠ " } else { "❌" };
            info!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>9.1}% {:>9.4}% {}",
                  name, m.n, m.wins, m.losses, m.expired, m.wr(), m.avg_pnl(), marker);
        };

        print_row("Baseline (128f)", &metrics_base);
        if v3_model.is_some() {
            print_row("Direction v3 (32f)", &metrics_v3);

            info!("");
            info!("  Confidence-gated v3:");
            for (gi, &gate) in gates.iter().enumerate() {
                let m = &metrics_gated[gi];
                if m.n > 0 {
                    let coverage = m.n as f64 / n_super as f64 * 100.0;
                    let name = format!("  v3 gate≥{:.2}", gate);
                    let marker = if m.wr() >= 65.0 { "✅" } else if m.wr() >= 55.0 { "⚠ " } else { "❌" };
                    info!("  {:<20} {:>8} {:>8} {:>8} {:>8} {:>9.1}% {:>9.4}% {} cov={:.0}%",
                          name, m.n, m.wins, m.losses, m.expired, m.wr(), m.avg_pnl(), marker, coverage);
                }
            }
        }

        if n_super > 0 {
            info!("  Base-V3 agree: {}/{} ({:.1}%)",
                  agree_count, n_super, agree_count as f64 / n_super as f64 * 100.0);
        }
    }

    let total = total_start.elapsed();
    info!("");
    info!("Direction v3 backtest complete in {:.1}s ({:.1}min)",
          total.as_secs_f64(), total.as_secs_f64() / 60.0);
    info!("Target: direction accuracy >= 60% (WR on super points)");

    Ok(())
}
