// strategies/ml_entry_strategy/src/bin/backtest.rs
//
// Super Entry Strategy Backtester v2
//
// Key improvements over v1:
//   - Cross-TF heuristic direction: uses trend/supertrend from higher+lower TFs
//   - WFO-aware date filtering: only evaluate trades after training cutoff
//   - Pre-loads ALL TFs into MultiTfStore for cross-referencing
//   - Enhanced reporting: heuristic vs ML direction, per-bucket WR, OOS-only stats
//
// USAGE:
//   cargo run --release -p ml_entry_strategy --bin super_entry_backtest
//
// ENV VARS:
//   SUPER_ENTRY_USE_GPU=true         — use GPU for inference
//   WFO_MIN_DATE=2026-01-13          — only count trades after this date (WFO OOS)
//   HEURISTIC_DIR_CONFIDENCE=0.3     — min heuristic confidence to override ML direction
//   HEURISTIC_DIR_MODE=override      — "override" (replace ML), "filter" (reject if disagree), "off"
//
// WORKFLOW:
//   1. Load models (super_entry + direction per TF)
//   2. Pre-load ALL TF candle data into MultiTfStore
//   3. For each (symbol, tf), run pipeline → predict → score → signal
//   4. For each signal, apply cross-TF heuristic direction
//   5. Simulate trade with final direction
//   6. Aggregate & report

use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{fetch_all_candles_for_tf, CandleWithIndicators};
use ml_entry_strategy::pipeline::SuperEntryPipeline;

// ─────────────────────────────────────────────────────────────────────
// Cross-TF Heuristic Direction System
// ─────────────────────────────────────────────────────────────────────

/// TF hierarchy for cross-TF lookups
fn get_higher_tf(tf: i32) -> Option<i32> {
    match tf {
        1 => Some(5),
        5 => Some(15),
        15 => Some(60),
        60 => Some(240),
        240 => Some(1440),
        _ => None,
    }
}

fn get_lower_tf(tf: i32) -> Option<i32> {
    match tf {
        5 => Some(1),
        15 => Some(5),
        60 => Some(15),
        240 => Some(60),
        1440 => Some(240),
        _ => None,
    }
}

/// Store for cross-TF candle lookups.
/// Pre-loads all TFs so we can quickly find the higher/lower TF candle
/// at any given timestamp for any symbol.
struct MultiTfStore {
    /// tf_minutes -> symbol -> sorted Vec<CandleWithIndicators>
    data: HashMap<i32, HashMap<String, Vec<CandleWithIndicators>>>,
}

impl MultiTfStore {
    fn new() -> Self {
        Self { data: HashMap::new() }
    }

    fn insert_tf(&mut self, tf: i32, grouped: HashMap<String, Vec<CandleWithIndicators>>) {
        self.data.insert(tf, grouped);
    }

    /// Find the latest candle for (symbol, tf) at time <= target_time.
    /// Uses binary search for O(log n) lookup.
    fn find_candle_at(
        &self,
        symbol: &str,
        tf: i32,
        target_time: DateTime<Utc>,
    ) -> Option<&CandleWithIndicators> {
        let tf_data = self.data.get(&tf)?;
        let candles = tf_data.get(symbol)?;
        if candles.is_empty() {
            return None;
        }
        // Binary search: find rightmost candle with time <= target_time
        let idx = candles.partition_point(|c| c.time <= target_time);
        if idx > 0 {
            Some(&candles[idx - 1])
        } else {
            None
        }
    }

    /// Get candles for a specific TF (all symbols)
    fn get_tf(&self, tf: i32) -> Option<&HashMap<String, Vec<CandleWithIndicators>>> {
        self.data.get(&tf)
    }
}

/// Heuristic direction mode
#[derive(Debug, Clone, Copy, PartialEq)]
enum HeuristicMode {
    /// Replace ML direction with heuristic when confident
    Override,
    /// Reject signal if heuristic disagrees with ML direction
    Filter,
    /// Disable heuristic — use pure ML direction
    Off,
}

/// Sign of a float: +1, -1, or 0
fn sign_f64(v: f64) -> i32 {
    if v > 0.01 { 1 } else if v < -0.01 { -1 } else { 0 }
}

/// Compute cross-TF heuristic direction using weighted voting.
///
/// Votes from 3 sources:
///   Current TF:  supertrend_dir, trend, trend_short  → weight=1 each → max ±3
///   Higher TF:   supertrend_dir, trend, trend_short  → weight=2 each → max ±6  (dominant)
///   Lower TF:    trend_short only                    → weight=1       → max ±1  (timing)
///
/// Total max = 10 points
/// Direction = sign(total)
/// Confidence = |total| / 10.0
///
/// Returns (direction, confidence, n_sources_used)
fn compute_heuristic_direction(
    current_candle: &CandleWithIndicators,
    current_tf: i32,
    store: &MultiTfStore,
) -> (i8, f32, u8) {
    let mut score: i32 = 0;
    let max_points: i32 = 10;
    let mut n_sources: u8 = 1;

    // ── Current TF (weight=1 each, max ±3) ──
    score += sign_f64(current_candle.supertrend_dir);
    score += sign_f64(current_candle.trend);
    score += sign_f64(current_candle.trend_short);

    // ── Higher TF (weight=2 each, max ±6) — THE KEY FOR DIRECTION ──
    if let Some(htf) = get_higher_tf(current_tf) {
        if let Some(htf_candle) = store.find_candle_at(
            &current_candle.symbol, htf, current_candle.time,
        ) {
            score += 2 * sign_f64(htf_candle.supertrend_dir);
            score += 2 * sign_f64(htf_candle.trend);
            score += 2 * sign_f64(htf_candle.trend_short);
            n_sources += 1;
        }
    }

    // ── Lower TF (weight=1, trend_short only — for timing) ──
    if let Some(ltf) = get_lower_tf(current_tf) {
        if let Some(ltf_candle) = store.find_candle_at(
            &current_candle.symbol, ltf, current_candle.time,
        ) {
            score += sign_f64(ltf_candle.trend_short);
            n_sources += 1;
        }
    }

    let direction: i8 = if score > 0 { 1 } else { -1 };
    let confidence = (score.abs() as f32) / (max_points as f32);

    (direction, confidence, n_sources)
}

// ─────────────────────────────────────────────────────────────────────
// Trade Simulation
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TradeResult {
    symbol: String,
    tf_minutes: i32,
    direction: i8,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
    outcome: TradeOutcome,
    bars_to_outcome: usize,
    p_super: f32,
    combined_score: f32,
    // v2 fields
    ml_direction: i8,
    heuristic_direction: i8,
    heuristic_confidence: f32,
    heuristic_sources: u8,
    entry_time: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeOutcome {
    Win,
    Loss,
    Expired,
}

fn simulate_trade(
    candles: &[CandleWithIndicators],
    signal_idx: usize,
    direction: i8,
    entry_price: f64,
    tp_pct: f64,
    sl_pct: f64,
    max_hold: usize,
) -> TradeResult {
    let tp_price = if direction == 1 {
        entry_price * (1.0 + tp_pct / 100.0)
    } else {
        entry_price * (1.0 - tp_pct / 100.0)
    };
    let sl_price = if direction == 1 {
        entry_price * (1.0 - sl_pct / 100.0)
    } else {
        entry_price * (1.0 + sl_pct / 100.0)
    };

    let end_idx = (signal_idx + max_hold).min(candles.len() - 1);

    for k in (signal_idx + 1)..=end_idx {
        let high = candles[k].high;
        let low = candles[k].low;

        if direction == 1 {
            if low <= sl_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: sl_price,
                    pnl_pct: (sl_price - entry_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0, combined_score: 0.0,
                    ml_direction: 0, heuristic_direction: 0,
                    heuristic_confidence: 0.0, heuristic_sources: 0,
                    entry_time: candles[signal_idx].time,
                };
            }
            if high >= tp_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: tp_price,
                    pnl_pct: (tp_price - entry_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0, combined_score: 0.0,
                    ml_direction: 0, heuristic_direction: 0,
                    heuristic_confidence: 0.0, heuristic_sources: 0,
                    entry_time: candles[signal_idx].time,
                };
            }
        } else {
            if high >= sl_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: sl_price,
                    pnl_pct: (entry_price - sl_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0, combined_score: 0.0,
                    ml_direction: 0, heuristic_direction: 0,
                    heuristic_confidence: 0.0, heuristic_sources: 0,
                    entry_time: candles[signal_idx].time,
                };
            }
            if low <= tp_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: tp_price,
                    pnl_pct: (entry_price - tp_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0, combined_score: 0.0,
                    ml_direction: 0, heuristic_direction: 0,
                    heuristic_confidence: 0.0, heuristic_sources: 0,
                    entry_time: candles[signal_idx].time,
                };
            }
        }
    }

    // Expired
    let last_close = candles[end_idx].close;
    let pnl = if direction == 1 {
        (last_close - entry_price) / entry_price * 100.0
    } else {
        (entry_price - last_close) / entry_price * 100.0
    };

    TradeResult {
        symbol: candles[signal_idx].symbol.clone(),
        tf_minutes: 0,
        direction,
        entry_price,
        exit_price: last_close,
        pnl_pct: pnl,
        outcome: TradeOutcome::Expired,
        bars_to_outcome: end_idx - signal_idx,
        p_super: 0.0, combined_score: 0.0,
        ml_direction: 0, heuristic_direction: 0,
        heuristic_confidence: 0.0, heuristic_sources: 0,
        entry_time: candles[signal_idx].time,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
struct TfMetrics {
    total: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    total_pnl: f64,
    pnl_values: Vec<f64>,
    total_candles: usize,
    // Direction stats
    ml_dir_used: usize,
    heuristic_dir_used: usize,
    heuristic_override_count: usize,
    heuristic_filter_reject: usize,
}

#[allow(dead_code)]
impl TfMetrics {
    fn win_rate(&self) -> f64 {
        if self.total > 0 { self.wins as f64 / self.total as f64 * 100.0 } else { 0.0 }
    }
    fn avg_pnl(&self) -> f64 {
        if self.total > 0 { self.total_pnl / self.total as f64 } else { 0.0 }
    }
    fn sharpe(&self) -> f64 {
        if self.total < 2 { return 0.0; }
        let mean = self.avg_pnl();
        let var = self.pnl_values.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / self.total as f64;
        if var > 0.0 { mean / var.sqrt() } else { 0.0 }
    }
    fn coverage(&self) -> f64 {
        if self.total_candles > 0 { self.total as f64 / self.total_candles as f64 * 100.0 } else { 0.0 }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = SuperEntryConfig::from_env();
    let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false);

    // WFO date filter: only count trades after this date
    let wfo_min_date: Option<DateTime<Utc>> = std::env::var("WFO_MIN_DATE")
        .ok()
        .and_then(|s| {
            NaiveDate::parse_from_str(s.trim(), "%Y-%m-%d")
                .ok()
                .map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc())
        });

    // Heuristic direction mode
    // Default: "filter" — reject signals where heuristic disagrees with ML.
    // This preserves ML direction quality while adding cross-TF confirmation.
    // "override" mode was tested and found to HURT performance (WR drops from 64% to 45%).
    let heuristic_mode = match std::env::var("HEURISTIC_DIR_MODE")
        .unwrap_or_else(|_| "filter".to_string())
        .to_lowercase()
        .as_str()
    {
        "override" => HeuristicMode::Override,
        "off" => HeuristicMode::Off,
        _ => HeuristicMode::Filter,
    };

    let heuristic_min_confidence: f32 = std::env::var("HEURISTIC_DIR_CONFIDENCE")
        .unwrap_or_else(|_| "0.3".to_string())
        .parse()
        .unwrap_or(0.3);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║   SUPER ENTRY STRATEGY BACKTESTER v2 (Cross-TF Heuristic)   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config:");
    println!("    P threshold:       {}", config.p_threshold);
    println!("    Warmup bars:       {}", config.warmup_bars);
    println!("    Lookahead:         {}", config.lookahead_bars);
    println!("    Max hold bars:     {}", config.effective_max_hold());
    println!("    SL fraction:       {}", config.sl_fraction);
    println!("    GPU:               {}", use_gpu);
    println!("    Heuristic mode:    {:?}", heuristic_mode);
    println!("    Heuristic min conf: {:.2}", heuristic_min_confidence);
    if let Some(ref d) = wfo_min_date {
        println!("    WFO OOS from:      {}", d.format("%Y-%m-%d"));
    } else {
        println!("    WFO OOS from:      (ALL data — no WFO filter)");
    }
    println!();

    for &tf in SuperEntryConfig::timeframes() {
        let limit = match tf {
            1 => 5000, 5 => 12000, 15 => 12000, 60 => 12000,
            240 => 12000, 1440 => 3700, _ => 5000,
        };
        println!(
            "    TF {:>5}m:  TP={:.2}%  SL={:.2}%  candles={:<6}  higher={:?}  lower={:?}",
            tf, config.target_pct_for_tf(tf), config.sl_pct_for_tf(tf), limit,
            get_higher_tf(tf), get_lower_tf(tf),
        );
    }
    println!();

    // Initialize pipeline
    let pipeline = match SuperEntryPipeline::new(config.clone(), use_gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to initialize pipeline: {}", e);
            return Ok(());
        }
    };
    if !pipeline.has_models() {
        eprintln!("No super_entry models found! Train first.");
        return Ok(());
    }

    let pool = PgPool::connect(&db_url).await?;

    // ── PHASE 1: Pre-load ALL TF data ──────────────────────────────
    println!("  Loading candle data for all TFs...");
    let t0_load = std::time::Instant::now();
    let mut store = MultiTfStore::new();

    // All TFs we need (active TFs + their higher/lower for cross-TF)
    let active_tfs = SuperEntryConfig::timeframes();
    let mut all_tfs_needed: Vec<i32> = active_tfs.to_vec();
    for &tf in active_tfs {
        if let Some(h) = get_higher_tf(tf) {
            if !all_tfs_needed.contains(&h) { all_tfs_needed.push(h); }
        }
        if let Some(l) = get_lower_tf(tf) {
            if !all_tfs_needed.contains(&l) { all_tfs_needed.push(l); }
        }
    }
    all_tfs_needed.sort_unstable();
    all_tfs_needed.dedup();

    for &tf in &all_tfs_needed {
        let limit = match tf {
            1 => 5000, 5 => 12000, 15 => 12000, 60 => 12000,
            240 => 12000, 1440 => 3700, _ => 5000,
        };
        let t0 = std::time::Instant::now();
        let grouped = fetch_all_candles_for_tf(&pool, tf, limit).await?;
        let n_symbols = grouped.len();
        let n_candles: usize = grouped.values().map(|v| v.len()).sum();
        println!("    TF {:>5}m: {} symbols, {} candles ({:.1}s)",
            tf, n_symbols, n_candles, t0.elapsed().as_secs_f64());
        store.insert_tf(tf, grouped);
    }
    println!("  All data loaded in {:.1}s\n", t0_load.elapsed().as_secs_f64());

    // ── PHASE 2: Process each active TF ────────────────────────────
    let mut metrics_by_tf: HashMap<i32, TfMetrics> = HashMap::new();
    let mut all_trades: Vec<TradeResult> = Vec::new();
    let t0_total = std::time::Instant::now();

    for &tf in active_tfs {
        let t0_tf = std::time::Instant::now();
        let target_pct = config.target_pct_for_tf(tf);
        let sl_pct = config.sl_pct_for_tf(tf);
        let mut tf_metrics = TfMetrics::default();

        let tf_data = match store.get_tf(tf) {
            Some(d) => d,
            None => continue,
        };

        for (_symbol, candles) in tf_data {
            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            let results = pipeline.process_candles(candles, tf, use_gpu)?;
            tf_metrics.total_candles += results.len();

            for result in &results {
                if let Some(ref signal) = result.signal {
                    let candle_idx = result.candle_index;
                    let entry_time = candles[candle_idx].time;

                    // WFO date filter check
                    if let Some(min_date) = wfo_min_date {
                        if entry_time < min_date {
                            continue;
                        }
                    }

                    let ml_direction = signal.side as i8;
                    let entry_price = signal.entry_price;

                    // ── Cross-TF Heuristic Direction ──
                    let (heuristic_dir, heur_conf, heur_sources) =
                        if heuristic_mode != HeuristicMode::Off {
                            compute_heuristic_direction(
                                &candles[candle_idx], tf, &store,
                            )
                        } else {
                            (ml_direction, 0.0, 0)
                        };

                    let final_direction = match heuristic_mode {
                        HeuristicMode::Override => {
                            if heur_conf >= heuristic_min_confidence {
                                if heuristic_dir != ml_direction {
                                    tf_metrics.heuristic_override_count += 1;
                                }
                                tf_metrics.heuristic_dir_used += 1;
                                heuristic_dir
                            } else {
                                tf_metrics.ml_dir_used += 1;
                                ml_direction
                            }
                        }
                        HeuristicMode::Filter => {
                            if heur_conf >= heuristic_min_confidence
                                && heuristic_dir != ml_direction
                            {
                                tf_metrics.heuristic_filter_reject += 1;
                                continue; // Skip this signal
                            }
                            tf_metrics.ml_dir_used += 1;
                            ml_direction
                        }
                        HeuristicMode::Off => {
                            tf_metrics.ml_dir_used += 1;
                            ml_direction
                        }
                    };

                    // Simulate trade
                    let mut trade = simulate_trade(
                        candles, candle_idx, final_direction,
                        entry_price, target_pct, sl_pct,
                        config.effective_max_hold(),
                    );
                    trade.tf_minutes = tf;
                    trade.p_super = signal.p_super;
                    trade.combined_score = signal.final_score;
                    trade.ml_direction = ml_direction;
                    trade.heuristic_direction = heuristic_dir;
                    trade.heuristic_confidence = heur_conf;
                    trade.heuristic_sources = heur_sources;

                    tf_metrics.total += 1;
                    tf_metrics.total_pnl += trade.pnl_pct;
                    tf_metrics.pnl_values.push(trade.pnl_pct);
                    match trade.outcome {
                        TradeOutcome::Win => tf_metrics.wins += 1,
                        TradeOutcome::Loss => tf_metrics.losses += 1,
                        TradeOutcome::Expired => tf_metrics.expired += 1,
                    }
                    all_trades.push(trade);
                }
            }
        }

        println!("  TF {:>5}m: {} trades in {:.1}s (heuristic_overrides={}, filter_rejects={})",
            tf, tf_metrics.total, t0_tf.elapsed().as_secs_f64(),
            tf_metrics.heuristic_override_count,
            tf_metrics.heuristic_filter_reject);
        metrics_by_tf.insert(tf, tf_metrics);
    }

    println!("\n  Total backtest time: {:.1}s\n", t0_total.elapsed().as_secs_f64());

    // ── RESULTS ────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║              SUPER ENTRY BACKTEST RESULTS v2 (Cross-TF Heuristic)            ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");

    if let Some(ref d) = wfo_min_date {
        println!("  ⚠️  WFO OOS filter: only trades after {}", d.format("%Y-%m-%d"));
    }
    println!("  Heuristic mode: {:?}, min_confidence={:.2}\n", heuristic_mode, heuristic_min_confidence);

    println!(
        "{:<6} {:>7} {:>6} {:>6} {:>7} {:>8} {:>9} {:>8} {:>8}  {:>6} {:>6}",
        "TF", "Trades", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL%", "Sharpe", "Cover%",
        "HeurOv", "FltRej"
    );
    println!("{}", "-".repeat(95));

    let mut tfs: Vec<i32> = metrics_by_tf.keys().copied().collect();
    tfs.sort();

    for tf in &tfs {
        let m = &metrics_by_tf[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        println!(
            "{:<6} {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}% {:>8.3} {:>7.2}%  {:>6} {:>6}",
            tf_name, m.total, m.wins, m.losses, m.expired,
            m.win_rate(), m.avg_pnl(), m.sharpe(), m.coverage(),
            m.heuristic_override_count, m.heuristic_filter_reject,
        );
    }

    // Overall
    let total_trades = all_trades.len();
    let total_wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
    let total_losses = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Loss).count();
    let total_expired = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Expired).count();
    let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

    println!("{}", "-".repeat(95));
    println!(
        "TOTAL  {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}%",
        total_trades, total_wins, total_losses, total_expired,
        if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 }
    );
    println!();

    // ── P(SUPER) BUCKET ANALYSIS ──────────────────────────────────
    println!("======== P(SUPER) BUCKET ANALYSIS ========");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "0.55-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90+"
    );

    let p_buckets: Vec<(f32, f32)> = vec![
        (0.55, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01),
    ];

    let mut by_tf_trades: HashMap<i32, Vec<&TradeResult>> = HashMap::new();
    for t in &all_trades {
        by_tf_trades.entry(t.tf_minutes).or_default().push(t);
    }
    let mut sorted_tfs: Vec<i32> = by_tf_trades.keys().copied().collect();
    sorted_tfs.sort();

    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let mut cells: Vec<String> = Vec::new();
        for (lo, hi) in &p_buckets {
            let bucket: Vec<&&TradeResult> = group.iter()
                .filter(|t| t.p_super >= *lo && t.p_super < *hi)
                .collect();
            let n = bucket.len();
            if n == 0 { cells.push("  -  ".to_string()); }
            else {
                let w = bucket.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
                cells.push(format!("{:.0}%({}) ", w as f64 / n as f64 * 100.0, n));
            }
        }
        println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            tf_name, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    println!();

    // ── DIRECTION BREAKDOWN: ML vs Heuristic ─────────────────────
    println!("======== DIRECTION: ML vs HEURISTIC ========");
    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "TF", "ML=Heur", "ML≠Heur", "MLonly", "->WR", "HeurWR%", "MLonlyWR%"
    );
    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        let agree: Vec<&&TradeResult> = group.iter()
            .filter(|t| t.ml_direction == t.heuristic_direction && t.heuristic_confidence >= heuristic_min_confidence)
            .collect();
        let disagree: Vec<&&TradeResult> = group.iter()
            .filter(|t| t.ml_direction != t.heuristic_direction && t.heuristic_confidence >= heuristic_min_confidence)
            .collect();
        let ml_only: Vec<&&TradeResult> = group.iter()
            .filter(|t| t.heuristic_confidence < heuristic_min_confidence)
            .collect();

        let wr = |trades: &Vec<&&TradeResult>| -> f64 {
            if trades.is_empty() { return 0.0; }
            trades.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64
                / trades.len() as f64 * 100.0
        };

        println!(
            "{:<6} {:>8} {:>8} {:>8} {:>7.1}% {:>9.1}% {:>9.1}%",
            tf_name, agree.len(), disagree.len(), ml_only.len(),
            wr(&agree), wr(&disagree), wr(&ml_only),
        );
    }
    println!();

    // ── HEURISTIC CONFIDENCE BUCKETS ─────────────────────────────
    println!("======== HEURISTIC CONFIDENCE BUCKETS (WinRate by confidence) ========");
    println!(
        "{:<6} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "TF", "conf<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"
    );
    let conf_buckets: Vec<(f32, f32)> = vec![
        (0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01),
    ];
    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let mut cells: Vec<String> = Vec::new();
        for (lo, hi) in &conf_buckets {
            let bucket: Vec<&&TradeResult> = group.iter()
                .filter(|t| t.heuristic_confidence >= *lo && t.heuristic_confidence < *hi)
                .collect();
            let n = bucket.len();
            if n == 0 { cells.push("   -   ".to_string()); }
            else {
                let w = bucket.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
                cells.push(format!("{:.0}%({}) ", w as f64 / n as f64 * 100.0, n));
            }
        }
        println!("{:<6} {:>12} {:>12} {:>12} {:>12} {:>12}",
            tf_name, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    println!();

    // ── LONG vs SHORT breakdown ──────────────────────────────────
    println!("======== DIRECTION BREAKDOWN (LONG vs SHORT) ========");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "TF", "LONG", "SHORT", "LongWR%", "ShortWR%", "LongPnL%", "ShortPnL%"
    );
    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let longs: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == 1).collect();
        let shorts: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == -1).collect();
        let wr_l = if !longs.is_empty() {
            longs.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / longs.len() as f64 * 100.0
        } else { 0.0 };
        let wr_s = if !shorts.is_empty() {
            shorts.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / shorts.len() as f64 * 100.0
        } else { 0.0 };
        let pnl_l = if !longs.is_empty() { longs.iter().map(|t| t.pnl_pct).sum::<f64>() / longs.len() as f64 } else { 0.0 };
        let pnl_s = if !shorts.is_empty() { shorts.iter().map(|t| t.pnl_pct).sum::<f64>() / shorts.len() as f64 } else { 0.0 };
        println!("{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.4}% {:>9.4}%",
            tf_name, longs.len(), shorts.len(), wr_l, wr_s, pnl_l, pnl_s);
    }
    println!();

    // Export CSV
    let csv_output = std::env::var("SUPER_ENTRY_BACKTEST_CSV").unwrap_or_default();
    if !csv_output.is_empty() {
        export_backtest_csv(&all_trades, &csv_output)?;
        println!("  Backtest CSV: {}", csv_output);
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         BACKTEST COMPLETE                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn export_backtest_csv(trades: &[TradeResult], path: &str) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    writeln!(file,
        "symbol,tf_minutes,direction,entry_price,exit_price,pnl_pct,outcome,bars_to_outcome,\
         p_super,combined_score,ml_direction,heuristic_direction,heuristic_confidence,\
         heuristic_sources,entry_time"
    )?;
    for t in trades {
        writeln!(file,
            "{},{},{},{:.6},{:.6},{:.6},{},{},{:.4},{:.4},{},{},{:.4},{},{}",
            t.symbol, t.tf_minutes, t.direction, t.entry_price, t.exit_price,
            t.pnl_pct,
            match t.outcome { TradeOutcome::Win => "win", TradeOutcome::Loss => "loss", TradeOutcome::Expired => "expired" },
            t.bars_to_outcome, t.p_super, t.combined_score,
            t.ml_direction, t.heuristic_direction, t.heuristic_confidence,
            t.heuristic_sources, t.entry_time.to_rfc3339(),
        )?;
    }
    Ok(())
}
