// strategies/super_level_strategy/src/bin/backtest.rs
//
// Super Level Strategy Backtester
//
// USAGE:
//   cargo run --release -p super_level_strategy --bin super_level_backtest
//
// Evaluates all 5 models ensemble on historical data.
// Shows detailed statistics per model, per TF, per scenario.

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;

use super_level_strategy::config::SuperLevelConfig;
use super_level_strategy::dataset::fetch_all_candles_for_tf;
use super_level_strategy::pipeline::{SuperLevelPipeline, candle_limit_for_tf};
use super_level_strategy::scorer::{SuperLevelDecision, RejectReason};
use ml_entry_strategy::dataset::CandleWithIndicators;

#[derive(Debug, Clone)]
struct TradeResult {
    symbol: String,
    tf_minutes: i32,
    direction: i8,
    is_bounce: bool,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
    outcome: TradeOutcome,
    bars_to_outcome: usize,
    p_eval: f32,
    p_level: f32,
    p_entry: f32,
    p_direction: f32,
    p_bounce: f32,
    combined_score: f32,
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
                    tf_minutes: 0, direction, is_bounce: false,
                    entry_price, exit_price: sl_price,
                    pnl_pct: (sl_price - entry_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_eval: 0.0, p_level: 0.0, p_entry: 0.0,
                    p_direction: 0.0, p_bounce: 0.0, combined_score: 0.0,
                };
            }
            if high >= tp_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, is_bounce: false,
                    entry_price, exit_price: tp_price,
                    pnl_pct: (tp_price - entry_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_eval: 0.0, p_level: 0.0, p_entry: 0.0,
                    p_direction: 0.0, p_bounce: 0.0, combined_score: 0.0,
                };
            }
        } else {
            if high >= sl_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, is_bounce: false,
                    entry_price, exit_price: sl_price,
                    pnl_pct: (entry_price - sl_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_eval: 0.0, p_level: 0.0, p_entry: 0.0,
                    p_direction: 0.0, p_bounce: 0.0, combined_score: 0.0,
                };
            }
            if low <= tp_price {
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, direction, is_bounce: false,
                    entry_price, exit_price: tp_price,
                    pnl_pct: (entry_price - tp_price) / entry_price * 100.0,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_eval: 0.0, p_level: 0.0, p_entry: 0.0,
                    p_direction: 0.0, p_bounce: 0.0, combined_score: 0.0,
                };
            }
        }
    }

    let last_close = candles[end_idx].close;
    let pnl = if direction == 1 {
        (last_close - entry_price) / entry_price * 100.0
    } else {
        (entry_price - last_close) / entry_price * 100.0
    };

    TradeResult {
        symbol: candles[signal_idx].symbol.clone(),
        tf_minutes: 0, direction, is_bounce: false,
        entry_price, exit_price: last_close, pnl_pct: pnl,
        outcome: TradeOutcome::Expired,
        bars_to_outcome: end_idx - signal_idx,
        p_eval: 0.0, p_level: 0.0, p_entry: 0.0,
        p_direction: 0.0, p_bounce: 0.0, combined_score: 0.0,
    }
}

#[derive(Debug, Default)]
#[allow(dead_code)]
struct TfMetrics {
    total: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    total_pnl: f64,
    pnl_values: Vec<f64>,
    total_candles: usize,
    // Rejection funnel
    reject_weak_level: usize,
    reject_bad_entry: usize,
    reject_weak_dir: usize,
    reject_unclear: usize,
    reject_eval: usize,
}

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
    fn expired_pct(&self) -> f64 {
        if self.total > 0 { self.expired as f64 / self.total as f64 * 100.0 } else { 0.0 }
    }
    fn coverage(&self) -> f64 {
        if self.total_candles > 0 { self.total as f64 / self.total_candles as f64 * 100.0 } else { 0.0 }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = SuperLevelConfig::from_env();
    let use_gpu = std::env::var("SUPER_LEVEL_USE_GPU")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          SUPER LEVEL STRATEGY BACKTESTER                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config:");
    println!("    P threshold:    {}", config.p_threshold);
    println!("    Warmup bars:    {}", config.warmup_bars);
    println!("    Lookahead:      {}", config.lookahead_bars);
    println!("    Max hold:       {}", config.effective_max_hold());
    println!("    SL fraction:    {}", config.sl_fraction);
    println!("    Formation bars: {}", config.level_params.formation_bars);
    println!("    Touch zone %:   {}", config.level_params.touch_zone_pct);
    println!("    GPU:            {}", use_gpu);
    println!();

    for &tf in SuperLevelConfig::timeframes() {
        println!(
            "    TF {:>5}m:  TP={:.2}%  SL={:.2}%  candles={}",
            tf,
            config.target_pct_for_tf(tf),
            config.sl_pct_for_tf(tf),
            candle_limit_for_tf(tf),
        );
    }
    println!();

    let pipeline = match SuperLevelPipeline::new(config.clone(), use_gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to init pipeline: {}", e);
            eprintln!("Train models first: python trainer/src/train_super_level.py");
            return Ok(());
        }
    };

    if !pipeline.has_models() {
        eprintln!("No super_level models found!");
        eprintln!("Steps:");
        eprintln!("  1. Build dataset: cargo run --release -p super_level_strategy --bin super_level_dataset");
        eprintln!("  2. Train models:  python trainer/src/train_super_level.py");
        return Ok(());
    }

    let pool = PgPool::connect(&db_url).await?;
    let mut metrics_by_tf: HashMap<i32, TfMetrics> = HashMap::new();
    let mut all_trades: Vec<TradeResult> = Vec::new();

    let t0_total = std::time::Instant::now();

    for &tf in SuperLevelConfig::timeframes() {
        let t0_tf = std::time::Instant::now();
        let target_pct = config.target_pct_for_tf(tf);
        let sl_pct = config.sl_pct_for_tf(tf);
        let limit = candle_limit_for_tf(tf);

        let mut tf_metrics = TfMetrics::default();

        let grouped = match fetch_all_candles_for_tf(&pool, tf, limit).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to fetch TF {}m: {}", tf, e);
                continue;
            }
        };

        println!("  TF {:>5}m: fetched {} symbols in {:.1}s...",
            tf, grouped.len(), t0_tf.elapsed().as_secs_f64());

        for (_symbol, candles) in &grouped {
            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            let results = pipeline.process_candles(candles, tf)?;
            tf_metrics.total_candles += results.len();

            for result in &results {
                // Track rejection funnel
                if let Some(ref dec) = result.decision {
                    match dec {
                        SuperLevelDecision::NoSignal { reason, .. } => {
                            match reason {
                                RejectReason::WeakLevel => tf_metrics.reject_weak_level += 1,
                                RejectReason::BadEntry => tf_metrics.reject_bad_entry += 1,
                                RejectReason::WeakDirection => tf_metrics.reject_weak_dir += 1,
                                RejectReason::UnclearScenario => tf_metrics.reject_unclear += 1,
                                RejectReason::EvaluatorRejected => tf_metrics.reject_eval += 1,
                            }
                        }
                        _ => {}
                    }
                }

                if let Some(ref signal) = result.signal {
                    let direction = signal.side as i8;
                    let entry_price = signal.entry_price;

                    let mut trade = simulate_trade(
                        candles, result.candle_index, direction,
                        entry_price, target_pct, sl_pct,
                        config.effective_max_hold(),
                    );
                    trade.tf_minutes = tf;
                    trade.is_bounce = signal.is_bounce;
                    trade.p_eval = signal.p_eval;
                    trade.p_level = signal.p_level;
                    trade.p_entry = signal.p_entry;
                    trade.p_direction = signal.p_direction;
                    trade.p_bounce = signal.p_bounce;
                    trade.combined_score = signal.final_score;

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

        println!("  TF {:>5}m: {} trades in {:.1}s",
            tf, tf_metrics.total, t0_tf.elapsed().as_secs_f64());
        metrics_by_tf.insert(tf, tf_metrics);
    }

    println!("\n  Total backtest time: {:.1}s\n", t0_total.elapsed().as_secs_f64());

    // ══════════════════════════════════════════════════════════════
    // RESULTS
    // ══════════════════════════════════════════════════════════════

    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                  SUPER LEVEL BACKTEST RESULTS                                ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Main table
    println!(
        "{:<6} {:>7} {:>6} {:>6} {:>7} {:>8} {:>9} {:>8} {:>8}",
        "TF", "Trades", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL%", "Sharpe", "Cover%"
    );
    println!("{}", "-".repeat(75));

    let mut tfs: Vec<i32> = metrics_by_tf.keys().copied().collect();
    tfs.sort();

    for &tf in &tfs {
        let m = &metrics_by_tf[&tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        println!(
            "{:<6} {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}% {:>8.3} {:>7.2}%",
            tf_name, m.total, m.wins, m.losses, m.expired,
            m.win_rate(), m.avg_pnl(), m.sharpe(), m.coverage()
        );
    }

    let total_trades = all_trades.len();
    let total_wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
    let total_losses = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Loss).count();
    let total_expired = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Expired).count();
    let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

    println!("{}", "-".repeat(75));
    println!(
        "TOTAL  {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}%",
        total_trades, total_wins, total_losses, total_expired,
        if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 }
    );
    println!();

    // ── REJECTION FUNNEL ──
    println!("======== REJECTION FUNNEL (per TF) ========");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "Candles", "WeakLvl", "BadEntry", "WeakDir", "Unclear", "EvalRej", "Signals"
    );
    for &tf in &tfs {
        let m = &metrics_by_tf[&tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        println!(
            "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            tf_name, m.total_candles, m.reject_weak_level, m.reject_bad_entry,
            m.reject_weak_dir, m.reject_unclear, m.reject_eval, m.total,
        );
    }
    println!();

    // ── P(EVAL) BUCKET ANALYSIS ──
    println!("======== P(EVAL) BUCKET ANALYSIS ========");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "0.55-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90+"
    );

    let p_buckets: Vec<(f32, f32)> = vec![(0.55,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)];
    let mut by_tf_trades: HashMap<i32, Vec<&TradeResult>> = HashMap::new();
    for t in &all_trades { by_tf_trades.entry(t.tf_minutes).or_default().push(t); }

    for &tf in &tfs {
        let group = match by_tf_trades.get(&tf) { Some(g) => g, None => continue };
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let cells: Vec<String> = p_buckets.iter().map(|&(lo, hi)| {
            let bucket: Vec<&&TradeResult> = group.iter().filter(|t| t.p_eval >= lo && t.p_eval < hi).collect();
            if bucket.is_empty() { "  -  ".to_string() }
            else {
                let w = bucket.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
                format!("{:.0}%({}) ", w as f64 / bucket.len() as f64 * 100.0, bucket.len())
            }
        }).collect();
        println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            tf_name, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    println!();

    // ── DIRECTION BREAKDOWN ──
    println!("======== DIRECTION BREAKDOWN ========");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "TF", "LONG", "SHORT", "LongWR%", "ShortWR%", "LongPnL%", "ShortPnL%"
    );

    for &tf in &tfs {
        let group = match by_tf_trades.get(&tf) { Some(g) => g, None => continue };
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        let longs: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == 1).collect();
        let shorts: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == -1).collect();
        let long_wr = if !longs.is_empty() { longs.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / longs.len() as f64 * 100.0 } else { 0.0 };
        let short_wr = if !shorts.is_empty() { shorts.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / shorts.len() as f64 * 100.0 } else { 0.0 };
        let long_pnl = if !longs.is_empty() { longs.iter().map(|t| t.pnl_pct).sum::<f64>() / longs.len() as f64 } else { 0.0 };
        let short_pnl = if !shorts.is_empty() { shorts.iter().map(|t| t.pnl_pct).sum::<f64>() / shorts.len() as f64 } else { 0.0 };

        println!("{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.4}% {:>9.4}%",
            tf_name, longs.len(), shorts.len(), long_wr, short_wr, long_pnl, short_pnl);
    }
    println!();

    // ── BOUNCE vs BREAKOUT ──
    println!("======== BOUNCE vs BREAKOUT BREAKDOWN ========");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "TF", "Bounce", "Break", "BncWR%", "BrkWR%", "BncPnL%", "BrkPnL%"
    );
    for &tf in &tfs {
        let group = match by_tf_trades.get(&tf) { Some(g) => g, None => continue };
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let bounces: Vec<&&TradeResult> = group.iter().filter(|t| t.is_bounce).collect();
        let breaks: Vec<&&TradeResult> = group.iter().filter(|t| !t.is_bounce).collect();
        let bnc_wr = if !bounces.is_empty() { bounces.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / bounces.len() as f64 * 100.0 } else { 0.0 };
        let brk_wr = if !breaks.is_empty() { breaks.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / breaks.len() as f64 * 100.0 } else { 0.0 };
        let bnc_pnl = if !bounces.is_empty() { bounces.iter().map(|t| t.pnl_pct).sum::<f64>() / bounces.len() as f64 } else { 0.0 };
        let brk_pnl = if !breaks.is_empty() { breaks.iter().map(|t| t.pnl_pct).sum::<f64>() / breaks.len() as f64 } else { 0.0 };
        println!("{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.4}% {:>9.4}%",
            tf_name, bounces.len(), breaks.len(), bnc_wr, brk_wr, bnc_pnl, brk_pnl);
    }
    println!();

    // ── AVG BARS TO OUTCOME ──
    println!("======== AVG BARS TO OUTCOME ========");
    println!("{:<6} {:>10} {:>10} {:>10}", "TF", "AvgWinBars", "AvgLossBars", "AvgExpBars");
    for &tf in &tfs {
        let group = match by_tf_trades.get(&tf) { Some(g) => g, None => continue };
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let avg = |outcome: TradeOutcome| {
            let bars: Vec<f64> = group.iter().filter(|t| t.outcome == outcome).map(|t| t.bars_to_outcome as f64).collect();
            if bars.is_empty() { 0.0 } else { bars.iter().sum::<f64>() / bars.len() as f64 }
        };
        println!("{:<6} {:>10.1} {:>11.1} {:>10.1}",
            tf_name, avg(TradeOutcome::Win), avg(TradeOutcome::Loss), avg(TradeOutcome::Expired));
    }
    println!();

    // ── TOP 5 SYMBOLS ──
    println!("======== TOP 5 SYMBOLS BY PNL (per TF) ========");
    for &tf in &tfs {
        let group = match by_tf_trades.get(&tf) { Some(g) => g, None => continue };
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let mut by_sym: HashMap<&str, (f64, usize, usize)> = HashMap::new();
        for t in group {
            let e = by_sym.entry(t.symbol.as_str()).or_insert((0.0, 0, 0));
            e.0 += t.pnl_pct; e.1 += 1;
            if t.outcome == TradeOutcome::Win { e.2 += 1; }
        }
        let mut sorted: Vec<_> = by_sym.into_iter().map(|(s,(p,n,w))| (s,p,n,w)).collect();
        sorted.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        println!("  {} — top 5:", tf_name);
        for (sym, pnl, tot, wins) in sorted.iter().take(5) {
            let wr = if *tot > 0 { *wins as f64 / *tot as f64 * 100.0 } else { 0.0 };
            println!("    {:<12} PnL: {:>+8.2}%  trades: {:>4}  WR: {:>5.1}%", sym, pnl, tot, wr);
        }
    }
    println!();

    // CSV export
    let csv_output = std::env::var("SUPER_LEVEL_BACKTEST_CSV").unwrap_or_default();
    if !csv_output.is_empty() {
        export_csv(&all_trades, &csv_output)?;
        println!("CSV exported to: {}", csv_output);
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         SUPER LEVEL BACKTEST COMPLETE                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    Ok(())
}

fn export_csv(trades: &[TradeResult], path: &str) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "symbol,tf_minutes,direction,is_bounce,entry_price,exit_price,pnl_pct,outcome,bars,p_eval,p_level,p_entry,p_direction,p_bounce,combined_score")?;
    for t in trades {
        writeln!(f, "{},{},{},{},{:.6},{:.6},{:.6},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            t.symbol, t.tf_minutes, t.direction, t.is_bounce,
            t.entry_price, t.exit_price, t.pnl_pct,
            match t.outcome { TradeOutcome::Win=>"win", TradeOutcome::Loss=>"loss", TradeOutcome::Expired=>"expired" },
            t.bars_to_outcome, t.p_eval, t.p_level, t.p_entry, t.p_direction, t.p_bounce, t.combined_score)?;
    }
    Ok(())
}
