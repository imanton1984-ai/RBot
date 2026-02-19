// strategies/ml_entry_strategy/src/bin/backtest.rs
//
// Super Entry Strategy Backtester
//
// Evaluates the Super Entry model on historical data.
//
// USAGE:
//   cargo run --release -p ml_entry_strategy --bin super_entry_backtest
//
// WORKFLOW:
//   1. Load models (super_entry + direction per TF)
//   2. For each (symbol, tf), fetch 1000 candles
//   3. Starting from candle 301, run the pipeline:
//      - extract features → predict → score → generate signal
//   4. For each signal, simulate trade:
//      - TP = target_move_pct for this TF
//      - SL = sl_fraction * target_move_pct
//      - Look ahead 20 candles for TP/SL/Expired
//   5. Aggregate metrics: WinRate, AvgPnL, Sharpe, %expired, coverage
//
// OUTPUT:
//   Console report per TF + overall
//   Optional CSV export

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::info;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{fetch_candles_with_indicators, fetch_active_symbols, CandleWithIndicators};
use ml_entry_strategy::pipeline::SuperEntryPipeline;

/// Result of a simulated trade
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
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeOutcome {
    Win,
    Loss,
    Expired,
}

/// Simulate a trade from a signal forward.
///
/// Looks ahead `lookahead` candles for TP or SL hit.
fn simulate_trade(
    candles: &[CandleWithIndicators],
    signal_idx: usize,
    direction: i8,
    entry_price: f64,
    tp_pct: f64,
    sl_pct: f64,
    lookahead: usize,
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

    let end_idx = (signal_idx + lookahead).min(candles.len() - 1);

    for k in (signal_idx + 1)..=end_idx {
        let high = candles[k].high;
        let low = candles[k].low;

        if direction == 1 {
            // LONG: check SL first (worse case first)
            if low <= sl_price {
                let pnl = (sl_price - entry_price) / entry_price * 100.0;
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0, // will be set by caller
                    direction,
                    entry_price,
                    exit_price: sl_price,
                    pnl_pct: pnl,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0,
                    combined_score: 0.0,
                };
            }
            if high >= tp_price {
                let pnl = (tp_price - entry_price) / entry_price * 100.0;
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: tp_price,
                    pnl_pct: pnl,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0,
                    combined_score: 0.0,
                };
            }
        } else {
            // SHORT: check SL first
            if high >= sl_price {
                let pnl = (entry_price - sl_price) / entry_price * 100.0;
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: sl_price,
                    pnl_pct: pnl,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0,
                    combined_score: 0.0,
                };
            }
            if low <= tp_price {
                let pnl = (entry_price - tp_price) / entry_price * 100.0;
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: tp_price,
                    pnl_pct: pnl,
                    outcome: TradeOutcome::Win,
                    bars_to_outcome: k - signal_idx,
                    p_super: 0.0,
                    combined_score: 0.0,
                };
            }
        }
    }

    // Expired: use last candle close
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
        p_super: 0.0,
        combined_score: 0.0,
    }
}

/// Metrics for a group of trades
#[derive(Debug, Default)]
struct TfMetrics {
    total: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    total_pnl: f64,
    pnl_values: Vec<f64>,
    total_candles: usize, // total candles processed (for coverage)
}

#[allow(dead_code)]
impl TfMetrics {
    fn win_rate(&self) -> f64 {
        if self.total > 0 {
            self.wins as f64 / self.total as f64 * 100.0
        } else {
            0.0
        }
    }

    fn avg_pnl(&self) -> f64 {
        if self.total > 0 {
            self.total_pnl / self.total as f64
        } else {
            0.0
        }
    }

    fn sharpe(&self) -> f64 {
        if self.total < 2 {
            return 0.0;
        }
        let mean = self.avg_pnl();
        let variance = self
            .pnl_values
            .iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>()
            / self.total as f64;
        if variance > 0.0 {
            mean / variance.sqrt()
        } else {
            0.0
        }
    }

    fn expired_pct(&self) -> f64 {
        if self.total > 0 {
            self.expired as f64 / self.total as f64 * 100.0
        } else {
            0.0
        }
    }

    fn coverage(&self) -> f64 {
        if self.total_candles > 0 {
            self.total as f64 / self.total_candles as f64 * 100.0
        } else {
            0.0
        }
    }
}

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

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         SUPER ENTRY STRATEGY BACKTESTER                      ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config:");
    println!("    P threshold:    {}", config.p_threshold);
    println!("    Warmup bars:    {}", config.warmup_bars);
    println!("    Lookahead:      {}", config.lookahead_bars);
    println!("    SL fraction:    {}", config.sl_fraction);
    println!("    GPU:            {}", use_gpu);
    println!();

    for &tf in SuperEntryConfig::timeframes() {
        println!(
            "    TF {:>5}m:  TP={:.2}%  SL={:.2}%",
            tf,
            config.target_pct_for_tf(tf),
            config.sl_pct_for_tf(tf)
        );
    }
    println!();

    // Initialize pipeline
    let pipeline = match SuperEntryPipeline::new(config.clone(), use_gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to initialize pipeline: {}", e);
            eprintln!("Make sure models are trained first: ./scripts/super_entry.sh --train-only");
            return Ok(());
        }
    };

    if !pipeline.has_models() {
        eprintln!("No super_entry models found!");
        eprintln!("Train models first:");
        eprintln!("  1. Build dataset:  cargo run --release -p ml_entry_strategy --bin super_entry_dataset");
        eprintln!("  2. Train models:   python trainer/src/train_super_entry.py");
        return Ok(());
    }

    let pool = PgPool::connect(&db_url).await?;
    let symbols = fetch_active_symbols(&pool).await?;
    info!("Found {} active symbols", symbols.len());

    let mut metrics_by_tf: HashMap<i32, TfMetrics> = HashMap::new();
    let mut all_trades: Vec<TradeResult> = Vec::new();

    for &tf in SuperEntryConfig::timeframes() {
        let target_pct = config.target_pct_for_tf(tf);
        let sl_pct = config.sl_pct_for_tf(tf);

        let mut tf_metrics = TfMetrics::default();

        for symbol in &symbols {
            let candles = fetch_candles_with_indicators(&pool, symbol, tf, 1000).await?;

            if candles.len() < config.warmup_bars + config.lookahead_bars + 1 {
                continue;
            }

            let results = pipeline.process_candles(&candles, tf, use_gpu)?;
            tf_metrics.total_candles += results.len();

            for result in &results {
                if let Some(ref signal) = result.signal {
                    let direction = signal.side as i8;
                    let entry_price = signal.entry_price;

                    let mut trade = simulate_trade(
                        &candles,
                        result.candle_index,
                        direction,
                        entry_price,
                        target_pct,
                        sl_pct,
                        config.lookahead_bars,
                    );

                    trade.tf_minutes = tf;
                    trade.p_super = signal.p_super;
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

        metrics_by_tf.insert(tf, tf_metrics);
    }

    // ── Print results ──────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SUPER ENTRY BACKTEST RESULTS                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "{:<6} {:>7} {:>6} {:>6} {:>7} {:>8} {:>9} {:>8} {:>8}",
        "TF", "Trades", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL%", "Sharpe", "Cover%"
    );
    println!("{}", "-".repeat(75));

    let mut tfs: Vec<i32> = metrics_by_tf.keys().copied().collect();
    tfs.sort();

    for tf in &tfs {
        let m = &metrics_by_tf[tf];
        let tf_name = match tf {
            1 => "1m",
            5 => "5m",
            15 => "15m",
            60 => "1h",
            240 => "4h",
            1440 => "1d",
            _ => "??",
        };

        println!(
            "{:<6} {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}% {:>8.3} {:>7.2}%",
            tf_name,
            m.total,
            m.wins,
            m.losses,
            m.expired,
            m.win_rate(),
            m.avg_pnl(),
            m.sharpe(),
            m.coverage()
        );
    }

    // Overall
    let total_trades = all_trades.len();
    let total_wins = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
    let total_losses = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Loss).count();
    let total_expired = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Expired).count();
    let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

    println!("{}", "-".repeat(75));
    println!(
        "TOTAL  {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}%",
        total_trades,
        total_wins,
        total_losses,
        total_expired,
        if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 }
    );
    println!();

    // ── Detailed breakdown for calibration ────────────────────
    println!("======== P(SUPER) BUCKET ANALYSIS (calibration helper) ========");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "0.55-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90+"
    );

    let p_buckets: Vec<(f32, f32, &str)> = vec![
        (0.55, 0.60, "0.55-0.60"),
        (0.60, 0.70, "0.60-0.70"),
        (0.70, 0.80, "0.70-0.80"),
        (0.80, 0.90, "0.80-0.90"),
        (0.90, 1.01, "0.90+"),
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
        for (lo, hi, _) in &p_buckets {
            let in_bucket: Vec<&&TradeResult> = group.iter()
                .filter(|t| t.p_super >= *lo && t.p_super < *hi)
                .collect();
            let n = in_bucket.len();
            if n == 0 {
                cells.push("  -  ".to_string());
            } else {
                let w = in_bucket.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
                cells.push(format!("{:.0}%({}) ", w as f64 / n as f64 * 100.0, n));
            }
        }
        println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            tf_name, cells[0], cells[1], cells[2], cells[3], cells[4]);
    }
    println!();

    // ── Direction breakdown ─────────────────────────────────
    println!("======== DIRECTION BREAKDOWN ========");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "TF", "LONG", "SHORT", "LongWR%", "ShortWR%", "LongPnL%", "ShortPnL%"
    );

    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        let longs: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == 1).collect();
        let shorts: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == -1).collect();

        let long_wr = if !longs.is_empty() {
            longs.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / longs.len() as f64 * 100.0
        } else { 0.0 };
        let short_wr = if !shorts.is_empty() {
            shorts.iter().filter(|t| t.outcome == TradeOutcome::Win).count() as f64 / shorts.len() as f64 * 100.0
        } else { 0.0 };
        let long_pnl = if !longs.is_empty() {
            longs.iter().map(|t| t.pnl_pct).sum::<f64>() / longs.len() as f64
        } else { 0.0 };
        let short_pnl = if !shorts.is_empty() {
            shorts.iter().map(|t| t.pnl_pct).sum::<f64>() / shorts.len() as f64
        } else { 0.0 };

        println!(
            "{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.4}% {:>9.4}%",
            tf_name, longs.len(), shorts.len(), long_wr, short_wr, long_pnl, short_pnl
        );
    }
    println!();

    // ── Avg bars to outcome ─────────────────────────────────
    println!("======== AVG BARS TO OUTCOME ========");
    println!(
        "{:<6} {:>10} {:>10} {:>10}",
        "TF", "AvgWinBars", "AvgLossBars", "AvgExpBars"
    );
    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        let win_bars: Vec<f64> = group.iter().filter(|t| t.outcome == TradeOutcome::Win).map(|t| t.bars_to_outcome as f64).collect();
        let loss_bars: Vec<f64> = group.iter().filter(|t| t.outcome == TradeOutcome::Loss).map(|t| t.bars_to_outcome as f64).collect();
        let exp_bars: Vec<f64> = group.iter().filter(|t| t.outcome == TradeOutcome::Expired).map(|t| t.bars_to_outcome as f64).collect();

        let avg_win = if !win_bars.is_empty() { win_bars.iter().sum::<f64>() / win_bars.len() as f64 } else { 0.0 };
        let avg_loss = if !loss_bars.is_empty() { loss_bars.iter().sum::<f64>() / loss_bars.len() as f64 } else { 0.0 };
        let avg_exp = if !exp_bars.is_empty() { exp_bars.iter().sum::<f64>() / exp_bars.len() as f64 } else { 0.0 };

        println!("{:<6} {:>10.1} {:>11.1} {:>10.1}", tf_name, avg_win, avg_loss, avg_exp);
    }
    println!();

    // ── Top 5 symbols per TF ─────────────────────────────────
    println!("======== TOP 5 SYMBOLS BY PNL (per TF) ========");
    for tf in &sorted_tfs {
        let group = &by_tf_trades[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        let mut by_symbol: HashMap<&str, (f64, usize, usize)> = HashMap::new();
        for t in group {
            let entry = by_symbol.entry(t.symbol.as_str()).or_insert((0.0, 0, 0));
            entry.0 += t.pnl_pct;
            entry.1 += 1;
            if t.outcome == TradeOutcome::Win { entry.2 += 1; }
        }

        let mut sorted_symbols: Vec<(&str, f64, usize, usize)> = by_symbol.iter()
            .map(|(&s, &(pnl, total, wins))| (s, pnl, total, wins))
            .collect();
        sorted_symbols.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        println!("  {} — top 5:", tf_name);
        for (sym, pnl, total, wins) in sorted_symbols.iter().take(5) {
            let wr = if *total > 0 { *wins as f64 / *total as f64 * 100.0 } else { 0.0 };
            println!("    {:<12} PnL: {:>+8.2}%  trades: {:>4}  WR: {:>5.1}%", sym, pnl, total, wr);
        }
    }
    println!();

    // Export CSV if requested
    let csv_output = std::env::var("SUPER_ENTRY_BACKTEST_CSV")
        .unwrap_or_default();

    if !csv_output.is_empty() {
        export_backtest_csv(&all_trades, &csv_output)?;
        println!("Backtest results exported to: {}", csv_output);
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         BACKTEST COMPLETE                                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

/// Export backtest results to CSV
fn export_backtest_csv(trades: &[TradeResult], path: &str) -> Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;
    writeln!(
        file,
        "symbol,tf_minutes,direction,entry_price,exit_price,pnl_pct,outcome,bars_to_outcome,p_super,combined_score"
    )?;

    for t in trades {
        let outcome_str = match t.outcome {
            TradeOutcome::Win => "win",
            TradeOutcome::Loss => "loss",
            TradeOutcome::Expired => "expired",
        };
        writeln!(
            file,
            "{},{},{},{:.6},{:.6},{:.6},{},{},{:.4},{:.4}",
            t.symbol,
            t.tf_minutes,
            t.direction,
            t.entry_price,
            t.exit_price,
            t.pnl_pct,
            outcome_str,
            t.bars_to_outcome,
            t.p_super,
            t.combined_score,
        )?;
    }

    Ok(())
}
