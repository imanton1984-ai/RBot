// strategies/ewmac_strategy/src/bin/backtest.rs
//
// EWMAC Strategy Backtester
//
// Evaluates the EWMAC strategy on historical data.
//
// USAGE:
//   cargo run --release -p ewmac_strategy --bin ewmac_backtest
//
// ENV VARS:
//   DATABASE_URL           - PostgreSQL connection string
//   EWMAC_MIN_FORECAST     - Minimum |forecast| for signal (default: 5)
//   EWMAC_MAX_HOLD_BARS    - Max bars to hold (default: 30)
//   EWMAC_WARMUP_BARS      - Warmup period (default: 300)
//   EWMAC_FDM              - Forecast diversification multiplier (default: 1.2)
//   EWMAC_BACKTEST_CSV     - If set, export results to this CSV file
//
// WORKFLOW:
//   1. For each (symbol, tf), fetch 1000 candles
//   2. Compute EWMAC indicators from raw OHLC
//   3. Generate signals where |forecast| >= min_forecast
//   4. For each signal, simulate trade:
//      - TP = ATR * tp_mult
//      - SL = ATR * sl_mult
//      - Max hold = max_hold_bars
//   5. Aggregate metrics: WinRate, AvgPnL, Sharpe, %expired, coverage

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;

use ewmac_strategy::config::EwmacConfig;
use ewmac_strategy::dataset::{Candle, fetch_all_candles_for_tf};
use ewmac_strategy::pipeline::EwmacPipeline;

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
    forecast: f64,
    signal_strength: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum TradeOutcome {
    Win,
    Loss,
    Expired,
}

/// Simulate a trade from a signal forward.
///
/// Looks ahead `max_hold` candles for TP or SL hit.
/// If neither is hit, force-close at market (Expired).
fn simulate_trade(
    candles: &[Candle],
    signal_idx: usize,
    direction: i8,
    entry_price: f64,
    tp_price: f64,
    sl_price: f64,
    max_hold: usize,
) -> TradeResult {
    let end_idx = (signal_idx + max_hold).min(candles.len() - 1);

    for k in (signal_idx + 1)..=end_idx {
        let high = candles[k].high;
        let low = candles[k].low;

        if direction == 1 {
            // LONG: check SL first (worse case first)
            if low <= sl_price {
                let pnl = (sl_price - entry_price) / entry_price * 100.0;
                return TradeResult {
                    symbol: candles[signal_idx].symbol.clone(),
                    tf_minutes: 0,
                    direction,
                    entry_price,
                    exit_price: sl_price,
                    pnl_pct: pnl,
                    outcome: TradeOutcome::Loss,
                    bars_to_outcome: k - signal_idx,
                    forecast: 0.0,
                    signal_strength: 0.0,
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
                    forecast: 0.0,
                    signal_strength: 0.0,
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
                    forecast: 0.0,
                    signal_strength: 0.0,
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
                    forecast: 0.0,
                    signal_strength: 0.0,
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
        forecast: 0.0,
        signal_strength: 0.0,
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
    total_candles: usize,
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
        let variance = self.pnl_values.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / self.total as f64;
        if variance > 0.0 { mean / variance.sqrt() } else { 0.0 }
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

    let config = EwmacConfig::from_env();

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         EWMAC STRATEGY BACKTESTER                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config:");
    println!("    Min forecast:   {}", config.min_forecast);
    println!("    Max forecast:   {}", config.max_forecast);
    println!("    Warmup bars:    {}", config.warmup_bars);
    println!("    Max hold bars:  {}", config.max_hold_bars);
    println!("    Cooldown bars:  {}", config.cooldown_bars);
    println!("    Min pairs agree:{}", config.min_pairs_agree);
    println!("    FDM:            {}", config.fdm);
    println!("    Min ATR%:       {}", config.min_atr_pct);
    println!("    Pairs:          {}", config.pairs.len());
    println!();

    for &tf in EwmacConfig::timeframes() {
        let (sl_m, tp_m) = config.atr_mults_for_tf(tf);
        println!("    TF {:>5}m:  SL=ATR*{:.1}  TP=ATR*{:.1}", tf, sl_m, tp_m);
    }
    println!();

    // Initialize pipeline (no models to load — instant)
    let pipeline = EwmacPipeline::new(config.clone());

    let pool = PgPool::connect(&db_url).await?;

    let mut metrics_by_tf: HashMap<i32, TfMetrics> = HashMap::new();
    let mut all_trades: Vec<TradeResult> = Vec::new();

    let t0_total = std::time::Instant::now();

    for &tf in EwmacConfig::timeframes() {
        let t0_tf = std::time::Instant::now();

        let mut tf_metrics = TfMetrics::default();

        let grouped = match fetch_all_candles_for_tf(&pool, tf, 1000).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to fetch candles for TF {}m: {}", tf, e);
                continue;
            }
        };

        println!("  TF {:>5}m: fetched {} symbols in {:.1}s, processing...",
            tf, grouped.len(), t0_tf.elapsed().as_secs_f64());

        for (_symbol, candles) in &grouped {
            if candles.len() < config.warmup_bars + config.max_hold_bars + 1 {
                continue;
            }

            let results = pipeline.process_candles(candles, tf)?;
            tf_metrics.total_candles += results.len();

            for result in &results {
                if let Some(ref signal) = result.signal {
                    let direction = signal.side as i8;
                    let entry_price = signal.entry_price;

                    let mut trade = simulate_trade(
                        candles,
                        result.candle_index,
                        direction,
                        entry_price,
                        signal.tp_price,
                        signal.sl_price,
                        config.max_hold_bars,
                    );

                    trade.tf_minutes = tf;
                    trade.forecast = signal.forecast;
                    trade.signal_strength = signal.signal_strength;

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

    // ── Print results ──────────────────────────────────────
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    EWMAC BACKTEST RESULTS                                    ║");
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
            1 => "1m", 5 => "5m", 15 => "15m", 60 => "1h", 240 => "4h", 1440 => "1d", _ => "??",
        };

        println!(
            "{:<6} {:>7} {:>6} {:>6} {:>7} {:>7.1}% {:>8.4}% {:>8.3} {:>7.2}%",
            tf_name, m.total, m.wins, m.losses, m.expired,
            m.win_rate(), m.avg_pnl(), m.sharpe(), m.coverage()
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
        total_trades, total_wins, total_losses, total_expired,
        if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 }
    );
    println!();

    // ── Forecast bucket analysis ────────────────────────────
    println!("======== FORECAST BUCKET ANALYSIS ========");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10}",
        "TF", "5-10", "10-15", "15-20", "20"
    );

    let f_buckets: Vec<(f64, f64, &str)> = vec![
        (5.0, 10.0, "5-10"),
        (10.0, 15.0, "10-15"),
        (15.0, 20.0, "15-20"),
        (20.0, 21.0, "20"),
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
        for (lo, hi, _) in &f_buckets {
            let in_bucket: Vec<&&TradeResult> = group.iter()
                .filter(|t| t.forecast.abs() >= *lo && t.forecast.abs() < *hi)
                .collect();
            let n = in_bucket.len();
            if n == 0 {
                cells.push("  -  ".to_string());
            } else {
                let w = in_bucket.iter().filter(|t| t.outcome == TradeOutcome::Win).count();
                cells.push(format!("{:.0}%({}) ", w as f64 / n as f64 * 100.0, n));
            }
        }
        println!("{:<6} {:>10} {:>10} {:>10} {:>10}",
            tf_name, cells[0], cells[1], cells[2], cells[3]);
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
    let csv_output = std::env::var("EWMAC_BACKTEST_CSV").unwrap_or_default();
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
        "symbol,tf_minutes,direction,entry_price,exit_price,pnl_pct,outcome,bars_to_outcome,forecast,signal_strength"
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
            t.symbol, t.tf_minutes, t.direction,
            t.entry_price, t.exit_price, t.pnl_pct,
            outcome_str, t.bars_to_outcome,
            t.forecast, t.signal_strength,
        )?;
    }

    Ok(())
}
