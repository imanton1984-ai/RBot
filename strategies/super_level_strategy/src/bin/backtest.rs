// strategies/super_level_strategy/src/bin/backtest.rs
//
// Super Level Strategy Backtester
//
// Оценивает Level-First Sniper Strategy на исторических данных.
//
// USAGE:
//   cargo run --release -p super_level_strategy --bin super_level_backtest
//
// WORKFLOW:
//   1. Загрузка ML моделей (super_entry, direction) — reuse SuperEntryPipeline
//   2. Для каждой (symbol, tf) пары:
//      a. Fetch ALL candles with indicators (reuse ml_entry_strategy::dataset)
//      b. Batch ML inference (zero-copy CUDA)
//      c. Incremental EWMAC computation
//      d. 5-phase filter per bar
//   3. Для каждого сигнала — симуляция сделки:
//      - TP1/TP2/TP3 (частичное закрытие 50% на TP1 → SL в безубыток)
//      - SL (ATR-based, bounce/breakout)
//      - Max hold bars → Expired
//   4. Агрегированные метрики по TF, сценарию, фазам
//
// OUTPUT:
//   Детальный консольный отчёт + опциональный CSV
//
// ENV:
//   SUPER_LEVEL_BACKTEST_CSV=path.csv — export trades to CSV

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;

use ml_entry_strategy::config::SuperEntryConfig;
use ml_entry_strategy::dataset::{fetch_all_candles_for_tf, CandleWithIndicators};

use super_level_strategy::config::SuperLevelConfig;
use super_level_strategy::phases::{Scenario, phase5_risk_levels};
use super_level_strategy::pipeline::{SuperLevelPipeline, PhaseStats};

// ═══════════════════════════════════════════════════════════
// TRADE SIMULATION
// ═══════════════════════════════════════════════════════════

/// Результат симулированной сделки
#[derive(Debug, Clone)]
struct TradeResult {
    symbol: String,
    tf_minutes: i32,
    direction: i8,
    scenario: Scenario,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
    outcome: TradeOutcome,
    bars_to_outcome: usize,
    p_super: f32,
    ewmac_forecast: f64,
    level_price: f64,
    level_strength: f32,
    distance_atr: f32,
    // Partial close tracking
    tp1_hit: bool,
    partial_pnl_pct: f64, // PnL от частичного закрытия на TP1
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
enum TradeOutcome {
    WinTP1,      // Закрыто 50% на TP1 + остаток на SL_BE или TP2/TP3
    WinTP2,      // Достигнут TP2
    WinTP3,      // Достигнут TP3
    Loss,        // SL hit
    Expired,     // Max hold bars
    PartialWin,  // TP1 hit, остаток закрыт по BE или expired
}

impl TradeOutcome {
    fn is_win(&self) -> bool {
        matches!(self, Self::WinTP1 | Self::WinTP2 | Self::WinTP3 | Self::PartialWin)
    }
}

/// Симуляция сделки с ЧАСТИЧНЫМ ЗАКРЫТИЕМ (50% на TP1 → SL в безубыток)
fn simulate_trade(
    candles: &[CandleWithIndicators],
    entry_idx: usize,
    direction: i8,
    entry_price: f64,
    atr: f64,
    scenario: Scenario,
    config: &SuperLevelConfig,
) -> TradeResult {
    let risk = phase5_risk_levels(entry_price, atr, direction, scenario, config);
    let max_hold = config.max_hold_bars;
    let end_idx = (entry_idx + max_hold).min(candles.len() - 1);
    let partial_close_frac = config.partial_close_pct / 100.0;

    let mut tp1_hit = false;
    let mut sl_price = risk.sl_price;
    let mut partial_pnl = 0.0;

    for k in (entry_idx + 1)..=end_idx {
        let high = candles[k].high;
        let low = candles[k].low;

        if direction == 1 {
            // LONG
            // Check SL first (worst case)
            if low <= sl_price {
                let remaining_frac = if tp1_hit { 1.0 - partial_close_frac } else { 1.0 };
                let sl_pnl = (sl_price - entry_price) / entry_price * 100.0 * remaining_frac;
                let total_pnl = partial_pnl + sl_pnl;
                return make_trade_result(
                    candles, entry_idx, direction, scenario, entry_price,
                    sl_price, total_pnl,
                    if tp1_hit { TradeOutcome::PartialWin } else { TradeOutcome::Loss },
                    k - entry_idx, tp1_hit, partial_pnl,
                );
            }

            // Check TP3 (best case)
            if high >= risk.tp3_price {
                let remaining_frac = if tp1_hit { 1.0 - partial_close_frac } else { 1.0 };
                let tp3_pnl = (risk.tp3_price - entry_price) / entry_price * 100.0 * remaining_frac;
                let total_pnl = partial_pnl + tp3_pnl;
                return make_trade_result(
                    candles, entry_idx, direction, scenario, entry_price,
                    risk.tp3_price, total_pnl, TradeOutcome::WinTP3,
                    k - entry_idx, true, partial_pnl,
                );
            }

            // Check TP2
            if high >= risk.tp2_price && tp1_hit {
                let remaining_frac = 1.0 - partial_close_frac;
                let tp2_pnl = (risk.tp2_price - entry_price) / entry_price * 100.0 * remaining_frac;
                let total_pnl = partial_pnl + tp2_pnl;
                return make_trade_result(
                    candles, entry_idx, direction, scenario, entry_price,
                    risk.tp2_price, total_pnl, TradeOutcome::WinTP2,
                    k - entry_idx, true, partial_pnl,
                );
            }

            // Check TP1 (partial close)
            if !tp1_hit && high >= risk.tp1_price {
                tp1_hit = true;
                partial_pnl = (risk.tp1_price - entry_price) / entry_price * 100.0 * partial_close_frac;
                // SL moves to breakeven
                if config.trail_sl_to_breakeven {
                    sl_price = entry_price;
                }
            }
        } else {
            // SHORT
            // Check SL first
            if high >= sl_price {
                let remaining_frac = if tp1_hit { 1.0 - partial_close_frac } else { 1.0 };
                let sl_pnl = (entry_price - sl_price) / entry_price * 100.0 * remaining_frac;
                let total_pnl = partial_pnl + sl_pnl;
                return make_trade_result(
                    candles, entry_idx, direction, scenario, entry_price,
                    sl_price, total_pnl,
                    if tp1_hit { TradeOutcome::PartialWin } else { TradeOutcome::Loss },
                    k - entry_idx, tp1_hit, partial_pnl,
                );
            }

            // Check TP3
            if low <= risk.tp3_price {
                let remaining_frac = if tp1_hit { 1.0 - partial_close_frac } else { 1.0 };
                let tp3_pnl = (entry_price - risk.tp3_price) / entry_price * 100.0 * remaining_frac;
                let total_pnl = partial_pnl + tp3_pnl;
                return make_trade_result(
                    candles, entry_idx, direction, scenario, entry_price,
                    risk.tp3_price, total_pnl, TradeOutcome::WinTP3,
                    k - entry_idx, true, partial_pnl,
                );
            }

            // Check TP2
            if low <= risk.tp2_price && tp1_hit {
                let remaining_frac = 1.0 - partial_close_frac;
                let tp2_pnl = (entry_price - risk.tp2_price) / entry_price * 100.0 * remaining_frac;
                let total_pnl = partial_pnl + tp2_pnl;
                return make_trade_result(
                    candles, entry_idx, direction, scenario, entry_price,
                    risk.tp2_price, total_pnl, TradeOutcome::WinTP2,
                    k - entry_idx, true, partial_pnl,
                );
            }

            // Check TP1
            if !tp1_hit && low <= risk.tp1_price {
                tp1_hit = true;
                partial_pnl = (entry_price - risk.tp1_price) / entry_price * 100.0 * partial_close_frac;
                if config.trail_sl_to_breakeven {
                    sl_price = entry_price;
                }
            }
        }
    }

    // Expired: close at last candle
    let last_close = candles[end_idx].close;
    let remaining_frac = if tp1_hit { 1.0 - partial_close_frac } else { 1.0 };
    let expired_pnl = if direction == 1 {
        (last_close - entry_price) / entry_price * 100.0 * remaining_frac
    } else {
        (entry_price - last_close) / entry_price * 100.0 * remaining_frac
    };
    let total_pnl = partial_pnl + expired_pnl;

    make_trade_result(
        candles, entry_idx, direction, scenario, entry_price,
        last_close, total_pnl,
        if tp1_hit { TradeOutcome::PartialWin } else { TradeOutcome::Expired },
        end_idx - entry_idx, tp1_hit, partial_pnl,
    )
}

fn make_trade_result(
    candles: &[CandleWithIndicators],
    entry_idx: usize,
    direction: i8,
    scenario: Scenario,
    entry_price: f64,
    exit_price: f64,
    pnl_pct: f64,
    outcome: TradeOutcome,
    bars: usize,
    tp1_hit: bool,
    partial_pnl: f64,
) -> TradeResult {
    TradeResult {
        symbol: candles[entry_idx].symbol.clone(),
        tf_minutes: 0, // set by caller
        direction,
        scenario,
        entry_price,
        exit_price,
        pnl_pct,
        outcome,
        bars_to_outcome: bars,
        p_super: 0.0,   // set by caller
        ewmac_forecast: 0.0,
        level_price: 0.0,
        level_strength: 0.0,
        distance_atr: 0.0,
        tp1_hit,
        partial_pnl_pct: partial_pnl,
    }
}

// ═══════════════════════════════════════════════════════════
// METRICS
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Default)]
struct TfMetrics {
    total: usize,
    wins: usize,
    losses: usize,
    expired: usize,
    partial_wins: usize,
    tp1_hits: usize,
    tp2_hits: usize,
    tp3_hits: usize,
    bounce_trades: usize,
    breakout_trades: usize,
    total_pnl: f64,
    pnl_values: Vec<f64>,
    total_candles: usize,
}

#[allow(dead_code)]
impl TfMetrics {
    fn win_rate(&self) -> f64 {
        if self.total > 0 { self.wins as f64 / self.total as f64 * 100.0 } else { 0.0 }
    }
    fn win_rate_inclusive(&self) -> f64 {
        // Counts PartialWin as win (TP1 was hit)
        if self.total > 0 {
            (self.wins + self.partial_wins) as f64 / self.total as f64 * 100.0
        } else { 0.0 }
    }
    fn avg_pnl(&self) -> f64 {
        if self.total > 0 { self.total_pnl / self.total as f64 } else { 0.0 }
    }
    fn sharpe(&self) -> f64 {
        if self.total < 2 { return 0.0; }
        let mean = self.avg_pnl();
        let variance: f64 = self.pnl_values.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / self.total as f64;
        if variance > 0.0 { mean / variance.sqrt() } else { 0.0 }
    }
    fn coverage(&self) -> f64 {
        if self.total_candles > 0 { self.total as f64 / self.total_candles as f64 * 100.0 } else { 0.0 }
    }
    fn max_drawdown(&self) -> f64 {
        let mut peak = 0.0f64;
        let mut max_dd = 0.0f64;
        let mut equity = 0.0f64;
        for pnl in &self.pnl_values {
            equity += pnl;
            if equity > peak { peak = equity; }
            let dd = peak - equity;
            if dd > max_dd { max_dd = dd; }
        }
        max_dd
    }
    fn profit_factor(&self) -> f64 {
        let gross_profit: f64 = self.pnl_values.iter().filter(|&&p| p > 0.0).sum();
        let gross_loss: f64 = self.pnl_values.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
        if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY }
    }
}

// ═══════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = SuperLevelConfig::from_env();
    let ml_config = SuperEntryConfig::from_env();
    let use_gpu = std::env::var("SUPER_ENTRY_USE_GPU")
        .unwrap_or_default()
        .parse::<bool>()
        .unwrap_or(false);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║       SUPER LEVEL STRATEGY BACKTESTER (Level-First)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  Config:");
    println!("    Radar ATR dist:   ≤ {:.2}", config.radar_distance_atr);
    println!("    Min level str:    {:.1}", config.min_level_strength);
    println!("    EWMAC threshold:  {:.1}", config.ewmac_trend_threshold);
    println!("    ADX threshold:    {:.1}", config.adx_trend_threshold);
    println!("    ML P threshold:   {:.2}", config.ml_p_threshold);
    println!("    Dir match req:    {}", config.require_ml_direction_match);
    println!("    Entry window:     {} bars", config.entry_window_bars);
    println!("    Max hold:         {} bars", config.max_hold_bars);
    println!("    Partial close:    {:.0}% on TP1", config.partial_close_pct);
    println!("    Trail to BE:      {}", config.trail_sl_to_breakeven);
    println!("    Warmup:           {} bars", config.warmup_bars);
    println!("    GPU:              {}", use_gpu);
    println!();
    println!("  ATR Multipliers:");
    println!("    Bounce:  SL={:.2} TP1={:.2} TP2={:.2} TP3={:.2}",
        config.bounce_atr.sl_mult, config.bounce_atr.tp1_mult,
        config.bounce_atr.tp2_mult, config.bounce_atr.tp3_mult);
    println!("    Breakout: SL={:.2} TP1={:.2} TP2={:.2} TP3={:.2}",
        config.breakout_atr.sl_mult, config.breakout_atr.tp1_mult,
        config.breakout_atr.tp2_mult, config.breakout_atr.tp3_mult);
    println!();

    // Initialize pipeline
    let pipeline = match SuperLevelPipeline::new(config.clone(), ml_config, use_gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to initialize pipeline: {}", e);
            eprintln!("Make sure super_entry models are trained first.");
            return Ok(());
        }
    };

    if !pipeline.has_models() {
        eprintln!("No super_entry models found! Train them first.");
        return Ok(());
    }

    let pool = PgPool::connect(&db_url).await?;

    let mut metrics_by_tf: HashMap<i32, TfMetrics> = HashMap::new();
    let mut all_trades: Vec<TradeResult> = Vec::new();

    let backtest_limit = |tf: i32| -> usize {
        match tf {
            1 => 5000, 5 => 12000, 15 => 12000, 60 => 12000,
            240 => 12000, 1440 => 3700, _ => 5000,
        }
    };

    let t0_total = std::time::Instant::now();

    for &tf in SuperLevelConfig::timeframes() {
        let t0_tf = std::time::Instant::now();
        let limit = backtest_limit(tf);
        let mut tf_metrics = TfMetrics::default();

        // Batch fetch: one SQL per TF (reuse ml_entry_strategy::dataset)
        let grouped = match fetch_all_candles_for_tf(&pool, tf, limit).await {
            Ok(g) => g,
            Err(e) => {
                eprintln!("Failed to fetch candles for TF {}m: {}", tf, e);
                continue;
            }
        };

        println!("  TF {:>5}m: fetched {} symbols (limit={}) in {:.1}s",
            tf, grouped.len(), limit, t0_tf.elapsed().as_secs_f64());

        let mut all_phase_stats = PhaseStats::default();

        for (_symbol, candles) in &grouped {
            let need = config.warmup_bars + config.entry_window_bars + config.max_hold_bars + 10;
            if candles.len() < need {
                continue;
            }

            // Run full pipeline (ML batch + EWMAC + 5 phases)
            let outputs = pipeline.process_candles(candles, tf, use_gpu)?;
            let stats = PhaseStats::from_outputs(&outputs);

            // Accumulate phase stats
            all_phase_stats.total_candles += stats.total_candles;
            all_phase_stats.phase1_passed += stats.phase1_passed;
            all_phase_stats.phase2_passed += stats.phase2_passed;
            all_phase_stats.phase3_passed += stats.phase3_passed;
            all_phase_stats.phase4_passed += stats.phase4_passed;
            all_phase_stats.phase5_signals += stats.phase5_signals;
            all_phase_stats.reject_no_level += stats.reject_no_level;
            all_phase_stats.reject_context += stats.reject_context;
            all_phase_stats.reject_ml += stats.reject_ml;
            all_phase_stats.reject_entry += stats.reject_entry;
            all_phase_stats.bounce_signals += stats.bounce_signals;
            all_phase_stats.breakout_signals += stats.breakout_signals;

            tf_metrics.total_candles += outputs.len();

            // Simulate trades for signals that passed all phases
            for output in &outputs {
                if !output.is_signal() {
                    continue;
                }

                let pr = &output.phase_result;
                let scenario = pr.scenario.unwrap_or(Scenario::Bounce);
                let direction = pr.direction;
                let entry_idx = output.candle_index + pr.entry_offset;

                if entry_idx + config.max_hold_bars >= candles.len() {
                    continue;
                }

                let atr = candles[output.candle_index].atr;

                let mut trade = simulate_trade(
                    candles, entry_idx, direction, pr.entry_price,
                    atr, scenario, &config,
                );

                trade.tf_minutes = tf;
                trade.p_super = pr.p_super;
                trade.ewmac_forecast = output.ewmac_forecast;
                trade.level_price = pr.level_price;
                trade.level_strength = pr.level_strength;
                trade.distance_atr = pr.distance_atr;

                tf_metrics.total += 1;
                tf_metrics.total_pnl += trade.pnl_pct;
                tf_metrics.pnl_values.push(trade.pnl_pct);

                if trade.tp1_hit { tf_metrics.tp1_hits += 1; }

                match trade.outcome {
                    TradeOutcome::WinTP1 => { tf_metrics.wins += 1; }
                    TradeOutcome::WinTP2 => { tf_metrics.wins += 1; tf_metrics.tp2_hits += 1; }
                    TradeOutcome::WinTP3 => { tf_metrics.wins += 1; tf_metrics.tp3_hits += 1; }
                    TradeOutcome::Loss => { tf_metrics.losses += 1; }
                    TradeOutcome::Expired => { tf_metrics.expired += 1; }
                    TradeOutcome::PartialWin => { tf_metrics.partial_wins += 1; }
                }

                match scenario {
                    Scenario::Bounce => tf_metrics.bounce_trades += 1,
                    Scenario::Breakout => tf_metrics.breakout_trades += 1,
                }

                all_trades.push(trade);
            }
        }

        // Print phase funnel for this TF
        println!("  TF {:>5}m: {} signals / {} candles ({:.2}%) | {} trades in {:.1}s",
            tf, all_phase_stats.phase5_signals, all_phase_stats.total_candles,
            if all_phase_stats.total_candles > 0 {
                all_phase_stats.phase5_signals as f64 / all_phase_stats.total_candles as f64 * 100.0
            } else { 0.0 },
            tf_metrics.total, t0_tf.elapsed().as_secs_f64());
        println!("           Funnel: Radar={} → Context={} → ML={} → Entry={} → Signal={}",
            all_phase_stats.phase1_passed, all_phase_stats.phase2_passed,
            all_phase_stats.phase3_passed, all_phase_stats.phase4_passed,
            all_phase_stats.phase5_signals);

        metrics_by_tf.insert(tf, tf_metrics);
    }

    println!("\n  Total backtest time: {:.1}s\n", t0_total.elapsed().as_secs_f64());

    // ═══════════════════════════════════════════════════════
    // PRINT RESULTS
    // ═══════════════════════════════════════════════════════

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                    SUPER LEVEL BACKTEST RESULTS (Level-First Sniper)                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!(
        "{:<6} {:>7} {:>6} {:>6} {:>5} {:>6} {:>8} {:>8} {:>8} {:>7} {:>7} {:>8}",
        "TF", "Trades", "Wins", "Loss", "Exp", "PWin", "WR%", "WR+P%", "AvgPnL", "Sharpe", "PF", "MaxDD%"
    );
    println!("{}", "─".repeat(100));

    let mut tfs: Vec<i32> = metrics_by_tf.keys().copied().collect();
    tfs.sort();

    for tf in &tfs {
        let m = &metrics_by_tf[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

        println!(
            "{:<6} {:>7} {:>6} {:>6} {:>5} {:>6} {:>7.1}% {:>7.1}% {:>7.4}% {:>7.3} {:>7.2} {:>7.2}%",
            tf_name,
            m.total, m.wins, m.losses, m.expired, m.partial_wins,
            m.win_rate(), m.win_rate_inclusive(), m.avg_pnl(),
            m.sharpe(), m.profit_factor(), m.max_drawdown(),
        );
    }

    // Overall
    let total_trades = all_trades.len();
    let total_wins = all_trades.iter().filter(|t| t.outcome.is_win()).count();
    let total_losses = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Loss).count();
    let total_expired = all_trades.iter().filter(|t| t.outcome == TradeOutcome::Expired).count();
    let total_partial = all_trades.iter().filter(|t| t.outcome == TradeOutcome::PartialWin).count();
    let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

    println!("{}", "─".repeat(100));
    println!(
        "TOTAL  {:>7} {:>6} {:>6} {:>5} {:>6} {:>7.1}% {:>7.1}% {:>7.4}%",
        total_trades, total_wins, total_losses, total_expired, total_partial,
        if total_trades > 0 { total_wins as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { (total_wins + total_partial) as f64 / total_trades as f64 * 100.0 } else { 0.0 },
        if total_trades > 0 { total_pnl / total_trades as f64 } else { 0.0 },
    );
    println!();

    // ── TP1 Hit Rate ─────────────────────────────────────
    println!("══════ TP1 HIT RATE (Безубыток) ══════");
    for tf in &tfs {
        let m = &metrics_by_tf[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
        let tp1_pct = if m.total > 0 { m.tp1_hits as f64 / m.total as f64 * 100.0 } else { 0.0 };
        println!("  {}: TP1 hit {}/{} ({:.1}%) | TP2: {} | TP3: {}",
            tf_name, m.tp1_hits, m.total, tp1_pct, m.tp2_hits, m.tp3_hits);
    }
    println!();

    // ── Scenario breakdown ─────────────────────────────────
    println!("══════ SCENARIO BREAKDOWN (Bounce vs Breakout) ══════");
    println!(
        "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "Bounce#", "BounceWR%", "BouncePnL", "Break#", "BreakWR%", "BreakPnL"
    );

    let mut by_tf_trades: HashMap<i32, Vec<&TradeResult>> = HashMap::new();
    for t in &all_trades {
        by_tf_trades.entry(t.tf_minutes).or_default().push(t);
    }

    for tf in &tfs {
        if let Some(group) = by_tf_trades.get(tf) {
            let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };

            let bounces: Vec<&&TradeResult> = group.iter().filter(|t| t.scenario == Scenario::Bounce).collect();
            let breakouts: Vec<&&TradeResult> = group.iter().filter(|t| t.scenario == Scenario::Breakout).collect();

            let bounce_wr = if !bounces.is_empty() {
                bounces.iter().filter(|t| t.outcome.is_win()).count() as f64 / bounces.len() as f64 * 100.0
            } else { 0.0 };
            let bounce_pnl = if !bounces.is_empty() {
                bounces.iter().map(|t| t.pnl_pct).sum::<f64>() / bounces.len() as f64
            } else { 0.0 };

            let break_wr = if !breakouts.is_empty() {
                breakouts.iter().filter(|t| t.outcome.is_win()).count() as f64 / breakouts.len() as f64 * 100.0
            } else { 0.0 };
            let break_pnl = if !breakouts.is_empty() {
                breakouts.iter().map(|t| t.pnl_pct).sum::<f64>() / breakouts.len() as f64
            } else { 0.0 };

            println!(
                "{:<6} {:>10} {:>9.1}% {:>9.4}% {:>10} {:>9.1}% {:>9.4}%",
                tf_name, bounces.len(), bounce_wr, bounce_pnl,
                breakouts.len(), break_wr, break_pnl,
            );
        }
    }
    println!();

    // ── Direction breakdown ─────────────────────────────────
    println!("══════ DIRECTION BREAKDOWN ══════");
    println!(
        "{:<6} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "TF", "LONG", "SHORT", "LongWR%", "ShortWR%", "LongPnL%", "ShortPnL%"
    );
    for tf in &tfs {
        if let Some(group) = by_tf_trades.get(tf) {
            let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
            let longs: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == 1).collect();
            let shorts: Vec<&&TradeResult> = group.iter().filter(|t| t.direction == -1).collect();
            let long_wr = if !longs.is_empty() { longs.iter().filter(|t| t.outcome.is_win()).count() as f64 / longs.len() as f64 * 100.0 } else { 0.0 };
            let short_wr = if !shorts.is_empty() { shorts.iter().filter(|t| t.outcome.is_win()).count() as f64 / shorts.len() as f64 * 100.0 } else { 0.0 };
            let long_pnl = if !longs.is_empty() { longs.iter().map(|t| t.pnl_pct).sum::<f64>() / longs.len() as f64 } else { 0.0 };
            let short_pnl = if !shorts.is_empty() { shorts.iter().map(|t| t.pnl_pct).sum::<f64>() / shorts.len() as f64 } else { 0.0 };
            println!(
                "{:<6} {:>8} {:>8} {:>9.1}% {:>9.1}% {:>9.4}% {:>9.4}%",
                tf_name, longs.len(), shorts.len(), long_wr, short_wr, long_pnl, short_pnl,
            );
        }
    }
    println!();

    // ── Top 5 symbols ────────────────────────────────────
    println!("══════ TOP 5 SYMBOLS BY PNL (per TF) ══════");
    for tf in &tfs {
        if let Some(group) = by_tf_trades.get(tf) {
            let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", 1440=>"1d", _=>"??" };
            let mut by_symbol: HashMap<&str, (f64, usize, usize)> = HashMap::new();
            for t in group {
                let entry = by_symbol.entry(t.symbol.as_str()).or_insert((0.0, 0, 0));
                entry.0 += t.pnl_pct;
                entry.1 += 1;
                if t.outcome.is_win() { entry.2 += 1; }
            }
            let mut sorted: Vec<_> = by_symbol.iter().map(|(&s, &(pnl, total, wins))| (s, pnl, total, wins)).collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            println!("  {} — top 5:", tf_name);
            for (sym, pnl, total, wins) in sorted.iter().take(5) {
                let wr = if *total > 0 { *wins as f64 / *total as f64 * 100.0 } else { 0.0 };
                println!("    {:<12} PnL: {:>+8.2}%  trades: {:>4}  WR: {:>5.1}%", sym, pnl, total, wr);
            }
        }
    }
    println!();

    // ── Comparison with pure ML (Super Entry) ──────────────
    println!("══════ COMPARISON HINT ══════");
    println!("  Run pure ML backtest for comparison:");
    println!("    cargo run --release -p ml_entry_strategy --bin super_entry_backtest");
    println!("  Super Level should show:");
    println!("    ✅ 3-5x fewer trades (quality > quantity)");
    println!("    ✅ Higher WR% (65-72% TP1 hit rate)");
    println!("    ✅ Lower max drawdown (entry agent saves from bad entries)");
    println!("    ✅ Higher Sharpe ratio");
    println!();

    // ── Export CSV ─────────────────────────────────────────
    if let Ok(csv_path) = std::env::var("SUPER_LEVEL_BACKTEST_CSV") {
        if !csv_path.is_empty() {
            export_csv(&all_trades, &csv_path)?;
            println!("Trades exported to: {}", csv_path);
        }
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║         SUPER LEVEL BACKTEST COMPLETE                        ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}

fn export_csv(trades: &[TradeResult], path: &str) -> Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "symbol,tf,dir,scenario,entry,exit,pnl_pct,outcome,bars,p_super,ewmac,level_price,level_str,dist_atr,tp1_hit,partial_pnl")?;
    for t in trades {
        let scenario = match t.scenario { Scenario::Bounce => "bounce", Scenario::Breakout => "breakout" };
        let outcome = match t.outcome {
            TradeOutcome::WinTP1 => "win_tp1",
            TradeOutcome::WinTP2 => "win_tp2",
            TradeOutcome::WinTP3 => "win_tp3",
            TradeOutcome::Loss => "loss",
            TradeOutcome::Expired => "expired",
            TradeOutcome::PartialWin => "partial_win",
        };
        writeln!(f, "{},{},{},{},{:.6},{:.6},{:.6},{},{},{:.4},{:.2},{:.6},{:.2},{:.3},{},{:.6}",
            t.symbol, t.tf_minutes, t.direction, scenario,
            t.entry_price, t.exit_price, t.pnl_pct, outcome,
            t.bars_to_outcome, t.p_super, t.ewmac_forecast,
            t.level_price, t.level_strength, t.distance_atr,
            t.tp1_hit, t.partial_pnl_pct,
        )?;
    }
    Ok(())
}
