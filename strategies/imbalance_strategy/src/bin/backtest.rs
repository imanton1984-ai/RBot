// strategies/imbalance_strategy/src/bin/backtest.rs
//
// Imbalance Candle Strategy — Honest Backtest v4
//
// v4 CHANGES:
//   1. Config loaded from config/imbalance.toml (no recompile needed)
//   2. TF pairs configurable and individually enable/disable
//   3. Improved statistics with entry mode display
//
// NO LOOK-AHEAD BIAS:
//   - Parent candle is fully CLOSED before we look at it
//   - Entry on child TF with confirmation only
//   - TP/SL checked on future candles only after entry
//
// USAGE:
//   cargo build --release -p imbalance_strategy --bin imbalance_backtest
//   ./target/release/imbalance_backtest [config_path]

use anyhow::Result;
use chrono::{DateTime, Utc};
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;
use tracing::info;

use imbalance_strategy::imbalance::{
    Candle, ImbalanceConfig,
};
use imbalance_strategy::confirmation::{
    SimTrade, TradeOutcome, process_symbol,
};

// ─────────────────────────────────────────────────────────────────────
// DB row types
// ─────────────────────────────────────────────────────────────────────

#[derive(sqlx::FromRow)]
struct CandleRow {
    time: DateTime<Utc>,
    symbol: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    rsi: f64,
    stoch_k: f64,
    stoch_d: f64,
    williams: f64,
    adx: f64,
    atr: f64,
    bb_upper: f64,
    bb_lower: f64,
    ema_20: f64,
    ema_50: f64,
    volume_spike: f64,
    trend: f64,
    trend_short: f64,
    supertrend_dir: f64,
    macd_hist: f64,
    cmf: f64,
    mfi: f64,
    obv: f64,
}

impl CandleRow {
    fn into_candle(self) -> Candle {
        Candle {
            time: self.time,
            symbol: self.symbol,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            rsi: self.rsi,
            stoch_k: self.stoch_k,
            stoch_d: self.stoch_d,
            williams: self.williams,
            adx: self.adx,
            atr: self.atr,
            bb_upper: self.bb_upper,
            bb_lower: self.bb_lower,
            ema_20: self.ema_20,
            ema_50: self.ema_50,
            volume_spike: self.volume_spike,
            trend: self.trend,
            trend_short: self.trend_short,
            supertrend_dir: self.supertrend_dir,
            macd_hist: self.macd_hist,
            cmf: self.cmf,
            mfi: self.mfi,
            obv: self.obv,
        }
    }
}

#[derive(sqlx::FromRow)]
struct CandleRowOhlcv {
    time: DateTime<Utc>,
    symbol: String,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

impl CandleRowOhlcv {
    fn into_candle(self) -> Candle {
        let atr_approx = (self.high - self.low) * 0.5;
        Candle {
            time: self.time,
            symbol: self.symbol,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            rsi: 50.0,
            stoch_k: 50.0,
            stoch_d: 50.0,
            williams: -50.0,
            adx: 25.0,
            atr: atr_approx,
            bb_upper: self.close,
            bb_lower: self.close,
            ema_20: self.close,
            ema_50: self.close,
            volume_spike: 1.0,
            trend: 0.0,
            trend_short: 0.0,
            supertrend_dir: 0.0,
            macd_hist: 0.0,
            cmf: 0.0,
            mfi: 50.0,
            obv: 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Batch DB Fetch
// ─────────────────────────────────────────────────────────────────────

fn candle_table(tf_minutes: i32) -> &'static str {
    match tf_minutes {
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        1440 => "market.candles_1d",
        _ => "market.candles_1d",
    }
}

async fn fetch_all_with_indicators(
    pool: &PgPool,
    tf_minutes: i32,
) -> Result<HashMap<String, Vec<Candle>>> {
    let table = candle_table(tf_minutes);

    let sql = format!(
        r#"
        SELECT
            c.time, c.symbol,
            c.open, c.high, c.low, c.close, c.volume,
            COALESCE(i.rsi, 50.0)::FLOAT8 as rsi,
            COALESCE(i.stoch_k, 50.0)::FLOAT8 as stoch_k,
            COALESCE(i.stoch_d, 50.0)::FLOAT8 as stoch_d,
            COALESCE(i.williams, -50.0)::FLOAT8 as williams,
            COALESCE(i.adx, 25.0)::FLOAT8 as adx,
            COALESCE(i.atr, 0.001)::FLOAT8 as atr,
            COALESCE(i.bb_upper, c.close)::FLOAT8 as bb_upper,
            COALESCE(i.bb_lower, c.close)::FLOAT8 as bb_lower,
            COALESCE(i.ema_20, c.close)::FLOAT8 as ema_20,
            COALESCE(i.ema_50, c.close)::FLOAT8 as ema_50,
            COALESCE(i.volume_spike, 1.0)::FLOAT8 as volume_spike,
            COALESCE(i.trend, 0.0)::FLOAT8 as trend,
            COALESCE(i.trend_short, 0.0)::FLOAT8 as trend_short,
            COALESCE(i.supertrend_dir, 0.0)::FLOAT8 as supertrend_dir,
            COALESCE(i.macd_hist, 0.0)::FLOAT8 as macd_hist,
            COALESCE(i.cmf, 0.0)::FLOAT8 as cmf,
            COALESCE(i.mfi, 50.0)::FLOAT8 as mfi,
            COALESCE(i.obv, 0.0)::FLOAT8 as obv
        FROM {table} c
        JOIN market.pairs p ON p.symbol = c.symbol AND p.is_active = true
        LEFT JOIN market.indicators_wide i
            ON i.symbol_id = p.symbol_id AND i.time = c.time AND i.tf_minutes = $1
        ORDER BY c.symbol, c.time ASC
        "#
    );

    let rows: Vec<CandleRow> = sqlx::query_as::<_, CandleRow>(&sql)
        .bind(tf_minutes as i16)
        .fetch_all(pool)
        .await?;

    let total_rows = rows.len();
    let mut grouped: HashMap<String, Vec<Candle>> = HashMap::new();
    for r in rows {
        let sym = r.symbol.clone();
        grouped.entry(sym).or_default().push(r.into_candle());
    }

    info!("    TF {}m (with indicators): {} rows, {} symbols",
          tf_minutes, total_rows, grouped.len());

    Ok(grouped)
}

async fn fetch_all_ohlcv(
    pool: &PgPool,
    tf_minutes: i32,
) -> Result<HashMap<String, Vec<Candle>>> {
    let table = candle_table(tf_minutes);

    let sql = format!(
        r#"
        SELECT c.time, c.symbol, c.open, c.high, c.low, c.close, c.volume
        FROM {table} c
        WHERE c.symbol IN (SELECT symbol FROM market.pairs WHERE is_active = true)
        ORDER BY c.symbol, c.time ASC
        "#
    );

    let rows: Vec<CandleRowOhlcv> = sqlx::query_as::<_, CandleRowOhlcv>(&sql)
        .fetch_all(pool)
        .await?;

    let total_rows = rows.len();
    let mut grouped: HashMap<String, Vec<Candle>> = HashMap::new();
    for r in rows {
        let sym = r.symbol.clone();
        grouped.entry(sym).or_default().push(r.into_candle());
    }

    info!("    TF {}m (OHLCV only): {} rows, {} symbols",
          tf_minutes, total_rows, grouped.len());

    Ok(grouped)
}

async fn fetch_active_symbols(pool: &PgPool) -> Result<Vec<String>> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT symbol FROM market.pairs WHERE is_active = true ORDER BY symbol"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(s,)| s).collect())
}

// ─────────────────────────────────────────────────────────────────────
// Statistics Printing
// ─────────────────────────────────────────────────────────────────────

fn print_stats(label: &str, trades: &[&SimTrade]) {
    if trades.is_empty() {
        info!("    {} — no trades", label);
        return;
    }
    let n = trades.len();
    let tp = trades.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
    let sl = trades.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
    let exp = trades.iter().filter(|t| t.outcome == TradeOutcome::Expired).count();
    let exp_pos = trades.iter()
        .filter(|t| t.outcome == TradeOutcome::Expired && t.pnl_pct > 0.0)
        .count();
    let wr = tp as f64 / n as f64 * 100.0;
    let total_pnl: f64 = trades.iter().map(|t| t.pnl_pct).sum();
    let avg_pnl = total_pnl / n as f64;
    let avg_hold: f64 = trades.iter().map(|t| t.hold_candles as f64).sum::<f64>() / n as f64;

    // Compute win rate including profitable expired
    let winning = trades.iter().filter(|t| t.pnl_pct > 0.0).count();
    let real_wr = winning as f64 / n as f64 * 100.0;

    info!("    📊 {} ({} trades):", label, n);
    info!("      TP WinRate: {:.1}%  RealWR: {:.1}%  (TP:{} SL:{} Expired:{} [{}+/{}−])",
          wr, real_wr, tp, sl, exp, exp_pos, exp - exp_pos);
    info!("      Avg P&L: {:.2}%,  Total P&L: {:.2}%", avg_pnl, total_pnl);
    info!("      Avg Hold: {:.1} candles", avg_hold);

    // Profit factor
    let gross_profit: f64 = trades.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
    let gross_loss: f64 = trades.iter().filter(|t| t.pnl_pct < 0.0).map(|t| t.pnl_pct.abs()).sum();
    let pf = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };
    info!("      Profit Factor: {:.2}  (gross +{:.1}% / -{:.1}%)", pf, gross_profit, gross_loss);
}

fn print_score_buckets(trades: &[&SimTrade]) {
    info!("    ─── By Score Bucket ───");
    let buckets: &[(f64, f64)] = &[
        (0.0, 0.50), (0.50, 0.65), (0.65, 0.75),
        (0.75, 0.85), (0.85, 0.95), (0.95, 1.01),
    ];
    for &(lo, hi) in buckets {
        let b: Vec<_> = trades.iter()
            .filter(|t| t.score >= lo && t.score < hi)
            .collect();
        if b.is_empty() { continue; }
        let bt = b.len();
        let btp = b.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let bsl = b.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
        let bwin = b.iter().filter(|t| t.pnl_pct > 0.0).count();
        let bwr = bwin as f64 / bt as f64 * 100.0;
        let bavg: f64 = b.iter().map(|t| t.pnl_pct).sum::<f64>() / bt as f64;
        let btot: f64 = b.iter().map(|t| t.pnl_pct).sum();
        info!("      [{:.2}-{:.2}): {} trades, RealWR: {:.1}% (TP:{} SL:{}), AvgPnL: {:.2}%, TotalPnL: {:.2}%",
              lo, hi, bt, bwr, btp, bsl, bavg, btot);
    }
}

fn print_move_size_buckets(trades: &[&SimTrade]) {
    info!("    ─── By Parent Move Size ───");
    let buckets: &[(f64, f64, &str)] = &[
        (0.0, 5.0, "< 5%"),
        (5.0, 10.0, "5-10%"),
        (10.0, 15.0, "10-15%"),
        (15.0, 20.0, "15-20%"),
        (20.0, 30.0, "20-30%"),
        (30.0, 100.0, "30%+"),
    ];
    for &(lo, hi, label) in buckets {
        let b: Vec<_> = trades.iter()
            .filter(|t| t.parent_move_pct >= lo && t.parent_move_pct < hi)
            .collect();
        if b.is_empty() { continue; }
        let bt = b.len();
        let btp = b.iter().filter(|t| t.outcome == TradeOutcome::TpHit).count();
        let bsl = b.iter().filter(|t| t.outcome == TradeOutcome::SlHit).count();
        let bwin = b.iter().filter(|t| t.pnl_pct > 0.0).count();
        let bwr = bwin as f64 / bt as f64 * 100.0;
        let bavg: f64 = b.iter().map(|t| t.pnl_pct).sum::<f64>() / bt as f64;
        let btot: f64 = b.iter().map(|t| t.pnl_pct).sum();
        info!("      {}: {} trades, RealWR: {:.1}% (TP:{} SL:{}), AvgPnL: {:.2}%, TotalPnL: {:.2}%",
              label, bt, bwr, btp, bsl, bavg, btot);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();

    // ─── Logging: both console and file ───
    let log_path = "logs/imbalance_backtest.log";
    std::fs::create_dir_all("logs")?;
    let log_file = std::fs::OpenOptions::new()
        .create(true).write(true).truncate(true).open(log_path)?;

    use tracing_subscriber::fmt;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let rust_log = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    let filter = if !rust_log.contains("sqlx") {
        format!("{},sqlx::query=error", rust_log)
    } else {
        rust_log
    };

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(fmt::layer().with_target(false).with_ansi(false)
            .with_writer(std::sync::Mutex::new(log_file)))
        .with(tracing_subscriber::EnvFilter::new(filter))
        .init();

    // ─── Configuration ───
    let config_path = std::env::args().nth(1)
        .unwrap_or_else(|| "config/imbalance.toml".to_string());

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let config = ImbalanceConfig::from_toml(&config_path);

    info!("╔════════════════════════════════════════════════════════════════╗");
    info!("║  Imbalance Candle Strategy — Honest Backtest v4               ║");
    info!("║  Config: {}",  config_path);
    info!("║  No Look-Ahead Bias — Simulates Real-Time Trading             ║");
    info!("╚════════════════════════════════════════════════════════════════╝");
    config.log_summary();

    let active_pairs = config.active_tf_pairs();
    if active_pairs.is_empty() {
        info!("  ⚠️  No active TF pairs. Enable at least one in config.");
        return Ok(());
    }

    // ─── Database ───
    let pool = PgPool::connect(&db_url).await?;
    let total_start = std::time::Instant::now();

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 1: Batch load candle data (only for active TF pairs)
    // ═══════════════════════════════════════════════════════════════════
    info!("");
    info!("  ══ Phase 1: Batch loading candle data ══");
    let phase1_start = std::time::Instant::now();

    // Collect unique TFs needed
    let mut parent_tfs: Vec<i32> = active_pairs.iter().map(|p| p.parent_tf).collect();
    let mut child_tfs: Vec<i32> = active_pairs.iter().map(|p| p.child_tf).collect();
    parent_tfs.sort();
    parent_tfs.dedup();
    child_tfs.sort();
    child_tfs.dedup();

    // TFs that are ONLY children (not also parents) → OHLCV only (fast)
    let child_only_tfs: Vec<i32> = child_tfs.iter()
        .filter(|c| !parent_tfs.contains(c))
        .copied()
        .collect();

    let mut tf_data: HashMap<i32, HashMap<String, Vec<Candle>>> = HashMap::new();

    // Fetch parent TFs with indicators
    for &tf in &parent_tfs {
        let data = fetch_all_with_indicators(&pool, tf).await?;
        tf_data.insert(tf, data);
    }

    // Fetch child-only TFs with OHLCV (fast, no indicators JOIN)
    for &tf in &child_only_tfs {
        let data = fetch_all_ohlcv(&pool, tf).await?;
        tf_data.insert(tf, data);
    }

    let phase1_secs = phase1_start.elapsed().as_secs_f64();
    info!("  Data loaded in {:.1}s ({:.1}min)", phase1_secs, phase1_secs / 60.0);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 2: Process all symbols
    // ═══════════════════════════════════════════════════════════════════
    info!("");
    info!("  ══ Phase 2: Processing symbols ══");
    let phase2_start = std::time::Instant::now();

    let symbols = fetch_active_symbols(&pool).await?;
    info!("  {} active symbols, {} active TF pairs", symbols.len(), active_pairs.len());

    let mut all_trades: Vec<SimTrade> = Vec::new();
    let mut symbols_processed = 0u32;
    let mut symbols_skipped = 0u32;

    for (si, symbol) in symbols.iter().enumerate() {
        let mut sym_trades = 0u32;

        for tf_pair in active_pairs {
            let parent_candles = match tf_data.get(&tf_pair.parent_tf)
                .and_then(|m| m.get(symbol.as_str()))
            {
                Some(c) if c.len() >= 50 => c,
                _ => continue,
            };

            let child_candles = match tf_data.get(&tf_pair.child_tf)
                .and_then(|m| m.get(symbol.as_str()))
            {
                Some(c) if c.len() >= 50 => c,
                _ => continue,
            };

            let trades = process_symbol(parent_candles, child_candles, tf_pair, &config);

            sym_trades += trades.len() as u32;
            all_trades.extend(trades);
        }

        if sym_trades > 0 {
            symbols_processed += 1;
        } else {
            symbols_skipped += 1;
        }

        if (si + 1) % 20 == 0 || si == 0 {
            info!("  [{}/{}] {} — {} sym trades (total: {}, {:.1}s)",
                  si + 1, symbols.len(), symbol, sym_trades,
                  all_trades.len(), total_start.elapsed().as_secs_f64());
        }
    }

    let phase2_secs = phase2_start.elapsed().as_secs_f64();
    info!("  Processing done in {:.1}s", phase2_secs);

    // ═══════════════════════════════════════════════════════════════════
    // PHASE 3: RESULTS
    // ═══════════════════════════════════════════════════════════════════
    let elapsed = total_start.elapsed();

    info!("");
    info!("╔════════════════════════════════════════════════════════════════╗");
    info!("║  IMBALANCE CANDLE STRATEGY — BACKTEST RESULTS v4              ║");
    info!("╚════════════════════════════════════════════════════════════════╝");
    info!("  Symbols processed: {} (skipped: {})", symbols_processed, symbols_skipped);
    info!("  Total trades: {}", all_trades.len());
    info!("  Data load: {:.1}s, Processing: {:.1}s, Total: {:.1}s ({:.1}min)",
          phase1_secs, phase2_secs, elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    info!("");

    if all_trades.is_empty() {
        info!("  ⚠️  No trades generated. Check thresholds and data availability.");
        return Ok(());
    }

    // ─── Per TF Pair Results ───
    for tf_pair in active_pairs {
        let tf_trades: Vec<&SimTrade> = all_trades.iter()
            .filter(|t| t.parent_tf == tf_pair.parent_tf && t.child_tf == tf_pair.child_tf)
            .collect();

        if tf_trades.is_empty() {
            info!("  ═══ {}m → {}m (≥{:.0}%) — no trades ═══",
                  tf_pair.parent_tf, tf_pair.child_tf, tf_pair.min_move_pct);
            info!("");
            continue;
        }

        info!("  ═══════════════════════════════════════════════════");
        info!("  ═══ {}m → {}m (≥{:.0}%) — {} trades ═══",
              tf_pair.parent_tf, tf_pair.child_tf, tf_pair.min_move_pct, tf_trades.len());
        info!("  ═══════════════════════════════════════════════════");

        print_stats("ALL", &tf_trades);
        info!("");

        // By signal type
        let cont: Vec<_> = tf_trades.iter()
            .filter(|t| t.signal_type == imbalance_strategy::imbalance::SignalType::Continuation)
            .copied().collect();
        let rev: Vec<_> = tf_trades.iter()
            .filter(|t| t.signal_type == imbalance_strategy::imbalance::SignalType::Reversal)
            .copied().collect();
        if !cont.is_empty() { print_stats("CONTINUATION", &cont); }
        if !rev.is_empty() { print_stats("REVERSAL", &rev); }
        info!("");

        // By trade direction
        let longs: Vec<_> = tf_trades.iter()
            .filter(|t| t.trade_direction == imbalance_strategy::imbalance::TradeDirection::Long)
            .copied().collect();
        let shorts: Vec<_> = tf_trades.iter()
            .filter(|t| t.trade_direction == imbalance_strategy::imbalance::TradeDirection::Short)
            .copied().collect();
        print_stats("LONG", &longs);
        if !shorts.is_empty() {
            print_stats("SHORT", &shorts);
            info!("");
            info!("    ─── SHORT: By Score Bucket ───");
            print_score_buckets(&shorts);
            info!("    ─── SHORT: By Parent Move Size ───");
            print_move_size_buckets(&shorts);
        }
        info!("");

        // Score buckets
        print_score_buckets(&tf_trades);
        info!("");

        // Move size buckets
        print_move_size_buckets(&tf_trades);
        info!("");

        // Hold time
        info!("    ─── By Hold Time ───");
        for hold in 1..=config.max_hold_candles {
            let h: Vec<_> = tf_trades.iter().filter(|t| t.hold_candles == hold).collect();
            if h.is_empty() { continue; }
            let ht = h.len();
            let hwin = h.iter().filter(|t| t.pnl_pct > 0.0).count();
            let hwr = hwin as f64 / ht as f64 * 100.0;
            let havg: f64 = h.iter().map(|t| t.pnl_pct).sum::<f64>() / ht as f64;
            info!("      hold={}: {} trades, RealWR: {:.1}%, AvgPnL: {:.2}%",
                  hold, ht, hwr, havg);
        }

        // Sample trades
        let samples: Vec<_> = tf_trades.iter().take(5).collect();
        if !samples.is_empty() {
            info!("");
            info!("    ─── Sample Trades (first 5) ───");
            for t in &samples {
                info!("      {} {} {}→{} {} {} score={:.2} move={:.1}% entry={:.4} exit={:.4} hold={} PnL={:.2}% {}",
                      t.symbol, t.imbalance_time.format("%Y-%m-%d %H:%M"),
                      t.parent_tf, t.child_tf,
                      t.signal_type, t.trade_direction,
                      t.score, t.parent_move_pct,
                      t.entry_price, t.exit_price,
                      t.hold_candles, t.pnl_pct, t.outcome);
            }
        }

        info!("");
    }

    // ─── Overall Summary ───
    info!("  ═══════════════════════════════════════════════════");
    info!("  ═══ OVERALL SUMMARY ═══");
    info!("  ═══════════════════════════════════════════════════");

    let all_refs: Vec<&SimTrade> = all_trades.iter().collect();
    print_stats("ALL TRADES", &all_refs);
    info!("");

    // By signal type
    let all_cont: Vec<_> = all_refs.iter()
        .filter(|t| t.signal_type == imbalance_strategy::imbalance::SignalType::Continuation)
        .copied().collect();
    let all_rev: Vec<_> = all_refs.iter()
        .filter(|t| t.signal_type == imbalance_strategy::imbalance::SignalType::Reversal)
        .copied().collect();
    if !all_cont.is_empty() { print_stats("ALL CONTINUATION", &all_cont); }
    if !all_rev.is_empty() { print_stats("ALL REVERSAL", &all_rev); }
    info!("");

    // By trade direction (LONG vs SHORT) — overall
    let all_longs: Vec<_> = all_refs.iter()
        .filter(|t| t.trade_direction == imbalance_strategy::imbalance::TradeDirection::Long)
        .copied().collect();
    let all_shorts: Vec<_> = all_refs.iter()
        .filter(|t| t.trade_direction == imbalance_strategy::imbalance::TradeDirection::Short)
        .copied().collect();
    print_stats("ALL LONG", &all_longs);
    if !all_shorts.is_empty() {
        print_stats("ALL SHORT", &all_shorts);
        info!("");
        info!("    ─── ALL SHORT: By Score Bucket ───");
        print_score_buckets(&all_shorts);
        info!("    ─── ALL SHORT: By Parent Move Size ───");
        print_move_size_buckets(&all_shorts);
        info!("    ─── ALL SHORT: By Hold Time ───");
        for hold in 1..=config.max_hold_candles {
            let h: Vec<_> = all_shorts.iter().filter(|t| t.hold_candles == hold).collect();
            if h.is_empty() { continue; }
            let ht = h.len();
            let hwin = h.iter().filter(|t| t.pnl_pct > 0.0).count();
            let hwr = hwin as f64 / ht as f64 * 100.0;
            let havg: f64 = h.iter().map(|t| t.pnl_pct).sum::<f64>() / ht as f64;
            info!("      hold={}: {} trades, RealWR: {:.1}%, AvgPnL: {:.2}%",
                  hold, ht, hwr, havg);
        }
    }
    info!("");

    // Score & move buckets
    print_score_buckets(&all_refs);
    info!("");
    print_move_size_buckets(&all_refs);
    info!("");

    // Top 10 symbols
    info!("    ─── Top 10 Symbols by Trade Count ───");
    let mut sym_stats: HashMap<&str, (usize, usize, f64)> = HashMap::new();
    for t in &all_trades {
        let e = sym_stats.entry(&t.symbol).or_default();
        e.0 += 1;
        if t.pnl_pct > 0.0 { e.1 += 1; }
        e.2 += t.pnl_pct;
    }
    let mut sym_list: Vec<_> = sym_stats.iter().collect();
    sym_list.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    for (sym, (count, wins, pnl)) in sym_list.iter().take(10) {
        info!("      {}: {} trades, RealWR: {:.1}%, TotalPnL: {:.2}%",
              sym, count, *wins as f64 / *count as f64 * 100.0, pnl);
    }

    // Top 10 best/worst
    info!("");
    info!("    ─── Top 10 Best Trades ───");
    let mut sorted = all_trades.clone();
    sorted.sort_by(|a, b| b.pnl_pct.partial_cmp(&a.pnl_pct).unwrap_or(std::cmp::Ordering::Equal));
    for (i, t) in sorted.iter().take(10).enumerate() {
        info!("      #{}: {} {} {}→{} {} {} score={:.2} PnL={:.2}% hold={} {}",
              i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
              t.parent_tf, t.child_tf,
              t.signal_type, t.trade_direction,
              t.score, t.pnl_pct, t.hold_candles, t.outcome);
    }

    info!("");
    info!("    ─── Top 10 Worst Trades ───");
    for (i, t) in sorted.iter().rev().take(10).enumerate() {
        info!("      #{}: {} {} {}→{} {} {} score={:.2} PnL={:.2}% hold={} {}",
              i + 1, t.symbol, t.entry_time.format("%Y-%m-%d %H:%M"),
              t.parent_tf, t.child_tf,
              t.signal_type, t.trade_direction,
              t.score, t.pnl_pct, t.hold_candles, t.outcome);
    }

    // ─── TRADING CONFIGURATION SUMMARY ───
    info!("  ═══════════════════════════════════════════════════");
    info!("  ═══ TRADING CONFIGURATION SUMMARY ═══");
    info!("  ═══════════════════════════════════════════════════");
    info!("");
    info!("    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐");
    info!("    │  SCORE TIER ANALYSIS — Which score thresholds to use for trading?                               │");
    info!("    ├──────────┬────────┬─────────┬──────────┬──────────┬──────────┬─────────────────────────────────────┤");
    info!("    │  Score    │ Trades │ RealWR  │ AvgPnL   │ TotalPnL │ PF       │ Recommendation                     │");
    info!("    ├──────────┼────────┼─────────┼──────────┼──────────┼──────────┼─────────────────────────────────────┤");

    let score_tiers: &[(f64, f64, &str)] = &[
        (0.75, 0.80, "0.75-0.80"),
        (0.80, 0.85, "0.80-0.85"),
        (0.85, 0.90, "0.85-0.90"),
        (0.90, 0.95, "0.90-0.95"),
        (0.95, 1.01, "0.95+   "),
        (0.75, 1.01, "ALL≥0.75"),
        (0.80, 1.01, "ALL≥0.80"),
        (0.85, 1.01, "ALL≥0.85"),
    ];

    for &(lo, hi, label) in score_tiers {
        let b: Vec<_> = all_refs.iter()
            .filter(|t| t.score >= lo && t.score < hi)
            .collect();
        if b.is_empty() { continue; }
        let bt = b.len();
        let bwin = b.iter().filter(|t| t.pnl_pct > 0.0).count();
        let bwr = bwin as f64 / bt as f64 * 100.0;
        let bavg: f64 = b.iter().map(|t| t.pnl_pct).sum::<f64>() / bt as f64;
        let btot: f64 = b.iter().map(|t| t.pnl_pct).sum();
        let gp: f64 = b.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
        let gl: f64 = b.iter().filter(|t| t.pnl_pct < 0.0).map(|t| t.pnl_pct.abs()).sum();
        let bpf = if gl > 0.0 { gp / gl } else { f64::INFINITY };

        let rec = if bpf >= 1.4 && bwr >= 55.0 && bt >= 30 {
            "🟢 STRONG — use for live"
        } else if bpf >= 1.2 && bwr >= 50.0 && bt >= 20 {
            "🟡 GOOD — acceptable for live"
        } else if bpf >= 1.0 && bavg > 0.0 {
            "🟠 MARGINAL — paper trade first"
        } else {
            "🔴 SKIP — negative expectancy"
        };

        info!("    │  {}│ {:>6} │ {:>6.1}% │ {:>7.2}% │ {:>7.1}% │ {:>7.2} │ {} │",
              label, bt, bwr, bavg, btot, bpf, rec);
    }
    info!("    └──────────┴────────┴─────────┴──────────┴──────────┴──────────┴─────────────────────────────────────┘");
    info!("");

    // ─── Cross-analysis: Score × TF pair ───
    info!("    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐");
    info!("    │  SCORE × TF PAIR — Best combinations                                                           │");
    info!("    ├──────────────────────────┬────────┬─────────┬──────────┬──────────┬──────────────────────────────┤");
    info!("    │  Combination             │ Trades │ RealWR  │ AvgPnL   │ PF       │ Recommendation              │");
    info!("    ├──────────────────────────┼────────┼─────────┼──────────┼──────────┼──────────────────────────────┤");

    for tf_pair in active_pairs {
        let tf_label = format!("{}→{}", tf_pair.parent_tf, tf_pair.child_tf);
        for &(lo, hi, score_label) in &[(0.75_f64, 0.85, "0.75-0.85"), (0.85, 0.95, "0.85-0.95"), (0.95, 1.01, "0.95+")] {
            let b: Vec<_> = all_refs.iter()
                .filter(|t| t.parent_tf == tf_pair.parent_tf
                    && t.child_tf == tf_pair.child_tf
                    && t.score >= lo && t.score < hi)
                .collect();
            if b.is_empty() { continue; }
            let bt = b.len();
            let bwin = b.iter().filter(|t| t.pnl_pct > 0.0).count();
            let bwr = bwin as f64 / bt as f64 * 100.0;
            let bavg: f64 = b.iter().map(|t| t.pnl_pct).sum::<f64>() / bt as f64;
            let gp: f64 = b.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
            let gl: f64 = b.iter().filter(|t| t.pnl_pct < 0.0).map(|t| t.pnl_pct.abs()).sum();
            let bpf = if gl > 0.0 { gp / gl } else { f64::INFINITY };

            let rec = if bpf >= 1.4 && bwr >= 55.0 && bt >= 15 {
                "🟢 STRONG"
            } else if bpf >= 1.15 && bwr >= 50.0 {
                "🟡 OK"
            } else {
                "🔴 WEAK"
            };

            let combo = format!("{} score {}", tf_label, score_label);
            info!("    │  {:24}│ {:>6} │ {:>6.1}% │ {:>7.2}% │ {:>7.2} │ {:28}│",
                  combo, bt, bwr, bavg, bpf, rec);
        }
    }
    info!("    └──────────────────────────┴────────┴─────────┴──────────┴──────────┴──────────────────────────────┘");
    info!("");

    // ─── Cross-analysis: Score × Move Size ───
    info!("    ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐");
    info!("    │  SCORE × MOVE SIZE — Optimal entry conditions                                                   │");
    info!("    ├──────────────────────────┬────────┬─────────┬──────────┬──────────┬──────────────────────────────┤");
    info!("    │  Combination             │ Trades │ RealWR  │ AvgPnL   │ PF       │ Status                      │");
    info!("    ├──────────────────────────┼────────┼─────────┼──────────┼──────────┼──────────────────────────────┤");

    let move_tiers: &[(f64, f64, &str)] = &[
        (18.0, 20.0, "18-20%"),
        (20.0, 25.0, "20-25%"),
        (25.0, 30.0, "25-30%"),
        (30.0, 50.0, "30-50%"),
        (50.0, 200.0, "50%+  "),
    ];
    let score_groups: &[(f64, f64, &str)] = &[
        (0.75, 0.85, "S.75-.85"),
        (0.85, 1.01, "S.85+  "),
    ];

    for &(mlo, mhi, mlabel) in move_tiers {
        for &(slo, shi, slabel) in score_groups {
            let b: Vec<_> = all_refs.iter()
                .filter(|t| t.parent_move_pct >= mlo && t.parent_move_pct < mhi
                    && t.score >= slo && t.score < shi)
                .collect();
            if b.is_empty() { continue; }
            let bt = b.len();
            let bwin = b.iter().filter(|t| t.pnl_pct > 0.0).count();
            let bwr = bwin as f64 / bt as f64 * 100.0;
            let bavg: f64 = b.iter().map(|t| t.pnl_pct).sum::<f64>() / bt as f64;
            let gp: f64 = b.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
            let gl: f64 = b.iter().filter(|t| t.pnl_pct < 0.0).map(|t| t.pnl_pct.abs()).sum();
            let bpf = if gl > 0.0 { gp / gl } else { f64::INFINITY };

            let status = if bpf >= 1.4 && bwr >= 55.0 && bt >= 10 {
                "🟢 SWEET SPOT"
            } else if bpf >= 1.15 && bavg > 0.0 {
                "🟡 ACCEPTABLE"
            } else {
                "🔴 AVOID"
            };

            let combo = format!("Move {} {}", mlabel, slabel);
            info!("    │  {:24}│ {:>6} │ {:>6.1}% │ {:>7.2}% │ {:>7.2} │ {:28}│",
                  combo, bt, bwr, bavg, bpf, status);
        }
    }
    info!("    └──────────────────────────┴────────┴─────────┴──────────┴──────────┴──────────────────────────────┘");
    info!("");

    // ─── Trading Recommendations ───
    info!("  ═══════════════════════════════════════════════════");
    info!("  ═══ TRADING RECOMMENDATIONS ═══");
    info!("  ═══════════════════════════════════════════════════");

    // Find best score threshold
    let mut best_threshold = 0.75f64;
    let mut best_pf = 0.0f64;
    for threshold in &[0.75, 0.80, 0.85, 0.90, 0.95] {
        let b: Vec<_> = all_refs.iter().filter(|t| t.score >= *threshold).collect();
        if b.len() < 20 { continue; }
        let gp: f64 = b.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
        let gl: f64 = b.iter().filter(|t| t.pnl_pct < 0.0).map(|t| t.pnl_pct.abs()).sum();
        let pf = if gl > 0.0 { gp / gl } else { 0.0 };
        if pf > best_pf {
            best_pf = pf;
            best_threshold = *threshold;
        }
    }

    let best_trades: Vec<_> = all_refs.iter().filter(|t| t.score >= best_threshold).collect();
    let best_n = best_trades.len();
    let best_wr = best_trades.iter().filter(|t| t.pnl_pct > 0.0).count() as f64 / best_n as f64 * 100.0;
    let best_avg: f64 = best_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / best_n as f64;

    info!("");
    info!("    💡 OPTIMAL SCORE THRESHOLD: ≥{:.2}", best_threshold);
    info!("       → {} trades, RealWR {:.1}%, AvgPnL {:.2}%, PF {:.2}", best_n, best_wr, best_avg, best_pf);
    info!("");

    // Position sizing recommendation based on Kelly Criterion
    let kelly_f = if best_pf > 1.0 {
        let avg_win: f64 = {
            let wins: Vec<_> = best_trades.iter().filter(|t| t.pnl_pct > 0.0).collect();
            if wins.is_empty() { 0.0 } else { wins.iter().map(|t| t.pnl_pct).sum::<f64>() / wins.len() as f64 }
        };
        let avg_loss: f64 = {
            let losses: Vec<_> = best_trades.iter().filter(|t| t.pnl_pct < 0.0).collect();
            if losses.is_empty() { 1.0 } else { losses.iter().map(|t| t.pnl_pct.abs()).sum::<f64>() / losses.len() as f64 }
        };
        let win_prob = best_wr / 100.0;
        let b = avg_win / avg_loss;
        let kelly = win_prob - (1.0 - win_prob) / b;
        kelly.max(0.0)
    } else {
        0.0
    };

    info!("    📐 POSITION SIZING (Kelly Criterion):");
    info!("       Full Kelly: {:.1}% of capital per trade", kelly_f * 100.0);
    info!("       Half Kelly (recommended): {:.1}% of capital per trade", kelly_f * 50.0);
    info!("       Quarter Kelly (conservative): {:.1}% of capital per trade", kelly_f * 25.0);
    info!("");

    // Expected monthly trades
    let first_time = all_trades.iter().map(|t| t.entry_time).min();
    let last_time = all_trades.iter().map(|t| t.entry_time).max();
    if let (Some(first), Some(last)) = (first_time, last_time) {
        let months = (last - first).num_days() as f64 / 30.0;
        if months > 1.0 {
            let trades_per_month = all_trades.len() as f64 / months;
            let filtered_per_month = best_n as f64 / months;
            info!("    📅 EXPECTED FREQUENCY:");
            info!("       All signals: ~{:.0} trades/month", trades_per_month);
            info!("       Score ≥{:.2}: ~{:.0} trades/month", best_threshold, filtered_per_month);
            info!("       Data span: {:.0} months ({} — {})",
                  months, first.format("%Y-%m"), last.format("%Y-%m"));
            info!("");
        }
    }

    // Verdict
    info!("");
    let n = all_trades.len();
    let winning = all_trades.iter().filter(|t| t.pnl_pct > 0.0).count();
    let real_wr = winning as f64 / n as f64 * 100.0;
    let avg_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / n as f64;
    let total_pnl: f64 = all_trades.iter().map(|t| t.pnl_pct).sum();

    let gross_profit: f64 = all_trades.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
    let gross_loss: f64 = all_trades.iter().filter(|t| t.pnl_pct < 0.0).map(|t| t.pnl_pct.abs()).sum();
    let pf = if gross_loss > 0.0 { gross_profit / gross_loss } else { f64::INFINITY };

    if real_wr > 50.0 && avg_pnl > 0.3 && pf > 1.15 {
        info!("  🟢 VERDICT: STRONG EDGE (RealWR={:.1}%, AvgPnL={:.2}%, PF={:.2}, TotalPnL={:.2}%)",
              real_wr, avg_pnl, pf, total_pnl);
    } else if avg_pnl > 0.05 && total_pnl > 0.0 && pf > 1.0 {
        info!("  🟡 VERDICT: POSITIVE edge (RealWR={:.1}%, AvgPnL={:.2}%, PF={:.2}, TotalPnL={:.2}%)",
              real_wr, avg_pnl, pf, total_pnl);
    } else {
        info!("  🔴 VERDICT: NO edge (RealWR={:.1}%, AvgPnL={:.2}%, PF={:.2}, TotalPnL={:.2}%)",
              real_wr, avg_pnl, pf, total_pnl);
    }

    info!("");
    info!("  Total time: {:.1}s ({:.1}min)", elapsed.as_secs_f64(), elapsed.as_secs_f64() / 60.0);
    info!("  Config: {}", config_path);
    info!("  Log: {}", log_path);

    Ok(())
}
