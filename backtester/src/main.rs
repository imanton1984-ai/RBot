// backtester/src/main.rs
//
// Signal Quality Backtester
//
// Evaluates historical trade signals against real candle data to determine
// if they would have been profitable. Results are used to:
//   1. Train the Signal Quality XGBoost model
//   2. Measure win-rate, avg PnL, Sharpe ratio per TF/symbol
//   3. Calibrate min_final_score thresholds
//
// WORKFLOW:
//   1. Read signals from trade.final_signals (WHERE reason IS NOT NULL)
//   2. For each signal, fetch future candles from market.candles_XX
//   3. Walk forward: check if TP1/TP2/TP3 or SL is hit first
//   4. Record outcome in trade.backtest_results
//   5. Export features + labels to CSV for XGBoost training
//   6. Print summary statistics (win-rate, avg PnL, by TF/symbol)

mod evaluator;
mod types;

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::PgPool;
use std::collections::HashMap;

use evaluator::SignalEvaluator;
use types::*;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let pool = PgPool::connect(&db_url).await?;

    // Config from env vars
    let min_score: f64 = std::env::var("BACKTEST_MIN_SCORE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0.55);

    let max_signals: i64 = std::env::var("BACKTEST_MAX_SIGNALS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(500_000);

    let timeout_bars: usize = std::env::var("BACKTEST_TIMEOUT_BARS")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(100);

    let csv_output = std::env::var("BACKTEST_CSV_OUTPUT")
        .unwrap_or_else(|_| "backtest_results.csv".to_string());

    tracing::info!("=== Signal Quality Backtester ===");
    tracing::info!("min_score={}, max_signals={}, timeout_bars={}", min_score, max_signals, timeout_bars);

    // Ensure results table exists
    create_backtest_table(&pool).await?;

    // Load signals
    let signals = load_signals(&pool, min_score, max_signals).await?;
    tracing::info!("Loaded {} signals for backtesting", signals.len());

    if signals.is_empty() {
        tracing::warn!("No signals found. Ensure trade.final_signals has data with reason IS NOT NULL.");
        return Ok(());
    }

    // Evaluate signals
    let evaluator = SignalEvaluator::new(pool.clone(), timeout_bars);
    let mut results: Vec<BacktestResult> = Vec::with_capacity(signals.len());
    let mut processed = 0u64;
    let mut win_count = 0u64;
    let mut loss_count = 0u64;
    let mut expired_count = 0u64;
    let mut total_pnl = 0.0f64;

    for signal in &signals {
        match evaluator.evaluate(signal).await {
            Ok(Some(result)) => {
                match result.outcome {
                    Outcome::Win { .. } => win_count += 1,
                    Outcome::Loss { .. } => loss_count += 1,
                    Outcome::Expired { .. } => expired_count += 1,
                }
                total_pnl += result.pnl_pct;
                results.push(result);
            }
            Ok(None) => {
                // Not enough future candle data to evaluate
            }
            Err(e) => {
                tracing::warn!("Error evaluating signal {} {}: {}", signal.symbol, signal.time, e);
            }
        }

        processed += 1;
        if processed % 1000 == 0 {
            tracing::info!(
                "Processed {}/{} signals: {} wins, {} losses, {} expired",
                processed, signals.len(), win_count, loss_count, expired_count
            );
        }
    }

    tracing::info!(
        "Evaluation complete: {} results from {} signals",
        results.len(), signals.len()
    );

    // Save results to DB
    let saved = save_results(&pool, &results).await?;
    tracing::info!("Saved {} backtest results to trade.backtest_results", saved);

    // Export CSV for XGBoost training — global + per-TF
    export_training_csv(&results, &csv_output)?;
    tracing::info!("Exported training CSV to {}", csv_output);

    // Export per-TF CSVs
    let mut by_tf: HashMap<i16, Vec<&BacktestResult>> = HashMap::new();
    for r in &results {
        by_tf.entry(r.tf_minutes).or_default().push(r);
    }
    for (tf, group) in &by_tf {
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", _=>"unknown" };
        let tf_csv = format!("backtest_results_{}.csv", tf_name);
        let owned: Vec<BacktestResult> = group.iter().map(|r| (*r).clone()).collect();
        export_training_csv(&owned, &tf_csv)?;
        tracing::info!("Exported {} {} results to {}", owned.len(), tf_name, tf_csv);
    }

    // Print summary statistics
    print_summary(&results);

    Ok(())
}

/// Load signals with reason JSON from the database
async fn load_signals(pool: &PgPool, min_score: f64, max_rows: i64) -> Result<Vec<SignalForBacktest>> {
    let rows = sqlx::query_as::<_, SignalForBacktest>(
        r#"
        SELECT
            s.symbol_id, COALESCE(s.symbol, p.symbol) as symbol,
            s.tf_minutes, s.side, s.final_score,
            s.ml_score, s.heur_score,
            s.entry_price, s.sl_price, s.tp1_price, s.tp2_price, s.tp3_price,
            s.reason,
            s.price10_target, s.price10_score,
            s.bounce_prob, s.bounce_score,
            s.breakout_prob, s.breakout_score,
            s.time, s.time_ms
        FROM trade.final_signals s
        LEFT JOIN market.pairs p ON p.symbol_id = s.symbol_id
        WHERE s.reason IS NOT NULL
          AND s.final_score >= $1
          AND s.entry_price IS NOT NULL
          AND s.sl_price IS NOT NULL
          AND s.tp1_price IS NOT NULL
          AND s.tf_minutes != 1440
        ORDER BY s.time ASC
        LIMIT $2
        "#,
    )
    .bind(min_score as f32)
    .bind(max_rows)
    .fetch_all(pool)
    .await?;

    Ok(rows)
}

/// Create backtest results table if it doesn't exist
async fn create_backtest_table(pool: &PgPool) -> Result<()> {
    // sqlx requires one statement per query() call
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS trade.backtest_results (
            id              BIGSERIAL,
            signal_time     TIMESTAMPTZ NOT NULL,
            signal_time_ms  BIGINT NOT NULL,
            symbol          TEXT NOT NULL,
            symbol_id       BIGINT NOT NULL,
            tf_minutes      SMALLINT NOT NULL,
            side            SMALLINT NOT NULL,
            final_score     REAL NOT NULL,
            entry_price     REAL NOT NULL,
            sl_price        REAL NOT NULL,
            tp1_price       REAL NOT NULL,
            tp2_price       REAL,
            tp3_price       REAL,
            outcome         TEXT NOT NULL,
            pnl_pct         DOUBLE PRECISION NOT NULL,
            exit_price      DOUBLE PRECISION,
            bars_to_outcome INT NOT NULL,
            max_favorable   DOUBLE PRECISION NOT NULL,
            max_adverse     DOUBLE PRECISION NOT NULL,
            tp1_hit         BOOLEAN NOT NULL DEFAULT false,
            tp2_hit         BOOLEAN NOT NULL DEFAULT false,
            tp3_hit         BOOLEAN NOT NULL DEFAULT false,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
            PRIMARY KEY (symbol_id, tf_minutes, signal_time)
        )
        "#,
    )
    .execute(pool)
    .await?;

    // Add tp1_hit/tp2_hit/tp3_hit columns if they don't exist (migration for existing tables)
    for col in &["tp1_hit", "tp2_hit", "tp3_hit"] {
        let sql = format!(
            "ALTER TABLE trade.backtest_results ADD COLUMN IF NOT EXISTS {} BOOLEAN NOT NULL DEFAULT false",
            col
        );
        let _ = sqlx::query(&sql).execute(pool).await;
    }

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_backtest_results_symbol_time ON trade.backtest_results(symbol, signal_time DESC)"
    ).execute(pool).await?;

    sqlx::query(
        "CREATE INDEX IF NOT EXISTS ix_backtest_results_outcome ON trade.backtest_results(outcome, final_score DESC)"
    ).execute(pool).await?;

    Ok(())
}

/// Save evaluation results to database
async fn save_results(pool: &PgPool, results: &[BacktestResult]) -> Result<usize> {
    let mut saved = 0;

    // Use batch INSERT for efficiency
    for chunk in results.chunks(500) {
        let mut time_v: Vec<chrono::DateTime<chrono::Utc>> = Vec::with_capacity(chunk.len());
        let mut time_ms_v: Vec<i64> = Vec::with_capacity(chunk.len());
        let mut symbol_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut symbol_id_v: Vec<i64> = Vec::with_capacity(chunk.len());
        let mut tf_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut side_v: Vec<i16> = Vec::with_capacity(chunk.len());
        let mut score_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut entry_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut sl_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut tp1_v: Vec<f32> = Vec::with_capacity(chunk.len());
        let mut tp2_v: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
        let mut tp3_v: Vec<Option<f32>> = Vec::with_capacity(chunk.len());
        let mut outcome_v: Vec<String> = Vec::with_capacity(chunk.len());
        let mut pnl_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut exit_v: Vec<Option<f64>> = Vec::with_capacity(chunk.len());
        let mut bars_v: Vec<i32> = Vec::with_capacity(chunk.len());
        let mut fav_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut adv_v: Vec<f64> = Vec::with_capacity(chunk.len());
        let mut tp1_hit_v: Vec<bool> = Vec::with_capacity(chunk.len());
        let mut tp2_hit_v: Vec<bool> = Vec::with_capacity(chunk.len());
        let mut tp3_hit_v: Vec<bool> = Vec::with_capacity(chunk.len());

        for r in chunk {
            time_v.push(r.signal_time);
            time_ms_v.push(r.signal_time_ms);
            symbol_v.push(r.symbol.clone());
            symbol_id_v.push(r.symbol_id);
            tf_v.push(r.tf_minutes);
            side_v.push(r.side);
            score_v.push(r.final_score);
            entry_v.push(r.entry_price);
            sl_v.push(r.sl_price);
            tp1_v.push(r.tp1_price);
            tp2_v.push(r.tp2_price);
            tp3_v.push(r.tp3_price);
            outcome_v.push(r.outcome.as_str().to_string());
            pnl_v.push(r.pnl_pct);
            exit_v.push(r.exit_price);
            bars_v.push(r.bars_to_outcome as i32);
            fav_v.push(r.max_favorable);
            adv_v.push(r.max_adverse);
            tp1_hit_v.push(r.tp1_hit);
            tp2_hit_v.push(r.tp2_hit);
            tp3_hit_v.push(r.tp3_hit);
        }

        sqlx::query(
            r#"
            INSERT INTO trade.backtest_results
            (signal_time, signal_time_ms, symbol, symbol_id, tf_minutes, side, final_score,
             entry_price, sl_price, tp1_price, tp2_price, tp3_price,
             outcome, pnl_pct, exit_price, bars_to_outcome, max_favorable, max_adverse,
             tp1_hit, tp2_hit, tp3_hit)
            SELECT * FROM UNNEST(
                $1::timestamptz[], $2::bigint[], $3::text[], $4::bigint[],
                $5::smallint[], $6::smallint[], $7::real[],
                $8::real[], $9::real[], $10::real[], $11::real[], $12::real[],
                $13::text[], $14::double precision[], $15::double precision[],
                $16::int[], $17::double precision[], $18::double precision[],
                $19::boolean[], $20::boolean[], $21::boolean[]
            )
            ON CONFLICT (symbol_id, tf_minutes, signal_time) DO UPDATE SET
                outcome = EXCLUDED.outcome,
                pnl_pct = EXCLUDED.pnl_pct,
                exit_price = EXCLUDED.exit_price,
                bars_to_outcome = EXCLUDED.bars_to_outcome,
                max_favorable = EXCLUDED.max_favorable,
                max_adverse = EXCLUDED.max_adverse,
                tp1_hit = EXCLUDED.tp1_hit,
                tp2_hit = EXCLUDED.tp2_hit,
                tp3_hit = EXCLUDED.tp3_hit
            "#,
        )
        .bind(&time_v).bind(&time_ms_v).bind(&symbol_v).bind(&symbol_id_v)
        .bind(&tf_v).bind(&side_v).bind(&score_v)
        .bind(&entry_v).bind(&sl_v).bind(&tp1_v).bind(&tp2_v).bind(&tp3_v)
        .bind(&outcome_v).bind(&pnl_v).bind(&exit_v)
        .bind(&bars_v).bind(&fav_v).bind(&adv_v)
        .bind(&tp1_hit_v).bind(&tp2_hit_v).bind(&tp3_hit_v)
        .execute(pool)
        .await?;

        saved += chunk.len();
    }

    Ok(saved)
}

/// Export results + signal features as CSV for XGBoost training
fn export_training_csv(results: &[BacktestResult], path: &str) -> Result<()> {
    use std::io::Write;

    let mut file = std::fs::File::create(path)?;

    // Header: signal features + derived features + outcome labels + partial close tracking
    writeln!(file,
        "symbol,tf_minutes,side,final_score,ml_score,heur_score,\
         entry_price,sl_pct,tp1_pct,tp2_pct,tp3_pct,risk_reward,risk_reward_tp2,\
         predictors_score,raw_signals_score,indicators_score,market_score,\
         coverage_score,consensus_score,\
         price10_score,bounce_prob,bounce_score,breakout_prob,breakout_score,\
         trend_strength,momentum_strength,volatility_regime,volume_spike_score,\
         level_aware,market_quality_score,\
         quality_multiplier,quality_grade,original_score,\
         pred_vs_raw,pred_vs_ind,component_std,component_min,ml_heur_gap,score_per_risk,\
         atr_pct,market_factor,score_factor,\
         tp1_hit,tp2_hit,tp3_hit,\
         label_win,label_tp_level,label_pnl_pct,label_max_favorable,label_max_adverse,label_bars"
    )?;

    for r in results {
        let reason = &r.reason_json;
        let get = |k: &str| reason.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let debug_get = |k: &str| {
            reason.get("debug").and_then(|d| d.get(k)).and_then(|v| v.as_f64())
                .or_else(|| reason.get(k).and_then(|v| v.as_f64()))
                .unwrap_or(0.0)
        };

        let ep = r.entry_price as f64;
        let sl_pct = if ep > 0.0 { (r.sl_price as f64 - ep).abs() / ep } else { 0.0 };
        let tp1_pct = if ep > 0.0 { (r.tp1_price as f64 - ep).abs() / ep } else { 0.0 };
        let tp2_pct = r.tp2_price.map(|t| if ep > 0.0 { (t as f64 - ep).abs() / ep } else { 0.0 }).unwrap_or(0.0);
        let tp3_pct = r.tp3_price.map(|t| if ep > 0.0 { (t as f64 - ep).abs() / ep } else { 0.0 }).unwrap_or(0.0);
        let rr = if sl_pct > 0.0 { tp1_pct / sl_pct } else { 0.0 };
        let rr2 = if sl_pct > 0.0 { tp2_pct / sl_pct } else { 0.0 };

        let level_aware = reason.get("level_aware").and_then(|v| v.as_bool()).unwrap_or(false);

        // Quality scorer info (from reason JSON)
        let quality_mult = get("quality_multiplier");
        let quality_grade = reason.get("quality_grade").and_then(|v| v.as_str()).unwrap_or("?");
        let original_score = get("original_score");

        // Derived features for better ML training
        let pred = debug_get("predictors_score");
        let raw = debug_get("raw_signals_score");
        let ind = debug_get("indicators_score");
        let mkt = debug_get("market_score");
        let pred_vs_raw = if raw > 0.0001 { pred / raw } else { 0.0 };
        let pred_vs_ind = if ind > 0.0001 { pred / ind } else { 0.0 };
        let scores = [pred, raw, ind, mkt];
        let mean_s = scores.iter().sum::<f64>() / 4.0;
        let var_s = scores.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / 4.0;
        let component_std = var_s.sqrt();
        let component_min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let ml_s = r.ml_score.unwrap_or(0.0) as f64;
        let heur_s = r.heur_score.unwrap_or(0.0) as f64;
        let ml_heur_gap = (ml_s - heur_s).abs();
        let score_per_risk = if sl_pct > 0.0 { r.final_score as f64 / sl_pct } else { 0.0 };

        let label_win = if matches!(r.outcome, Outcome::Win { .. }) { 1.0 } else { 0.0 };
        let label_tp = match &r.outcome {
            Outcome::Win { tp_level, .. } => *tp_level as f64,
            Outcome::Loss { .. } => -1.0,
            Outcome::Expired { .. } => 0.0,
        };

        writeln!(file,
            "{},{},{},{:.4},{:.4},{:.4},\
             {:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.4},\
             {:.4},{:.4},{:.4},{:.4},\
             {:.4},{:.4},\
             {:.4},{:.4},{:.4},{:.4},{:.4},\
             {:.4},{:.4},{:.4},{:.4},\
             {},{:.4},\
             {:.4},{},{:.4},\
             {:.4},{:.4},{:.6},{:.4},{:.4},{:.2},\
             {:.6},{:.4},{:.4},\
             {},{},{},\
             {:.1},{:.1},{:.6},{:.6},{:.6},{}",
            r.symbol, r.tf_minutes, r.side, r.final_score,
            ml_s, heur_s,
            ep, sl_pct, tp1_pct, tp2_pct, tp3_pct, rr, rr2,
            pred, raw, ind, mkt,
            debug_get("coverage_score"), debug_get("consensus_score"),
            r.price10_score.unwrap_or(0.0),
            r.bounce_prob.unwrap_or(0.0), r.bounce_score.unwrap_or(0.0),
            r.breakout_prob.unwrap_or(0.0), r.breakout_score.unwrap_or(0.0),
            debug_get("trend_strength"), debug_get("momentum_strength"),
            debug_get("volatility_regime"), debug_get("volume_spike_score"),
            if level_aware { 1 } else { 0 },
            get("market_quality_score"),
            quality_mult, quality_grade, original_score,
            pred_vs_raw, pred_vs_ind, component_std, component_min, ml_heur_gap, score_per_risk,
            get("atr_pct"), get("market_factor"), get("score_factor"),
            if r.tp1_hit { 1 } else { 0 },
            if r.tp2_hit { 1 } else { 0 },
            if r.tp3_hit { 1 } else { 0 },
            label_win, label_tp, r.pnl_pct, r.max_favorable, r.max_adverse, r.bars_to_outcome
        )?;
    }

    Ok(())
}

/// Print summary statistics grouped by TF
fn print_summary(results: &[BacktestResult]) {
    if results.is_empty() {
        println!("\nNo results to summarize.");
        return;
    }

    println!("\n======== BACKTEST SUMMARY ========");
    println!("{:<6} {:>6} {:>6} {:>6} {:>8} {:>8} {:>8} {:>10}",
        "TF", "Total", "Wins", "Loss", "Exprd", "WinRate", "AvgPnL", "Sharpe");

    // Group by tf_minutes
    let mut by_tf: HashMap<i16, Vec<&BacktestResult>> = HashMap::new();
    for r in results {
        by_tf.entry(r.tf_minutes).or_default().push(r);
    }

    let mut tfs: Vec<i16> = by_tf.keys().copied().collect();
    tfs.sort();

    for tf in tfs {
        let group = &by_tf[&tf];
        let total = group.len();
        let wins = group.iter().filter(|r| matches!(r.outcome, Outcome::Win { .. })).count();
        let losses = group.iter().filter(|r| matches!(r.outcome, Outcome::Loss { .. })).count();
        let expired = group.iter().filter(|r| matches!(r.outcome, Outcome::Expired { .. })).count();
        let avg_pnl: f64 = group.iter().map(|r| r.pnl_pct).sum::<f64>() / total as f64;
        let win_rate = if total > 0 { wins as f64 / total as f64 * 100.0 } else { 0.0 };

        // Simple Sharpe approximation
        let pnls: Vec<f64> = group.iter().map(|r| r.pnl_pct).collect();
        let mean = avg_pnl;
        let variance = pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / total as f64;
        let sharpe = if variance > 0.0 { mean / variance.sqrt() } else { 0.0 };

        let tf_name = match tf {
            1 => "1m", 5 => "5m", 15 => "15m", 60 => "1h", 240 => "4h",
            _ => "??",
        };

        println!("{:<6} {:>6} {:>6} {:>6} {:>8} {:>7.1}% {:>7.4}% {:>10.3}",
            tf_name, total, wins, losses, expired, win_rate, avg_pnl * 100.0, sharpe);
    }

    // Overall
    let total = results.len();
    let wins = results.iter().filter(|r| matches!(r.outcome, Outcome::Win { .. })).count();
    let avg_pnl: f64 = results.iter().map(|r| r.pnl_pct).sum::<f64>() / total as f64;
    println!("-------------------------------");
    println!("TOTAL: {} signals, {} wins ({:.1}%), avg PnL: {:.4}%",
        total, wins, wins as f64 / total as f64 * 100.0, avg_pnl * 100.0);
    println!("==================================\n");

    // TP Level distribution (partial close tracking)
    println!("======== TP LEVEL DISTRIBUTION (Partial Close) ========");
    println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>10} {:>12}",
        "TF", "TP1_hit", "TP2_hit", "TP3_hit", "SL_only", "AvgPnL+", "AvgPnL-");

    let mut tfs_tp: Vec<i16> = by_tf.keys().copied().collect();
    tfs_tp.sort();
    for tf in &tfs_tp {
        let group = &by_tf[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", _=>"??" };

        let tp1_cnt = group.iter().filter(|r| r.tp1_hit).count();
        let tp2_cnt = group.iter().filter(|r| r.tp2_hit).count();
        let tp3_cnt = group.iter().filter(|r| r.tp3_hit).count();
        let sl_only = group.iter().filter(|r| matches!(r.outcome, Outcome::Loss { .. })).count();

        // Average PnL for wins vs losses
        let win_pnls: Vec<f64> = group.iter()
            .filter(|r| matches!(r.outcome, Outcome::Win { .. }))
            .map(|r| r.pnl_pct).collect();
        let loss_pnls: Vec<f64> = group.iter()
            .filter(|r| matches!(r.outcome, Outcome::Loss { .. }))
            .map(|r| r.pnl_pct).collect();
        let avg_win_pnl = if !win_pnls.is_empty() {
            win_pnls.iter().sum::<f64>() / win_pnls.len() as f64
        } else { 0.0 };
        let avg_loss_pnl = if !loss_pnls.is_empty() {
            loss_pnls.iter().sum::<f64>() / loss_pnls.len() as f64
        } else { 0.0 };

        println!("{:<6} {:>8} {:>8} {:>8} {:>10} {:>9.4}% {:>10.4}%",
            tf_name, tp1_cnt, tp2_cnt, tp3_cnt, sl_only,
            avg_win_pnl * 100.0, avg_loss_pnl * 100.0);
    }
    println!("=====================================================\n");

    // Score breakdown table: TF × score bucket
    println!("======== WIN RATE BY SCORE BUCKET ========");
    println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "TF", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70-0.75", "0.75-0.80", "0.80+");

    let buckets: Vec<(f32, f32, &str)> = vec![
        (0.55, 0.60, "0.55-0.60"),
        (0.60, 0.65, "0.60-0.65"),
        (0.65, 0.70, "0.65-0.70"),
        (0.70, 0.75, "0.70-0.75"),
        (0.75, 0.80, "0.75-0.80"),
        (0.80, 1.01, "0.80+"),
    ];

    let tfs2: Vec<i16> = by_tf.keys().copied().collect::<Vec<_>>().into_iter().collect();
    let mut tfs_sorted = tfs2; tfs_sorted.sort();
    for tf in &tfs_sorted {
        let group = &by_tf[tf];
        let tf_name = match tf { 1=>"1m", 5=>"5m", 15=>"15m", 60=>"1h", 240=>"4h", _=>"??" };

        let mut cells: Vec<String> = Vec::new();
        for (lo, hi, _) in &buckets {
            let in_bucket: Vec<&&BacktestResult> = group.iter()
                .filter(|r| r.final_score >= *lo && r.final_score < *hi)
                .collect();
            let n = in_bucket.len();
            if n == 0 {
                cells.push("  -  ".to_string());
            } else {
                let w = in_bucket.iter().filter(|r| matches!(r.outcome, Outcome::Win { .. })).count();
                cells.push(format!("{:.0}% ({})", w as f64 / n as f64 * 100.0, n));
            }
        }

        println!("{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            tf_name, cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]);
    }
    println!("==========================================\n");
}
