// backtester/src/main.rs
//
// Signal Quality Backtester
//
// Evaluates historical trade signals against real candle data to determine
// if they would have been profitable. Results are used to:
//   1. Train the Signal Quality XGBoost model
//   2. Measure win-rate, avg PnL, Sharpe ratio per TF/symbol
//   3. Calibrate min_final_score thresholds
//   4. Generate Entry Policy training dataset (for Entry Agent)
//   5. Compare Baseline vs Entry Agent evaluation
//
// WORKFLOW:
//   1. Read signals from trade.final_signals (WHERE reason IS NOT NULL)
//   2. For each signal, fetch future candles from market.candles_XX
//   3. Walk forward: check if TP1/TP2/TP3 or SL is hit first
//   4. Record outcome in trade.backtest_results
//   5. Export features + labels to CSV for XGBoost training
//   6. Export Entry Policy dataset (features + elapsed + remaining + labels)
//   7. Compare Baseline vs Entry Agent performance
//   8. Print summary statistics (win-rate, avg PnL, by TF/symbol)

mod evaluator;
mod types;
mod entry_policy_labeler;
mod entry_agent_evaluator;
mod entry_agent_real_evaluator;

use anyhow::Result;
use dotenvy::dotenv;
use sqlx::{PgPool, Row};
use std::collections::HashMap;

use evaluator::SignalEvaluator;
use types::*;
use entry_policy_labeler::*;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt::init();

    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://postgres:postgres@localhost:5433/timescaledb_binance".to_string());

    let pool = PgPool::connect(&db_url).await?;

    // Config from env vars
    // V9: Default min_score raised from 0.55 → 0.60 to filter weak signals.
    // Score bucket analysis showed 0.55-0.60 has marginal WR (~61% on 5m),
    // while 0.60+ has significantly better WR (~64-71%).
    let min_score: f64 = std::env::var("BACKTEST_MIN_SCORE")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0.60);

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

    for signal in &signals {
        match evaluator.evaluate(signal).await {
            Ok(Some(result)) => {
                match result.outcome {
                    Outcome::Win { .. } => win_count += 1,
                    Outcome::Loss { .. } => loss_count += 1,
                    Outcome::Expired { .. } => expired_count += 1,
                }
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

    // Export Entry Policy dataset (for training Entry Agent)
    let entry_csv = std::env::var("ENTRY_POLICY_CSV_OUTPUT")
        .unwrap_or_else(|_| "entry_policy_dataset.csv".to_string());
    export_entry_policy_dataset(&pool, &results, &entry_csv).await?;
    tracing::info!("Exported Entry Policy dataset to {}", entry_csv);

    // Run Entry Agent comparison (enabled by default, disable with BACKTEST_COMPARE_ENTRY_AGENT=false)
    let run_agent = std::env::var("BACKTEST_COMPARE_ENTRY_AGENT")
        .map(|v| v != "false" && v != "0")
        .unwrap_or(true);
    if run_agent {
        tracing::info!("Running Entry Agent comparison evaluation...");
        run_entry_agent_comparison(&pool, &results, timeout_bars).await?;
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

/// Export Entry Policy training dataset
///
/// This generates a CSV with features + labels for training the Entry Agent models.
/// For each backtest result (setup), we:
///   1. Fetch future candles with ATR from market.indicators_wide
///   2. Run expert labeling to find optimal entry bar
///   3. Generate examples: WAIT for bars before optimal, ENTER at optimal, or CANCEL if none profitable
///   4. Export features + elapsed + remaining + label_enter + label_cancel + weight
async fn export_entry_policy_dataset(
    pool: &PgPool,
    results: &[BacktestResult],
    output_path: &str,
) -> Result<()> {
    use entry_policy_labeler::*;
    use std::io::Write;

    tracing::info!("Generating Entry Policy dataset...");

    let mut file = std::fs::File::create(output_path)?;

    // Write header
    // Features will be expanded from reason_json + standard signal features
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
         impulse_phase,momentum_acceleration,rsi_slope,volume_impulse_confirm,\
         nearest_support_dist_atr,nearest_resistance_dist_atr,sr_position,\
         ema_stack,price_vs_emas,bb_position,bb_width,\
         elapsed,remaining,label_enter,label_cancel,weight"
    )?;

    let mut total_examples = 0;
    let mut enter_examples = 0;
    let mut cancel_examples = 0;

    // Configuration for simulation (must match your trading logic!)
    let sim_cfg = SimCfg {
        window_bars: 10,       // Default, will be overridden per TF
        max_hold_bars: 12,     // Match backtester timeout
        sl_atr_mult: 1.0,
        rr1: 1.0,
        rr2: 1.5,
        rr3: 2.0,
        tp1_close_pct: 0.50,
        tp2_close_pct: 0.30,
        tp3_close_pct: 0.20,
    };

    // Fetch ATR series for each result and generate examples
    for result in results {
        // Get TF-specific config
        let window_bars = get_window_bars_for_tf(result.tf_minutes);
        let max_hold_bars = get_max_hold_bars_for_tf(result.tf_minutes);

        let cfg = SimCfg {
            window_bars,
            max_hold_bars,
            ..sim_cfg
        };

        // Fetch future candles with ATR for this symbol/TF
        let candles = fetch_candles_with_atr(
            pool,
            result.symbol_id,
            result.tf_minutes,
            result.signal_time,
            window_bars + max_hold_bars + 5, // Buffer
        ).await?;

        if candles.len() < window_bars {
            // Not enough data
            continue;
        }

        // Convert to OhlcBar
        let ohlc_bars: Vec<OhlcBar> = candles.into_iter().map(|c| OhlcBar {
            high: c.high,
            low: c.low,
            close: c.close,
            atr: (c.atr as f64).max(1e-9),
        }).collect();

        // Run expert labeling
        let side = result.side as i8;
        let label = find_best_entry(side, 0, &ohlc_bars, cfg);

        // Feature extractor closure
        let reason = &result.reason_json;
        let get = |k: &str| reason.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
        let debug_get = |k: &str| {
            reason.get("debug").and_then(|d| d.get(k)).and_then(|v| v.as_f64())
                .or_else(|| reason.get(k).and_then(|v| v.as_f64()))
                .unwrap_or(0.0)
        };

        let ep = result.entry_price as f64;
        let sl_pct = if ep > 0.0 { (result.sl_price as f64 - ep).abs() / ep } else { 0.0 };
        let tp1_pct = if ep > 0.0 { (result.tp1_price as f64 - ep).abs() / ep } else { 0.0 };
        let tp2_pct = result.tp2_price.map(|t| if ep > 0.0 { (t as f64 - ep).abs() / ep } else { 0.0 }).unwrap_or(0.0);
        let tp3_pct = result.tp3_price.map(|t| if ep > 0.0 { (t as f64 - ep).abs() / ep } else { 0.0 }).unwrap_or(0.0);
        let rr = if sl_pct > 0.0 { tp1_pct / sl_pct } else { 0.0 };
        let rr2 = if sl_pct > 0.0 { tp2_pct / sl_pct } else { 0.0 };

        let level_aware = reason.get("level_aware").and_then(|v| v.as_bool()).unwrap_or(false);
        let quality_mult = get("quality_multiplier");
        let quality_grade = reason.get("quality_grade").and_then(|v| v.as_str()).unwrap_or("?");
        let original_score = get("original_score");

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
        let ml_s = result.ml_score.unwrap_or(0.0) as f64;
        let heur_s = result.heur_score.unwrap_or(0.0) as f64;
        let ml_heur_gap = (ml_s - heur_s).abs();
        let score_per_risk = if sl_pct > 0.0 { result.final_score as f64 / sl_pct } else { 0.0 };

        let atr_pct = get("atr_pct");
        let market_factor = get("market_factor");
        let score_factor = get("score_factor");

        // Base feature vector (same for all bars in this setup)
        let base_features = vec![
            result.tf_minutes as f32,
            result.side as f32,
            result.final_score,
            ml_s as f32,
            heur_s as f32,
            ep as f32,
            sl_pct as f32,
            tp1_pct as f32,
            tp2_pct as f32,
            tp3_pct as f32,
            rr as f32,
            rr2 as f32,
            pred as f32,
            raw as f32,
            ind as f32,
            mkt as f32,
            debug_get("coverage_score") as f32,
            debug_get("consensus_score") as f32,
            result.price10_score.unwrap_or(0.0) as f32,
            result.bounce_prob.unwrap_or(0.0) as f32,
            result.bounce_score.unwrap_or(0.0) as f32,
            result.breakout_prob.unwrap_or(0.0) as f32,
            result.breakout_score.unwrap_or(0.0) as f32,
            debug_get("trend_strength") as f32,
            debug_get("momentum_strength") as f32,
            debug_get("volatility_regime") as f32,
            debug_get("volume_spike_score") as f32,
            if level_aware { 1.0 } else { 0.0 },
            get("market_quality_score") as f32,
            quality_mult as f32,
            // quality_grade is string, skip for now or encode
            0.0, // placeholder for grade
            original_score as f32,
            pred_vs_raw as f32,
            pred_vs_ind as f32,
            component_std as f32,
            component_min as f32,
            ml_heur_gap as f32,
            score_per_risk as f32,
            atr_pct as f32,
            market_factor as f32,
            score_factor as f32,
        ];

        // Generate examples
        let examples = build_examples_for_setup(
            side,
            0, // t0 = 0 (relative to signal)
            &label,
            window_bars,
            |_bar_idx| {
                // For now, use same base features for all bars
                // In v2, you could add bar-specific features (e.g., current RSI, etc.)
                base_features.clone()
            },
        );

        // Write examples to CSV
        for ex in &examples {
            total_examples += 1;
            if ex.label_enter == 1 {
                enter_examples += 1;
            }
            if ex.label_cancel == 1 {
                cancel_examples += 1;
            }

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
                 {:.4},{:.4},{:.4},{:.4},\
                 {:.4},{:.4},{:.4},\
                 {:.4},{:.4},{:.4},{:.4},\
                 {},{},{},{},{:.4}",
                result.symbol, result.tf_minutes, result.side,
                result.final_score, ml_s, heur_s,
                ep, sl_pct, tp1_pct, tp2_pct, tp3_pct, rr, rr2,
                pred, raw, ind, mkt,
                debug_get("coverage_score"), debug_get("consensus_score"),
                result.price10_score.unwrap_or(0.0),
                result.bounce_prob.unwrap_or(0.0), result.bounce_score.unwrap_or(0.0),
                result.breakout_prob.unwrap_or(0.0), result.breakout_score.unwrap_or(0.0),
                debug_get("trend_strength"), debug_get("momentum_strength"),
                debug_get("volatility_regime"), debug_get("volume_spike_score"),
                if level_aware { 1 } else { 0 },
                get("market_quality_score"),
                quality_mult, quality_grade, original_score,
                pred_vs_raw, pred_vs_ind, component_std, component_min, ml_heur_gap, score_per_risk,
                atr_pct, market_factor, score_factor,
                debug_get("impulse_phase"), debug_get("momentum_acceleration"),
                debug_get("rsi_slope"), debug_get("volume_impulse_confirm"),
                debug_get("nearest_support_dist_atr"), debug_get("nearest_resistance_dist_atr"),
                debug_get("sr_position"),
                debug_get("ema_stack"), debug_get("price_vs_emas"),
                debug_get("bb_position"), debug_get("bb_width"),
                ex.elapsed, ex.remaining, ex.label_enter, ex.label_cancel, ex.weight
            )?;
        }
    }

    tracing::info!(
        "Entry Policy dataset: {} total examples ({} ENTER, {} CANCEL)",
        total_examples, enter_examples, cancel_examples
    );

    Ok(())
}

/// Candle with ATR for entry policy labeling
#[derive(Debug, Clone)]
struct CandleWithAtr {
    high: f64,
    low: f64,
    close: f64,
    atr: f32,  // Matches DB type (FLOAT4)
}

/// Fetch candles with ATR from database
async fn fetch_candles_with_atr(
    pool: &PgPool,
    symbol_id: i64,
    tf_minutes: i16,
    after_time: chrono::DateTime<chrono::Utc>,
    limit: usize,
) -> Result<Vec<CandleWithAtr>> {
    // Join candles with indicators to get ATR
    let candle_table = match tf_minutes {
        1 => "market.candles_1m",
        5 => "market.candles_5m",
        15 => "market.candles_15m",
        60 => "market.candles_1h",
        240 => "market.candles_4h",
        _ => "market.candles_1h",
    };

    let sql = format!(
        "SELECT c.high, c.low, c.close, COALESCE(i.atr, 0.001) as atr \
         FROM {} c \
         LEFT JOIN market.indicators_wide i \
           ON c.symbol_id = i.symbol_id AND c.time = i.time AND i.tf_minutes = $4 \
         WHERE c.symbol_id = $1 AND c.time > $2 \
         ORDER BY c.time ASC LIMIT $3",
        candle_table
    );

    let rows = sqlx::query::<sqlx::Postgres>(&sql)
        .bind(symbol_id)
        .bind(after_time)
        .bind(limit as i64)
        .bind(tf_minutes as i16)
        .fetch_all(pool)
        .await?;

    let mut candles = Vec::with_capacity(rows.len());
    for row in rows {
        let high: f64 = row.get("high");
        let low: f64 = row.get("low");
        let close: f64 = row.get("close");
        let atr: f32 = row.get("atr"); // ATR is FLOAT4 in DB

        candles.push(CandleWithAtr {
            high,
            low,
            close,
            atr: atr.max(0.001), // Ensure non-zero
        });
    }

    Ok(candles)
}

/// Run comparison between Baseline and Entry Agent evaluation
async fn run_entry_agent_comparison(
    pool: &PgPool,
    baseline_results: &[BacktestResult],
    timeout_bars: usize,
) -> Result<()> {
    use entry_agent_real_evaluator::{RealEntryAgentEvaluator, EntryAgentComparison};
    
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        ENTRY AGENT: Loading Models...                       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("\n");
    
    // Check env var for GPU usage, default to false for safety
    let use_gpu = std::env::var("ENTRY_AGENT_USE_GPU")
        .unwrap_or_default()
        .to_lowercase()
        .parse::<bool>()
        .unwrap_or(false);
    
    tracing::info!("Entry Agent GPU acceleration: {}", use_gpu);
    let evaluator = match RealEntryAgentEvaluator::new(pool, timeout_bars, use_gpu) {
        Ok(e) => e,
        Err(e) => {
            println!("Failed to create Entry Agent evaluator: {}", e);
            println!("Continuing with baseline-only evaluation...");
            return Ok(());
        }
    };
    
    // Report model/fallback status
    if evaluator.has_models() {
        println!("✅ Entry Agent models loaded successfully!");
        println!("   Evaluating signals with real model inference...\n");
    } else if evaluator.is_using_expert_fallback() {
        println!("⚠️  Entry Agent models not found — using expert labeling fallback");
        println!("   This simulates the OPTIMAL entry timing (upper bound for agent).");
        println!("   To train real models, run: python scripts/train_entry_policy.py\n");
    }
    
    // Evaluate all signals with Entry Agent
    let mut agent_results = Vec::with_capacity(baseline_results.len());
    let mut processed = 0usize;
    let mut entered = 0usize;
    let mut cancelled = 0usize;
    let mut expired = 0usize;
    
    for result in baseline_results {
        // Reconstruct SignalForBacktest from BacktestResult
        let signal = SignalForBacktest {
            symbol_id: result.symbol_id,
            symbol: result.symbol.clone(),
            tf_minutes: result.tf_minutes,
            side: result.side,
            final_score: result.final_score,
            ml_score: result.ml_score,
            heur_score: result.heur_score,
            entry_price: Some(result.entry_price),
            sl_price: Some(result.sl_price),
            tp1_price: Some(result.tp1_price),
            tp2_price: result.tp2_price,
            tp3_price: result.tp3_price,
            reason: Some(result.reason_json.clone()),
            price10_target: None,
            price10_score: result.price10_score,
            bounce_prob: result.bounce_prob,
            bounce_score: result.bounce_score,
            breakout_prob: result.breakout_prob,
            breakout_score: result.breakout_score,
            time: result.signal_time,
            time_ms: result.signal_time_ms,
        };
        
        match evaluator.evaluate(&signal).await {
            Ok(Some(agent_result)) => {
                if agent_result.was_entered() {
                    entered += 1;
                } else if agent_result.was_cancelled() {
                    cancelled += 1;
                } else {
                    expired += 1;
                }
                agent_results.push(agent_result);
            }
            Ok(None) => {}
            Err(e) => {
                tracing::warn!("Entry Agent evaluation error: {}", e);
            }
        }
        
        processed += 1;
        if processed % 1000 == 0 {
            println!("  Processed {}/{} signals: {} entered, {} cancelled, {} expired",
                processed, baseline_results.len(), entered, cancelled, expired);
        }
    }
    
    println!("\n  Entry Agent evaluation complete: {} results\n", agent_results.len());
    
    // Build full comparison with per-TF breakdown
    let comparison = EntryAgentComparison::from_results(baseline_results, &agent_results);
    comparison.print();
    
    Ok(())
}

