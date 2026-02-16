// signal_quality/trainer.rs
//
// Training data preparation for the Signal Quality ML model.
//
// This module reads trade.final_signals from the database, extracts features
// from the reason JSON, and prepares training data for XGBoost.
//
// The full training workflow:
//   1. Backtester evaluates each historical signal → TradeOutcome
//   2. trainer::prepare_training_data() reads signals + outcomes from DB
//   3. Exports to CSV or DMatrix format
//   4. Python/Rust XGBoost trains the model
//   5. Model saved as .ubj → loaded by MlQualityScorer
//
// NOT YET WIRED INTO PIPELINE — needs backtester module first.

use anyhow::Result;
use sqlx::PgPool;
use super::types::*;

/// Prepares training data from the database.
/// Reads trade.final_signals and joins with backtest outcomes (future table).
pub struct TrainingDataBuilder {
    pool: PgPool,
    min_score: f64,
    max_rows: i64,
}

impl TrainingDataBuilder {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool,
            min_score: 0.55,
            max_rows: 500_000,
        }
    }

    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    pub fn with_max_rows(mut self, max_rows: i64) -> Self {
        self.max_rows = max_rows;
        self
    }

    /// Load signals that have a reason JSON (required for feature extraction).
    /// Returns raw rows for feature building.
    pub async fn load_signals_with_reason(&self) -> Result<Vec<SignalRow>> {
        let rows = sqlx::query_as::<_, SignalRow>(
            r#"
            SELECT 
                symbol, tf_minutes, side, final_score,
                ml_score, heur_score,
                entry_price, sl_price, tp1_price, tp2_price, tp3_price,
                reason,
                price10_target, price10_score,
                bounce_prob, bounce_score,
                breakout_prob, breakout_score,
                time, time_ms
            FROM trade.final_signals
            WHERE reason IS NOT NULL
              AND final_score >= $1
            ORDER BY time DESC
            LIMIT $2
            "#,
        )
        .bind(self.min_score)
        .bind(self.max_rows)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Convert raw DB rows to SignalFeatures for training.
    pub fn build_features(rows: &[SignalRow]) -> Vec<SignalFeatures> {
        rows.iter()
            .filter_map(|row| {
                let reason = row.reason.as_ref()?;
                Some(SignalFeatures::from_reason_json(
                    reason,
                    &row.symbol,
                    row.tf_minutes,
                    row.side,
                    row.entry_price.unwrap_or(0.0) as f64,
                    row.sl_price.unwrap_or(0.0) as f64,
                    row.tp1_price.unwrap_or(0.0) as f64,
                    row.tp2_price.unwrap_or(0.0) as f64,
                    row.tp3_price.unwrap_or(0.0) as f64,
                    row.ml_score.map(|x| x as f64),
                    row.heur_score.map(|x| x as f64),
                    row.price10_target,
                    row.price10_score.map(|x| x as f64),
                    row.bounce_prob.map(|x| x as f64),
                    row.bounce_score.map(|x| x as f64),
                    row.breakout_prob.map(|x| x as f64),
                    row.breakout_score.map(|x| x as f64),
                ))
            })
            .collect()
    }

    /// Export features to CSV for Python XGBoost training.
    /// Label column must be added by backtester (win_probability or pnl).
    pub fn export_features_csv(features: &[SignalFeatures], path: &str) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Header
        let names = SignalFeatures::feature_names();
        writeln!(file, "{}", names.join(","))?;

        // Data
        for f in features {
            let vec = f.to_feature_vector();
            let line: Vec<String> = vec.iter().map(|v| format!("{:.6}", v)).collect();
            writeln!(file, "{}", line.join(","))?;
        }

        tracing::info!(
            target: "signal_quality",
            "Exported {} features to {}",
            features.len(),
            path
        );

        Ok(())
    }

    /// Export features + outcomes for XGBoost training.
    /// This is the FULL training pipeline step after backtester provides outcomes.
    pub fn export_training_csv(
        examples: &[TrainingExample],
        path: &str,
    ) -> Result<()> {
        use std::io::Write;

        let mut file = std::fs::File::create(path)?;

        // Header: feature columns + label columns
        let names = SignalFeatures::feature_names();
        let label_names = vec!["label_win", "label_pnl_pct", "label_max_favorable", "label_max_adverse"];
        let all_names: Vec<&str> = names.iter().copied()
            .chain(label_names.iter().copied())
            .collect();
        writeln!(file, "{}", all_names.join(","))?;

        // Data
        for ex in examples {
            let fvec = ex.features.to_feature_vector();
            let mut line: Vec<String> = fvec.iter().map(|v| format!("{:.6}", v)).collect();

            // Label: binary win (1 if reached TP1, 0 otherwise)
            let win = if ex.outcome.result >= 1.0 { 1.0 } else { 0.0 };
            line.push(format!("{:.6}", win));
            line.push(format!("{:.6}", ex.outcome.pnl_pct));
            line.push(format!("{:.6}", ex.outcome.max_favorable));
            line.push(format!("{:.6}", ex.outcome.max_adverse));

            writeln!(file, "{}", line.join(","))?;
        }

        tracing::info!(
            target: "signal_quality",
            "Exported {} training examples to {}",
            examples.len(),
            path
        );

        Ok(())
    }
}

/// Raw signal row from database (for sqlx::FromRow).
/// Uses sqlx FromRow derive when connected to DB.
#[derive(Debug, Clone)]
pub struct SignalRow {
    pub symbol: String,
    pub tf_minutes: i16,
    pub side: i16,
    pub final_score: f32,
    pub ml_score: Option<f32>,
    pub heur_score: Option<f32>,
    pub entry_price: Option<f32>,
    pub sl_price: Option<f32>,
    pub tp1_price: Option<f32>,
    pub tp2_price: Option<f32>,
    pub tp3_price: Option<f32>,
    pub reason: Option<serde_json::Value>,
    pub price10_target: Option<f64>,
    pub price10_score: Option<f32>,
    pub bounce_prob: Option<f32>,
    pub bounce_score: Option<f32>,
    pub breakout_prob: Option<f32>,
    pub breakout_score: Option<f32>,
    pub time: chrono::DateTime<chrono::Utc>,
    pub time_ms: i64,
}

// Manual FromRow implementation (sqlx::FromRow derive requires sqlx feature in Cargo.toml)
impl<'r> sqlx::FromRow<'r, sqlx::postgres::PgRow> for SignalRow {
    fn from_row(row: &'r sqlx::postgres::PgRow) -> std::result::Result<Self, sqlx::Error> {
        use sqlx::Row;
        Ok(Self {
            symbol: row.try_get("symbol")?,
            tf_minutes: row.try_get("tf_minutes")?,
            side: row.try_get("side")?,
            final_score: row.try_get("final_score")?,
            ml_score: row.try_get("ml_score")?,
            heur_score: row.try_get("heur_score")?,
            entry_price: row.try_get("entry_price")?,
            sl_price: row.try_get("sl_price")?,
            tp1_price: row.try_get("tp1_price")?,
            tp2_price: row.try_get("tp2_price")?,
            tp3_price: row.try_get("tp3_price")?,
            reason: row.try_get("reason")?,
            price10_target: row.try_get("price10_target")?,
            price10_score: row.try_get("price10_score")?,
            bounce_prob: row.try_get("bounce_prob")?,
            bounce_score: row.try_get("bounce_score")?,
            breakout_prob: row.try_get("breakout_prob")?,
            breakout_score: row.try_get("breakout_score")?,
            time: row.try_get("time")?,
            time_ms: row.try_get("time_ms")?,
        })
    }
}

/// XGBoost training configuration (for the Python or Rust trainer script)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XgbTrainConfig {
    /// Number of boosting rounds
    pub n_rounds: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Max tree depth
    pub max_depth: u32,
    /// Subsample ratio per tree
    pub subsample: f64,
    /// Column sample ratio per tree
    pub colsample_bytree: f64,
    /// Objective function
    pub objective: String,
    /// Evaluation metric
    pub eval_metric: String,
    /// Use GPU for training
    pub use_gpu: bool,
}

impl Default for XgbTrainConfig {
    fn default() -> Self {
        Self {
            n_rounds: 500,
            learning_rate: 0.05,
            max_depth: 6,
            subsample: 0.8,
            colsample_bytree: 0.8,
            objective: "binary:logistic".to_string(), // Predict win probability
            eval_metric: "auc".to_string(),
            use_gpu: false,
        }
    }
}

impl XgbTrainConfig {
    /// Export config as JSON for the Python trainer script
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "n_rounds": self.n_rounds,
            "params": {
                "eta": self.learning_rate,
                "max_depth": self.max_depth,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "objective": self.objective,
                "eval_metric": self.eval_metric,
                "tree_method": if self.use_gpu { "gpu_hist" } else { "hist" },
                "verbosity": 1,
            },
            "feature_names": SignalFeatures::feature_names(),
            "model_output": "models/signal_quality_v1.ubj",
        })
    }
}
