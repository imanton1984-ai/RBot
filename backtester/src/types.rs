// backtester/src/types.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

/// Signal loaded from trade.final_signals for backtesting
#[derive(Debug, Clone, FromRow)]
pub struct SignalForBacktest {
    pub symbol_id: i64,
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
    #[allow(dead_code)]
    pub price10_target: Option<f64>,
    pub price10_score: Option<f32>,
    pub bounce_prob: Option<f32>,
    pub bounce_score: Option<f32>,
    pub breakout_prob: Option<f32>,
    pub breakout_score: Option<f32>,
    pub time: DateTime<Utc>,
    pub time_ms: i64,
}

/// Candle row fetched for outcome evaluation
#[derive(Debug, Clone, FromRow)]
pub struct CandleRow {
    #[allow(dead_code)]
    pub time: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
}

/// Trade outcome after evaluating a signal against future candles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Outcome {
    /// Price hit TP1 before stop-loss
    Win {
        tp_level: u8,  // Always 1 - TP1 only
        exit_price: f64,
    },
    /// Price hit stop-loss before TP1
    Loss {
        exit_price: f64,
    },
    /// Neither TP1 nor SL hit within timeout
    Expired {
        last_price: f64,
    },
}

impl Outcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Outcome::Win { .. } => "win_tp1",
            Outcome::Loss { .. } => "loss_sl",
            Outcome::Expired { .. } => "expired",
        }
    }
}

/// Complete backtest result for one signal
#[derive(Debug, Clone)]
pub struct BacktestResult {
    pub signal_time: DateTime<Utc>,
    pub signal_time_ms: i64,
    pub symbol: String,
    pub symbol_id: i64,
    pub tf_minutes: i16,
    pub side: i16,
    pub final_score: f32,
    pub ml_score: Option<f32>,
    pub heur_score: Option<f32>,

    pub entry_price: f32,
    pub sl_price: f32,
    pub tp1_price: f32,
    pub tp2_price: Option<f32>,
    pub tp3_price: Option<f32>,

    pub price10_score: Option<f32>,
    pub bounce_prob: Option<f32>,
    pub bounce_score: Option<f32>,
    pub breakout_prob: Option<f32>,
    pub breakout_score: Option<f32>,

    pub outcome: Outcome,
    pub pnl_pct: f64,
    pub exit_price: Option<f64>,
    pub bars_to_outcome: usize,
    pub max_favorable: f64,
    pub max_adverse: f64,

    pub reason_json: serde_json::Value,
}
