// compute/src/predictors/types.rs

#[derive(Debug, Clone, PartialEq)]
pub enum PredictionAspect {
    PriceTarget,
    LevelBounce,
    LevelBreakout,
}

impl PredictionAspect {
    pub fn as_int(&self) -> i16 {
        match self {
            PredictionAspect::PriceTarget => 1,
            PredictionAspect::LevelBounce => 2,
            PredictionAspect::LevelBreakout => 3,
        }
    }

    pub fn from_int(value: i16) -> Option<Self> {
        match value {
            1 => Some(PredictionAspect::PriceTarget),
            2 => Some(PredictionAspect::LevelBounce),
            3 => Some(PredictionAspect::LevelBreakout),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CalcSource {
    Hard,
    Ml,
}

impl CalcSource {
    pub fn as_int(&self) -> i16 {
        match self {
            CalcSource::Hard => 1,
            CalcSource::Ml => 2,
        }
    }

    pub fn from_int(value: i16) -> Option<Self> {
        match value {
            1 => Some(CalcSource::Hard),
            2 => Some(CalcSource::Ml),
            _ => None,
        }
    }
}

// This will be the struct that is written to the database
#[derive(Debug, Clone)]
pub struct PredictionRow {
    // Time axes
    pub time: chrono::DateTime<chrono::Utc>,
    pub time_ms: i64,

    // Symbol and timeframe
    pub symbol_id: i64,
    pub symbol: String,
    pub tf_minutes: i32,

    // Horizon and aspect
    pub horizon_bars: i32,
    pub aspect: PredictionAspect,
    pub calc_source: CalcSource,

    // Predictor reference
    pub predictor_id: i64,

    // Score and value
    pub score_norm: f32,
    pub value: f64,

    // Value range (for uncertainty)
    pub value_low: Option<f64>,
    pub value_high: Option<f64>,

    // Direction
    pub side: Option<i16>,

    // Level info (if aspect is about levels)
    pub level_hash: Option<String>,
    pub level_kind: Option<i16>,
    pub level_price: Option<f64>,
    pub level_strength: Option<f32>,
    pub level_distance_atr: Option<f32>,

    // Context
    pub candle_is_final: bool,
    pub event_time_ms: Option<i64>,

    // Additional details
    pub details_json: Option<serde_json::Value>,

    // Unique key for upsert
    pub prediction_key: String,
}

pub struct PredictorId(pub i64);
pub struct PredictorMeta {
    pub predictor_id: i64,
    pub name: String,
    pub version: String,
    pub aspect: PredictionAspect,
    pub calc_source: CalcSource,
    pub framework: String,
    pub artifact_path: Option<String>,
    pub feature_schema_id: String,
}

pub struct FeatureSchemaId(pub String);
pub struct PredictionKey(pub String);