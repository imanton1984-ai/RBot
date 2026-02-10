// compute/src/raw_signal_types.rs

use common::{Symbol, Timeframe};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone)]
pub struct RawSignal {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,

    pub indicator_id: i16,
    pub signal_kind: i16,
    pub signal_sub_id: i16, // NEW: чтобы не биться за один ключ при upsert

    pub side: i16,
    pub score: f32,
    pub value: f32,

    pub details: Option<JsonValue>,

    // NEW: мета и контекст (пока default в processor)
    pub candle_is_final: bool,
    pub calc_source: i16,
    pub event_time_ms: Option<i64>,

    // NEW: снапшоты для последующего анализа / отладки
    pub features_json: Option<JsonValue>,
    pub scores_json: Option<JsonValue>,
    pub predictors_json: Option<JsonValue>,
}