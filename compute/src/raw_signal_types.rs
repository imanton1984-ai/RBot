use common::{Symbol, Timeframe};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone)]
pub struct RawSignal {
    pub symbol: Symbol,
    pub timeframe: Timeframe,
    pub timestamp: i64,
    pub indicator_id: i16,
    pub signal_kind: i16,
    pub side: i16,
    pub score: f32,
    pub value: f32,
    pub details: Option<JsonValue>,
}
