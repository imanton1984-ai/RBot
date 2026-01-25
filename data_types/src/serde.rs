//! Serialization utilities for the trading bot

use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Helper function to serialize with serde_json
pub fn serialize_json<T>(data: &T) -> Result<String, serde_json::Error>
where
    T: SerdeSerialize,
{
    serde_json::to_string(data)
}

/// Helper function to deserialize with serde_json
pub fn deserialize_json<T>(json_str: &str) -> Result<T, serde_json::Error>
where
    T: for<'de> SerdeDeserialize<'de>,
{
    serde_json::from_str(json_str)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Candle;

    #[test]
    fn test_json_serialization() {
        let candle = Candle {
            time: chrono::Utc::now(),
            symbol: "BTCUSDT".to_string(),
            timeframe: crate::types::Timeframe::Min1,
            open: 10000.0,
            high: 10100.0,
            low: 9900.0,
            close: 10050.0,
            volume: 100.0,
            is_final: true,
        };

        let json_str = serialize_json(&candle).unwrap();
        let deserialized: Candle = deserialize_json(&json_str).unwrap();
        assert_eq!(candle, deserialized);
    }
}