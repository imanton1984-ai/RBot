//! Serialization utilities for the trading bot

use rkyv::{Archive, Deserialize, Serialize};
use serde::{Deserialize as SerdeDeserialize, Serialize as SerdeSerialize};

/// Trait for types that can be serialized with rkyv
pub trait RkyvSerializable: Archive + Deserialize<'static> + Serialize {
    /// Serialize to bytes using rkyv
    fn serialize_rkyv(&self) -> Result<Vec<u8>, rkyv::ser::Error> {
        let mut serializer = rkyv::ser::serializers::AllocSerializer::<1024>::default();
        rkyv::ser::serialize(&self, &mut serializer)?;
        Ok(serializer.into_vec())
    }

    /// Deserialize from bytes using rkyv
    fn deserialize_rkyv(bytes: &[u8]) -> Result<Self::Archived, rkyv::de::Error> 
    where 
        Self: Sized,
    {
        rkyv::from_bytes(bytes)
    }
}

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
    T: SerdeDeserialize<'static>,
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