pub mod enums;
pub mod error;
pub mod serde;
pub mod timeframe;
pub mod types;

// Re-export commonly used types at the crate root for convenience
pub use crate::timeframe::Timeframe;
pub use crate::types::Candle;

// Также можно реэкспортировать всё из подмодулей, если нужно:
// pub use crate::enums::*;
// pub use crate::error::*;
// pub use crate::serde::{serialize_json, deserialize_json};
