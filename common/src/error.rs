//! Error types for the trading bot

use thiserror::Error;

/// Main error type for the trading bot
#[derive(Error, Debug)]
pub enum TradingBotError {
    /// Error related to data parsing
    #[error("Data parsing error: {0}")]
    ParseError(String),

    /// Error related to network communication
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),

    /// Error related to database operations
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),

    /// Error related to serialization/deserialization
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// Error related to candle building
    #[error("Candle building error: {0}")]
    CandleBuildError(String),

    /// Error related to indicator calculation
    #[error("Indicator calculation error: {0}")]
    IndicatorError(String),

    /// Error related to order management
    #[error("Order management error: {0}")]
    OrderError(String),

    /// Error related to position tracking
    #[error("Position tracking error: {0}")]
    PositionError(String),

    /// Error related to configuration
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Error related to GPU operations
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Error related to ML inference
    #[error("ML inference error: {0}")]
    MlError(String),

    /// Error related to invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Error related to missing data
    #[error("Missing data: {0}")]
    MissingData(String),
}