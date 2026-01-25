pub mod config;

// Реэкспортируем data_types для удобства, чтобы везде подключать только common
pub use data_types::*;
pub use config::*;
pub mod health;