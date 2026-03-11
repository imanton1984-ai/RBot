// strategies/super_level_strategy/src/lib.rs
//
// Super Level Strategy — Level-First Sniper Strategy
//
// Архитектура "Level-First" — уровни (ликвидность) являются фундаментом.
// ML, EWMAC, Эвристика — слуги-помощники, подтверждающие вход.
//
// 5 ФАЗ:
//   1. Radar      — Поиск "Зоны Убийства" (цена у сильного уровня, ≤0.5 ATR)
//   2. Context    — Определение сценария: Bounce (отскок) или Breakout (пробой)
//   3. ML Valid.  — Подтверждение через Super Entry модель (p_super ≥ threshold)
//   4. Entry Agent — Тайминг входа (ENTER/WAIT/CANCEL на младшем TF)
//   5. Risk Mgmt  — ATR-based SL/TP (bounce: tight, breakout: wide)
//
// ПЕРЕИСПОЛЬЗУЕТ:
//   - ml_entry_strategy    — SuperEntryPipeline (zero-copy CUDA inference)
//   - ewmac_strategy       — EwmacCalculator    (trend context)
//   - predictors::level_view — LevelView        (уровни S/R)
//   - indicators::sr_levels  — calculate_sr_levels (вычисление уровней)
//   - entry_policy::agent   — EntryAgent        (тайминг входа)

pub mod config;
pub mod phases;
pub mod pipeline;

// Re-exports
pub use config::SuperLevelConfig;
pub use phases::{PhaseResult, Scenario};
pub use pipeline::SuperLevelPipeline;
