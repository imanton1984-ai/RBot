// strategies/super_level_strategy/src/lib.rs
//
// Super Level Strategy — ML-based Level-First Trading
//
// 5 ML моделей:
//   1. Level     — качество/сила уровня (по реальным касаниям)
//   2. Entry     — определяет оптимальную точку входа у уровня
//   3. Direction  — предсказывает направление (LONG/SHORT)
//   4. BounceBreak — предсказывает отскок или пробой
//   5. Evaluator  — финальный вердикт по всем моделям
//
// Суть: быстрые сделки у уровней. Цена близко от уровня =
// движение вот-вот наступит. Модели определяют силу, направление,
// сценарий и итоговую оценку.

pub mod config;
pub mod dataset;
pub mod model;
pub mod scorer;
pub mod signal_generator;
pub mod pipeline;

// Re-exports
pub use config::SuperLevelConfig;
pub use dataset::{SuperLevelExample, PriceLevel, LevelStrength, LevelType};
pub use model::{SuperLevelModelManager, SuperLevelPrediction};
pub use scorer::{SuperLevelScorer, SuperLevelDecision};
pub use signal_generator::{SignalGenerator, SuperLevelSignal};
pub use pipeline::SuperLevelPipeline;
