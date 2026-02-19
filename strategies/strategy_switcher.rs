// strategies/strategy_switcher.rs
//
// Strategy Switcher — централизованный переключатель стратегий
//
// Позволяет:
//   - Выбирать активную стратегию (или несколько одновременно)
//   - Переключать стратегии через конфиг, env-переменные или WebUI API
//   - Управлять приоритетами стратегий при конфликте сигналов
//
// ИСПОЛЬЗОВАНИЕ:
//   1. Через env: ACTIVE_STRATEGY=super_entry,default
//   2. Через конфиг: config/strategies.toml
//   3. Через WebUI API: POST /api/strategy/switch { "strategy": "super_entry" }
//
// БУДУЩЕЕ РАСШИРЕНИЕ:
//   - Добавить новые стратегии: impl StrategyProvider
//   - Добавить runtime-переключение через WebSocket
//   - Добавить A/B тестирование стратегий в production
//   - Добавить ensemble mode (комбинирование сигналов нескольких стратегий)

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════
// ТИПЫ СТРАТЕГИЙ
// ═══════════════════════════════════════════════════════════

/// Идентификатор стратегии
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyId {
    /// Основная стратегия (entry_policy + signal_quality + predictors)
    Default,
    /// Super Entry Strategy (ML-модель поиска супер-входов)
    SuperEntry,
    /// Комбинированный режим: Default + SuperEntry фильтр
    Combined,
}

impl StrategyId {
    /// Из строки
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "default" | "main" | "base" => Some(Self::Default),
            "super_entry" | "super-entry" | "superentry" => Some(Self::SuperEntry),
            "combined" | "both" | "all" => Some(Self::Combined),
            _ => None,
        }
    }

    /// В строку
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::SuperEntry => "super_entry",
            Self::Combined => "combined",
        }
    }

    /// Описание для UI
    pub fn description(&self) -> &'static str {
        match self {
            Self::Default => "Основная стратегия (Entry Policy + Signal Quality + Predictors)",
            Self::SuperEntry => "Super Entry: ML-модель поиска точек с сильным движением",
            Self::Combined => "Комбинированный: Default + Super Entry фильтр качества",
        }
    }
}

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ═══════════════════════════════════════════════════════════
// КОНФИГ СТРАТЕГИЙ
// ═══════════════════════════════════════════════════════════

/// Конфигурация стратегии
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Уникальный ID
    pub id: StrategyId,
    /// Включена ли стратегия
    pub enabled: bool,
    /// Приоритет (0 = наивысший, при конфликте сигналов выигрывает стратегия с меньшим priority)
    pub priority: u8,
    /// Вес стратегии в ensemble mode (0.0-1.0)
    pub weight: f64,
    /// Таймфреймы на которых работает стратегия
    pub timeframes: Vec<i32>,
    /// Дополнительные параметры (JSON)
    pub params: HashMap<String, serde_json::Value>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            id: StrategyId::Default,
            enabled: true,
            priority: 0,
            weight: 1.0,
            timeframes: vec![1, 5, 15, 60, 240],
            params: HashMap::new(),
        }
    }
}

impl StrategyConfig {
    /// Конфиг для Super Entry
    pub fn super_entry_default() -> Self {
        Self {
            id: StrategyId::SuperEntry,
            enabled: false, // По умолчанию выключена (нужны обученные модели)
            priority: 1,
            weight: 0.5,
            timeframes: vec![1, 5, 15, 60, 240, 1440],
            params: HashMap::new(),
        }
    }

    /// Конфиг для Combined mode
    pub fn combined_default() -> Self {
        Self {
            id: StrategyId::Combined,
            enabled: false,
            priority: 0,
            weight: 1.0,
            timeframes: vec![1, 5, 15, 60, 240],
            params: HashMap::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════
// ПЕРЕКЛЮЧАТЕЛЬ СТРАТЕГИЙ
// ═══════════════════════════════════════════════════════════

/// Централизованный менеджер стратегий.
///
/// Потокобезопасный (Arc<RwLock<...>>), можно шарить между tokio tasks.
///
/// # Пример использования (будущее)
/// ```rust,no_run
/// let switcher = StrategySwitcher::from_env();
///
/// // Проверить активную стратегию
/// if switcher.is_active(StrategyId::SuperEntry) {
///     let signals = super_entry_pipeline.run_all(&pool, false).await?;
///     // ...
/// }
///
/// // Переключить стратегию (из WebUI API)
/// switcher.activate(StrategyId::SuperEntry);
/// switcher.deactivate(StrategyId::Default);
///
/// // Получить все активные стратегии
/// for strategy in switcher.active_strategies() {
///     println!("Active: {} (priority={})", strategy.id, strategy.priority);
/// }
/// ```
pub struct StrategySwitcher {
    strategies: Arc<RwLock<HashMap<StrategyId, StrategyConfig>>>,
}

impl StrategySwitcher {
    /// Создать новый переключатель с дефолтными стратегиями
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        strategies.insert(StrategyId::Default, StrategyConfig::default());
        strategies.insert(StrategyId::SuperEntry, StrategyConfig::super_entry_default());
        strategies.insert(StrategyId::Combined, StrategyConfig::combined_default());

        Self {
            strategies: Arc::new(RwLock::new(strategies)),
        }
    }

    /// Создать из переменных окружения
    ///
    /// Читает:
    ///   - `ACTIVE_STRATEGY` — ID активной стратегии (default/super_entry/combined)
    ///   - `SUPER_ENTRY_ENABLED` — включить Super Entry (true/false)
    pub fn from_env() -> Self {
        let switcher = Self::new();

        // Активировать стратегию из env
        if let Ok(strategy_str) = std::env::var("ACTIVE_STRATEGY") {
            for name in strategy_str.split(',') {
                if let Some(id) = StrategyId::from_str(name.trim()) {
                    switcher.activate(id);
                }
            }
        }

        // Super Entry из отдельного env
        if let Ok(val) = std::env::var("SUPER_ENTRY_ENABLED") {
            if val == "true" || val == "1" {
                switcher.activate(StrategyId::SuperEntry);
            }
        }

        switcher
    }

    /// Проверить активна ли стратегия
    pub fn is_active(&self, id: StrategyId) -> bool {
        self.strategies
            .read()
            .unwrap()
            .get(&id)
            .map_or(false, |c| c.enabled)
    }

    /// Активировать стратегию
    pub fn activate(&self, id: StrategyId) {
        if let Some(config) = self.strategies.write().unwrap().get_mut(&id) {
            config.enabled = true;
        }
    }

    /// Деактивировать стратегию
    pub fn deactivate(&self, id: StrategyId) {
        if let Some(config) = self.strategies.write().unwrap().get_mut(&id) {
            config.enabled = false;
        }
    }

    /// Переключить на единственную стратегию (деактивирует все остальные)
    pub fn switch_to(&self, id: StrategyId) {
        let mut strategies = self.strategies.write().unwrap();
        for (sid, config) in strategies.iter_mut() {
            config.enabled = *sid == id;
        }
    }

    /// Получить список активных стратегий (сортировка по приоритету)
    pub fn active_strategies(&self) -> Vec<StrategyConfig> {
        let strategies = self.strategies.read().unwrap();
        let mut active: Vec<StrategyConfig> = strategies
            .values()
            .filter(|c| c.enabled)
            .cloned()
            .collect();
        active.sort_by_key(|c| c.priority);
        active
    }

    /// Получить конфиг стратегии по ID
    pub fn get_config(&self, id: StrategyId) -> Option<StrategyConfig> {
        self.strategies.read().unwrap().get(&id).cloned()
    }

    /// Обновить конфиг стратегии
    pub fn update_config(&self, config: StrategyConfig) {
        self.strategies
            .write()
            .unwrap()
            .insert(config.id, config);
    }

    /// Получить summary для UI
    pub fn summary(&self) -> Vec<StrategySummary> {
        let strategies = self.strategies.read().unwrap();
        let mut summaries: Vec<StrategySummary> = strategies
            .values()
            .map(|c| StrategySummary {
                id: c.id.as_str().to_string(),
                name: c.id.description().to_string(),
                enabled: c.enabled,
                priority: c.priority,
                weight: c.weight,
                timeframes: c.timeframes.clone(),
            })
            .collect();
        summaries.sort_by_key(|s| s.priority);
        summaries
    }
}

impl Default for StrategySwitcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary для API/UI
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategySummary {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub priority: u8,
    pub weight: f64,
    pub timeframes: Vec<i32>,
}

// ═══════════════════════════════════════════════════════════
// СИГНАЛЬНЫЙ КОНФЛИКТ-РЕЗОЛВЕР (для Combined mode)
// ═══════════════════════════════════════════════════════════

/// Как разрешать конфликты между стратегиями
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Побеждает стратегия с высшим приоритетом
    HighestPriority,
    /// Побеждает стратегия с высшим confidence / score
    HighestScore,
    /// Взвешенное среднее сигналов
    WeightedAverage,
    /// Вход только если ОБЕ стратегии согласны
    Consensus,
    /// Вход если ЛЮБАЯ стратегия даёт сигнал
    AnySignal,
}

impl Default for ConflictResolution {
    fn default() -> Self {
        Self::Consensus
    }
}

// ═══════════════════════════════════════════════════════════
// ТЕСТЫ
// ═══════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_id_from_str() {
        assert_eq!(StrategyId::from_str("default"), Some(StrategyId::Default));
        assert_eq!(StrategyId::from_str("super_entry"), Some(StrategyId::SuperEntry));
        assert_eq!(StrategyId::from_str("super-entry"), Some(StrategyId::SuperEntry));
        assert_eq!(StrategyId::from_str("combined"), Some(StrategyId::Combined));
        assert_eq!(StrategyId::from_str("unknown"), None);
    }

    #[test]
    fn test_switcher_default() {
        let switcher = StrategySwitcher::new();
        assert!(switcher.is_active(StrategyId::Default));
        assert!(!switcher.is_active(StrategyId::SuperEntry));
    }

    #[test]
    fn test_activate_deactivate() {
        let switcher = StrategySwitcher::new();

        switcher.activate(StrategyId::SuperEntry);
        assert!(switcher.is_active(StrategyId::SuperEntry));

        switcher.deactivate(StrategyId::SuperEntry);
        assert!(!switcher.is_active(StrategyId::SuperEntry));
    }

    #[test]
    fn test_switch_to() {
        let switcher = StrategySwitcher::new();

        switcher.switch_to(StrategyId::SuperEntry);
        assert!(!switcher.is_active(StrategyId::Default));
        assert!(switcher.is_active(StrategyId::SuperEntry));
        assert!(!switcher.is_active(StrategyId::Combined));
    }

    #[test]
    fn test_active_strategies_sorted() {
        let switcher = StrategySwitcher::new();
        switcher.activate(StrategyId::SuperEntry);

        let active = switcher.active_strategies();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].id, StrategyId::Default); // priority 0
        assert_eq!(active[1].id, StrategyId::SuperEntry); // priority 1
    }

    #[test]
    fn test_summary() {
        let switcher = StrategySwitcher::new();
        let summary = switcher.summary();
        assert_eq!(summary.len(), 3);
    }
}
