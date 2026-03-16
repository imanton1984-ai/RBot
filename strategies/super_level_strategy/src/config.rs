// strategies/super_level_strategy/src/config.rs
//
// Configuration for Super Level Strategy (ML-based, 5 models)
//
// 5 моделей:
//   1. Level     — определяет силу уровня по касаниям (strong/medium/weak)
//   2. Entry     — определяет оптимальную точку входа у уровня
//   3. Direction  — предсказывает направление (LONG/SHORT)
//   4. BounceBreak — предсказывает отскок или пробой
//   5. Evaluator  — оценивает все 4 модели, выдаёт финальный вердикт
//
// TP/SL берём из super_entry config (tf_target_move_pct).
// Warmup: 500 свечей (формирование уровней, индикаторов).
// Обучение: time-based split (% от текущего количества свечей).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::OnceLock;

// ═══════════════════════════════════════════════════════════════
// TP/SL из super_entry (процент хода по таймфрейму)
// ═══════════════════════════════════════════════════════════════

/// Target move percentage thresholds per timeframe — из super_entry.
pub fn tf_target_move_pct() -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    m.insert(1, 1.2);
    m.insert(5, 2.8);
    m.insert(15, 3.5);
    m.insert(60, 5.0);
    m.insert(240, 7.5);
    m.insert(1440, 10.0);
    m
}

pub fn get_target_move_pct(tf_minutes: i32) -> Option<f64> {
    tf_target_move_pct().get(&tf_minutes).copied()
}

// ═══════════════════════════════════════════════════════════════
// LEVEL COMPUTATION PARAMETERS
// ═══════════════════════════════════════════════════════════════

/// Параметры вычисления уровней по касаниям
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelParams {
    /// % от цены для кластеризации касаний в один уровень.
    /// Если две точки ближе чем sensitivity_pct*price — это один уровень.
    pub sensitivity_pct: f64,

    /// Количество свечей для формирования уровней (lookback).
    /// 500-1000 свечей — рекомендуемый диапазон.
    pub formation_bars: usize,

    /// Минимальное количество касаний для Strong уровня (≥5)
    pub strong_touches: usize,

    /// Минимальное количество касаний для Medium уровня (3-4)
    pub medium_touches: usize,

    /// Максимальная дистанция до уровня в ATR для активации.
    pub max_distance_atr: f64,

    /// Толщина зоны касания (% от цены).
    /// Если цена подходит ближе чем touch_zone_pct — это "касание".
    pub touch_zone_pct: f64,

    /// Сколько свечей между касаниями чтобы считался новый тач
    /// (не считаем одно продолжительное нахождение у уровня как N касаний).
    pub min_bars_between_touches: usize,
}

impl Default for LevelParams {
    fn default() -> Self {
        Self {
            sensitivity_pct: 0.15,
            formation_bars: 500,
            strong_touches: 5,
            medium_touches: 3,
            max_distance_atr: 0.5,
            touch_zone_pct: 0.12,
            min_bars_between_touches: 3,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// MAIN CONFIG
// ═══════════════════════════════════════════════════════════════

/// Полная конфигурация Super Level Strategy
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuperLevelConfig {
    /// Warmup свечей для индикаторов + формирования уровней.
    pub warmup_bars: usize,

    /// Lookahead (свечей вперёд для лейблинга)
    pub lookahead_bars: usize,

    /// Минимальный P(evaluator) для финального сигнала
    pub p_threshold: f64,

    /// SL fraction от target move
    pub sl_fraction: f64,

    /// Max hold bars перед принудительным закрытием
    pub max_hold_bars: usize,

    /// Target пороги по ТФ (из super_entry)
    pub tf_targets: HashMap<i32, f64>,

    /// Параметры вычисления уровней
    pub level_params: LevelParams,

    // ── Model paths ───────────────────────────────────────────
    pub level_model_template: String,
    pub entry_model_template: String,
    pub direction_model_template: String,
    pub bounce_break_model_template: String,
    pub evaluator_model_template: String,

    // ── Training split ────────────────────────────────────────
    /// % свечей для обучения (0-100). Остаток — на тест.
    /// Time-based: первые train_pct% — train, остальные — test.
    pub train_pct: f64,

    /// % свечей на подготовку/warmup (не используются для train/test).
    pub prep_pct: f64,
}

impl Default for SuperLevelConfig {
    fn default() -> Self {
        Self {
            warmup_bars: 500,
            lookahead_bars: 25,
            p_threshold: 0.55,
            sl_fraction: 0.65,
            max_hold_bars: 25,
            tf_targets: tf_target_move_pct(),
            level_params: LevelParams::default(),
            level_model_template: "models/slvl_level_v1_tf{tf}.ubj".to_string(),
            entry_model_template: "models/slvl_entry_v1_tf{tf}.ubj".to_string(),
            direction_model_template: "models/slvl_dir_v1_tf{tf}.ubj".to_string(),
            bounce_break_model_template: "models/slvl_bb_v1_tf{tf}.ubj".to_string(),
            evaluator_model_template: "models/slvl_eval_v1_tf{tf}.ubj".to_string(),
            train_pct: 65.0,
            prep_pct: 5.0,
        }
    }
}

impl SuperLevelConfig {
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("SUPER_LEVEL_WARMUP") {
            if let Ok(n) = v.parse() { cfg.warmup_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_LOOKAHEAD") {
            if let Ok(n) = v.parse() { cfg.lookahead_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_P_THRESHOLD") {
            if let Ok(n) = v.parse() { cfg.p_threshold = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_SL_FRACTION") {
            if let Ok(n) = v.parse() { cfg.sl_fraction = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_MAX_HOLD") {
            if let Ok(n) = v.parse() { cfg.max_hold_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_TRAIN_PCT") {
            if let Ok(n) = v.parse::<f64>() { cfg.train_pct = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_FORMATION_BARS") {
            if let Ok(n) = v.parse() { cfg.level_params.formation_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_SENSITIVITY_PCT") {
            if let Ok(n) = v.parse() { cfg.level_params.sensitivity_pct = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_MAX_DIST_ATR") {
            if let Ok(n) = v.parse() { cfg.level_params.max_distance_atr = n; }
        }

        cfg
    }

    pub fn effective_max_hold(&self) -> usize {
        if self.max_hold_bars > 0 { self.max_hold_bars } else { self.lookahead_bars }
    }

    pub fn target_pct_for_tf(&self, tf_minutes: i32) -> f64 {
        self.tf_targets
            .get(&tf_minutes)
            .copied()
            .unwrap_or_else(|| get_target_move_pct(tf_minutes).unwrap_or(2.0))
    }

    pub fn sl_pct_for_tf(&self, tf_minutes: i32) -> f64 {
        self.target_pct_for_tf(tf_minutes) * self.sl_fraction
    }

    pub fn timeframes() -> &'static [i32] {
        static TIMEFRAMES: OnceLock<Vec<i32>> = OnceLock::new();
        TIMEFRAMES.get_or_init(|| {
            let raw = std::env::var("SUPER_LEVEL_TIMEFRAMES")
                .unwrap_or_else(|_| "5,15,60,240,1440".to_string());
            let mut tfs: Vec<i32> = raw
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .filter(|&m| [1, 5, 15, 60, 240, 1440].contains(&m))
                .collect();
            tfs.sort_unstable();
            tfs.dedup();
            if tfs.is_empty() {
                tfs = vec![5, 15, 60, 240, 1440];
            }
            tracing::info!("SuperLevel timeframes: {:?}", tfs);
            tfs
        })
    }

    pub fn all_timeframes() -> &'static [i32] {
        &[1, 5, 15, 60, 240, 1440]
    }

    pub fn model_path(&self, template: &str, tf_minutes: i32) -> String {
        template.replace("{tf}", &tf_minutes.to_string())
    }
}

// ═══════════════════════════════════════════════════════════════
// FEATURE NAMES (синхронизированы с Python trainer)
// ═══════════════════════════════════════════════════════════════

/// Индикаторные фичи (33) — такие же как в ml_entry_strategy
pub const INDICATOR_FEATURES: &[&str] = &[
    "rsi", "cci", "stoch_k", "stoch_d", "williams",
    "macd", "macd_signal", "macd_hist",
    "adx", "sma", "ema_20", "ema_50", "ema_200",
    "bb_upper", "bb_mid", "bb_lower", "atr",
    "obv", "vwap", "volume_spike",
    "trend", "trend_short", "poc",
    "alligator_jaw", "alligator_teeth", "alligator_lips",
    "mfi", "fibo_pivot", "fibo_r1", "fibo_s1",
    "supertrend", "supertrend_dir", "cmf",
];

/// Производные фичи (19) — как в ml_entry_strategy
pub const DERIVED_FEATURES: &[&str] = &[
    "rsi_norm", "cci_norm", "stoch_norm", "williams_norm",
    "bb_position", "bb_width_pct", "atr_pct",
    "price_vs_sma", "price_vs_ema20", "price_vs_ema50",
    "price_vs_ema200", "price_vs_vwap",
    "macd_norm", "obv_change_pct", "volume_spike_flag",
    "mfi_norm", "price_vs_fibo_pivot", "price_vs_supertrend",
    "alligator_spread",
];

/// Фичи уровней (вычисляются для ближайшего уровня) — ГЛАВНЫЕ для стратегии
pub const LEVEL_FEATURES: &[&str] = &[
    // Ближайший уровень поддержки
    "nearest_sup_dist_atr",       // расстояние до ближайшей поддержки в ATR
    "nearest_sup_dist_pct",       // расстояние до ближайшей поддержки в %
    "nearest_sup_touches",        // количество касаний ближайшей поддержки
    "nearest_sup_strength",       // сила: 0=нет, 1=weak, 2=medium, 3=strong
    "nearest_sup_bars_since_touch", // баров с последнего касания
    "nearest_sup_age_bars",       // возраст уровня (баров с первого касания)
    // Ближайший уровень сопротивления
    "nearest_res_dist_atr",
    "nearest_res_dist_pct",
    "nearest_res_touches",
    "nearest_res_strength",
    "nearest_res_bars_since_touch",
    "nearest_res_age_bars",
    // Общие
    "in_level_zone",              // 1.0 если цена в зоне касания (любого уровня)
    "channel_width_atr",          // ширина канала sup-res в ATR
    "channel_position",           // позиция в канале 0=support, 1=resistance
    "total_levels_nearby",        // количество уровней в пределах 2 ATR
    "approach_velocity_3",        // скорость приближения к уровню за 3 свечи
    "approach_velocity_5",        // скорость приближения за 5 свечей
];

/// Динамические фичи (такой же lookback как в super_entry, но сокращённый)
pub const DYNAMIC_LOOKBACK_WINDOWS: &[usize] = &[3, 5, 10, 15, 25];

pub const DYNAMIC_FEATURES: &[&str] = &[
    // Window 3
    "price_return_lb3", "atr_ratio_lb3", "rsi_slope_lb3",
    "trend_persist_lb3", "adx_slope_lb3", "macd_hist_slope_lb3",
    // Window 5
    "price_return_lb5", "atr_ratio_lb5", "rsi_slope_lb5",
    "trend_persist_lb5", "adx_slope_lb5", "macd_hist_slope_lb5",
    // Window 10
    "price_return_lb10", "atr_ratio_lb10", "rsi_slope_lb10",
    "trend_persist_lb10", "adx_slope_lb10", "macd_hist_slope_lb10",
    // Window 15
    "price_return_lb15", "atr_ratio_lb15", "rsi_slope_lb15",
    "trend_persist_lb15", "adx_slope_lb15", "macd_hist_slope_lb15",
    // Window 25
    "price_return_lb25", "atr_ratio_lb25", "rsi_slope_lb25",
    "trend_persist_lb25", "adx_slope_lb25", "macd_hist_slope_lb25",
    // Aggregate
    "supertrend_consistency", "trend_alignment",
    "price_accel", "volume_trend_ratio",
];

/// All feature names (indicators + derived + level + dynamic)
pub fn all_feature_names() -> Vec<&'static str> {
    let mut names: Vec<&str> = INDICATOR_FEATURES.to_vec();
    names.extend_from_slice(DERIVED_FEATURES);
    names.extend_from_slice(LEVEL_FEATURES);
    names.extend_from_slice(DYNAMIC_FEATURES);
    names
}

pub fn total_feature_count() -> usize {
    INDICATOR_FEATURES.len() + DERIVED_FEATURES.len() + LEVEL_FEATURES.len() + DYNAMIC_FEATURES.len()
}

pub fn static_feature_count() -> usize {
    INDICATOR_FEATURES.len() + DERIVED_FEATURES.len()
}

pub fn level_feature_count() -> usize {
    LEVEL_FEATURES.len()
}

pub fn dynamic_feature_count() -> usize {
    DYNAMIC_FEATURES.len()
}

pub fn max_dynamic_lookback() -> usize {
    *DYNAMIC_LOOKBACK_WINDOWS.last().unwrap_or(&15)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SuperLevelConfig::default();
        assert_eq!(cfg.warmup_bars, 500);
        assert_eq!(cfg.lookahead_bars, 25);
        assert!((cfg.p_threshold - 0.55).abs() < 1e-6);
        assert_eq!(cfg.max_hold_bars, 25);
    }

    #[test]
    fn test_feature_count() {
        assert_eq!(INDICATOR_FEATURES.len(), 33);
        assert_eq!(DERIVED_FEATURES.len(), 19);
        assert_eq!(LEVEL_FEATURES.len(), 18);
        assert_eq!(DYNAMIC_FEATURES.len(), 34); // 6 × 5 windows + 4 aggregate
        assert_eq!(total_feature_count(), 104);
    }

    #[test]
    fn test_target_pct() {
        let cfg = SuperLevelConfig::default();
        assert!((cfg.target_pct_for_tf(60) - 5.0).abs() < 1e-6);
        assert!((cfg.sl_pct_for_tf(60) - 3.25).abs() < 1e-6);
    }
}
