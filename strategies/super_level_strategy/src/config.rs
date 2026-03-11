// strategies/super_level_strategy/src/config.rs
//
// Configuration for Super Level Strategy
//
// Основана на SuperEntryConfig + параметры уровней из signal_params.toml
// Все пороги настраиваются через env-переменные (SUPER_LEVEL_*)

use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

/// ATR-множители для SL/TP в зависимости от сценария
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AtrMultipliers {
    pub sl_mult: f64,
    pub tp1_mult: f64,
    pub tp2_mult: f64,
    pub tp3_mult: f64,
}

impl AtrMultipliers {
    /// Bounce — короткий и безопасный (из signal_params.toml [atr_bounce])
    pub fn bounce() -> Self {
        Self {
            sl_mult: 0.55,
            tp1_mult: 0.75,
            tp2_mult: 1.4,
            tp3_mult: 2.2,
        }
    }

    /// Breakout — шире (из signal_params.toml [atr_breakout])
    pub fn breakout() -> Self {
        Self {
            sl_mult: 0.75,
            tp1_mult: 1.1,
            tp2_mult: 1.8,
            tp3_mult: 2.8,
        }
    }
}

/// Полная конфигурация Super Level Strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperLevelConfig {
    // ── ФАЗА 1: Radar ─────────────────────────────────────────
    /// Максимальная дистанция до уровня в ATR для активации (≤ 0.5 ATR)
    pub radar_distance_atr: f32,
    /// Только strong уровни (strength >= этого порога)
    pub min_level_strength: f32,
    /// Чувствительность кластеризации уровней (% от цены)
    pub sr_sensitivity_pct: f64,
    /// Сколько свечей назад смотреть для вычисления SR уровней
    pub sr_lookback_bars: usize,

    // ── ФАЗА 2: Context ────────────────────────────────────────
    /// Порог EWMAC forecast для определения сильного тренда (пробой)
    pub ewmac_trend_threshold: f64,
    /// Порог ADX для определения сильного тренда
    pub adx_trend_threshold: f64,
    /// Порог RSI для overbought (отскок от resistance)
    pub rsi_overbought: f64,
    /// Порог RSI для oversold (отскок от support)
    pub rsi_oversold: f64,
    /// Minm approach_velocity для подтверждения пробоя
    pub min_approach_velocity: f32,

    // ── ФАЗА 3: ML Validator ───────────────────────────────────
    /// Минимальный P(super) для подтверждения входа
    pub ml_p_threshold: f64,
    /// Требуемая согласованность направления ML с направлением от уровня
    pub require_ml_direction_match: bool,

    // ── ФАЗА 4: Entry Agent (симуляция) ────────────────────────
    /// Окно ожидания (количество свечей для Entry Agent)
    pub entry_window_bars: usize,
    /// Порог отмены входа (если цена пробивает уровень на entry_cancel_atr ATR)
    pub entry_cancel_atr: f64,

    // ── ФАЗА 5: Risk Management ───────────────────────────────
    /// ATR-мультипликаторы для Bounce
    pub bounce_atr: AtrMultipliers,
    /// ATR-мультипликаторы для Breakout
    pub breakout_atr: AtrMultipliers,
    /// % позиции для частичного закрытия на TP1
    pub partial_close_pct: f64,
    /// Перевести SL в безубыток после TP1
    pub trail_sl_to_breakeven: bool,

    // ── Общие ──────────────────────────────────────────────────
    /// Warmup свечей (300 — для индикаторов)
    pub warmup_bars: usize,
    /// Max hold bars перед принудительным закрытием
    pub max_hold_bars: usize,
}

impl Default for SuperLevelConfig {
    fn default() -> Self {
        Self {
            // Phase 1: Radar — УЖЕСТОЧЕНО: 0.3 ATR, только самые сильные уровни
            radar_distance_atr: 0.35,     // V2: 0.5 → 0.3 (ближе к уровню = сильнее реакция)
            min_level_strength: 0.86,    // V2: 0.7 → 0.85 (только strong/strong_like уровни)
            sr_sensitivity_pct: 0.3,     // 0.3% кластеризация
            sr_lookback_bars: 500,       // 100 свечей назад для SR

            // Phase 2: Context — РАСШИРЕНО для 5m + ужесточено для высоких TF
            ewmac_trend_threshold: 8.0,  // V2: 10 → 8 (больше свечей считаются "в тренде")
            adx_trend_threshold: 22.0,   // V2: 25 → 22 (чуть легче определить тренд)
            rsi_overbought: 60.0,        // V2: 75 → 65 (шире полоса для bounce)
            rsi_oversold: 30.0,          // V2: 25 → 35 (шире полоса для bounce)
            min_approach_velocity: 0.3,

            // Phase 3: ML Validator — УЖЕСТОЧЕНО: требуем более уверенный ML
            ml_p_threshold: 0.65,        // V2: 0.55 → 0.62 (отсекаем слабые ML сигналы)
            require_ml_direction_match: true,

            // Phase 4: Entry Agent — входим ТОЛЬКО при паттерне
            entry_window_bars: 4,        // V2: 5 → 4 (быстрее решаем)
            entry_cancel_atr: 0.8,       // V2: 1.0 → 0.8 (быстрее cancel)

            // Phase 5: Risk Management
            bounce_atr: AtrMultipliers::bounce(),
            breakout_atr: AtrMultipliers::breakout(),
            partial_close_pct: 50.0,
            trail_sl_to_breakeven: true,

            // General
            warmup_bars: 300,
            max_hold_bars: 25,
        }
    }
}

impl SuperLevelConfig {
    /// Load config from environment variables with defaults
    pub fn from_env() -> Self {
        let mut cfg = Self::default();

        if let Ok(v) = std::env::var("SUPER_LEVEL_RADAR_ATR") {
            if let Ok(n) = v.parse() { cfg.radar_distance_atr = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_MIN_STRENGTH") {
            if let Ok(n) = v.parse() { cfg.min_level_strength = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_SR_LOOKBACK") {
            if let Ok(n) = v.parse() { cfg.sr_lookback_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_EWMAC_THRESHOLD") {
            if let Ok(n) = v.parse() { cfg.ewmac_trend_threshold = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_ADX_THRESHOLD") {
            if let Ok(n) = v.parse() { cfg.adx_trend_threshold = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_ML_THRESHOLD") {
            if let Ok(n) = v.parse() { cfg.ml_p_threshold = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_ENTRY_WINDOW") {
            if let Ok(n) = v.parse() { cfg.entry_window_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_MAX_HOLD") {
            if let Ok(n) = v.parse() { cfg.max_hold_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_WARMUP") {
            if let Ok(n) = v.parse() { cfg.warmup_bars = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_PARTIAL_CLOSE") {
            if let Ok(n) = v.parse() { cfg.partial_close_pct = n; }
        }
        if let Ok(v) = std::env::var("SUPER_LEVEL_REQUIRE_DIR_MATCH") {
            cfg.require_ml_direction_match = v == "true" || v == "1";
        }

        cfg
    }

    /// Timeframes for SuperLevel (inherited from SuperEntry)
    pub fn timeframes() -> &'static [i32] {
        static TIMEFRAMES: OnceLock<Vec<i32>> = OnceLock::new();
        TIMEFRAMES.get_or_init(|| {
            let raw = std::env::var("SUPER_LEVEL_TIMEFRAMES")
                .unwrap_or_else(|_| "5,15,60,240".to_string());
            let mut tfs: Vec<i32> = raw
                .split(',')
                .filter_map(|s| s.trim().parse::<i32>().ok())
                .filter(|&m| [1, 5, 15, 60, 240, 1440].contains(&m))
                .collect();
            tfs.sort_unstable();
            tfs.dedup();
            if tfs.is_empty() {
                tfs = vec![5, 15, 60, 240];
            }
            tracing::info!("SuperLevel timeframes: {:?}", tfs);
            tfs
        })
    }

    /// Get ATR multipliers for a given scenario
    pub fn atr_mults_for_scenario(&self, is_breakout: bool) -> &AtrMultipliers {
        if is_breakout {
            &self.breakout_atr
        } else {
            &self.bounce_atr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = SuperLevelConfig::default();
        assert!((cfg.radar_distance_atr - 0.3).abs() < 1e-6);
        assert!((cfg.ml_p_threshold - 0.62).abs() < 1e-6);
        assert_eq!(cfg.warmup_bars, 300);
        assert_eq!(cfg.max_hold_bars, 20);
        assert!(cfg.trail_sl_to_breakeven);
    }

    #[test]
    fn test_atr_bounce_multipliers() {
        let m = AtrMultipliers::bounce();
        assert!((m.sl_mult - 0.75).abs() < 1e-6);
        assert!((m.tp1_mult - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_atr_breakout_multipliers() {
        let m = AtrMultipliers::breakout();
        assert!((m.sl_mult - 0.75).abs() < 1e-6);
        assert!((m.tp1_mult - 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_scenario_selection() {
        let cfg = SuperLevelConfig::default();
        let bounce = cfg.atr_mults_for_scenario(false);
        assert!((bounce.sl_mult - 0.75).abs() < 1e-6);
        let breakout = cfg.atr_mults_for_scenario(true);
        assert!((breakout.sl_mult - 0.75).abs() < 1e-6);
    }
}
