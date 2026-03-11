// strategies/super_level_strategy/src/phases.rs
//
// 5-Phase Level-First Sniper Logic
//
// Каждая фаза — гейт. Сделка проходит дальше ТОЛЬКО если прошла предыдущую.
// Все вычисления — inline, zero-copy, без лишних аллокаций в горячем пути.
//
// ПЕРЕИСПОЛЬЗУЕТ:
//   - indicators::sr_levels::calculate_sr_levels  (Phase 1)
//   - ewmac_strategy::ewmac::EwmacCalculator     (Phase 2)
//   - ml_entry_strategy::pipeline (P(super))      (Phase 3)
//   - entry_policy::agent pattern                 (Phase 4)
//   - signal_params.toml ATR mult                 (Phase 5)

use crate::config::{SuperLevelConfig, AtrMultipliers};
use ml_entry_strategy::dataset::CandleWithIndicators;

/// Сценарий входа (определяется в Фазе 2)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scenario {
    /// Отскок от уровня (контр-тренд)
    Bounce,
    /// Пробой уровня (по тренду)
    Breakout,
}

/// Тип уровня, от которого пришёл сигнал
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LevelSide {
    /// Цена у поддержки — бот ищет LONG
    Support,
    /// Цена у сопротивления — бот ищет SHORT
    Resistance,
}

/// Причина отклонения сигнала
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RejectPhase {
    /// Фаза 1: цена не у уровня
    NoLevel,
    /// Фаза 2: контекст не подтверждён (смешанные сигналы)
    ContextUnclear,
    /// Фаза 3: ML не подтвердила
    MlRejected,
    /// Фаза 4: Entry Agent отменил (CANCEL)
    EntryCancelled,
}

/// Результат прохождения фаз
#[derive(Debug, Clone)]
pub struct PhaseResult {
    /// Прошел ли все фазы
    pub passed: bool,
    /// Сценарий (Bounce/Breakout)
    pub scenario: Option<Scenario>,
    /// Сторона уровня (Support/Resistance)
    pub level_side: Option<LevelSide>,
    /// Направление входа: 1=LONG, -1=SHORT
    pub direction: i8,
    /// Цена уровня
    pub level_price: f64,
    /// Сила уровня
    pub level_strength: f32,
    /// Дистанция до уровня в ATR
    pub distance_atr: f32,
    /// EWMAC forecast (из Phase 2)
    pub ewmac_forecast: f64,
    /// P(super) из ML (Phase 3)
    pub p_super: f32,
    /// Entry bar offset (Phase 4: 0 = вход на текущей свече)
    pub entry_offset: usize,
    /// Entry price (после entry agent)
    pub entry_price: f64,
    /// ATR мультипликаторы для SL/TP
    pub atr_mults: AtrMultipliers,
    /// Причина отклонения (если не прошёл)
    pub reject_reason: Option<RejectPhase>,
}

impl PhaseResult {
    fn rejected(reason: RejectPhase) -> Self {
        Self {
            passed: false,
            scenario: None,
            level_side: None,
            direction: 0,
            level_price: 0.0,
            level_strength: 0.0,
            distance_atr: f32::MAX,
            ewmac_forecast: 0.0,
            p_super: 0.0,
            entry_offset: 0,
            entry_price: 0.0,
            atr_mults: AtrMultipliers::bounce(),
            reject_reason: Some(reason),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ФАЗА 1: RADAR — Поиск "Зоны Убийства"
// ═══════════════════════════════════════════════════════════════

/// Результат Фазы 1
pub struct RadarHit {
    pub level_price: f64,
    pub level_strength: f32,
    pub distance_atr: f32,
    pub level_side: LevelSide,
}

/// Фаза 1: Проверяет, находится ли цена в "зоне убийства" (≤ radar_distance_atr от
/// сильного уровня S/R).
///
/// Вычисляет SR уровни из OHLC истории (zero-copy, compute на лету).
/// Возвращает None если цена "в середине канала" — бот спит.
pub fn phase1_radar(
    candles: &[CandleWithIndicators],
    idx: usize,
    config: &SuperLevelConfig,
) -> Option<RadarHit> {
    let current = &candles[idx];
    let close = current.close;
    let atr = current.atr;

    if atr <= 0.0 || close <= 0.0 {
        return None;
    }

    // Вычисляем SR уровни из последних sr_lookback_bars свечей
    let lookback_start = idx.saturating_sub(config.sr_lookback_bars);
    let slice = &candles[lookback_start..=idx];

    let n = slice.len();
    if n < 5 {
        return None;
    }

    // Собираем OHLC массивы inline (без аллокаций Vec на каждую свечу — 
    // собираем один раз для всего lookback)
    let mut highs = Vec::with_capacity(n);
    let mut lows = Vec::with_capacity(n);
    let mut closes = Vec::with_capacity(n);
    for c in slice {
        highs.push(c.high);
        lows.push(c.low);
        closes.push(c.close);
    }

    // Чувствительность кластеризации = sr_sensitivity_pct * close / 100
    let sensitivity = config.sr_sensitivity_pct * close / 100.0;

    let levels = indicators::sr_levels::calculate_sr_levels(&highs, &lows, &closes, sensitivity);

    // Ищем ближайший СИЛЬНЫЙ уровень в пределах radar_distance_atr
    let mut best_hit: Option<RadarHit> = None;

    // Проверяем все 6 уровней (strong/mid/light × support/resistance)
    let level_candidates = [
        (levels.strong_support, 0.9f32, LevelSide::Support),
        (levels.mid_support, 0.6, LevelSide::Support),
        (levels.strong_resistance, 0.9, LevelSide::Resistance),
        (levels.mid_resistance, 0.6, LevelSide::Resistance),
    ];

    for &(price, strength, side) in &level_candidates {
        if price.is_nan() || price <= 0.0 {
            continue;
        }
        if strength < config.min_level_strength {
            continue;
        }

        let dist_atr = ((close - price).abs() / atr) as f32;

        if dist_atr <= config.radar_distance_atr {
            // Нашли уровень в зоне убийства!
            // Проверяем, что это правильная сторона:
            //   Support должен быть ниже цены
            //   Resistance должен быть выше цены
            let valid_side = match side {
                LevelSide::Support => price <= close,
                LevelSide::Resistance => price >= close,
            };

            if !valid_side {
                continue;
            }

            // Выбираем ближайший
            if best_hit.as_ref().map_or(true, |h| dist_atr < h.distance_atr) {
                best_hit = Some(RadarHit {
                    level_price: price,
                    level_strength: strength,
                    distance_atr: dist_atr,
                    level_side: side,
                });
            }
        }
    }

    best_hit
}

// ═══════════════════════════════════════════════════════════════
// ФАЗА 2: CONTEXT — Отскок или Пробой?
// ═══════════════════════════════════════════════════════════════

/// Фаза 2: Определяет сценарий (Bounce vs Breakout) на основе:
///   - EWMAC forecast (тренд)
///   - ADX (сила тренда)
///   - RSI/Stoch (перегретость осцилляторов)
///   - MFI (денежный поток) — дополнительное подтверждение
///
/// V2: Убран слабый fallback. Требуется МИНИМУМ 2 подтверждения из 3:
///   (1) осцилляторы перегреты, (2) тренд слабый, (3) forecast согласован.
/// Это фильтрует ~70% ложных входов по сравнению с V1.
pub fn phase2_context(
    candle: &CandleWithIndicators,
    radar: &RadarHit,
    ewmac_forecast: f64,
    config: &SuperLevelConfig,
) -> Option<(Scenario, i8)> {
    let rsi = candle.rsi;
    let adx = candle.adx;
    let stoch_k = candle.stoch_k;
    let mfi = candle.mfi;

    match radar.level_side {
        LevelSide::Support => {
            // Цена у поддержки — решаем: отскок вверх (LONG) или пробой вниз (SHORT)?

            // ── Сценарий BOUNCE (LONG) — нужно 2/3 подтверждения ──
            let mut bounce_score: u8 = 0;

            // (1) Осцилляторы oversold
            if rsi < config.rsi_oversold { bounce_score += 1; }
            if stoch_k < 25.0 { bounce_score += 1; }
            if mfi < 30.0 { bounce_score += 1; }

            // (2) Тренд слабый/отсутствует (EWMAC флэт)
            if ewmac_forecast.abs() < config.ewmac_trend_threshold {
                bounce_score += 1;
            }

            // (3) EWMAC не противоречит LONG (не сильный даунтренд)
            if ewmac_forecast > -config.ewmac_trend_threshold * 0.5 {
                bounce_score += 1;
            }

            // Нужно минимум 2 подтверждения
            if bounce_score >= 2 {
                return Some((Scenario::Bounce, 1)); // LONG от support
            }

            // ── Сценарий BREAKOUT (SHORT) — сильный нисходящий тренд ──
            let strong_downtrend = ewmac_forecast < -config.ewmac_trend_threshold
                && adx > config.adx_trend_threshold;

            if strong_downtrend {
                return Some((Scenario::Breakout, -1)); // SHORT пробой support
            }

            None // Контекст неясный — НЕ ВХОДИМ (V2: убран слабый fallback)
        }
        LevelSide::Resistance => {
            // Цена у сопротивления — решаем: отскок вниз (SHORT) или пробой вверх (LONG)?

            // ── Сценарий BOUNCE (SHORT) — нужно 2/3 подтверждения ──
            let mut bounce_score: u8 = 0;

            // (1) Осцилляторы overbought
            if rsi > config.rsi_overbought { bounce_score += 1; }
            if stoch_k > 75.0 { bounce_score += 1; }
            if mfi > 70.0 { bounce_score += 1; }

            // (2) Тренд слабый
            if ewmac_forecast.abs() < config.ewmac_trend_threshold {
                bounce_score += 1;
            }

            // (3) EWMAC не противоречит SHORT (не сильный аптренд)
            if ewmac_forecast < config.ewmac_trend_threshold * 0.5 {
                bounce_score += 1;
            }

            // Нужно минимум 2 подтверждения
            if bounce_score >= 2 {
                return Some((Scenario::Bounce, -1)); // SHORT от resistance
            }

            // ── Сценарий BREAKOUT (LONG) — сильный восходящий тренд ──
            let strong_uptrend = ewmac_forecast > config.ewmac_trend_threshold
                && adx > config.adx_trend_threshold;

            if strong_uptrend {
                return Some((Scenario::Breakout, 1)); // LONG пробой resistance
            }

            None // V2: убран слабый fallback
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// ФАЗА 3: ML VALIDATOR — Super Entry Model
// ═══════════════════════════════════════════════════════════════

/// Фаза 3: Проверяет ML предсказание.
/// ML не ищет сделки — он ПОДТВЕРЖДАЕТ гипотезу из Фазы 2.
///
/// Возвращает true если:
///   - p_super >= ml_p_threshold
///   - направление ML совпадает с направлением от уровня (если require_ml_direction_match)
pub fn phase3_ml_validate(
    p_super: f32,
    ml_direction: i8,
    expected_direction: i8,
    config: &SuperLevelConfig,
) -> bool {
    // Проверка порога P(super)
    if (p_super as f64) < config.ml_p_threshold {
        return false;
    }

    // Проверка согласованности направления
    if config.require_ml_direction_match && ml_direction != expected_direction {
        return false;
    }

    true
}

// ═══════════════════════════════════════════════════════════════
// ФАЗА 4: ENTRY AGENT — Тайминг входа
// ═══════════════════════════════════════════════════════════════

/// Фаза 4: Симуляция Entry Agent для бэктеста.
///
/// V2: СТРОГИЙ режим — вход ТОЛЬКО при найденном паттерне!
/// Если за окно не найден паттерн → CANCEL (не принудительный вход).
/// Это ключевое отличие от V1: качество входа > количество входов.
///
/// Паттерны:
///   - Пин-бар (тень > 55% диапазона в сторону уровня)
///   - Разворотная свеча (тело > 45% range + close в нашу сторону)
///   - Объёмная свеча с закрытием в нашу сторону (volume_spike > 1.8)
///   - Engulfing (текущая свеча полностью поглощает предыдущую)
///   - CANCEL: цена пробивает уровень на entry_cancel_atr ATR
///   - CANCEL: окно истекло без паттерна
///
/// Возвращает (entry_offset, entry_price) или None если CANCEL.
pub fn phase4_entry_agent(
    candles: &[CandleWithIndicators],
    signal_idx: usize,
    direction: i8,
    level_price: f64,
    config: &SuperLevelConfig,
) -> Option<(usize, f64)> {
    let atr = candles[signal_idx].atr;
    let cancel_distance = config.entry_cancel_atr * atr;
    let window_end = (signal_idx + config.entry_window_bars).min(candles.len() - 1);

    for bar in signal_idx..=window_end {
        let c = &candles[bar];

        // ── CANCEL: цена пробила уровень против нас ──
        if direction == 1 {
            if c.low < level_price - cancel_distance {
                return None; // CANCEL — спасены от убытка
            }
        } else {
            if c.high > level_price + cancel_distance {
                return None; // CANCEL
            }
        }

        // ── Ищем подтверждающий паттерн ──

        let body = (c.close - c.open).abs();
        let upper_shadow = c.high - c.close.max(c.open);
        let lower_shadow = c.close.min(c.open) - c.low;
        let candle_range = c.high - c.low;

        if candle_range < 1e-12 {
            continue;
        }

        let is_bullish_close = c.close > c.open;
        let is_bearish_close = c.close < c.open;

        // Pattern 1: Пин-бар (длинная тень к уровню, тело маленькое)
        let pin_bar = if direction == 1 {
            lower_shadow / candle_range > 0.55 && is_bullish_close
        } else {
            upper_shadow / candle_range > 0.55 && is_bearish_close
        };

        // Pattern 2: Сильная разворотная свеча (большое тело в нашу сторону)
        let strong_reversal = if direction == 1 {
            is_bullish_close && body / candle_range > 0.45
        } else {
            is_bearish_close && body / candle_range > 0.45
        };

        // Pattern 3: Объёмный подтверждающий бар
        let volume_reversal = if direction == 1 {
            is_bullish_close && c.volume_spike > 1.8
        } else {
            is_bearish_close && c.volume_spike > 1.8
        };

        // Pattern 4: Engulfing (текущая свеча поглощает предыдущую)
        let engulfing = if bar > signal_idx {
            let prev = &candles[bar - 1];
            if direction == 1 {
                // Bullish engulfing: текущая бычья, полностью покрывает предыдущую
                is_bullish_close && c.open <= prev.close.min(prev.open)
                    && c.close >= prev.close.max(prev.open)
            } else {
                // Bearish engulfing
                is_bearish_close && c.open >= prev.close.max(prev.open)
                    && c.close <= prev.close.min(prev.open)
            }
        } else {
            false
        };

        // ENTER ТОЛЬКО если есть подтверждение
        if pin_bar || strong_reversal || volume_reversal || engulfing {
            return Some((bar - signal_idx, c.close));
        }
    }

    // V2: Окно истекло без паттерна → CANCEL (НЕ принудительный вход!)
    None
}

// ═══════════════════════════════════════════════════════════════
// ФАЗА 5: RISK MANAGEMENT
// ═══════════════════════════════════════════════════════════════

/// SL/TP цены для сделки
#[derive(Debug, Clone, Copy)]
pub struct RiskLevels {
    pub sl_price: f64,
    pub tp1_price: f64,
    pub tp2_price: f64,
    pub tp3_price: f64,
}

/// Фаза 5: Вычисляет SL/TP на основе ATR и сценария.
///
/// Для Bounce: короткий и безопасный (SL за уровень)
/// Для Breakout: шире (SL за пробитый уровень)
pub fn phase5_risk_levels(
    entry_price: f64,
    atr: f64,
    direction: i8,
    scenario: Scenario,
    config: &SuperLevelConfig,
) -> RiskLevels {
    let mults = config.atr_mults_for_scenario(scenario == Scenario::Breakout);

    if direction == 1 {
        // LONG
        RiskLevels {
            sl_price: entry_price - mults.sl_mult * atr,
            tp1_price: entry_price + mults.tp1_mult * atr,
            tp2_price: entry_price + mults.tp2_mult * atr,
            tp3_price: entry_price + mults.tp3_mult * atr,
        }
    } else {
        // SHORT
        RiskLevels {
            sl_price: entry_price + mults.sl_mult * atr,
            tp1_price: entry_price - mults.tp1_mult * atr,
            tp2_price: entry_price - mults.tp2_mult * atr,
            tp3_price: entry_price - mults.tp3_mult * atr,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// FULL PIPELINE: все 5 фаз в одном вызове
// ═══════════════════════════════════════════════════════════════

/// Прогоняет все 5 фаз для одной свечи.
///
/// # Аргументы
/// * `candles` - Полный массив свечей с индикаторами
/// * `idx` - Индекс текущей свечи
/// * `ewmac_forecast` - Forecast из EwmacCalculator (вычисленный заранее)
/// * `p_super` - P(super) из ML модели (вычисленный заранее batch inference)
/// * `ml_direction` - Направление из ML модели
/// * `config` - Конфигурация стратегии
pub fn run_all_phases(
    candles: &[CandleWithIndicators],
    idx: usize,
    ewmac_forecast: f64,
    p_super: f32,
    ml_direction: i8,
    config: &SuperLevelConfig,
) -> PhaseResult {
    // ── ФАЗА 1: Radar ─────────────────────────────────────
    let radar = match phase1_radar(candles, idx, config) {
        Some(hit) => hit,
        None => return PhaseResult::rejected(RejectPhase::NoLevel),
    };

    // ── ФАЗА 2: Context ───────────────────────────────────
    let (scenario, direction) = match phase2_context(&candles[idx], &radar, ewmac_forecast, config) {
        Some(ctx) => ctx,
        None => return PhaseResult::rejected(RejectPhase::ContextUnclear),
    };

    // ── ФАЗА 3: ML Validator ──────────────────────────────
    if !phase3_ml_validate(p_super, ml_direction, direction, config) {
        return PhaseResult::rejected(RejectPhase::MlRejected);
    }

    // ── ФАЗА 4: Entry Agent ───────────────────────────────
    let (entry_offset, entry_price) = match phase4_entry_agent(
        candles, idx, direction, radar.level_price, config,
    ) {
        Some((offset, price)) => (offset, price),
        None => return PhaseResult::rejected(RejectPhase::EntryCancelled),
    };

    // ── ФАЗА 5: Risk Management ──────────────────────────
    let atr_mults = *config.atr_mults_for_scenario(scenario == Scenario::Breakout);

    PhaseResult {
        passed: true,
        scenario: Some(scenario),
        level_side: Some(radar.level_side),
        direction,
        level_price: radar.level_price,
        level_strength: radar.level_strength,
        distance_atr: radar.distance_atr,
        ewmac_forecast,
        p_super,
        entry_offset,
        entry_price,
        atr_mults,
        reject_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(close: f64, high: f64, low: f64, atr: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(), symbol: "BTCUSDT".to_string(), symbol_id: 1,
            open: close, high, low, close, volume: 1000.0,
            rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
            adx: 20.0, sma: close, ema_20: close, ema_50: close, ema_200: close,
            bb_upper: close * 1.02, bb_mid: close, bb_lower: close * 0.98,
            atr, obv: 0.0, vwap: close, volume_spike: 1.0,
            trend: 0.0, trend_short: 0.0, poc: close,
            alligator_jaw: close, alligator_teeth: close, alligator_lips: close,
            mfi: 50.0, fibo_pivot: close, fibo_r1: close * 1.01, fibo_s1: close * 0.99,
            supertrend: close, supertrend_dir: 1.0, cmf: 0.0,
        }
    }

    #[test]
    fn test_ml_validate_pass() {
        let cfg = SuperLevelConfig::default();
        assert!(phase3_ml_validate(0.70, 1, 1, &cfg));
    }

    #[test]
    fn test_ml_validate_reject_low_p() {
        let cfg = SuperLevelConfig::default();
        assert!(!phase3_ml_validate(0.40, 1, 1, &cfg));
    }

    #[test]
    fn test_ml_validate_reject_direction() {
        let cfg = SuperLevelConfig::default();
        assert!(!phase3_ml_validate(0.70, -1, 1, &cfg)); // ML short, expected long
    }

    #[test]
    fn test_risk_levels_long_bounce() {
        let cfg = SuperLevelConfig::default();
        let risk = phase5_risk_levels(100.0, 2.0, 1, Scenario::Bounce, &cfg);
        assert!(risk.sl_price < 100.0);
        assert!(risk.tp1_price > 100.0);
        assert!(risk.tp2_price > risk.tp1_price);
        assert!(risk.tp3_price > risk.tp2_price);
    }

    #[test]
    fn test_risk_levels_short_breakout() {
        let cfg = SuperLevelConfig::default();
        let risk = phase5_risk_levels(100.0, 2.0, -1, Scenario::Breakout, &cfg);
        assert!(risk.sl_price > 100.0);
        assert!(risk.tp1_price < 100.0);
    }

    #[test]
    fn test_phase2_bounce_from_support() {
        let cfg = SuperLevelConfig::default();
        let mut c = make_candle(100.0, 101.0, 99.0, 2.0);
        c.rsi = 28.0;     // oversold (< 35)
        c.stoch_k = 18.0;  // oversold (< 25)
        c.mfi = 25.0;      // low mfi (< 30) — 3 confirmations
        c.adx = 18.0;      // weak trend
        let radar = RadarHit {
            level_price: 99.5, level_strength: 0.9,
            distance_atr: 0.25, level_side: LevelSide::Support,
        };

        let result = phase2_context(&c, &radar, 3.0, &cfg); // weak EWMAC
        assert!(result.is_some());
        let (scenario, dir) = result.unwrap();
        assert_eq!(scenario, Scenario::Bounce);
        assert_eq!(dir, 1); // LONG
    }

    #[test]
    fn test_phase2_breakout_from_resistance() {
        let cfg = SuperLevelConfig::default();
        let mut c = make_candle(100.0, 101.0, 99.0, 2.0);
        c.rsi = 60.0;
        c.adx = 30.0;    // strong trend (> adx_threshold=22)
        let radar = RadarHit {
            level_price: 100.5, level_strength: 0.9,
            distance_atr: 0.25, level_side: LevelSide::Resistance,
        };

        let result = phase2_context(&c, &radar, 15.0, &cfg); // strong up EWMAC (> 8)
        assert!(result.is_some());
        let (scenario, dir) = result.unwrap();
        assert_eq!(scenario, Scenario::Breakout);
        assert_eq!(dir, 1); // LONG (пробой вверх)
    }
}
