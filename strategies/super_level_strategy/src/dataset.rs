// strategies/super_level_strategy/src/dataset.rs
//
// Dataset Builder for Super Level Strategy
//
// Ключевое отличие от ml_entry_strategy:
//   - Уровни вычисляются по реальным КАСАНИЯМ (touches), не swing-points
//   - Модель видит distance_to_level, touch_count, level_strength, etc.
//   - Уровни "помнят" историю (старый уровень может стать актуальным)
//   - 5 разных лейблов для 5 моделей
//
// CandleWithIndicators переиспользуется из ml_entry_strategy.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;

use ml_entry_strategy::dataset::CandleWithIndicators;
use crate::config::{
    LevelParams,
    DYNAMIC_LOOKBACK_WINDOWS,
};

// ═══════════════════════════════════════════════════════════════
// LEVEL COMPUTATION: настоящий подсчёт касаний
// ═══════════════════════════════════════════════════════════════

/// Сила уровня (по количеству касаний)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LevelStrength {
    Weak   = 1, // 1-2 касания
    Medium = 2, // 3-4 касания
    Strong = 3, // ≥5 касаний
}

/// Тип уровня
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LevelType {
    Support,
    Resistance,
}

/// Обнаруженный ценовой уровень с историей касаний
#[derive(Debug, Clone)]
pub struct PriceLevel {
    /// Цена центра кластера
    pub price: f64,
    /// Количество уникальных касаний
    pub touch_count: usize,
    /// Индекс первого касания (из candles)
    pub first_touch_idx: usize,
    /// Индекс последнего касания
    pub last_touch_idx: usize,
    /// Тип уровня
    pub level_type: LevelType,
    /// Сила по количеству касаний
    pub strength: LevelStrength,
    /// Все индексы касаний (для отладки)
    pub touch_indices: Vec<usize>,
}

/// Кластер для группировки ценовых точек
struct LevelCluster {
    center: f64,
    sum: f64,
    count: usize,
    touch_indices: Vec<usize>,
    is_support: bool,
    is_resistance: bool,
}

impl LevelCluster {
    fn new(price: f64, idx: usize, is_support: bool) -> Self {
        Self {
            center: price,
            sum: price,
            count: 1,
            touch_indices: vec![idx],
            is_support,
            is_resistance: !is_support,
        }
    }

    fn add(&mut self, price: f64, idx: usize, is_support: bool) {
        self.sum += price;
        self.count += 1;
        self.center = self.sum / self.count as f64;
        self.touch_indices.push(idx);
        if is_support { self.is_support = true; }
        if !is_support { self.is_resistance = true; }
    }
}

/// Вычисляет уровни по реальным касаниям на окне свечей [start..=end].
///
/// Алгоритм:
///   1. Проходим свечи в окне, фиксируем "касания":
///      - High подходит к потенциальному уровню сопротивления
///      - Low подходит к потенциальному уровню поддержки
///   2. Ищем свинг-хаи и свинг-лоу (pivot_depth=2)
///   3. Кластеризуем близкие pivot-точки (sensitivity_pct% от цены)
///   4. Для каждого кластера СЧИТАЕМ уникальные касания:
///      - Проходим ВСЕ свечи заново
///      - Если high или low попал в зону кластера — это touch
///      - Между тачами должно быть min_bars_between_touches
///   5. Присваиваем силу: Strong(≥5) / Medium(3-4) / Weak(1-2)
pub fn compute_levels(
    candles: &[CandleWithIndicators],
    start: usize,
    end: usize,
    params: &LevelParams,
) -> Vec<PriceLevel> {
    if end <= start + 5 || end >= candles.len() {
        return Vec::new();
    }

    let slice = &candles[start..=end];
    let n = slice.len();
    let ref_price = slice[n - 1].close;

    if ref_price <= 0.0 {
        return Vec::new();
    }

    // Step 1: Найти свинг-хаи и свинг-лоу
    let pivot_depth: usize = 2;
    let mut pivot_points: Vec<(f64, usize, bool)> = Vec::new(); // (price, global_idx, is_support)

    for i in pivot_depth..(n.saturating_sub(pivot_depth)) {
        let global_idx = start + i;

        // Swing high?
        let mut is_high = true;
        for d in 1..=pivot_depth {
            if slice[i].high <= slice[i - d].high || slice[i].high <= slice[i + d].high {
                is_high = false;
                break;
            }
        }
        if is_high {
            pivot_points.push((slice[i].high, global_idx, false)); // resistance
        }

        // Swing low?
        let mut is_low = true;
        for d in 1..=pivot_depth {
            if slice[i].low >= slice[i - d].low || slice[i].low >= slice[i + d].low {
                is_low = false;
                break;
            }
        }
        if is_low {
            pivot_points.push((slice[i].low, global_idx, true)); // support
        }
    }

    if pivot_points.is_empty() {
        return Vec::new();
    }

    // Step 2: Кластеризуем pivot-точки
    let sensitivity = params.sensitivity_pct / 100.0 * ref_price;
    let mut clusters: Vec<LevelCluster> = Vec::new();

    for &(price, idx, is_sup) in &pivot_points {
        let mut added = false;
        for cluster in &mut clusters {
            if (cluster.center - price).abs() <= sensitivity {
                cluster.add(price, idx, is_sup);
                added = true;
                break;
            }
        }
        if !added {
            clusters.push(LevelCluster::new(price, idx, is_sup));
        }
    }

    // Step 3: Для каждого кластера считаем РЕАЛЬНЫЕ касания
    let touch_zone = params.touch_zone_pct / 100.0;
    let min_gap = params.min_bars_between_touches;

    let mut levels: Vec<PriceLevel> = Vec::new();

    for cluster in &clusters {
        let level_price = cluster.center;
        let zone_low = level_price * (1.0 - touch_zone);
        let zone_high = level_price * (1.0 + touch_zone);

        // Подсчитываем уникальные касания: проход по всем свечам
        let mut touches: Vec<usize> = Vec::new();
        let mut last_touch_idx: Option<usize> = None;

        for i in 0..n {
            let c = &slice[i];
            let global_i = start + i;

            // Цена попала в зону уровня?
            let in_zone = c.low <= zone_high && c.high >= zone_low;

            if in_zone {
                // Проверяем минимальный gap между касаниями
                let is_new_touch = match last_touch_idx {
                    Some(prev) => (global_i - prev) >= min_gap,
                    None => true,
                };

                if is_new_touch {
                    touches.push(global_i);
                    last_touch_idx = Some(global_i);
                }
            }
        }

        if touches.is_empty() {
            continue;
        }

        let touch_count = touches.len();
        let strength = if touch_count >= params.strong_touches {
            LevelStrength::Strong
        } else if touch_count >= params.medium_touches {
            LevelStrength::Medium
        } else {
            LevelStrength::Weak
        };

        // Определяем тип: support если цена в основном сверху, resistance — снизу
        let level_type = if cluster.is_support && !cluster.is_resistance {
            LevelType::Support
        } else if cluster.is_resistance && !cluster.is_support {
            LevelType::Resistance
        } else {
            // Оба типа — определяем по текущей позиции цены
            if ref_price > level_price {
                LevelType::Support
            } else {
                LevelType::Resistance
            }
        };

        levels.push(PriceLevel {
            price: level_price,
            touch_count,
            first_touch_idx: *touches.first().unwrap(),
            last_touch_idx: *touches.last().unwrap(),
            level_type,
            strength,
            touch_indices: touches,
        });
    }

    // Сортируем по количеству касаний (сильнейшие первые)
    levels.sort_by(|a, b| b.touch_count.cmp(&a.touch_count));
    levels
}

// ═══════════════════════════════════════════════════════════════
// LEVEL FEATURES: фичи для ML на основе уровней
// ═══════════════════════════════════════════════════════════════

/// Фичи уровней для одной свечи. Порядок совпадает с LEVEL_FEATURES.
pub fn compute_level_features(
    candles: &[CandleWithIndicators],
    idx: usize,
    levels: &[PriceLevel],
) -> Vec<f64> {
    let n_feats = crate::config::level_feature_count();
    let c = &candles[idx];
    let close = c.close;
    let atr = c.atr;

    if close <= 0.0 || atr <= 0.0 || levels.is_empty() {
        return vec![0.0; n_feats];
    }

    // Найти ближайшую поддержку (ниже цены)
    let mut nearest_sup: Option<&PriceLevel> = None;
    let mut nearest_sup_dist = f64::MAX;

    // Найти ближайшее сопротивление (выше цены)
    let mut nearest_res: Option<&PriceLevel> = None;
    let mut nearest_res_dist = f64::MAX;

    for level in levels {
        let dist = (close - level.price).abs();
        match level.level_type {
            LevelType::Support => {
                if level.price <= close && dist < nearest_sup_dist {
                    nearest_sup_dist = dist;
                    nearest_sup = Some(level);
                }
            }
            LevelType::Resistance => {
                if level.price >= close && dist < nearest_res_dist {
                    nearest_res_dist = dist;
                    nearest_res = Some(level);
                }
            }
        }
    }

    // Также ищем ближайший по абсолютному расстоянию (без ограничения стороны)
    // на случай если цена выше всех support или ниже всех resistance
    if nearest_sup.is_none() {
        // Берём ближайший support по расстоянию (даже если выше цены)
        for level in levels.iter().filter(|l| l.level_type == LevelType::Support) {
            let dist = (close - level.price).abs();
            if dist < nearest_sup_dist {
                nearest_sup_dist = dist;
                nearest_sup = Some(level);
            }
        }
    }
    if nearest_res.is_none() {
        for level in levels.iter().filter(|l| l.level_type == LevelType::Resistance) {
            let dist = (close - level.price).abs();
            if dist < nearest_res_dist {
                nearest_res_dist = dist;
                nearest_res = Some(level);
            }
        }
    }

    let mut feats = Vec::with_capacity(n_feats);

    // ── Nearest Support features (6) ──
    if let Some(sup) = nearest_sup {
        feats.push((close - sup.price).abs() / atr);                    // dist_atr
        feats.push((close - sup.price).abs() / close * 100.0);          // dist_pct
        feats.push(sup.touch_count as f64);                              // touches
        feats.push(sup.strength as i32 as f64);                         // strength (1-3)
        feats.push((idx - sup.last_touch_idx) as f64);                  // bars_since_touch
        feats.push((idx - sup.first_touch_idx) as f64);                 // age_bars
    } else {
        feats.extend_from_slice(&[99.0, 99.0, 0.0, 0.0, 999.0, 999.0]);
    }

    // ── Nearest Resistance features (6) ──
    if let Some(res) = nearest_res {
        feats.push((res.price - close).abs() / atr);
        feats.push((res.price - close).abs() / close * 100.0);
        feats.push(res.touch_count as f64);
        feats.push(res.strength as i32 as f64);
        feats.push((idx - res.last_touch_idx) as f64);
        feats.push((idx - res.first_touch_idx) as f64);
    } else {
        feats.extend_from_slice(&[99.0, 99.0, 0.0, 0.0, 999.0, 999.0]);
    }

    // ── In level zone ──
    let in_zone = levels.iter().any(|l| {
        let zone = l.price * (0.12 / 100.0); // touch_zone_pct
        close >= l.price - zone && close <= l.price + zone
    });
    feats.push(if in_zone { 1.0 } else { 0.0 });

    // ── Channel width in ATR ──
    let channel_width = match (nearest_sup, nearest_res) {
        (Some(s), Some(r)) => (r.price - s.price).abs() / atr,
        _ => 99.0,
    };
    feats.push(channel_width);

    // ── Channel position (0=support, 1=resistance) ──
    let channel_pos = match (nearest_sup, nearest_res) {
        (Some(s), Some(r)) => {
            let range = r.price - s.price;
            if range.abs() > 1e-12 { (close - s.price) / range } else { 0.5 }
        }
        _ => 0.5,
    };
    feats.push(channel_pos.clamp(0.0, 1.0));

    // ── Total levels nearby (within 2 ATR) ──
    let nearby_count = levels.iter().filter(|l| {
        (close - l.price).abs() / atr <= 2.0
    }).count();
    feats.push(nearby_count as f64);

    // ── Approach velocity (3 and 5 bars) ──
    // Скорость приближения к ближайшему уровню
    let nearest_level_price = nearest_sup
        .or(nearest_res)
        .map(|l| l.price)
        .unwrap_or(close);

    let approach_3 = if idx >= 3 {
        let dist_now = (close - nearest_level_price).abs();
        let dist_3 = (candles[idx - 3].close - nearest_level_price).abs();
        if dist_3 > 1e-12 { (dist_3 - dist_now) / dist_3 } else { 0.0 }
    } else { 0.0 };
    feats.push(approach_3);

    let approach_5 = if idx >= 5 {
        let dist_now = (close - nearest_level_price).abs();
        let dist_5 = (candles[idx - 5].close - nearest_level_price).abs();
        if dist_5 > 1e-12 { (dist_5 - dist_now) / dist_5 } else { 0.0 }
    } else { 0.0 };
    feats.push(approach_5);

    debug_assert_eq!(feats.len(), n_feats,
        "Level feature count mismatch: expected {}, got {}", n_feats, feats.len());
    feats
}

// ═══════════════════════════════════════════════════════════════
// DYNAMIC FEATURES (сокращённая версия из super_entry)
// ═══════════════════════════════════════════════════════════════

pub fn compute_dynamic_features(candles: &[CandleWithIndicators], t: usize) -> Vec<f64> {
    let n_dynamic = crate::config::dynamic_feature_count();
    let max_lb = crate::config::max_dynamic_lookback();

    if t < max_lb || t >= candles.len() {
        return vec![0.0; n_dynamic];
    }

    let cur = &candles[t];
    let close = cur.close;
    let safe_div = |a: f64, b: f64| -> f64 {
        if b.abs() > 1e-12 { a / b } else { 0.0 }
    };

    let mut feats = Vec::with_capacity(n_dynamic);

    // Per-window features (6 features × 5 windows = 30)
    for &lb in DYNAMIC_LOOKBACK_WINDOWS {
        let prev = &candles[t - lb];
        feats.push(safe_div(close - prev.close, close) * 100.0);  // price_return
        feats.push(safe_div(cur.atr, prev.atr) - 1.0);           // atr_ratio
        feats.push((cur.rsi - prev.rsi) / 100.0);                 // rsi_slope
        let trend_sum: f64 = (0..lb).map(|j| candles[t - j].trend).sum();
        feats.push(trend_sum / lb as f64);                          // trend_persist
        feats.push((cur.adx - prev.adx) / 100.0);                 // adx_slope
        feats.push(safe_div(cur.macd_hist - prev.macd_hist, close) * 1000.0); // macd_hist_slope
    }

    // Aggregate features (4)
    let st_window = 25.min(t + 1);
    let st_sum: f64 = (0..st_window).map(|j| candles[t - j].supertrend_dir).sum();
    feats.push(st_sum / st_window as f64); // supertrend_consistency

    feats.push(cur.trend * cur.trend_short); // trend_alignment

    // Price acceleration
    let ret5 = if t >= 5 { safe_div(close - candles[t-5].close, close) * 100.0 } else { 0.0 };
    let ret5p = if t >= 10 {
        safe_div(candles[t-5].close - candles[t-10].close, candles[t-5].close) * 100.0
    } else { 0.0 };
    feats.push(ret5 - ret5p); // price_accel

    // Volume trend ratio
    let vol_recent: f64 = (0..5.min(t+1)).map(|j| candles[t-j].volume).sum::<f64>() / 5.0_f64.min((t+1) as f64);
    let vol_prev: f64 = if t >= 5 {
        (5..10.min(t+1)).map(|j| candles[t-j].volume).sum::<f64>() / 5.0_f64.min((t-4) as f64)
    } else { vol_recent };
    feats.push(safe_div(vol_recent, vol_prev)); // volume_trend_ratio

    debug_assert_eq!(feats.len(), n_dynamic,
        "Dynamic feature count mismatch: expected {}, got {}", n_dynamic, feats.len());
    feats
}

// ═══════════════════════════════════════════════════════════════
// LABELED EXAMPLE
// ═══════════════════════════════════════════════════════════════

/// Labeled training example for Super Level Strategy (5 лейблов)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperLevelExample {
    pub symbol: String,
    pub tf_minutes: i32,
    pub timestamp: String,
    pub features: Vec<f64>,

    // ── Label 1: Level Quality (для Level Model) ──────────────
    /// Сила ближайшего уровня (1=weak, 2=medium, 3=strong, 0=нет уровня)
    pub level_strength: i32,
    /// Количество касаний ближайшего уровня
    pub level_touches: i32,

    // ── Label 2: Entry Quality (для Entry Model) ──────────────
    /// Was TP hit? (binary: 0/1)
    pub is_good_entry: bool,

    // ── Label 3: Direction (для Direction Model) ──────────────
    /// 1 = LONG, -1 = SHORT
    pub direction: i8,

    // ── Label 4: Bounce/Break (для BounceBreak Model) ──────
    /// 1 = bounce (отскок от уровня), 0 = break (пробой)
    pub is_bounce: bool,

    // ── Label 5: Trade Outcome (для Evaluator Model) ──────────
    /// Final trade outcome: 1=win (TP hit), 0=loss/expired
    pub is_win: bool,

    // ── Metadata ──────────────────────────────────────────────
    pub max_up_move_pct: f64,
    pub max_down_move_pct: f64,
    pub magnitude_pct: f64,
    pub nearest_level_dist_atr: f64,
}

/// Build labels for Level Strategy with proper touch-based levels.
///
/// For each candle from start_idx to n-lookahead:
///   1. Compute levels from history [start..idx] (sliding window)
///   2. Find nearest level, compute features
///   3. Look ahead to determine labels (TP/SL hit, bounce/break, direction)
pub fn build_labels(
    candles: &[CandleWithIndicators],
    start_idx: usize,
    lookahead: usize,
    target_move_pct: f64,
    sl_fraction: f64,
    tf_minutes: i32,
    level_params: &LevelParams,
) -> Vec<SuperLevelExample> {
    let n = candles.len();
    if n < start_idx + lookahead + 1 {
        return Vec::new();
    }

    let end_idx = n - lookahead;
    let mut examples = Vec::with_capacity(end_idx - start_idx);

    // Precompute levels for the entire formation window once,
    // then update incrementally. For simplicity (and correctness),
    // recompute every ~50 bars to capture new levels.
    let mut cached_levels: Vec<PriceLevel> = Vec::new();
    let mut last_level_update = 0;

    for t in start_idx..end_idx {
        let entry_price = candles[t].close;
        if entry_price <= 0.0 { continue; }

        // Recompute levels every 50 bars or on first iteration
        if cached_levels.is_empty() || t - last_level_update >= 50 {
            let formation_start = t.saturating_sub(level_params.formation_bars);
            cached_levels = compute_levels(candles, formation_start, t, level_params);
            last_level_update = t;
        }

        // ==== Build features ====
        // Static indicators + derived (52)
        let mut features = candles[t].full_features();
        // Level features (18)
        features.extend(compute_level_features(candles, t, &cached_levels));
        // Dynamic features (34)
        features.extend(compute_dynamic_features(candles, t));

        // ==== Find nearest level ====
        let atr = candles[t].atr.max(1e-12);
        let mut nearest_level: Option<&PriceLevel> = None;
        let mut nearest_dist = f64::MAX;

        for level in &cached_levels {
            let dist = (entry_price - level.price).abs();
            if dist < nearest_dist {
                nearest_dist = dist;
                nearest_level = Some(level);
            }
        }

        let nearest_dist_atr = nearest_dist / atr;
        let level_strength_val = nearest_level.map(|l| l.strength as i32).unwrap_or(0);
        let level_touches_val = nearest_level.map(|l| l.touch_count as i32).unwrap_or(0);

        // ==== Simulate trade (same first-touch logic as super_entry) ====
        let tp_long = entry_price * (1.0 + target_move_pct / 100.0);
        let sl_long = entry_price * (1.0 - (target_move_pct * sl_fraction) / 100.0);
        let tp_short = entry_price * (1.0 - target_move_pct / 100.0);
        let sl_short = entry_price * (1.0 + (target_move_pct * sl_fraction) / 100.0);

        let mut long_win = false;
        let mut short_win = false;
        let mut long_active = true;
        let mut short_active = true;
        let mut max_up: f64 = 0.0;
        let mut max_down: f64 = 0.0;

        for k in 1..=lookahead {
            let idx = t + k;
            if idx >= n { break; }
            let high = candles[idx].high;
            let low = candles[idx].low;

            let up_move = (high - entry_price) / entry_price * 100.0;
            let down_move = (entry_price - low) / entry_price * 100.0;
            if up_move > max_up { max_up = up_move; }
            if down_move > max_down { max_down = down_move; }

            if long_active {
                if low <= sl_long { long_active = false; }
                else if high >= tp_long { long_win = true; long_active = false; }
            }
            if short_active {
                if high >= sl_short { short_active = false; }
                else if low <= tp_short { short_win = true; short_active = false; }
            }
            if !long_active && !short_active { break; }
        }

        // Direction и is_super
        let (is_win, direction, magnitude) = if long_win && !short_win {
            (true, 1i8, target_move_pct)
        } else if short_win && !long_win {
            (true, -1i8, target_move_pct)
        } else if long_win && short_win {
            if max_up >= max_down { (true, 1i8, max_up) } else { (true, -1i8, max_down) }
        } else {
            if max_up >= max_down { (false, 1i8, max_up) } else { (false, -1i8, max_down) }
        };

        // ==== Bounce/Break label ====
        // Определяем: цена отскочила от уровня (bounce) или пробила (break)?
        let is_bounce = if let Some(level) = nearest_level {
            let level_price = level.price;
            match level.level_type {
                LevelType::Support => {
                    // Цена у поддержки: bounce = пошла вверх (max_up > max_down)
                    // break = пробила вниз (цена упала ниже уровня на >0.5%)
                    let broke_through = candles[t..(t + lookahead).min(n)]
                        .iter()
                        .any(|c| c.low < level_price * 0.995);
                    !broke_through
                }
                LevelType::Resistance => {
                    let broke_through = candles[t..(t + lookahead).min(n)]
                        .iter()
                        .any(|c| c.high > level_price * 1.005);
                    !broke_through
                }
            }
        } else {
            true // no level = default bounce
        };

        // ==== Is Good Entry (Label 2) ====
        // Good entry = is_win AND near a level (within 1 ATR)
        let is_good_entry = is_win && nearest_dist_atr < 1.0;

        examples.push(SuperLevelExample {
            symbol: candles[t].symbol.clone(),
            tf_minutes,
            timestamp: candles[t].time.to_rfc3339(),
            features,
            level_strength: level_strength_val,
            level_touches: level_touches_val,
            is_good_entry,
            direction,
            is_bounce,
            is_win,
            max_up_move_pct: max_up,
            max_down_move_pct: max_down,
            magnitude_pct: magnitude,
            nearest_level_dist_atr: nearest_dist_atr,
        });
    }

    examples
}

// ═══════════════════════════════════════════════════════════════
// DB fetch — переиспользуем из ml_entry_strategy
// ═══════════════════════════════════════════════════════════════

pub use ml_entry_strategy::dataset::{
    fetch_candles_with_indicators,
    fetch_all_candles_for_tf,
    fetch_active_symbols,
};

// ═══════════════════════════════════════════════════════════════
// CSV Export
// ═══════════════════════════════════════════════════════════════

/// Export dataset to CSV file (with all 5 labels)
pub fn export_dataset_csv(
    examples: &[SuperLevelExample],
    output_path: &str,
    feature_names: &[&str],
) -> Result<()> {
    let mut file = std::fs::File::create(output_path)?;

    // Header
    let mut header = String::from("symbol,tf_minutes,timestamp");
    for name in feature_names {
        header.push(',');
        header.push_str(name);
    }
    header.push_str(",level_strength,level_touches,is_good_entry,direction,is_bounce,is_win,max_up_move_pct,max_down_move_pct,magnitude_pct,nearest_level_dist_atr");
    writeln!(file, "{}", header)?;

    for ex in examples {
        let mut line = format!("{},{},{}", ex.symbol, ex.tf_minutes, ex.timestamp);
        for &val in &ex.features {
            line.push_str(&format!(",{:.6}", val));
        }
        line.push_str(&format!(
            ",{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6}",
            ex.level_strength,
            ex.level_touches,
            if ex.is_good_entry { 1 } else { 0 },
            ex.direction,
            if ex.is_bounce { 1 } else { 0 },
            if ex.is_win { 1 } else { 0 },
            ex.max_up_move_pct,
            ex.max_down_move_pct,
            ex.magnitude_pct,
            ex.nearest_level_dist_atr,
        ));
        writeln!(file, "{}", line)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_candle(close: f64, high: f64, low: f64) -> CandleWithIndicators {
        CandleWithIndicators {
            time: Utc::now(),
            symbol: "BTCUSDT".to_string(),
            symbol_id: 1,
            open: close, high, low, close, volume: 1000.0,
            rsi: 50.0, cci: 0.0, stoch_k: 50.0, stoch_d: 50.0, williams: -50.0,
            macd: 0.0, macd_signal: 0.0, macd_hist: 0.0,
            adx: 25.0, sma: close, ema_20: close, ema_50: close, ema_200: close,
            bb_upper: close * 1.02, bb_mid: close, bb_lower: close * 0.98,
            atr: close * 0.01, obv: 0.0, vwap: close, volume_spike: 1.0,
            trend: 0.0, trend_short: 0.0, poc: close,
            alligator_jaw: close, alligator_teeth: close, alligator_lips: close,
            mfi: 50.0, fibo_pivot: close, fibo_r1: close * 1.01, fibo_s1: close * 0.99,
            supertrend: close, supertrend_dir: 1.0, cmf: 0.0,
        }
    }

    #[test]
    fn test_compute_levels_basic() {
        // Create candles that touch 100.0 and 105.0 multiple times
        let mut candles = Vec::new();
        for i in 0..50 {
            let close = if i % 8 < 4 { 100.0 + (i % 4) as f64 } else { 105.0 - (i % 4) as f64 };
            let high = close + 0.5;
            let low = close - 0.5;
            candles.push(make_candle(close, high, low));
        }

        let params = LevelParams::default();
        let levels = compute_levels(&candles, 0, 49, &params);

        // Should find at least some levels
        assert!(!levels.is_empty(), "Should detect levels from zigzag pattern");
    }

    #[test]
    fn test_level_features_count() {
        let candles: Vec<CandleWithIndicators> = (0..60)
            .map(|_| make_candle(100.0, 101.0, 99.0))
            .collect();

        let levels = vec![PriceLevel {
            price: 99.5,
            touch_count: 5,
            first_touch_idx: 10,
            last_touch_idx: 50,
            level_type: LevelType::Support,
            strength: LevelStrength::Strong,
            touch_indices: vec![10, 20, 30, 40, 50],
        }];

        let feats = compute_level_features(&candles, 55, &levels);
        assert_eq!(feats.len(), crate::config::level_feature_count());
    }

    #[test]
    fn test_dynamic_features_count() {
        let candles: Vec<CandleWithIndicators> = (0..60)
            .map(|_| make_candle(100.0, 101.0, 99.0))
            .collect();

        let feats = compute_dynamic_features(&candles, 55);
        assert_eq!(feats.len(), crate::config::dynamic_feature_count());
    }
}
