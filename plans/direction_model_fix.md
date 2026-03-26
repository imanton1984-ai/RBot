# DIRECTION MODEL v3 — ФИНАЛЬНЫЙ ПОДХОД

## ПЕРЕОСМЫСЛЕНИЕ ЗАДАЧИ

**Проблема**: LONG vs SHORT binary classification = ~50% на любых фичах. 20 фичей ~50%, 128 фичей ~51%. Задача бинарной классификации принципиально неправильная для random walk на коротких TF.

**Решение**: Regression на `direction_quality` — не предсказывать sign движения, а предсказывать СКОРОСТЬ + DIRECTION. Быстрые TP = чистый сигнал, медленные = шум.

```
direction_quality = direction * 1/bars_to_tp

Примеры:
  LONG TP за 3 бара  → direction_quality = +0.333  -- сильный LONG сигнал
  SHORT TP за 2 бара → direction_quality = -0.500  -- сильный SHORT сигнал  
  LONG TP за 20 баров → direction_quality = +0.050  -- слабый, модель игнорирует
  No TP hit → direction_quality = ±0.010             -- шум, почти ноль
```

При inference: `sign(prediction) = direction`, `abs(prediction) = confidence`

---

## 32 CURATED FEATURES — 6 ДОМЕНОВ, БЕЗ ДУБЛИРОВАНИЯ

### GROUP 1: Trend alignment — 6 features
| Feature | Гипотеза для direction |
|---------|----------------------|
| `supertrend_dir` | Прямой direction сигнал |
| `htf_supertrend_dir` | HTF подтверждает направление — strongest signal |
| `trend` | Compute indicator trend |
| `trend_short` | Short-term trend |
| `trend_alignment` | Multi-indicator consensus |
| `ema_convergence_change` | EMA converging/diverging |

### GROUP 2: Momentum — 6 features  
| Feature | Гипотеза |
|---------|----------|
| `price_roc_lb1` | Мгновенный momentum |
| `price_roc_lb2` | 2-bar momentum |
| `price_accel_1bar` | Acceleration — ускоряется ли рынок |
| `macd_hist` | MACD histogram — classic momentum |
| `macd_hist_roc_lb1` | MACD momentum change |
| `rsi_slope_lb3` | RSI acceleration |

### GROUP 3: Market structure — 6 features
| Feature | Гипотеза |
|---------|----------|
| `dist_to_low_50` | Close к поддержке → вероятнее вверх |
| `dist_to_high_50` | Close к сопротивлению → вероятнее вниз |
| `bb_position` | Позиция в BB — mean reversion/breakout |
| `price_vs_vwap` | Цена выше/ниже VWAP — institutional bias |
| `price_vs_ema20` | Short-term price position |
| `high_low_pressure` | High-low range asymmetry |

### GROUP 4: Volume/Pressure — 4 features
| Feature | Гипотеза |
|---------|----------|
| `volume_up_vs_down_lb10` | Buy vs sell volume ratio |
| `obv_price_divergence` | Smart money divergence |
| `acute_wick_rejection_2bar` | Wick rejection = support/resistance |
| `cmf` | Chaikin Money Flow |

### GROUP 5: BTC relative — 5 features
| Feature | Гипотеза |
|---------|----------|
| `btc_return_lb5` | BTC momentum short — альты следуют |
| `btc_return_lb25` | BTC momentum medium |
| `btc_supertrend_dir` | BTC trend — рынок идёт куда BTC |
| `alt_vs_btc_return_lb5` | Relative strength — отстаёт? catching up? |
| `alt_vs_btc_return_lb25` | Relative strength medium |

### GROUP 6: Volatility context — 5 features
| Feature | Гипотеза |
|---------|----------|
| `bb_squeeze_pctl` | Volatility compression — breakout imminent |
| `atr_ratio_lb5` | ATR expanding/contracting |
| `bb_width_pct` | BB width — regime detection |
| `supertrend_consistency` | Supertrend stability — trending vs choppy |
| `volume_trend_ratio` | Volume confirming trend? |

**TOTAL: 32 features** — каждая с конкретной гипотезой, из разных доменов, минимальная корреляция.

---

## ЧТО ДЕЛАТЬ В КОДЕ — МИНИМАЛЬНЫЕ ИЗМЕНЕНИЯ

### 1. Python: train_direction_wfo.py — ПОЛНАЯ ПЕРЕРАБОТКА

```python
# Ключевые изменения:
# 1. Загрузить super_entry_dataset.csv — фичи уже есть!
#    - 26 из 32 фичей УЖЕ В ДАТАСЕТЕ: trend, supertrend_dir, macd_hist, bb_position, 
#      dist_to_low_50, dist_to_high_50, cmf, и все dynamic features
#    - НУЖНО ДОБАВИТЬ ТОЛЬКО: 5 BTC features + volume_up_vs_down_lb10
#
# 2. Task = reg:squarederror на direction_quality
#    - direction_quality = direction * 1/bars_to_tp (для is_super=True)
#    - Обучение ТОЛЬКО на is_super=True, direction!=0
#
# 3. Убрать Oracle — не нужен
#
# 4. Inference: sign = direction, abs = confidence

DIRECTION_V3_FEATURES = [
    # Из INDICATOR_FEATURES - сырые:
    "supertrend_dir", "trend", "trend_short", "macd_hist", "cmf",
    # Из DERIVED_FEATURES:
    "bb_position", "bb_width_pct", "price_vs_vwap", "price_vs_ema20",
    # Из DYNAMIC_FEATURES:
    "price_roc_lb1", "price_roc_lb2", "price_accel_1bar",
    "macd_hist_roc_lb1", "rsi_slope_lb3",
    "dist_to_low_50", "dist_to_high_50",
    "high_low_pressure", "supertrend_consistency", "trend_alignment",
    "ema_convergence_change", "volume_trend_ratio",
    "atr_ratio_lb5", "bb_squeeze_pctl", "obv_price_divergence",
    "acute_wick_rejection_2bar",
    "htf_supertrend_dir",
    # НОВЫЕ — нужно вычислить в dataset builder:
    "btc_return_lb5", "btc_return_lb25", "btc_supertrend_dir",
    "alt_vs_btc_return_lb5", "alt_vs_btc_return_lb25",
    "volume_up_vs_down_lb10",
]  # 32 фичи
```

### 2. Rust: dataset_builder — добавить 6 колонок

- `btc_return_lb5`, `btc_return_lb25`, `btc_supertrend_dir` — BTC context
- `alt_vs_btc_return_lb5`, `alt_vs_btc_return_lb25` — relative strength
- `volume_up_vs_down_lb10` — volume imbalance

Эти фичи вычисляются из уже загруженных BTC свечей + текущего символа. Добавить в существующий `dataset.rs` → `compute_dynamic_features_with_htf()` или как отдельный `compute_btc_features()`.

### 3. Rust: pipeline.rs + model.rs — inference на 32 фичах

Direction модель загружается **отдельно** от super_entry. Она использует SUBSET из 128 фичей + 6 новых. При inference:
- Берём нужные 32 из уже вычисленных 128
- Добавляем 6 новых
- Подаём в direction model

### 4. Scorer/Signal — confidence gate + HTF filter

```rust
// abs prediction = confidence  
let prediction = direction_model.predict(features_32);
let direction = if prediction > 0.0 { LONG } else { SHORT };
let confidence = prediction.abs();

// Confidence gate: skip если недостаточно уверен
if confidence < 0.05 {  // калибруется по WFO OOS
    return None;
}

// HTF hard rule
if htf_supertrend_dir < 0.0 && direction == LONG { return None; }
if htf_supertrend_dir > 0.0 && direction == SHORT { return None; }
```

---

## ПОРЯДОК РЕАЛИЗАЦИИ

```
[ ] 1. Расширить dataset_builder.rs — добавить 6 колонок BTC + volume в CSV
[ ] 2. Переписать train_direction_wfo.py — regression на direction_quality, 32 фичи
[ ] 3. Обучить и проверить OOS на WFO — цель AUC >= 0.60
[ ] 4. Интегрировать в pipeline.rs + model.rs — inference на 32 фичах
[ ] 5. Добавить confidence gate + HTF filter 
[ ] 6. Запустить direction_backtest.rs — проверить WR
```

---

## ПОЧЕМУ ЭТО ДОЛЖНО СРАБОТАТЬ

1. **Regression на direction_quality** — модель учит СКОРОСТЬ+НАПРАВЛЕНИЕ одновременно. Быстрые TP = чистый лейбл, шум отфильтрован через 1/bars_to_tp weight. В отличие от binary classification где 50% лейблов — шум.

2. **32 curated features** — каждая из отдельного домена: trend, momentum, structure, volume, BTC, volatility. Нет дублирования. XGBoost не тратит capacity на мусор.

3. **BTC context** — новые фичи btc_return + relative strength. В direction v2 WFO `btc_return_lb25` и `btc_supertrend_dir` были стабильно в TOP-5 important на ВСЕХ TF. Они РАБОТАЮТ, но в старой модели их НЕТ.

4. **abs = confidence** — выходит built-in confidence gate. Если предсказание близко к 0 → direction неизвестен → skip. Не нужен отдельный порог.

5. **HTF hard rule** — убираем 50% ошибок простым правилом без ML.
