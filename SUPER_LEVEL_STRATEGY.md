# 🎯 Super Level Strategy — Level-First Sniper

## Философия

**Уровни (ликвидность) — единственное место на графике, где действительно принимаются решения крупными игроками.**

Чистый ML часто пытается найти закономерности в "рыночном шуме" посреди канала, где цена движется хаотично. Super Level переворачивает архитектуру: **уровни — фундамент, ML/EWMAC/Эвристика — слуги-помощники, подтверждающие вход.**

Результат: стратегия снайпера. Торгует редко, но максимально безопасно.

---

## 🏛 Архитектура: 5 Фаз

Сделка проходит на следующий этап **ТОЛЬКО** если прошла предыдущий.

```
┌─────────────────┐
│  ФАЗА 1: RADAR  │ ← Цена у сильного S/R уровня (≤ 0.3 ATR)?
│  (Зона Убийства) │   Если нет → БОТ СПИТ
└────────┬────────┘
         │ ✅
┌────────▼────────┐
│ ФАЗА 2: CONTEXT │ ← Bounce (отскок) или Breakout (пробой)?
│  (EWMAC + RSI)  │   EWMAC forecast + ADX + RSI/Stoch/MFI
└────────┬────────┘   Нужно ≥2 подтверждения из 5
         │ ✅
┌────────▼────────┐
│ ФАЗА 3: ML      │ ← SuperEntry модель P(super) ≥ 0.62?
│ (Валидатор)      │   + направление ML совпадает с гипотезой?
└────────┬────────┘
         │ ✅
┌────────▼────────┐
│ ФАЗА 4: ENTRY   │ ← Ищет подтверждающий паттерн (пин-бар,
│ (Agent)          │   engulfing, volume reversal) в окне 4 свечей
└────────┬────────┘   Нет паттерна → CANCEL
         │ ✅
┌────────▼────────┐
│ ФАЗА 5: RISK    │ ← SL/TP по ATR (bounce: tight, breakout: wide)
│ (Управление)     │   50% закрытие на TP1 → SL в безубыток
└─────────────────┘
```

---

## 📂 Структура файлов

```
strategies/super_level_strategy/
├── Cargo.toml              # Зависимости (ml_entry_strategy, ewmac_strategy, etc.)
├── src/
│   ├── lib.rs              # Модули: config, phases, pipeline
│   ├── config.rs           # SuperLevelConfig — все параметры стратегии
│   ├── phases.rs           # 5-фазная логика (Radar→Context→ML→Entry→Risk)
│   ├── pipeline.rs         # Оркестрация (batch ML + EWMAC + per-bar filter)
│   └── bin/
│       └── backtest.rs     # Бэктестер с partial close и детальной статистикой
```

---

## 🚀 Запуск бэктеста

```bash
# Базовый запуск
cargo run --release -p super_level_strategy --bin super_level_backtest

# С экспортом CSV
SUPER_LEVEL_BACKTEST_CSV=trades.csv cargo run --release -p super_level_strategy --bin super_level_backtest

# С GPU (если есть CUDA)
SUPER_ENTRY_USE_GPU=true cargo run --release -p super_level_strategy --bin super_level_backtest
```

### Необходимые условия
1. **База данных** — TimescaleDB с данными в `market.candles_*` и `market.indicators_wide`
2. **ML модели** — `models/super_entry_v1_tf{X}.ubj` и `models/super_dir_v1_tf{X}.ubj` (из ml_entry_strategy)
3. **ENV** — `DATABASE_URL` (postgres connection string)

### Переучивать модели НЕ нужно
Стратегия использует существующие `super_entry` модели как валидатор (Phase 3). Новые модели не требуются.

---

## ⚙️ Параметры калибровки

Все параметры задаются через **env-переменные** или изменением defaults в [`config.rs`](strategies/super_level_strategy/src/config.rs).

### 🎚 Фаза 1: RADAR — Зона Убийства

| Параметр | ENV | Default | Описание | Эффект |
|----------|-----|---------|----------|--------|
| `radar_distance_atr` | `SUPER_LEVEL_RADAR_ATR` | **0.3** | Макс. дистанция до уровня в ATR | ↓ = меньше сигналов, но ближе к уровню |
| `min_level_strength` | `SUPER_LEVEL_MIN_STRENGTH` | **0.85** | Мин. сила уровня (0-1) | ↑ = только самые сильные, меньше входов |
| `sr_sensitivity_pct` | — | 0.3 | Кластеризация уровней (% от цены) | ↑ = больше уровней объединяются |
| `sr_lookback_bars` | `SUPER_LEVEL_SR_LOOKBACK` | 100 | Свечей назад для SR вычисления | ↑ = более "старые" уровни найдутся |

**Как калибровать Radar:**
```bash
# Меньше сделок, но точнее (снайпер)
SUPER_LEVEL_RADAR_ATR=0.2 SUPER_LEVEL_MIN_STRENGTH=0.9 cargo run ...

# Больше сделок (агрессивнее)
SUPER_LEVEL_RADAR_ATR=0.5 SUPER_LEVEL_MIN_STRENGTH=0.6 cargo run ...
```

### 🎚 Фаза 2: CONTEXT — Bounce vs Breakout

| Параметр | ENV | Default | Описание | Эффект |
|----------|-----|---------|----------|--------|
| `ewmac_trend_threshold` | `SUPER_LEVEL_EWMAC_THRESHOLD` | **8.0** | EWMAC forecast порог для тренда | ↑ = больше свечей в "weak trend" (bounce) |
| `adx_trend_threshold` | `SUPER_LEVEL_ADX_THRESHOLD` | **22.0** | ADX порог для определения тренда | ↑ = меньше breakout, больше bounce |
| `rsi_overbought` | — | 65.0 | RSI для overbought | ↓ = шире полоса, больше bounce SHORT |
| `rsi_oversold` | — | 35.0 | RSI для oversold | ↑ = шире полоса, больше bounce LONG |

**Как калибровать Context:**

Фаза 2 использует **систему подтверждений** (2 из 5 нужно):
1. RSI oversold/overbought
2. Stoch K < 25 / > 75
3. MFI < 30 / > 70
4. EWMAC flat (|forecast| < threshold)
5. EWMAC не противоречит направлению

```bash
# Больше bounce сигналов (расширить полосы)
SUPER_LEVEL_EWMAC_THRESHOLD=12 SUPER_LEVEL_ADX_THRESHOLD=28 cargo run ...

# Меньше bounce, больше breakout
SUPER_LEVEL_EWMAC_THRESHOLD=5 SUPER_LEVEL_ADX_THRESHOLD=18 cargo run ...
```

### 🎚 Фаза 3: ML VALIDATOR

| Параметр | ENV | Default | Описание | Эффект |
|----------|-----|---------|----------|--------|
| `ml_p_threshold` | `SUPER_LEVEL_ML_THRESHOLD` | **0.62** | Мин. P(super) от ML модели | ↑ = меньше сигналов, выше качество |
| `require_ml_direction_match` | `SUPER_LEVEL_REQUIRE_DIR_MATCH` | true | ML направление = уровень? | false = ML только фильтрует силу |

**Как калибровать ML:**
```bash
# Строгий ML (меньше входов, выше WR)
SUPER_LEVEL_ML_THRESHOLD=0.70 cargo run ...

# Мягкий ML (больше входов, ML только фильтрует мусор)
SUPER_LEVEL_ML_THRESHOLD=0.50 SUPER_LEVEL_REQUIRE_DIR_MATCH=false cargo run ...
```

### 🎚 Фаза 4: ENTRY AGENT

| Параметр | ENV | Default | Описание | Эффект |
|----------|-----|---------|----------|--------|
| `entry_window_bars` | `SUPER_LEVEL_ENTRY_WINDOW` | **4** | Свечей на поиск паттерна входа | ↑ = больше шансов найти паттерн |
| `entry_cancel_atr` | — | 0.8 | ATR дистанция для CANCEL | ↓ = быстрее отменяет (безопаснее) |

**Паттерны входа** (Phase 4, [`phases.rs:327`](strategies/super_level_strategy/src/phases.rs:327)):
- **Pin bar** — тень > 55% range в сторону уровня + close в нашу сторону
- **Strong reversal** — тело > 45% range + close в нашу сторону
- **Volume reversal** — volume_spike > 1.8 + close в нашу сторону
- **Engulfing** — текущая свеча полностью поглощает предыдущую
- **Нет паттерна → CANCEL** (ключевое отличие от V1!)

```bash
# Длинное окно (больше входов)
SUPER_LEVEL_ENTRY_WINDOW=6 cargo run ...

# Короткое окно (быстрее решаем)
SUPER_LEVEL_ENTRY_WINDOW=3 cargo run ...
```

### 🎚 Фаза 5: RISK MANAGEMENT

ATR-мультипликаторы из [`config/signal_params.toml`](config/signal_params.toml):

| Сценарий | SL | TP1 | TP2 | TP3 |
|----------|-----|-----|-----|-----|
| **Bounce** | 0.55×ATR | 0.75×ATR | 1.4×ATR | 2.2×ATR |
| **Breakout** | 0.75×ATR | 1.1×ATR | 1.8×ATR | 2.8×ATR |

| Параметр | ENV | Default | Описание |
|----------|-----|---------|----------|
| `partial_close_pct` | `SUPER_LEVEL_PARTIAL_CLOSE` | 50 | % позиции закрыть на TP1 |
| `trail_sl_to_breakeven` | — | true | SL → entry price после TP1 |
| `max_hold_bars` | `SUPER_LEVEL_MAX_HOLD` | 20 | Принудительное закрытие |

---

## 📊 Чтение результатов бэктеста

### Funnel (воронка фаз)
```
Funnel: Radar=138702 → Context=114256 → ML=8035 → Entry=6609 → Signal=6609
```
- **Radar**: сколько свечей было "у уровня"
- **Context**: прошли фильтр bounce/breakout (сколько имели ясный контекст)
- **ML**: прошли ML валидацию (p_super ≥ threshold)
- **Entry**: нашли подтверждающий паттерн входа
- **Signal**: финальные сделки

### Метрики
| Метрика | Описание |
|---------|----------|
| **WR%** | Чистый винрейт (только TP2/TP3 хиты) |
| **WR+P%** | Включает PartialWin (TP1 hit + остаток BE/expired) |
| **AvgPnL** | Средний PnL на сделку (должен быть >0) |
| **PF** | Profit Factor: gross_profit / gross_loss (>1 = прибыльно) |
| **MaxDD%** | Максимальная просадка (сумма PnL) |
| **TP1 Hit** | % сделок где цена достигла TP1 (ключевая метрика безубытка) |

### Интерпретация Expired

**Expired** — сделка не достигла ни SL, ни TP за `max_hold_bars`. Это НЕ всегда плохо:
- Если AvgPnL по expired > 0 → цена дрейфует в нашу сторону
- Если expired > 50% → `max_hold_bars` слишком мало или ATR TP/SL слишком далеко

---

## 🔧 Типичные сценарии калибровки

### Проблема: слишком много сделок
```bash
SUPER_LEVEL_RADAR_ATR=0.2 \
SUPER_LEVEL_ML_THRESHOLD=0.70 \
SUPER_LEVEL_MIN_STRENGTH=0.9 \
cargo run --release -p super_level_strategy --bin super_level_backtest
```

### Проблема: нет сигналов на 5m
```bash
# Расширить RSI/EWMAC пороги — файл config.rs Default:
# rsi_oversold: 35 → 40
# rsi_overbought: 65 → 60
# ewmac_trend_threshold: 8 → 12
```

### Проблема: низкий WR% (< 40%)
```bash
# Ужесточить ML + Entry Agent
SUPER_LEVEL_ML_THRESHOLD=0.68 \
SUPER_LEVEL_ENTRY_WINDOW=3 \
cargo run --release -p super_level_strategy --bin super_level_backtest
```

### Проблема: много expired
```bash
# Увеличить max_hold ИЛИ уменьшить TP мультипликаторы
SUPER_LEVEL_MAX_HOLD=30 \
cargo run --release -p super_level_strategy --bin super_level_backtest
```

### Проблема: большая просадка
```bash
# Тест с очень жёстким фильтром (снайпер-режим)
SUPER_LEVEL_RADAR_ATR=0.15 \
SUPER_LEVEL_ML_THRESHOLD=0.72 \
SUPER_LEVEL_MIN_STRENGTH=0.9 \
SUPER_LEVEL_ENTRY_WINDOW=3 \
cargo run --release -p super_level_strategy --bin super_level_backtest
```

---

## 📈 Текущие результаты V2 (2025-03)

| TF | Trades | TP1 Hit% | AvgPnL% | PF | Статус |
|----|--------|----------|---------|-----|--------|
| 5m | 6,609 | 8.4% | +0.093% | 1.12 | ⚠️ Много expired |
| 15m | 16,564 | 20.0% | +0.012% | 1.02 | ⚠️ Brake even |
| 1h | 8,978 | 30.3% | -0.028% | 0.98 | ❌ Убыточный |
| **4h** | **5,132** | **40.2%** | **+0.264%** | **1.15** | ✅ **Лучший TF** |

**Вывод**: 4h — основной рабочий TF. 5m/15m прибыльные но на грани. 1h требует доработки.

---

## 🔗 Переиспользование компонентов

| Компонент | Source | Использование |
|-----------|--------|---------------|
| ML inference | `ml_entry_strategy::SuperEntryPipeline` | Phase 3: batch zero-copy CUDA predict |
| EWMAC | `ewmac_strategy::EwmacCalculator` | Phase 2: incremental O(1) per bar |
| SR Levels | `indicators::sr_levels::calculate_sr_levels` | Phase 1: inline from OHLC |
| Candle data | `ml_entry_strategy::dataset::fetch_all_candles_for_tf` | Batch SQL per TF |
| Entry Agent | `predictors::entry_policy::agent` (pattern) | Phase 4: pattern-based |
| ATR params | `config/signal_params.toml` [atr_bounce/breakout] | Phase 5: SL/TP |

---

## 🛠 Будущие улучшения

1. **TF-specific пороги** — разные ML/ADX/RSI пороги для 5m vs 4h
2. **Обучить Level-ML модель** — features включают distance_to_level, touch_count, approach_velocity
3. **Multi-timeframe confirmation** — сигнал на 15m подтверждается 1h контекстом
4. **Order flow integration** — объём ордеров на уровне как дополнительный фильтр
5. **Dynamic position sizing** — размер позиции от strength × p_super × scenario
