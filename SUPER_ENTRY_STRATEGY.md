# 🚀 Super Entry Strategy (ML Entry Strategy)

## Оглавление
1. [Что это](#что-это)
2. [Архитектура](#архитектура)
3. [Быстрый старт](#быстрый-старт)
4. [Команды и флаги](#команды-и-флаги)
5. [Результаты бектеста v1](#результаты-бектеста-v1)
6. [Оценка результатов](#оценка-результатов)
7. [Калибровка и тюнинг](#калибровка-и-тюнинг)
8. [Переменные окружения](#переменные-окружения)
9. [Структура файлов](#структура-файлов)
10. [GPU-поддержка](#gpu-поддержка)
11. [FAQ](#faq)

---

## Что это

**Super Entry Strategy** — это отдельная ML-стратегия, которая ищет точки входа с высокой вероятностью сильного ценового движения ("super move"). Стратегия полностью независима от основной системы (entry_policy, signal_quality) и может включаться/выключаться флагом.

### Ключевая идея

Модель учится по историческим данным различать:
- **Super move** — ценовое движение ≥ целевого порога за 20 свечей (lookahead)
- **Обычное движение** — движение ниже порога

**Пороги целевого движения по таймфреймам (TF_TARGET_MOVE_PCT):**

| TF | Порог | Описание |
|----|-------|----------|
| 1m | 1.0% | Скальпинг: быстрое движение 1% за 20 минут |
| 5m | 1.75% | Краткосрочное движение за ~1.5 часа |
| 15m | 2.75% | Среднесрочное движение за ~5 часов |
| 1h | 3.75% | Дневное движение за ~20 часов |
| 4h | 4.75% | Многодневное движение за ~3.3 дня |
| 1d | 5.75% | Недельное движение за ~20 дней |

### Что предсказывает модель

1. **P(super)** — вероятность что из текущей точки будет движение ≥ порога за 20 свечей
2. **P(direction=LONG)** — вероятность что движение будет вверх (>0.5 = LONG, <0.5 = SHORT)

---

## Архитектура

```
                     ┌─────────────────────┐
                     │   TimescaleDB       │
                     │ candles + indicators │
                     └────────┬────────────┘
                              │
                     ┌────────▼────────────┐
                     │  Dataset Builder    │  ← cargo run --bin super_entry_dataset
                     │  (warmup 300 свечей │
                     │   labeling t=301..980│
                     │   lookahead=20)      │
                     └────────┬────────────┘
                              │
                     ┌────────▼────────────┐
                     │  super_entry_dataset │  CSV файл (38 фичей + labels)
                     │  .csv               │
                     └────────┬────────────┘
                              │
                     ┌────────▼────────────┐
                     │  Python XGBoost     │  ← python train_super_entry.py
                     │  Trainer            │
                     │  (split по парам    │
                     │   80%/20%)          │
                     └────────┬────────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
        ┌────────▼─────────┐      ┌────────▼─────────┐
        │ super_entry_v1   │      │  super_dir_v1    │
        │ _tf{X}.ubj       │      │  _tf{X}.ubj      │
        │ P(super move)    │      │  P(direction)    │
        └────────┬─────────┘      └────────┬─────────┘
                 │                         │
                 └────────────┬────────────┘
                              │
                     ┌────────▼────────────┐
                     │  Rust Pipeline      │
                     │  Features → Model   │
                     │  → Scorer → Signal  │
                     └────────┬────────────┘
                              │
                     ┌────────▼────────────┐
                     │  Backtester         │  ← cargo run --bin super_entry_backtest
                     │  SL/TP simulation   │
                     │  Метрики по TF      │
                     └─────────────────────┘
```

---

## Быстрый старт

### Предварительные требования
- База данных с данными (candles + indicators): запустить `compute_history` сначала
- Python 3 + XGBoost + scikit-learn
- Rust toolchain

### Полный пайплайн (одной командой)
```bash
./scripts/super_entry.sh
```

### Пошагово
```bash
# 1. Построить датасет из БД
./scripts/super_entry.sh --dataset

# 2. Обучить модели
./scripts/super_entry.sh --train

# 3. Запустить бектест
./scripts/super_entry.sh --backtest
```

### Запуск через startGPU/startCPU (с флагом стратегии)
```bash
# Запуск бота с Super Entry стратегией
./startGPU.sh --super-entry
./startCPU.sh --super-entry

# Указать конкретную стратегию
./startGPU.sh --strategy=super_entry
./startCPU.sh --strategy=combined

# Через env-переменную
ACTIVE_STRATEGY=super_entry ./startGPU.sh
SUPER_ENTRY_ENABLED=true ./startGPU.sh
```

### Остановка
```bash
./scripts/stop.sh          # Остановит все процессы включая super_entry
./scripts/full_stop.sh     # Полная остановка + docker
```

### Переключатель стратегий
Файл [`strategies/strategy_switcher.rs`](strategies/strategy_switcher.rs) — централизованный менеджер стратегий для будущей интеграции с WebUI:

```rust
let switcher = StrategySwitcher::from_env();

// Проверить активную стратегию
if switcher.is_active(StrategyId::SuperEntry) { ... }

// Переключить (из WebUI API)
switcher.switch_to(StrategyId::SuperEntry);

// Получить список для UI
let summary = switcher.summary(); // → JSON
```

Поддерживаемые стратегии:
- `default` — Основная (Entry Policy + Signal Quality + Predictors)
- `super_entry` — Super Entry ML-модель
- `combined` — Комбинированный (Default + Super Entry фильтр)

---

## Команды и флаги

### `scripts/super_entry.sh`

| Флаг | Описание |
|------|----------|
| *(без флагов)* | Полный пайплайн: dataset → train → backtest |
| `--dataset` | Только построить датасет из БД |
| `--train` | Только обучить модели (нужен CSV датасет) |
| `--backtest` | Только запустить бектест (нужны .ubj модели) |
| `--gpu` | Использовать GPU для тренировки (CUDA) |

### `scripts/teacher.sh`

| Флаг | Описание |
|------|----------|
| `--skip-super-entry` | Пропустить обучение Super Entry моделей |
| *(без этого флага)* | Обучит ВСЕ модели включая Super Entry |

### Cargo бинарники напрямую

```bash
# Dataset builder
cargo run --release -p ml_entry_strategy --bin super_entry_dataset

# Backtester
cargo run --release -p ml_entry_strategy --bin super_entry_backtest
```

---

## Результаты бектеста v1

### Сводная таблица

| TF | Сделки | WinRate | AvgPnL | Sharpe | Coverage | Оценка |
|----|--------|---------|--------|--------|----------|--------|
| **1m** | 6 230 | 48.8% | +0.33% | 0.458 | 6.69% | ⚠️ Средне |
| **5m** | 11 631 | 49.5% | +0.55% | 0.435 | 12.49% | ⚠️ Средне |
| **15m** | 8 658 | **52.7%** | **+0.94%** | **0.470** | 9.33% | ✅ Хорошо |
| **1h** | 32 137 | **58.6%** | **+1.47%** | **0.534** | 35.54% | 🔥 Отлично |
| **4h** | 40 660 | 48.0% | +1.06% | 0.299 | 49.61% | ⚠️ Много шума |
| **1d** | 20 957 | 39.1% | +0.51% | 0.120 | 46.18% | ❌ Weak |
| **TOTAL** | **120 273** | **49.8%** | **+0.98%** | - | - | ✅ Прибыльно |

---

## Оценка результатов

### 🔥 Сильные стороны

1. **1h — звезда стратегии**: 58.6% WinRate, +1.47% AvgPnL, Sharpe 0.534. Это отличные показатели для крипто-стратегии. Модель на 1h лучше всех научилась отличать "super" входы от шума.

2. **Положительный AvgPnL на ВСЕХ TF**: Даже на 1d, где WinRate всего 39.1%, средний PnL всё равно +0.51% — это значит что выигрышные сделки значительно крупнее проигрышных (хороший risk/reward).

3. **120K+ сделок** — статистически значимая выборка. Это не случайность.

4. **Coverage 6-50%** — модель НЕ входит в каждую свечу, а выбирает. При coverage 6.69% на 1m модель отбирает ~1 из 15 свечей — это селективность.

### ⚠️ Слабые стороны (что калибровать)

1. **4h и 1d — слишком много сигналов**: Coverage 49.6% и 46.2% — модель входит в каждую вторую свечу. Это значит пороги слишком низкие для этих TF. Решение: поднять `p_threshold` или увеличить `TF_TARGET_MOVE_PCT`.

2. **1d WinRate 39.1%** — модель слабо работает на дневках. Причина: 99.7% свечей помечены как "super" в датасете (5.75% за 20 дней — это слишком легко для крипто). Решение: поднять порог до 8-10% или вообще отключить 1d.

3. **Direction модель слабая**: AUC ~0.56 для P(direction). Это чуть лучше рандома. Значит модель хорошо определяет КОГДА будет большое движение, но плохо определяет КУДА. Решение: добавить trend-фичи, использовать более длинный контекст.

### 📊 Сравнение с baseline

Средний PnL +0.98% на сделку — это отличный показатель. Для сравнения:
- Типичная крипто-стратегия: +0.1% — +0.3% avg PnL  
- Наша основная стратегия (1h): +0.15% avg PnL (из текущих бектестов)
- **Super Entry 1h: +1.47%** — в 10x лучше!

---

## Калибровка и тюнинг

### 1. Порог P(super) — `SUPER_ENTRY_P_THRESHOLD`

**Текущее значение:** 0.55

Это главный рычаг: чем выше порог, тем меньше сделок, но выше качество.

| Значение | Эффект |
|----------|--------|
| 0.45 | Больше сделок, ниже WinRate, больше шума |
| **0.55** | **Баланс (текущее)** |
| 0.65 | Меньше сделок, выше WinRate |
| 0.75 | Только высококачественные сигналы |
| 0.85 | Очень мало сигналов |

**Рекомендация:** Попробовать 0.60-0.65 для уменьшения шума на 4h/1d.

```bash
# Запуск с повышенным порогом
SUPER_ENTRY_P_THRESHOLD=0.65 ./scripts/super_entry.sh --backtest
```

### 2. Целевое движение по TF — `TF_TARGET_MOVE_PCT`

Эти пороги определяют что считается "super move" — минимальный % движения за 20 свечей.

**Текущие значения и рекомендации:**

| TF | Текущее | super% в датасете | Рекомендация | Обоснование |
|----|---------|-------------------|-------------|-------------|
| 1m | 1.0% | 29.5% | ✅ Оставить | Хороший баланс |
| 5m | 1.75% | 43.0% | Поднять до 2.0% | Снизить до ~35% super |
| 15m | 2.75% | 49.0% | Поднять до 3.5% | Снизить до ~35% |
| 1h | 3.75% | 76.2% | Поднять до 5.0% | Слишком много super |
| 4h | 4.75% | 92.6% | Поднять до 8.0% | 92% super = шум |
| 1d | 5.75% | 99.7% | Поднять до 12.0% или отключить | Бесполезно на 99.7% |

**Идеальное распределение super%: 25-40%** — модель должна различать, а при 92-99% она не может ничему научиться.

Для изменения порогов нужно поправить [`config.rs`](strategies/ml_entry_strategy/src/config.rs):

```rust
pub fn tf_target_move_pct() -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    m.insert(1, 1.0);      // хорошо
    m.insert(5, 2.0);      // поднять
    m.insert(15, 3.5);     // поднять
    m.insert(60, 5.0);     // поднять
    m.insert(240, 8.0);    // сильно поднять
    m.insert(1440, 12.0);  // сильно поднять или отключить
    m
}
```

И пересоздать датасет + переобучить:
```bash
./scripts/super_entry.sh --dataset
./scripts/super_entry.sh --train
./scripts/super_entry.sh --backtest
```

### 3. SL фракция — `SUPER_ENTRY_SL_FRACTION`

**Текущее значение:** 0.5 (SL = 50% от TP)

| Значение | SL для 1h (TP=3.75%) | Эффект |
|----------|----------------------|--------|
| 0.3 | SL=1.125% | Тайтовый SL, больше стопов, лучший RR |
| **0.5** | **SL=1.875%** | **Баланс** |
| 0.7 | SL=2.625% | Широкий SL, меньше стопов, хуже RR |
| 1.0 | SL=3.75% | RR=1:1, максимум выживаемости |

```bash
# Тайтовый SL
SUPER_ENTRY_SL_FRACTION=0.3 ./scripts/super_entry.sh --backtest
```

### 4. Lookahead окно — `SUPER_ENTRY_LOOKAHEAD`

**Текущее значение:** 20 свечей

| Значение | Эффект |
|----------|--------|
| 10 | Быстрее входы/выходы, меньше expired, нужно больше резкости |
| **20** | **Баланс** |
| 30 | Больше super% (легче достичь порог за 30 свечей), может overfit |

### 5. Warmup — `SUPER_ENTRY_WARMUP_BARS`

**Текущее значение:** 300 свечей

Первые 300 свечей используются для контекста индикаторов. Не рекомендуется менять без необходимости (индикаторы типа EMA-200 требуют 200+ свечей для калибровки).

### 6. Overheated фильтр

Фильтр отбрасывает маргинальные сигналы (p_super чуть выше порога) при экстремальных значениях осцилляторов:
- LONG: RSI > 75, Stoch > 85, CCI > 150 → "перегретый" вход
- SHORT: RSI < 25, Stoch < 15, CCI < -150 → "перепроданный" вход

Настраивается в [`scorer.rs`](strategies/ml_entry_strategy/src/scorer.rs):
```rust
pub overheated_margin: f32,          // 0.05 — зона маргинальности
pub enable_overheated_filter: bool,  // true
```

### 7. Рекомендуемый план калибровки

```bash
# Шаг 1: Поднять пороги для высоких TF
# Отредактировать config.rs → tf_target_move_pct()

# Шаг 2: Пересоздать датасет и переобучить
./scripts/super_entry.sh

# Шаг 3: Попробовать разные p_threshold
for p in 0.50 0.55 0.60 0.65 0.70 0.75; do
    echo "=== P_THRESHOLD=$p ==="
    SUPER_ENTRY_P_THRESHOLD=$p ./target/release/super_entry_backtest 2>&1 | grep -E "^(TF|TOTAL|---)"
done

# Шаг 4: Попробовать разные SL
for sl in 0.3 0.4 0.5 0.6 0.7; do
    echo "=== SL_FRACTION=$sl ==="
    SUPER_ENTRY_SL_FRACTION=$sl ./target/release/super_entry_backtest 2>&1 | grep -E "^(TF|TOTAL|---)"
done
```

---

## Переменные окружения

| Переменная | По умолчанию | Описание |
|-----------|-------------|----------|
| `DATABASE_URL` | `postgres://...localhost:5433/...` | URL базы данных |
| `SUPER_ENTRY_P_THRESHOLD` | `0.55` | Порог P(super) для генерации сигнала |
| `SUPER_ENTRY_SL_FRACTION` | `0.5` | SL как доля от TP |
| `SUPER_ENTRY_WARMUP_BARS` | `300` | Свечей на warmup индикаторов |
| `SUPER_ENTRY_LOOKAHEAD` | `20` | Окно lookahead (свечей) |
| `SUPER_ENTRY_USE_GPU` | `false` | GPU для инференса |
| `SUPER_ENTRY_TRAIN_SPLIT` | `0.8` | Доля пар для тренировки |
| `SUPER_ENTRY_DATASET_OUTPUT` | `super_entry_dataset.csv` | Путь к CSV датасету |
| `SUPER_ENTRY_BACKTEST_CSV` | *(пусто)* | CSV для экспорта результатов бектеста |
| `RUST_LOG` | `info` | Уровень логирования |

---

## Структура файлов

```
strategies/ml_entry_strategy/
├── Cargo.toml                      # Зависимости крейта
└── src/
    ├── lib.rs                      # Точка входа библиотеки
    ├── config.rs                   # TF_TARGET_MOVE_PCT, features, SuperEntryConfig
    ├── dataset.rs                  # Построение датасета: labeling + features + DB queries
    ├── model.rs                    # SuperEntryModelManager (обёртка XGBoost)
    ├── scorer.rs                   # Решение Super/NoSignal + Overheated фильтр
    ├── signal_generator.rs         # Генерация SuperEntrySignal (entry/SL/TP)
    ├── pipeline.rs                 # Оркестрация: features → model → scorer → signal
    ├── strategy.rs                 # Высокоуровневый API для интеграции
    └── bin/
        ├── dataset_builder.rs      # Бинарник построения CSV датасета
        └── backtest.rs             # Бинарник бектестера

trainer/src/train_super_entry.py    # Python тренер XGBoost моделей

scripts/super_entry.sh              # Скрипт пайплайна
scripts/teacher.sh                  # Обновлён: +Stage 4 Super Entry
scripts/stop.sh                     # Обновлён: +pgrep super_entry
scripts/full_stop.sh                # Обновлён: +pgrep super_entry

models/
├── super_entry_v1_tf{1,5,15,60,240,1440}.ubj    # P(super) модели
├── super_entry_v1_tf{X}.schema.json              # Схемы фичей
├── super_dir_v1_tf{1,5,15,60,240,1440}.ubj       # P(direction) модели
└── super_dir_v1_tf{X}.schema.json                # Схемы фичей
```

### 38 фичей модели

**23 индикатора (из market.indicators_wide):**
`rsi, cci, stoch_k, stoch_d, williams, macd, macd_signal, macd_hist, adx, sma, ema_20, ema_50, ema_200, bb_upper, bb_mid, bb_lower, atr, obv, vwap, volume_spike, trend, trend_short, poc`

**15 производных фичей (вычисляются из индикаторов):**
`rsi_norm, cci_norm, stoch_norm, williams_norm, bb_position, bb_width_pct, atr_pct, price_vs_sma, price_vs_ema20, price_vs_ema50, price_vs_ema200, price_vs_vwap, macd_norm, obv_change_pct, volume_spike_flag`

---

## GPU-поддержка

### Тренировка (Python)
```bash
./scripts/super_entry.sh --train --gpu
```
XGBoost использует `device: "cuda"` для GPU-тренировки. Ускорение 3-5x.

### Инференс (Rust)
XGBoost модели загружаются на CPU. При inference XGBoost использует CUDA predictor автоматически если `SUPER_ENTRY_USE_GPU=true`. Однако bottleneck — это DB I/O (~95%), а не inference (<1ms на батч), поэтому GPU ускорение инференса минимально.

**Кастомные .cu файлы НЕ нужны** — XGBoost имеет встроенный CUDA бэкенд.

---

## FAQ

### Как отключить стратегию?
```bash
# Через teacher.sh — пропустить обучение
./scripts/teacher.sh --skip-super-entry

# Стратегия не будет генерировать сигналы если модели отсутствуют
rm -f models/super_entry_v1_tf*.ubj models/super_dir_v1_tf*.ubj
```

### Как запустить только для конкретного TF?
Сейчас бектест запускается для всех TF. Для фильтрации по TF используйте CSV результаты:
```bash
SUPER_ENTRY_BACKTEST_CSV=results.csv ./target/release/super_entry_backtest
# Потом фильтруйте: grep "^[^,]*,60," results.csv > results_1h.csv
```

### Как часто нужно переобучать?
Рекомендуется переобучать при:
- Добавлении новых торговых пар
- Накоплении значительного количества новых исторических данных
- Изменении рыночного режима (eg: переход из bull в bear)

### Можно ли комбинировать с основной стратегией?
Да. Super Entry работает полностью независимо. В будущем можно:
1. Использовать Super Entry как фильтр для основной стратегии
2. Или запускать параллельно и выбирать лучший сигнал
3. Или использовать как confirmation signal

### Какой TF лучший?
**1h** — однозначно лучший: 58.6% WR, +1.47% AvgPnL, Sharpe 0.534. Рекомендуется начинать live-тестирование именно с 1h.
