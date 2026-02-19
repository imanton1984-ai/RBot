# 🏗️ Architecture Refactoring: Strategy Separation

## Обзор изменений

Этот документ описывает рефакторинг архитектуры стратегий для четкого разделения пайплайнов и устранения смешивания между базовой стратегией (level_strategy) и Super Entry стратегией.

---

## 📋 Проблемы до рефакторинга

### 1. Смешение пайплайнов
```bash
./startGPU.sh --super-entry
# Запускал ВСЁ:
#   ✅ ingestor
#   ✅ compute_history (индикаторы → raw_signals → indicators_wide)
#   ✅ compute_realtime
#   ✅ predictors (ML price/levels модели)
#   ✅ trade_signals (сигналы для level_strategy)
#   ✅ super_entry_service (ML inference)
```

**Проблема**: Даже при запуске Super Entry стратегии генерились все данные для level_strategy, что:
- Тратило оперативную память
- Нагружало CPU/GPU
- Увеличивало время запуска
- Заполняло SSD лишними данными

### 2. Бектест в основном пайплайне
```bash
./scripts/super_entry.sh  # Автоматически запускался при старте
# dataset → train → backtest
```

**Проблема**: Бектест должен запускаться ОТДЕЛЬНО, а не при каждом старте бота.

### 3. Стратегии не выделены в отдельные модули
- `ml_entry_strategy` находился в `strategies/`
- `level_strategy` (базовая) была "размазана" по `compute/predictors/`
- Нет четкого разделения ответственности

---

## 🎯 Целевая архитектура

### Структура проекта после рефакторинга

```
strategies/
├── level_strategy/          # Базовая стратегия (переименована из "default")
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── config.rs        # LevelStrategyConfig
│       ├── pipeline.rs      # LevelStrategyPipeline
│       ├── signal_generator.rs
│       ├── strategy.rs      # Высокоуровневый API
│       └── bin/
│           ├── service.rs   # Production сервис
│           └── backtest.rs  # Бектестер
│
├── ml_entry_strategy/       # Super Entry (без изменений имени)
│   └── ... (существующая структура)
│
└── strategy_switcher.rs     # Централизованный переключатель
```

### Разделение пайплайнов

```
┌─────────────────────────────────────────────────────────────┐
│                    ОБЩИЕ МОДУЛИ                            │
│  common/ │ database/ │ ingestor/ │ compute/indicators/     │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────┐                     ┌───────────────────┐
│  LEVEL STRATEGY   │                     │   SUPER ENTRY     │
│                   │                     │                   │
│  compute_history  │                     │  compute_history  │
│    └─ indicators  │                     │    └─ indicators  │
│  compute_realtime │                     │  compute_realtime │
│    └─ indicators  │                     │    └─ indicators  │
│                   │                     │                   │
│  + predictors     │                     │  + ML inference   │
│  + trade_signals  │                     │  + super_entry_   │
│                   │                     │    signals        │
│  trade_signals    │                     │                   │
│  table            │                     │  super_entry_     │
│                   │                     │  signals table    │
└───────────────────┘                     └───────────────────┘
```

---

## 🔧 Изменения в скриптах

### 1. Переименование backtester скрипта

**До:**
```bash
./scripts/super_entry.sh              # Полный пайплайн с бектестом
```

**После:**
```bash
./scripts/super_entry_backtester.sh   # Только dataset/train/backtest
./scripts/super_entry_pipeline.sh     # Production service для генерации сигналов
```

### 2. Обновление startGPU.sh/startCPU.sh

**До:**
```bash
if [ "$RUN_SUPER_ENTRY" = true ]; then
    ./scripts/super_entry.sh --gpu    # Запускал dataset → train → backtest
fi
```

**После:**
```bash
if [ "$RUN_SUPER_ENTRY" = true ]; then
    ./scripts/super_entry_pipeline.sh # Запускает service в фоне
fi
# Бектест запускается ОТДЕЛЬНО:
#   ./scripts/super_entry_backtester.sh
```

### 3. Обновление run.sh

```bash
ACTIVE_STRATEGY="${ACTIVE_STRATEGY:-default}"

if [ "$ACTIVE_STRATEGY" = "super_entry" ]; then
    # Запускаем ТОЛЬКО super_entry_service
    start_svc "super_entry_service" "super_entry_service"
else
    # Запускаем level_strategy (через compute_history/compute_realtime)
    # predictors + trade_signals вычисляются внутри
fi
```

---

## 📊 Сравнение пайплайнов

| Этап | Level Strategy | Super Entry | Общий? |
|------|---------------|-------------|--------|
| `ingestor` | ✅ | ✅ | Да |
| `candles_*` таблицы | ✅ | ✅ | Да |
| `compute_history` (индикаторы) | ✅ | ✅ | Да |
| `indicators_wide` | ✅ | ✅ | Да |
| `raw_signals` | ✅ | ❌ | Нет |
| `compute/predictors` | ✅ | ❌ | Нет |
| `trade_signals` таблица | ✅ | ❌ | Нет |
| `super_entry_signals` таблица | ❌ | ✅ | Нет |

---

## 🚀 Использование

### Запуск level_strategy (базовая стратегия)

```bash
# CPU режим
./startCPU.sh

# GPU режим
./startGPU.sh

# Или явно указать стратегию
./startGPU.sh --strategy=level
```

### Запуск Super Entry стратегии

```bash
# Запуск production сервиса (генерация сигналов)
./startGPU.sh --super-entry

# Отдельный запуск backtest (НЕ при старте!)
./scripts/super_entry_backtester.sh --dataset
./scripts/super_entry_backtester.sh --train
./scripts/super_entry_backtester.sh --backtest

# Или полный цикл бектеста
./scripts/super_entry_backtester.sh
```

### Переключение стратегий

```bash
# Через env-переменную
ACTIVE_STRATEGY=level ./startGPU.sh
ACTIVE_STRATEGY=super_entry ./startGPU.sh

# Через флаг
./startGPU.sh --super-entry
```

---

## 🗂️ Структура таблиц БД

### Level Strategy
```sql
-- Предсказания
market.predictors

-- Торговые сигналы
market.trade_signals
```

### Super Entry Strategy
```sql
-- Super Entry сигналы
trade._090_super_entry_signals
```

---

## 📁 Переменные окружения

### Level Strategy
```bash
# Конфигурация
LEVEL_HORIZON_BARS=10
LEVEL_MIN_STORE_SCORE=0.50
MIN_FINAL_SCORE=0.60
LEVEL_PREFER_ML=true
LEVEL_MAX_LEVELS=2
LEVEL_USE_CUDA=false
LEVEL_USE_GPU_HISTORY=false
LEVEL_ML_BATCH_SIZE=4096

# Модели
model_path_price="models/price_v1_tf{tf}.ubj"
model_path_levels="models/levels_v1_tf{tf}.ubj"
```

### Super Entry Strategy
```bash
# Конфигурация
SUPER_ENTRY_P_THRESHOLD=0.55
SUPER_ENTRY_SL_FRACTION=0.5
SUPER_ENTRY_WARMUP_BARS=300
SUPER_ENTRY_LOOKAHEAD=20
SUPER_ENTRY_USE_GPU=false

# Модели
model_path_template="models/super_entry_v1_tf{tf}.ubj"
direction_model_path_template="models/super_dir_v1_tf{tf}.ubj"
```

---

## 🔄 Миграция

### Существующие данные
- Все существующие таблицы сохраняются
- `market.indicators_wide` используется обеими стратегиями
- `market.trade_signals` остается для level_strategy
- `trade._090_super_entry_signals` для Super Entry

### Обратная совместимость
- Старые скрипты переименованы с сохранением функциональности
- `super_entry.sh` → `super_entry_backtester.sh`
- ENV-переменные не изменены

---

## 🧪 Тестирование

### Level Strategy
```bash
# Запуск сервиса
cargo run --release -p level_strategy --bin level_strategy_service

# Бектест
cargo run --release -p level_strategy --bin level_strategy_backtest
```

### Super Entry Strategy
```bash
# Запуск сервиса
cargo run --release -p ml_entry_strategy --bin super_entry_service

# Бектест
cargo run --release -p ml_entry_strategy --bin super_entry_backtest
```

---

## 📝 Будущие улучшения

1. **StrategySwitcher интеграция**: Полная интеграция в `compute_history`/`compute_realtime` для динамического переключения пайплайнов

2. **Combined Mode**: Одновременный запуск обеих стратегий с консенсус-фильтрацией сигналов

3. **WebUI переключатель**: API для переключения стратегий через веб-интерфейс

4. **A/B тестирование**: Параллельный запуск стратегий для сравнения эффективности

---

## ⚠️ Критические замечания

1. **Не сломать существующий пайплайн**: Базовая стратегия должна работать как прежде
2. **Сохранить данные**: `indicators_wide` должен заполняться полностью для обеих стратегий
3. **GPU/CPU совместимость**: Оба режима должны работать на GPU и CPU машинах
4. **Backward compatibility**: Существующие бектесты должны продолжать работать

---

## 📚 Связанная документация

- [SUPER_ENTRY_STRATEGY.md](SUPER_ENTRY_STRATEGY.md) — документация Super Entry стратегии
- [strategies/strategy_switcher.rs](strategies/strategy_switcher.rs) — переключатель стратегий
- [scripts/super_entry_backtester.sh](scripts/super_entry_backtester.sh) — скрипт бектеста
- [scripts/super_entry_pipeline.sh](scripts/super_entry_pipeline.sh) — production пайплайн
