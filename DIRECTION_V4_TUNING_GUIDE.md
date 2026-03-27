# Direction v4 — Руководство по настройке параметров

## 🔧 Параметры генерации датасета (Rust, direction_dataset)

Все параметры задаются через ENV переменные при запуске `direction_dataset`:

```bash
# Полный пример запуска:
DIR_WINDOW_SIZE=20 \
DIR_PREDICTION_HORIZON=25 \
DIR_FEATURE_SET=ohlcv \
DIR_LABEL_METHOD=max_excursion \
DIRECTION_TF=15,60,240 \
./target/release/direction_dataset
```

### Окно паттерна (`DIR_WINDOW_SIZE`)
| Значение | Фичей | Эффект |
|----------|-------|--------|
| 10 | 50 | Меньше контекста, быстрее обучение, меньше переобучение |
| **20** (current) | **100** | **Оптимальный баланс** |
| 30 | 150 | Больше контекста, больше фичей → риск переобучения |
| 60 | 300 | Много контекста, очень медленное обучение |

**Где менять:** ENV `DIR_WINDOW_SIZE` при запуске dataset и backtest.  
**Когда менять:** Если на коротких TF (5m, 15m) accuracy низкая → попробуй уменьшить до 10-15.

### Горизонт предсказания (`DIR_PREDICTION_HORIZON`)

| Значение | Эффект |
|----------|--------|
| 5 | Очень короткий, больше шума в лейблах |
| 10 | Короткий, хорош для скальпинга |
| **25** (current) | **Совпадает с P(super) lookahead** |
| 50 | Длинный, сглаженные лейблы, меньше сделок |

**Где менять:** ENV `DIR_PREDICTION_HORIZON` + `--horizon` в тренировочном скрипте.  
**Важно:** Horizon должен совпадать в dataset, train, и backtest!

### Набор фичей (`DIR_FEATURE_SET`)

| Значение | Фичей/свечу | Описание |
|----------|-------------|----------|
| `ohlc` | 4 | Только OHLC relative |
| **`ohlcv`** (current) | **5** | **OHLC + объём** |
| `ohlcvb` | 8 | + body/wick анализ |
| `full` | 10 | + inter-bar dynamics |

**Где менять:** ENV `DIR_FEATURE_SET`.  
**Совет:** `ohlcv` — best baseline. `full` может дать +1-2% accuracy но с риском переобучения.

### Метод лейблирования (`DIR_LABEL_METHOD`)

| Значение | Логика |
|----------|--------|
| `final_return` | label = знак return за horizon баров (простой, шумный) |
| **`max_excursion`** (current) | **Цена КОГДА-ЛИБО достигла порога?** |

**Где менять:** ENV `DIR_LABEL_METHOD`.  
**MaxExcursion лучше** потому что совпадает с логикой TP/SL (нам важно, что цена дошла до TP).

### Пороги UP/DOWN (%) — автоматически по TF

Текущие (из P(super) targets):

| TF | Порог | Влияние |
|----|-------|---------|
| 5m | 2.8% | Строгий фильтр |
| 15m | 3.5% | |
| 60m | 5.0% | |
| 240m | 7.5% | |
| 1440m | 10.0% | |

**Переопределить:** ENV `DIR_UP_THRESHOLD=3.0` (применяется ко всем TF).  
**Меньше порог** → больше UP/DOWN, меньше FLAT → больше данных для обучения, но шумнее.  
**Больше порог** → меньше UP/DOWN, больше FLAT → чище лейблы, но меньше данных.

---

## 🎯 Параметры обучения (Python, train_direction_wfo.py)

```bash
python trainer/src/train_direction_wfo.py \
  --csv dataset/direction_v4_dataset.csv \
  --mode binary \
  --gpu \
  --timeframes 15,60,240
```

### Режим (`--mode`)
- `binary` — **рекомендуется**. UP vs DOWN, исключает FLAT. Confidence = P(class).
- `regression` — регрессия на future_return. Для экспериментов.

### XGBoost гиперпараметры (в коде)

| Параметр | Текущее | Что делает | Куда крутить |
|----------|---------|------------|-------------|
| `eta` | 0.03 | Learning rate | 0.01-0.05. Ниже = стабильнее, дольше |
| `max_depth` | 6 | Глубина деревьев | 4-8. Меньше = меньше переобучение |
| `min_child_weight` | 20 | Min samples в листе | 10-50. Больше = стабильнее |
| `lambda` (L2) | 2.0 | Регуляризация | 1.0-5.0. Больше = меньше переобучение |
| `alpha` (L1) | 0.5 | Спарсификация | 0.1-1.0 |
| `gamma` | 0.2 | Min loss reduction | 0.1-0.5 |
| `subsample` | 0.8 | Доля данных на дерево | 0.6-0.9 |
| `colsample_bytree` | 0.7 | Доля фичей на дерево | 0.5-0.8 |
| `num_boost_round` | 1500 | Max деревьев | 500-3000 |

**Где менять:** В файле `trainer/src/train_direction_wfo.py`, функция `train_binary()`, dict `params`.

---

## 🔄 Параметры бэктеста (Rust, direction_backtest)

```bash
DIRECTION_MODE=binary \
DIRECTION_STANDALONE=1 \
DIRECTION_TF=15,60,240 \
WFO_MIN_DATE=2026-01-13 \
./target/release/direction_backtest
```

| ENV | Описание | Default |
|-----|----------|---------|
| `DIRECTION_MODE` | `binary` или `regression` | `binary` |
| `DIRECTION_STANDALONE` | `1` = без super-фильтра | `0` |
| `DIRECTION_TF` | Какие TF тестировать | `15,60,240` |
| `WFO_MIN_DATE` | Считать trades только после даты | нет |
| `DIRECTION_GPU` | GPU для inference | `0` |

---

## 📋 Чеклист: как тестировать новую конфигурацию

1. **Собрать датасет** с новыми параметрами:
   ```bash
   DIR_WINDOW_SIZE=15 DIR_PREDICTION_HORIZON=20 ./target/release/direction_dataset
   ```

2. **Обучить модель** (WFO оценка + сохранение):
   ```bash
   python trainer/src/train_direction_wfo.py --csv dataset/direction_v4_dataset.csv --mode binary --gpu
   ```

3. **Бэктест**:
   ```bash
   DIR_WINDOW_SIZE=15 DIR_PREDICTION_HORIZON=20 DIRECTION_MODE=binary DIRECTION_STANDALONE=1 \
   ./target/release/direction_backtest
   ```

⚠️ **Важно:** `DIR_WINDOW_SIZE`, `DIR_PREDICTION_HORIZON`, `DIR_FEATURE_SET` должны СОВПАДАТЬ между dataset, train, и backtest!

Параметры по убыванию влияния на качество модели
🥇 #1: Feature Set → full вместо ohlcv (максимальное влияние)
Текущее: ohlcv = 5 фичей/свечу = 100 фичей (window=20).

Рекомендация: full = 10 фичей/свечу = 200 фичей.

Добавляет:

body_pct — размер тела свечи (бычья/медвежья)
upper_wick_ratio — верхняя тень (отбой от сопротивления)
lower_wick_ratio — нижняя тень (отбой от поддержки)
bar_return — % возврат от предыдущего close
gap_pct — гэп от предыдущего close к текущему open
Это ТО ЧТО трейдеры видят визуально — паттерны свечей (молот, доджи, поглощение). Модель сейчас видит только OHLC точки, но не понимает что "длинная нижняя тень = покупатели отбили цену". Ожидаемый эффект: +2-5% accuracy на gates.

DIR_FEATURE_SET=full ./target/release/direction_dataset
# Затем переобучить
🥈 #2: Prediction Horizon → попробовать 10-15 вместо 25
Текущее: horizon=25 (25 свечей вперёд).

Для 60m это 25 часов, для 240m это 4.2 дня.

Проблема: модель предсказывает далеко вперёд, но TP/SL часто срабатывают НАМНОГО раньше. Если TP=5% ловится за 5-10 баров, зачем учить модель на 25-баровое окно?

Рекомендация: попробовать horizon=10-15 для каждого TF.

DIR_PREDICTION_HORIZON=12 ./target/release/direction_dataset
python trainer/src/train_direction_wfo.py --horizon 12 --csv ... --gpu
Ожидаемый эффект: +1-3% accuracy, меньше шума в лейблах.

🥉 #3: Window Size → эксперименты 10/20/30/40
Текущее: window=20 (20 свечей истории).

Для разных TF оптимальное окно может быть разным:

15m: window=10-15 (2.5-3.75 часа контекста) — краткосрочные паттерны
60m: window=20-30 (20-30 часов контекста) — дневные паттерны
240m: window=30-50 (5-8 дней контекста) — недельные паттерны
Рекомендация: отдельные модели с разным window на каждый TF.

# Для 60m:
DIR_WINDOW_SIZE=30 DIRECTION_TF=60 ./target/release/direction_dataset
# Для 240m:  
DIR_WINDOW_SIZE=40 DIRECTION_TF=240 ./target/release/direction_dataset
#4: XGBoost — max_depth + num_boost_round
Текущее: max_depth=6, num_boost_round=1500, early_stopping=100.

Для модели с 200 фичами (full set) нужны более глубокие деревья и больше раундов:

max_depth=8 — деревья смогут находить более сложные паттерны
num_boost_round=5000 — больше деревьев с медленным learning rate
early_stopping=200 — дольше ждать улучшения
eta=0.01 — медленнее, но точнее (вместо 0.03)
Менять в train_direction_wfo.py, dict params.

#5: Label threshold — снизить для промежуточных TF
Текущие: 15m=3.5%, 60m=5.0%, 240m=7.5%.

Попробовать уменьшить на 30-40%:

15m: 2.0% (вместо 3.5%)
60m: 3.0% (вместо 5.0%)
240m: 5.0% (вместо 7.5%)
Это даст больше UP/DOWN лейблов (меньше FLAT) → больше данных для обучения → лучшая генерализация. Но потребуется уменьшить TP/SL пропорционально.

DIR_UP_THRESHOLD=3.0 ./target/release/direction_dataset
🚀 Рекомендуемый "max quality" конфиг для первого теста
# Шаг 1: Датасет с full features, horizon=12, window=25
DIR_FEATURE_SET=full \
DIR_WINDOW_SIZE=25 \
DIR_PREDICTION_HORIZON=12 \
DIR_LABEL_METHOD=max_excursion \
DIRECTION_TF=15,60,240 \
./target/release/direction_dataset

# Шаг 2: Обучение (долго, но качественно)
python trainer/src/train_direction_wfo.py \
  --csv dataset/direction_v4_dataset.csv \
  --mode binary \
  --gpu \
  --horizon 12 \
  --timeframes 15,60,240

# Шаг 3: Бэктест
DIR_FEATURE_SET=full \
DIR_WINDOW_SIZE=25 \
DIR_PREDICTION_HORIZON=12 \
DIRECTION_MODE=binary \
DIRECTION_STANDALONE=1 \
./target/release/direction_backtest
В тренировочном скрипте перед запуском поменять в params:

"eta": 0.01 (вместо 0.03)
"max_depth": 8 (вместо 6)
num_boost_round=5000 (вместо 1500)
early_stopping_rounds=200 (вместо 100)
Время обучения: ~30-60 минут на GPU вместо ~10 минут. Но результат должен быть ощутимо лучше.
