import os
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlalchemy
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
import onnx
from sklearn.model_selection import TimeSeriesSplit
import json
import hashlib

# --- КОНФИГУРАЦИЯ ---
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/timescaledb_binance")
SYMBOL = "BTCUSDT"
TIMEFRAMES = [1, 5, 15, 60, 240, 1440]  # All 6 timeframes: 1m, 5m, 15m, 1h, 4h, 1d
HORIZON = 10  # Предсказываем на 10 свечей вперед
MODEL_DIR = "../models"

# Список фичей, которые ДОЛЖНЫ совпадать с тем, что вы подаете в Rust (FeatureView)
FEATURES = [
    'close', 'high', 'low', 'open', 'volume',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'ema_20', 'ema_50', 'ema_200', 'sma',
    'bb_upper', 'bb_lower', 'bb_mid',
    'atr', 'adx', 'vwap', 'obv', 'cci',
    'stoch_k', 'stoch_d', 'williams_r',  # Fixed: was 'williams' in Python but mapped from 'williams' in DB
    'trend', 'trend_short'
]

def load_data(timeframe):
    print(f"Connecting to {DB_URL}...")
    engine = sqlalchemy.create_engine(DB_URL)

    # Формируем SQL запрос. Важно: порядок колонок должен быть жестким,
    # либо в Rust мы должны маппить их по именам.
    # Join candles and indicators tables to get both OHLCV and indicators
    query = f"""
        SELECT
            c.close, c.high, c.low, c.open, c.volume,
            i.rsi, i.macd, i.macd_signal, i.macd_hist,
            i.ema_20, i.ema_50, i.ema_200, i.sma,
            i.bb_upper, i.bb_lower, i.bb_mid,
            i.atr, i.adx, i.vwap, i.obv, i.cci,
            i.stoch_k, i.stoch_d, i.williams as williams_r,  -- Map williams to williams_r
            i.trend, i.trend_short
        FROM market.candles c
        INNER JOIN market.indicators_wide i
            ON c.symbol_id = i.symbol_id AND c.time = i.time AND c.time_ms = i.time_ms
        WHERE i.symbol = '{SYMBOL}' AND c.tf_minutes = {timeframe}
        ORDER BY c.time ASC
    """
    print(f"Fetching data for timeframe {timeframe}m...")
    df = pd.read_sql(query, engine)

    # Check if we have enough data
    if len(df) < 200:
        print(f"Warning: Not enough data for timeframe {timeframe}m (got {len(df)} rows). Skipping...")
        return None

    # Заменяем NULL на 0 (или среднее), чтобы XGBoost не падал при конвертации
    df = df.fillna(0.0)

    # Конвертируем все в float32 (требование ONNX Runtime в Rust)
    for c in df.columns:
        df[c] = df[c].astype(np.float32)

    return df

def train_price_model(df):
    print("Training Price Prediction Model...")

    # Target: Percentage return after N candles
    y = (df['close'].shift(-HORIZON) / df['close']) - 1
    X = df[FEATURES]

    # Fill infinities that might result from division by zero
    y = y.replace([np.inf, -np.inf], 0.0)

    # Remove last N rows where target is NaN
    X = X.iloc[:-HORIZON]
    y = y.iloc[:-HORIZON]

    # Check if we have enough data for training
    if len(X) < 50:  # Need at least 50 samples for meaningful training
        print(f"  Not enough data after removing {HORIZON} rows for target. Available: {len(X)} rows.")
        return None

    # TimeSeriesSplit for cross-validation (adjust splits based on available data)
    n_splits = min(5, len(X) // 20)  # At least 20 samples per fold
    if n_splits < 2:
        print("  Not enough data for cross-validation, skipping CV and training directly...")
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            objective='reg:squarederror',
            n_jobs=-1
        )
        model.fit(X, y)
        return model

    tscv = TimeSeriesSplit(n_splits=n_splits)

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        objective='reg:squarederror',
        n_jobs=-1
    )

    print(f"  Cross-validating with {n_splits} splits...")
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Skip folds with insufficient data
        if len(X_train) < 10 or len(X_test) < 5:
            continue

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"    Split score: {score:.4f}")

    if scores:
        print(f"  Average R^2 Score on cross-validation: {np.mean(scores):.4f}")
    else:
        print("  Could not perform cross-validation due to insufficient data in folds")

    # Final fit on all data
    print("  Fitting final model on all data...")
    model.fit(X, y)

    return model

def export_to_onnx(model, filename, features, schema_id):
    print(f"Exporting to {filename}...")

    # Описываем входной тензор: [None (любой batch size), кол-во фичей]
    # Используем правильный тип данных в зависимости от библиотеки
    initial_type = [('float_input', OnnxFloatTensorType([None, len(features)]))]

    try:
        onnx_model = convert_xgboost(model, initial_types=initial_type)

        os.makedirs(MODEL_DIR, exist_ok=True)
        full_path = os.path.join(MODEL_DIR, filename)
        onnx.save_model(onnx_model, full_path)
        print(f"Saved model to {full_path}")

        # Export metadata
        meta = {
            "schema_id": schema_id,
            "features": features
        }
        meta_path = os.path.join(MODEL_DIR, filename.replace(".onnx", ".json"))
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"Saved metadata to {meta_path}")
    except Exception as e:
        print(f"Error exporting model {filename}: {e}")
        # Continue without crashing the whole process

def train_level_model(df):
    print("Training Level Prediction Model...")

    # Target: классификация (пробой/отскок от уровня)
    # Для примера, создадим искусственную целевую переменную
    # В реальности это будет более сложная логика на основе SR уровней

    # Создаем фиктивные уровни (в реальности они будут из SR уровней)
    df_copy = df.copy()
    df_copy['avg_price'] = df_copy['close'].rolling(window=20).mean()
    df_copy['is_near_level'] = abs(df_copy['close'] - df_copy['avg_price']) < (df_copy['atr'] * 0.5)  # Рядом с уровнем

    # Целевая переменная: 1 если пробой, 0 если отскок (упрощенно)
    df_copy['next_direction'] = (df_copy['close'].shift(-HORIZON) > df_copy['close']).astype(int)

    y = df_copy['next_direction']
    X = df_copy[FEATURES]

    # Удаляем последние N строк, где нет target
    X = X.iloc[:-HORIZON]
    y = y.iloc[:-HORIZON]

    # Убираем строки с NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    # Check if we have enough data for training
    if len(X) < 50:  # Need at least 50 samples for meaningful training
        print(f"  Not enough data for level model. Available: {len(X)} rows after cleaning.")
        return None

    # Обучение модели классификации
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        n_jobs=-1
    )
    model.fit(X, y)

    # Оценка
    score = model.score(X, y)
    print(f"  Accuracy Score on train set: {score:.4f}")

    return model

if __name__ == "__main__":
    try:
        # Create schema_id from FEATURES list
        feature_string = ",".join(FEATURES)
        schema_id = hashlib.sha256(feature_string.encode('utf-8')).hexdigest()

        for timeframe in TIMEFRAMES:
            print(f"\n=== Training models for timeframe {timeframe}m ===")
            
            df = load_data(timeframe)
            if df is None or len(df) < 200:
                print(f"Not enough data to train for timeframe {timeframe}m! Run the bot to collect candles first.")
                continue
            
            # Обучаем модель предсказания цены
            price_model = train_price_model(df.copy())
            if price_model is not None:
                export_to_onnx(price_model, f"price_v1_tf{timeframe}.onnx", FEATURES, f"{schema_id}_tf{timeframe}")
            else:
                print(f"  Skipping price model export for timeframe {timeframe}m due to insufficient data.")

            # Обучаем модель предсказания уровней
            level_model = train_level_model(df.copy())
            if level_model is not None:
                export_to_onnx(level_model, f"levels_v1_tf{timeframe}.onnx", FEATURES, f"{schema_id}_tf{timeframe}")
            else:
                print(f"  Skipping level model export for timeframe {timeframe}m due to insufficient data.")

            print(f"Training complete for timeframe {timeframe}m.")

        print("\nAll training completed!")
    except Exception as e:
        print(f"Error: {e}")
