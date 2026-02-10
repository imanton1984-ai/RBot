import os
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlalchemy
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools import convert_xgboost
import onnx
from sklearn.model_selection import TimeSeriesSplit
import json
import hashlib

# --- КОНФИГУРАЦИЯ ---
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/timescaledb_binance")
SYMBOL = "BTCUSDT"
TIMEFRAME = 5 # 5 minutes
HORIZON = 10  # Предсказываем на 10 свечей вперед
MODEL_DIR = "../models"

# Список фичей, которые ДОЛЖНЫ совпадать с тем, что вы подаете в Rust (FeatureView)
FEATURES = [
    'close', 'high', 'low', 'open',
    'rsi', 'macd', 'macd_signal', 'macd_hist',
    'ema_20', 'ema_50', 'ema_200', 'sma',
    'bb_upper', 'bb_lower', 'bb_mid',
    'atr', 'adx', 'vwap', 'obv', 'cci',
    'stoch_k', 'stoch_d', 'williams',
    'trend', 'trend_short', 'volume'
]

def load_data():
    print(f"Connecting to {DB_URL}...")
    engine = sqlalchemy.create_engine(DB_URL)
    
    # Формируем SQL запрос. Важно: порядок колонок должен быть жестким, 
    # либо в Rust мы должны маппить их по именам.
    cols = ", ".join(FEATURES)
    query = f"""
        SELECT {cols}
        FROM market.indicators_wide
        WHERE symbol = '{SYMBOL}' AND tf_minutes = {TIMEFRAME}
        ORDER BY time ASC
    """
    print("Fetching data...")
    df = pd.read_sql(query, engine)
    
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
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    print("Cross-validating...")
    scores = []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        print(f"  Split score: {score:.4f}")

    print(f"Average R^2 Score on cross-validation: {np.mean(scores):.4f}")

    # Final fit on all data
    print("Fitting final model on all data...")
    model.fit(X, y)
    
    return model

def export_to_onnx(model, filename, features, schema_id):
    print(f"Exporting to {filename}...")
    
    # Описываем входной тензор: [None (любой batch size), кол-во фичей]
    initial_type = [('float_input', FloatTensorType([None, len(features)]))]
    
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

def train_level_model(df):
    print("Training Level Prediction Model...")
    
    # Target: классификация (пробой/отскок от уровня)
    # Для примера, создадим искусственную целевую переменную
    # В реальности это будет более сложная логика на основе SR уровней
    
    # Создаем фиктивные уровни (в реальности они будут из SR уровней)
    df['avg_price'] = df['close'].rolling(window=20).mean()
    df['is_near_level'] = abs(df['close'] - df['avg_price']) < (df['atr'] * 0.5)  # Рядом с уровнем
    
    # Целевая переменная: 1 если пробой, 0 если отскок (упрощенно)
    df['next_direction'] = (df['close'].shift(-HORIZON) > df['close']).astype(int)
    
    y = df['next_direction']
    X = df[FEATURES]
    
    # Удаляем последние N строк, где нет target
    X = X.iloc[:-HORIZON]
    y = y.iloc[:-HORIZON]
    
    # Убираем строки с NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]
    
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
    print(f"Accuracy Score on train set: {score:.4f}")
    
    return model

if __name__ == "__main__":
    try:
        df = load_data()
        if len(df) < 200:
            print("Not enough data to train! Run the bot to collect candles first.")
        else:
            # Create schema_id from FEATURES list
            feature_string = ",".join(FEATURES)
            schema_id = hashlib.sha256(feature_string.encode('utf-8')).hexdigest()

            # Обучаем модель предсказания цены
            price_model = train_price_model(df.copy())
            export_to_onnx(price_model, "price_v1.onnx", FEATURES, schema_id)
            
            # Обучаем модель предсказания уровней
            level_model = train_level_model(df.copy())
            export_to_onnx(level_model, "levels_v1.onnx", FEATURES, schema_id)
            
            print("Training complete.")
    except Exception as e:
        print(f"Error: {e}")
