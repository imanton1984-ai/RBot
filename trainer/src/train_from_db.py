import os
import pandas as pd
import numpy as np
import xgboost as xgb
import sqlalchemy
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools import convert_xgboost
import onnx

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
    
    # Target: Цена через N свечей
    # shift(-HORIZON) берет цену "из будущего" и ставит в текущую строку для обучения
    y = df['close'].shift(-HORIZON)
    X = df[FEATURES]
    
    # Удаляем последние N строк, где нет target
    X = X.iloc[:-HORIZON]
    y = y.iloc[:-HORIZON]
    
    # Обучение
    model = xgb.XGBRegressor(
        n_estimators=100, 
        max_depth=5, 
        learning_rate=0.05, 
        objective='reg:squarederror',
        n_jobs=-1
    )
    model.fit(X, y)
    
    # Оценка (простая)
    score = model.score(X, y)
    print(f"R^2 Score on train set: {score:.4f}")
    
    return model

def export_to_onnx(model, filename):
    print(f"Exporting to {filename}...")
    
    # Описываем входной тензор: [None (любой batch size), кол-во фичей]
    initial_type = [('float_input', FloatTensorType([None, len(FEATURES)]))]
    
    onnx_model = convert_xgboost(model, initial_types=initial_type)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    full_path = os.path.join(MODEL_DIR, filename)
    onnx.save_model(onnx_model, full_path)
    print(f"Saved model to {full_path}")

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
            # Обучаем модель предсказания цены
            price_model = train_price_model(df)
            export_to_onnx(price_model, "price_v1.onnx")
            
            # Обучаем модель предсказания уровней
            level_model = train_level_model(df)
            export_to_onnx(level_model, "levels_v1.onnx")
            
            print("Training complete.")
    except Exception as e:
        print(f"Error: {e}")