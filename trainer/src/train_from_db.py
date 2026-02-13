import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd

from sqlalchemy import create_engine
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/timescaledb_binance")
MODELS_DIR = os.getenv("MODELS_DIR", "models")

TIMEFRAMES = [1, 5, 15, 60, 240, 1440]  # minutes
HORIZON = int(os.getenv("HORIZON_BARS", "10"))

FEATURES = [
    "rsi","cci","stoch_k","stoch_d", "williams",
    "macd","macd_signal","macd_hist","adx","sma","ema_20","ema_50","ema_200",
    "bb_upper","bb_mid","bb_lower","atr",
    "obv","vwap","volume_spike",
    "alligator_jaw","alligator_teeth","alligator_lips",
    "trend","trend_short","poc",
]

def ensure_dirs():
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(tf_minutes: int) -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)
    
    # Сопоставляем минуты с именами таблиц в БД
    if tf_minutes < 60:
        table_name = f"candles_{tf_minutes}m"
    elif tf_minutes == 60:
        table_name = "candles_1h"
    elif tf_minutes == 240:
        table_name = "candles_4h"
    elif tf_minutes == 1440:
        table_name = "candles_1d"
    else:
        table_name = f"candles_{tf_minutes}m" # на всякий случай

    q = f"""
    SELECT
      i.time,
      i.tf_minutes,
      i.symbol,
      c.close,
      i.rsi, i.cci, i.stoch_k, i.stoch_d, i.williams,
      i.macd, i.macd_signal, i.macd_hist, i.adx, i.sma, i.ema_20, i.ema_50, i.ema_200,
      i.bb_upper, i.bb_mid, i.bb_lower, i.atr,
      i.obv, i.vwap, i.volume_spike,
      i.alligator_jaw, i.alligator_teeth, i.alligator_lips,
      i.trend, i.trend_short, i.poc
    FROM market.indicators_wide i
    JOIN market.{table_name} c 
      ON c.symbol_id = i.symbol_id AND c.time = i.time
    WHERE i.tf_minutes = {tf_minutes}
    ORDER BY i.time ASC
    """
    
    df = pd.read_sql(q, engine)
    return df

def export_xgb(model, model_name: str, schema: dict):
    ensure_dirs()
    model_path = os.path.join(MODELS_DIR, model_name)
    schema_path = os.path.join(MODELS_DIR, model_name.replace(".ubj", ".json"))

    # save model
    model.save_model(model_path)

    # save schema/meta
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print(f"  saved: {model_path}")
    print(f"  saved: {schema_path}")

def train_price_model(df: pd.DataFrame):
    print("Training Price Model (regression return)...")

    y = (df["close"].shift(-HORIZON) / df["close"]) - 1.0
    X = df[FEATURES]

    y = y.replace([np.inf, -np.inf], 0.0)
    X = X.iloc[:-HORIZON]
    y = y.iloc[:-HORIZON]

    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 50:
        print(f"  Not enough rows: {len(X)}")
        return None

    # Check if CUDA is available for XGBoost
    try:
        # Test if GPU is available for XGBoost
        xgb.train({'tree_method': 'hist', 'device': 'cuda'}, 
                  xgb.DMatrix(np.random.random((10, 10)), label=np.random.random(10)), 
                  num_boost_round=1)
        tree_method = "hist"
        device = "cuda"
    except:
        tree_method = "hist"  # hist is generally the best method for CPU as well
        device = "cpu"

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method=tree_method,
        device=device  # New parameter instead of predictor
    )

    # CV (optional)
    n_splits = min(5, len(X) // 200)
    if n_splits >= 2:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for tr, te in tscv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            scores.append(model.score(X.iloc[te], y.iloc[te]))
        print(f"  CV mean score: {float(np.mean(scores)):.4f}")

    model.fit(X, y)
    print(f"  train score: {model.score(X, y):.4f}")
    return model

def train_level_model(df: pd.DataFrame):
    print("Training Levels Model (binary direction)...")

    # target: direction after HORIZON candles
    future = df["close"].shift(-HORIZON)
    y = (future > df["close"]).astype(int)

    X = df[FEATURES]
    X = X.iloc[:-HORIZON]
    y = y.iloc[:-HORIZON]

    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    if len(X) < 50:
        print(f"  Not enough rows: {len(X)}")
        return None

    # Check if CUDA is available for XGBoost
    try:
        # Test if GPU is available for XGBoost
        xgb.train({'tree_method': 'hist', 'device': 'cuda'}, 
                  xgb.DMatrix(np.random.random((10, 10)), label=np.random.random(10)), 
                  num_boost_round=1)
        tree_method = "hist"
        device = "cuda"
    except:
        tree_method = "hist"  # hist is generally the best method for CPU as well
        device = "cpu"

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method=tree_method,
        device=device  # New parameter instead of predictor
    )
    model.fit(X, y)
    print(f"  train acc: {model.score(X, y):.4f}")
    return model

if __name__ == "__main__":
    ensure_dirs()

    feature_string = ",".join(FEATURES)
    schema_base = hashlib.sha256(feature_string.encode("utf-8")).hexdigest()

    for tf in TIMEFRAMES:
        print(f"\n=== timeframe {tf}m ===")
        df = load_data(tf)
        if df is None or len(df) < 200:
            print("  Not enough data. Run collector first.")
            continue

        price = train_price_model(df.copy())
        if price is not None:
            schema = {
                "schema_id": f"{schema_base}_tf{tf}",
                "features": FEATURES,
                "task": "price_regression_return",
                "horizon_bars": HORIZON,
                "tf_minutes": tf,
            }
            export_xgb(price, f"price_v1_tf{tf}.ubj", schema)

        levels = train_level_model(df.copy())
        if levels is not None:
            schema = {
                "schema_id": f"{schema_base}_tf{tf}",
                "features": FEATURES,
                "task": "levels_binary_direction",
                "horizon_bars": HORIZON,
                "tf_minutes": tf,
            }
            export_xgb(levels, f"levels_v1_tf{tf}.ubj", schema)

    print("\nAll training completed.")