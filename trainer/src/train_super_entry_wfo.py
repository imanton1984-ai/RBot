#!/usr/bin/env python3
"""
train_super_entry_wfo.py — Walk-Forward Optimization (WFO) training for Super Entry Strategy.

Replaces the naive pair-based split with professional time-based Walk-Forward Validation
with Purge-Embargo to prevent data leakage.

=== METHODOLOGY ===
Walk-Forward Optimization splits the time series into sequential train→test windows:

  [=== TRAIN 1 ===][TEST 1]
  [======= TRAIN 2 ========][TEST 2]
  [============ TRAIN 3 ============][TEST 3]
  ...

Key properties:
  - Training window EXPANDS (more data → better model each fold)
  - Test windows are contiguous, non-overlapping (cover the full OOS period)
  - Purge: remove training rows whose labels (lookahead=25 bars) leak into test period
  - Embargo: skip test rows whose features (lookback=50 bars) reach into training data
  - All symbols split at the SAME timestamps (no cross-sectional leakage)
  - Combined purge+embargo only trims rows near the boundary (not a blanket time gap)

=== WFO FOLD DESIGN (per TF) ===
Calculated from actual data availability:

  TF 1m:   3.5 days  → 2 folds × 0.75d test  | purge=25min, embargo=50min
  TF 5m:   18 days   → 3 folds × 3.5d test   | purge=2h, embargo=4.2h
  TF 15m:  124 days  → 5 folds × 14d test     | purge=6.3h, embargo=12.5h
  TF 60m:  488 days  → 5 folds × 60d test     | purge=25h, embargo=50h
  TF 240m: 1946 days → 6 folds × 120d test    | purge=100h, embargo=200h
  TF 1440m: 2052 days → 5 folds × 180d test   | purge=25d, embargo=50d

=== OUTPUT ===
  - models/super_entry_v1_tf{X}.ubj    — Final P(super) model (trained on ALL data)
  - models/super_dir_v1_tf{X}.ubj      — Final P(direction) model (trained on ALL data)
  - models/wfo_report_tf{X}.json       — Per-fold and aggregate OOS metrics
  - models/wfo_folds/super_entry_tf{X}_fold{N}.ubj — Per-fold models (optional)

Usage:
    python trainer/src/train_super_entry_wfo.py [--csv dataset/super_entry_dataset.csv] [--gpu]
    python trainer/src/train_super_entry_wfo.py --timeframes 15,60,240 --gpu
    python trainer/src/train_super_entry_wfo.py --evaluate-only  # WFO metrics without final model
"""

import argparse
import hashlib
import json
import os
import sys
import time as time_module
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score,
        f1_score, log_loss,
    )
except ImportError:
    print("ERROR: Required packages not installed.")
    print("  pip install xgboost scikit-learn pandas numpy")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE DEFINITIONS — must match Rust config.rs exactly
# ═════════════════════════════════════════════════════════════════════════════

INDICATOR_FEATURES = [
    "rsi", "cci", "stoch_k", "stoch_d", "williams",
    "macd", "macd_signal", "macd_hist",
    "adx", "sma", "ema_20", "ema_50", "ema_200",
    "bb_upper", "bb_mid", "bb_lower", "atr",
    "obv", "vwap", "volume_spike",
    "trend", "trend_short", "poc",
    "alligator_jaw", "alligator_teeth", "alligator_lips",
    "mfi", "fibo_pivot", "fibo_r1", "fibo_s1",
    "supertrend", "supertrend_dir", "cmf",
]

DERIVED_FEATURES = [
    "rsi_norm", "cci_norm", "stoch_norm", "williams_norm",
    "bb_position", "bb_width_pct", "atr_pct",
    "price_vs_sma", "price_vs_ema20", "price_vs_ema50",
    "price_vs_ema200", "price_vs_vwap",
    "macd_norm", "obv_change_pct", "volume_spike_flag",
    "mfi_norm", "price_vs_fibo_pivot", "price_vs_supertrend",
    "alligator_spread",
]

DYNAMIC_FEATURES = [
    # Window 3
    "price_return_lb3", "atr_ratio_lb3", "rsi_slope_lb3",
    "trend_persist_lb3", "trend_short_persist_lb3",
    "adx_slope_lb3", "macd_hist_slope_lb3", "ema20_direction_lb3",
    # Window 5
    "price_return_lb5", "atr_ratio_lb5", "rsi_slope_lb5",
    "trend_persist_lb5", "trend_short_persist_lb5",
    "adx_slope_lb5", "macd_hist_slope_lb5", "ema20_direction_lb5",
    # Window 10
    "price_return_lb10", "atr_ratio_lb10", "rsi_slope_lb10",
    "trend_persist_lb10", "trend_short_persist_lb10",
    "adx_slope_lb10", "macd_hist_slope_lb10", "ema20_direction_lb10",
    # Window 15
    "price_return_lb15", "atr_ratio_lb15", "rsi_slope_lb15",
    "trend_persist_lb15", "trend_short_persist_lb15",
    "adx_slope_lb15", "macd_hist_slope_lb15", "ema20_direction_lb15",
    # Window 25
    "price_return_lb25", "atr_ratio_lb25", "rsi_slope_lb25",
    "trend_persist_lb25", "trend_short_persist_lb25",
    "adx_slope_lb25", "macd_hist_slope_lb25", "ema20_direction_lb25",
    # Window 50
    "price_return_lb50", "atr_ratio_lb50", "rsi_slope_lb50",
    "trend_persist_lb50", "trend_short_persist_lb50",
    "adx_slope_lb50", "macd_hist_slope_lb50", "ema20_direction_lb50",
    # Aggregate (6)
    "supertrend_consistency", "trend_alignment",
    "price_accel", "volume_trend_ratio",
    "ema_convergence_change", "high_low_pressure",
    # v3: Rate-of-Change & Momentum Dynamics (14 features)
    # Price RoC — short-term momentum critical for direction
    "price_roc_lb1", "price_roc_lb2", "price_accel_1bar", "price_accel_3bar",
    # Volume RoC — volume dynamics
    "volume_roc_lb1", "volume_roc_lb3", "volume_roc_lb5", "volume_roc_lb10", "volume_accel",
    # BB Squeeze — volatility compression breakout predictor
    "bb_squeeze_pctl",
    # OBV Divergence — smart money detection
    "obv_price_divergence",
    # MACD Histogram Acceleration
    "macd_hist_roc_lb1", "macd_hist_roc_lb3", "macd_hist_accel",
]

STATIC_FEATURES = INDICATOR_FEATURES + DERIVED_FEATURES
ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

TIMEFRAMES = [1, 5, 15, 60, 240, 1440]

TF_TARGET_MOVE_PCT = {
    1: 1.2, 5: 2.8, 15: 3.5, 60: 5.0, 240: 7.5, 1440: 10.0,
}


# ═════════════════════════════════════════════════════════════════════════════
# WFO CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# From Rust config
LOOKAHEAD_BARS = 25
MAX_DYNAMIC_LOOKBACK = 50

# WFO parameters per timeframe.
#
# Designed based on actual data ranges in the dataset:
#   1m:    2026-03-10 .. 2026-03-13   (~3.5 days,   706K rows, 151 symbols)
#   5m:    2026-02-25 .. 2026-03-15   (~18 days,    783K rows, 151 symbols)
#   15m:   2025-11-11 .. 2026-03-15   (~124 days,  1.67M rows, 151 symbols)
#   60m:   2024-11-11 .. 2026-03-14   (~488 days,  1.32M rows, 149 symbols)
#   240m:  2020-11-09 .. 2026-03-09   (~1946 days, 726K rows,  144 symbols)
#   1440m: 2020-07-04 .. 2026-02-15   (~2052 days,  92K rows,  102 symbols)
#
# Key design constraints:
#   - test_days: long enough for statistical significance (min ~50 symbols × 100+ bars)
#   - min_train_days: enough data for XGBoost to learn without overfitting
#   - Not too many folds (training time), not too few (unstable metrics)
#   - Purge/embargo proportional to TF granularity

WFO_CONFIG: Dict[int, dict] = {
    1: {
        "n_splits": 2,
        "test_days": 0.75,        # 18 hours per fold
        "min_train_days": 1.5,    # 1.5 days minimum training
        "desc": "1m: very short history, 2 folds — limited statistical power",
    },
    5: {
        "n_splits": 3,
        "test_days": 3.5,         # 3.5 days per fold (~1008 candles/symbol)
        "min_train_days": 6.0,    # 6 days minimum training
        "desc": "5m: 18 days, 3 folds with expanding window",
    },
    15: {
        "n_splits": 5,
        "test_days": 14.0,        # 2 weeks per fold (~1344 candles/symbol)
        "min_train_days": 30.0,   # 1 month minimum training
        "desc": "15m: 4 months, 5 folds with fortnightly test windows",
    },
    60: {
        "n_splits": 5,
        "test_days": 60.0,        # 2 months per fold (~1440 candles/symbol)
        "min_train_days": 120.0,  # 4 months minimum training
        "desc": "1h: 16 months, 5 folds with bimonthly test windows",
    },
    240: {
        "n_splits": 6,
        "test_days": 120.0,       # 4 months per fold (~720 candles/symbol)
        "min_train_days": 365.0,  # 1 year minimum training
        "desc": "4h: 5.3 years, 6 folds with quarterly test windows",
    },
    1440: {
        "n_splits": 5,
        "test_days": 180.0,       # 6 months per fold (~180 candles/symbol)
        "min_train_days": 365.0,  # 1 year minimum training
        "desc": "1d: 5.6 years, 5 folds with semi-annual test windows",
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# WFO FOLD COMPUTATION (Time-Based with Purge & Embargo)
# ═════════════════════════════════════════════════════════════════════════════

def compute_wfo_folds(
    timestamps: pd.Series,
    tf_minutes: int,
    config: dict,
) -> List[Dict]:
    """
    Compute Walk-Forward fold boundaries based on global timestamps.

    Expanding window: each fold trains on ALL data from the beginning.
    Test windows are contiguous, non-overlapping, covering the end of the timeline.

    Purge & Embargo are applied to individual rows (not as a blanket time gap):
      - Purge: drop training rows within `LOOKAHEAD_BARS * tf_min` of the boundary
      - Embargo: skip test rows within `MAX_DYNAMIC_LOOKBACK * tf_min` of the boundary

    Timeline layout:
      [==========  TRAIN_fold1  ==========][test_1][test_2]...[test_N]
                                            <----- total test span ----->

    Returns list of fold dicts with raw boundaries (purge/embargo applied at split time).
    """
    n_splits = config["n_splits"]
    test_days = config["test_days"]
    min_train_days = config["min_train_days"]

    test_delta = timedelta(days=test_days)
    min_train_delta = timedelta(days=min_train_days)

    t_min = timestamps.min()
    t_max = timestamps.max()
    total_span = t_max - t_min

    # Total time allocated for test windows (from end of timeline)
    total_test_span = n_splits * test_delta
    initial_train_end = t_max - total_test_span

    if (initial_train_end - t_min) < min_train_delta:
        # Not enough training data — reduce n_splits or test_days
        avail_test = total_span - min_train_delta
        possible_splits = int(avail_test / test_delta)
        if possible_splits < 1:
            print(f"    ⚠️ Not enough data for any WFO fold. Need {min_train_days}d train + "
                  f"{test_days}d test, have {total_span.days:.1f}d total.")
            return []
        print(f"    ⚠️ Reducing n_splits from {n_splits} to {possible_splits} "
              f"(min_train={min_train_days}d constraint)")
        n_splits = possible_splits
        total_test_span = n_splits * test_delta
        initial_train_end = t_max - total_test_span

    folds = []
    for i in range(n_splits):
        # Test window for fold i
        test_start_raw = initial_train_end + i * test_delta
        test_end_raw = initial_train_end + (i + 1) * test_delta

        # Training: from data start to the test boundary
        train_start = t_min
        train_end_raw = test_start_raw  # contiguous boundary

        # Validate training period
        if (train_end_raw - train_start) < min_train_delta:
            print(f"    ⚠️ Fold {i}: training span < minimum. Skipping.")
            continue

        folds.append({
            "fold_idx": i,
            "train_start": train_start,
            "train_end_raw": train_end_raw,
            "test_start_raw": test_start_raw,
            "test_end_raw": test_end_raw,
        })

    return folds


def apply_fold_split(
    df: pd.DataFrame,
    fold: Dict,
    tf_minutes: int,
    ts_col: str = "ts",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/test for a WFO fold with purge & embargo.

    Purge: remove training rows whose forward labels overlap the test period.
      Rows where timestamp > train_end - lookahead_bars * bar_duration are dropped.

    Embargo: skip test rows whose backward features reach into training data.
      Rows where timestamp < test_start + max_dynamic_lookback * bar_duration are dropped.
    """
    purge_delta = timedelta(minutes=LOOKAHEAD_BARS * tf_minutes)
    embargo_delta = timedelta(minutes=MAX_DYNAMIC_LOOKBACK * tf_minutes)

    # Effective boundaries (after purge & embargo)
    effective_train_end = fold["train_end_raw"] - purge_delta
    effective_test_start = fold["test_start_raw"] + embargo_delta

    # Training: [train_start ... effective_train_end]
    train_mask = (df[ts_col] >= fold["train_start"]) & (df[ts_col] <= effective_train_end)

    # Testing: [effective_test_start ... test_end_raw]
    test_mask = (df[ts_col] >= effective_test_start) & (df[ts_col] <= fold["test_end_raw"])

    return df[train_mask].copy(), df[test_mask].copy()


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING & EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def train_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    use_gpu: bool = False,
    model_name: str = "",
    custom_params: Optional[dict] = None,
    num_boost_round: int = 500,
    early_stopping_rounds: int = 30,
) -> Tuple[xgb.Booster, dict]:
    """Train binary XGBoost classifier and evaluate on out-of-sample data."""

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    # Clean NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Class balance
    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    scale_pos_weight = n_neg / max(n_pos, 1.0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "verbosity": 0,
    }

    if custom_params:
        params.update(custom_params)

    print(f"    Training {model_name}... "
          f"(train={len(y_train)}, test={len(y_test)}, pos={n_pos:.0f}/{len(y_train)} "
          f"= {n_pos/len(y_train)*100:.1f}%)")

    evals = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )

    # Evaluate OOS
    y_pred_prob = model.predict(dtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics: Dict[str, float] = {}
    if len(np.unique(y_test)) > 1:
        metrics["auc"] = float(roc_auc_score(y_test, y_pred_prob))
        metrics["precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["logloss"] = float(log_loss(y_test, y_pred_prob))
    else:
        metrics["auc"] = 0.5
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0
        metrics["logloss"] = 1.0

    metrics["train_size"] = int(len(y_train))
    metrics["test_size"] = int(len(y_test))
    metrics["train_pos_rate"] = float(n_pos / len(y_train))
    metrics["test_pos_rate"] = float(y_test.mean())
    metrics["best_iteration"] = int(model.best_iteration)

    print(f"      → AUC={metrics['auc']:.4f}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
          f"LogLoss={metrics['logloss']:.4f}  (trees={model.best_iteration + 1})")

    # Top-5 feature importance
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    metrics["top5_features"] = [name for name, _ in sorted_imp[:5]]

    return model, metrics


def save_model(
    model: xgb.Booster,
    feature_names: List[str],
    tf: int,
    task_name: str,
    metrics: dict,
    output_dir: str,
    suffix: str = "",
) -> str:
    """Save model + schema + metadata. Returns path to .ubj file."""
    os.makedirs(output_dir, exist_ok=True)

    model_name = f"{task_name}_v1_tf{tf}{suffix}"

    ubj_path = os.path.join(output_dir, f"{model_name}.ubj")
    model.save_model(ubj_path)

    json_path = os.path.join(output_dir, f"{model_name}.json")
    model.save_model(json_path)

    schema = {
        "schema_id": hashlib.sha256(",".join(feature_names).encode()).hexdigest() + f"_tf{tf}",
        "features": feature_names,
        "task": task_name,
        "tf_minutes": tf,
        "outputs": 1,
        "objective": "binary:logistic",
        "training_method": "walk_forward_optimization",
    }
    schema_path = os.path.join(output_dir, f"{model_name}.schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    # Filter out non-serializable keys for metadata
    clean_metrics = {k: v for k, v in metrics.items()
                     if isinstance(v, (int, float, str, list, dict, bool))}

    meta = {
        "model_type": "xgboost_binary_classifier",
        "objective": "binary:logistic",
        "target": task_name,
        "n_features": len(feature_names),
        "metrics": clean_metrics,
        "best_iteration": model.best_iteration,
        "n_trees": model.best_iteration + 1,
        "strategy": "ml_entry_strategy",
        "training_method": "walk_forward_optimization",
        "target_move_pct": TF_TARGET_MOVE_PCT.get(tf, 0),
    }
    meta_path = os.path.join(output_dir, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return ubj_path


# ═════════════════════════════════════════════════════════════════════════════
# WFO AGGREGATION & STABILITY ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_fold_metrics(fold_metrics: List[dict], name: str) -> dict:
    """
    Aggregate OOS metrics across WFO folds.

    Computes: mean, std, min, max, coefficient of variation, performance trend.
    A stable model has low CV and non-negative trend.
    """
    if not fold_metrics:
        return {}

    metric_names = ["auc", "precision", "recall", "f1", "logloss"]
    agg: Dict[str, float] = {}

    for m in metric_names:
        values = [f[m] for f in fold_metrics if m in f and f.get("test_size", 0) > 0]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        agg[f"{m}_mean"] = float(arr.mean())
        agg[f"{m}_std"] = float(arr.std())
        agg[f"{m}_min"] = float(arr.min())
        agg[f"{m}_max"] = float(arr.max())

        # Coefficient of variation (lower = more stable)
        if arr.mean() > 1e-8:
            agg[f"{m}_cv"] = float(arr.std() / arr.mean())
        else:
            agg[f"{m}_cv"] = float("inf")

        # Linear trend across folds (positive = improving)
        if len(values) >= 3:
            x = np.arange(len(values), dtype=np.float64)
            slope = np.polyfit(x, arr, 1)[0]
            agg[f"{m}_trend_per_fold"] = float(slope)

    total_test = sum(f.get("test_size", 0) for f in fold_metrics)
    agg["total_oos_samples"] = total_test
    agg["n_valid_folds"] = len(fold_metrics)

    # Print summary
    print(f"    {name} (across {len(fold_metrics)} folds, {total_test} OOS samples):")
    for m in metric_names:
        if f"{m}_mean" in agg:
            trend_str = ""
            if f"{m}_trend_per_fold" in agg:
                t = agg[f"{m}_trend_per_fold"]
                arrow = "↑" if t > 0.005 else ("↓" if t < -0.005 else "→")
                trend_str = f"  trend={arrow}{abs(t):.4f}/fold"
            print(f"      {m:12s}  {agg[f'{m}_mean']:.4f} ± {agg[f'{m}_std']:.4f}  "
                  f"[{agg[f'{m}_min']:.4f} .. {agg[f'{m}_max']:.4f}]  "
                  f"CV={agg.get(f'{m}_cv', 0):.3f}{trend_str}")

    return agg


def collect_feature_importance(fold_metrics: List[dict]) -> List[Tuple[str, int]]:
    """
    Count how often each feature appears in top-5 across folds.
    Returns sorted list of (feature_name, count).
    """
    counts: Dict[str, int] = {}
    for fm in fold_metrics:
        for feat in fm.get("top5_features", []):
            counts[feat] = counts.get(feat, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# WFO TRAINING PIPELINE (per TF)
# ═════════════════════════════════════════════════════════════════════════════

def train_wfo_for_tf(
    tf: int,
    tf_df: pd.DataFrame,
    use_gpu: bool = False,
    output_dir: str = "models",
    save_fold_models: bool = True,
    evaluate_only: bool = False,
) -> dict:
    """
    Run full Walk-Forward Optimization for one timeframe.

    Steps:
      1. Compute WFO fold boundaries from timestamp range
      2. For each fold: apply purge+embargo, train models, evaluate OOS
      3. Aggregate OOS metrics across folds
      4. (Optional) Train final production model on ALL data
      5. Save WFO report

    Args:
        tf: timeframe in minutes
        tf_df: DataFrame for this TF (all symbols)
        use_gpu: use CUDA for XGBoost
        output_dir: where to save models
        save_fold_models: save per-fold models (disk-intensive)
        evaluate_only: only compute WFO metrics, skip final model training

    Returns: WFO report dict
    """
    config = WFO_CONFIG.get(tf)
    if config is None:
        print(f"  No WFO config for TF {tf}m. Skipping.")
        return {}

    t_start = time_module.time()

    print(f"\n{'═' * 75}")
    print(f"  WFO TRAINING — TF {tf}m")
    print(f"  {config['desc']}")
    print(f"  Data: {len(tf_df)} rows, {tf_df['symbol'].nunique()} symbols")
    print(f"  Folds: {config['n_splits']} × {config['test_days']}d test, "
          f"min_train={config['min_train_days']}d")
    print(f"  Purge: {LOOKAHEAD_BARS} bars × {tf}min = "
          f"{LOOKAHEAD_BARS * tf / 60:.1f}h")
    print(f"  Embargo: {MAX_DYNAMIC_LOOKBACK} bars × {tf}min = "
          f"{MAX_DYNAMIC_LOOKBACK * tf / 60:.1f}h")
    print(f"{'═' * 75}")

    # Parse timestamps
    tf_df = tf_df.copy()
    tf_df["ts"] = pd.to_datetime(tf_df["timestamp"], utc=True)

    # Compute fold boundaries
    folds = compute_wfo_folds(tf_df["ts"], tf, config)

    if not folds:
        print(f"  ❌ No valid folds for TF {tf}m!")
        return {}

    print(f"\n  {len(folds)} Walk-Forward folds:")
    for fold in folds:
        train_days = (fold["train_end_raw"] - fold["train_start"]).total_seconds() / 86400
        test_days_actual = (fold["test_end_raw"] - fold["test_start_raw"]).total_seconds() / 86400
        print(f"    Fold {fold['fold_idx']}: "
              f"Train [{fold['train_start'].strftime('%Y-%m-%d')} → "
              f"{fold['train_end_raw'].strftime('%Y-%m-%d')}] ({train_days:.0f}d) | "
              f"Test [{fold['test_start_raw'].strftime('%Y-%m-%d')} → "
              f"{fold['test_end_raw'].strftime('%Y-%m-%d')}] ({test_days_actual:.0f}d)")

    # ── Per-fold training & evaluation ──
    super_fold_metrics: List[dict] = []
    dir_fold_metrics: List[dict] = []
    fold_details: List[dict] = []

    fold_models_dir = os.path.join(output_dir, "wfo_folds")
    if save_fold_models:
        os.makedirs(fold_models_dir, exist_ok=True)

    for fold in folds:
        fi = fold["fold_idx"]
        print(f"\n  ── Fold {fi}/{len(folds) - 1} ──")

        # Split with purge & embargo
        train_df, test_df = apply_fold_split(tf_df, fold, tf, ts_col="ts")

        if len(train_df) < 100 or len(test_df) < 30:
            print(f"    ⚠️ Too few samples after purge/embargo "
                  f"(train={len(train_df)}, test={len(test_df)}). Skipping fold.")
            continue

        train_super_rate = train_df["is_super"].mean() * 100
        test_super_rate = test_df["is_super"].mean() * 100
        print(f"    Train: {len(train_df)} rows, {train_df['symbol'].nunique()} symbols, "
              f"super={train_super_rate:.1f}%")
        print(f"    Test:  {len(test_df)} rows, {test_df['symbol'].nunique()} symbols, "
              f"super={test_super_rate:.1f}%")

        # === Train P(super) model ===
        super_model, super_metrics = train_binary(
            train_df, test_df,
            label_col="is_super",
            feature_cols=ALL_FEATURES,
            use_gpu=use_gpu,
            model_name=f"P(super) fold {fi}",
        )
        super_fold_metrics.append(super_metrics)

        if save_fold_models:
            save_model(super_model, ALL_FEATURES, tf, "super_entry",
                       super_metrics, fold_models_dir, suffix=f"_fold{fi}")

        # === Train P(direction) model (super-only, conservative) ===
        # Train ONLY on is_super=True examples — non-super have ambiguous direction.
        # Conservative regularization preserves directional signal quality.
        train_super = train_df[train_df["is_super"] == 1].copy()
        test_super = test_df[test_df["is_super"] == 1].copy()

        dir_metrics: dict = {
            "auc": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "logloss": 1.0, "train_size": len(train_super),
            "test_size": len(test_super),
        }

        if len(train_super) >= 50 and len(test_super) >= 20:
            train_super.loc[:, "label_long"] = (train_super["direction"] == 1).astype(int)
            test_super.loc[:, "label_long"] = (test_super["direction"] == 1).astype(int)

            # Direction model: ANTI-OVERFIT conservative params
            dir_params = {
                "eta": 0.01,
                "max_depth": 5,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 50,
                "lambda": 5.0,
                "alpha": 1.0,
                "gamma": 0.5,
            }

            dir_model, dir_metrics = train_binary(
                train_super, test_super,
                label_col="label_long",
                feature_cols=ALL_FEATURES,
                use_gpu=use_gpu,
                model_name=f"P(dir) fold {fi}",
                custom_params=dir_params,
                num_boost_round=2000,
                early_stopping_rounds=80,
            )

            if save_fold_models:
                save_model(dir_model, ALL_FEATURES, tf, "super_dir",
                           dir_metrics, fold_models_dir, suffix=f"_fold{fi}")
        else:
            print(f"    ⚠️ Not enough super examples for direction model "
                  f"(train={len(train_super)}, test={len(test_super)}). Skipping.")

        dir_fold_metrics.append(dir_metrics)

        fold_details.append({
            "fold_idx": fi,
            "train_start": fold["train_start"].isoformat(),
            "train_end_raw": fold["train_end_raw"].isoformat(),
            "test_start_raw": fold["test_start_raw"].isoformat(),
            "test_end_raw": fold["test_end_raw"].isoformat(),
            "train_rows_after_purge": len(train_df),
            "test_rows_after_embargo": len(test_df),
            "train_symbols": int(train_df["symbol"].nunique()),
            "test_symbols": int(test_df["symbol"].nunique()),
            "super_metrics": super_metrics,
            "dir_metrics": dir_metrics,
        })

    # ══ AGGREGATE OOS METRICS ══
    print(f"\n  {'─' * 60}")
    print(f"  WFO Aggregate Out-of-Sample Metrics — TF {tf}m")
    print(f"  {'─' * 60}")

    super_agg = aggregate_fold_metrics(super_fold_metrics, "P(super)")
    dir_agg = aggregate_fold_metrics(dir_fold_metrics, "P(direction)")

    # Feature importance stability
    super_fi = collect_feature_importance(super_fold_metrics)
    dir_fi = collect_feature_importance(dir_fold_metrics)

    if super_fi:
        n_folds = len(super_fold_metrics)
        print(f"\n    Stable features P(super) (appeared in top-5 across {n_folds} folds):")
        for feat, cnt in super_fi[:8]:
            print(f"      {feat:35s}  {cnt}/{n_folds} folds")
    if dir_fi:
        n_folds = len(dir_fold_metrics)
        print(f"    Stable features P(dir) (appeared in top-5 across {n_folds} folds):")
        for feat, cnt in dir_fi[:8]:
            print(f"      {feat:35s}  {cnt}/{n_folds} folds")

    # ══ TRAIN FINAL PRODUCTION MODEL ══
    final_super_metrics = {}
    final_dir_metrics = {}

    if not evaluate_only:
        print(f"\n  ── FINAL model (trained on ALL data) ──")

        # For early stopping, use chronologically last 10% as validation
        # This doesn't affect OOS evaluation (already done above)
        n_val = max(int(len(tf_df) * 0.10), 100)
        val_df = tf_df.tail(n_val)
        train_all_df = tf_df.head(len(tf_df) - n_val)

        # Final P(super)
        final_super_model, final_super_metrics = train_binary(
            train_all_df, val_df,
            label_col="is_super",
            feature_cols=ALL_FEATURES,
            use_gpu=use_gpu,
            model_name="FINAL P(super)",
        )
        final_super_metrics["wfo_oos_auc_mean"] = super_agg.get("auc_mean", 0)
        final_super_metrics["wfo_oos_auc_std"] = super_agg.get("auc_std", 0)
        final_super_metrics["wfo_oos_f1_mean"] = super_agg.get("f1_mean", 0)

        save_model(final_super_model, ALL_FEATURES, tf, "super_entry",
                   final_super_metrics, output_dir)
        print(f"    ✅ Saved: {output_dir}/super_entry_v1_tf{tf}.ubj")

        # Final P(direction) — super-only, conservative
        all_super = tf_df[tf_df["is_super"] == 1].copy()
        if len(all_super) >= 100:
            n_val_dir = max(int(len(all_super) * 0.10), 50)
            val_dir = all_super.tail(n_val_dir)
            train_dir = all_super.head(len(all_super) - n_val_dir)

            train_dir.loc[:, "label_long"] = (train_dir["direction"] == 1).astype(int)
            val_dir.loc[:, "label_long"] = (val_dir["direction"] == 1).astype(int)

            dir_params = {
                "eta": 0.01, "max_depth": 5, "subsample": 0.7,
                "colsample_bytree": 0.7, "min_child_weight": 50,
                "lambda": 5.0, "alpha": 1.0, "gamma": 0.5,
            }

            final_dir_model, final_dir_metrics = train_binary(
                train_dir, val_dir,
                label_col="label_long",
                feature_cols=ALL_FEATURES,
                use_gpu=use_gpu,
                model_name="FINAL P(dir)",
                custom_params=dir_params,
                num_boost_round=2000,
                early_stopping_rounds=80,
            )
            final_dir_metrics["wfo_oos_auc_mean"] = dir_agg.get("auc_mean", 0)
            final_dir_metrics["wfo_oos_auc_std"] = dir_agg.get("auc_std", 0)

            save_model(final_dir_model, ALL_FEATURES, tf, "super_dir",
                       final_dir_metrics, output_dir)
            print(f"    ✅ Saved: {output_dir}/super_dir_v1_tf{tf}.ubj")
        else:
            print(f"    ⚠️ Not enough super examples ({len(all_super)}) for final direction model.")

    # ══ SAVE WFO REPORT ══
    elapsed = time_module.time() - t_start
    report = {
        "timeframe_minutes": tf,
        "training_method": "walk_forward_optimization",
        "wfo_config": {k: v for k, v in config.items()},
        "purge_bars": LOOKAHEAD_BARS,
        "embargo_bars": MAX_DYNAMIC_LOOKBACK,
        "purge_hours": LOOKAHEAD_BARS * tf / 60,
        "embargo_hours": MAX_DYNAMIC_LOOKBACK * tf / 60,
        "total_rows": len(tf_df),
        "total_symbols": int(tf_df["symbol"].nunique()),
        "timestamp_range": {
            "min": tf_df["ts"].min().isoformat(),
            "max": tf_df["ts"].max().isoformat(),
            "span_days": (tf_df["ts"].max() - tf_df["ts"].min()).total_seconds() / 86400,
        },
        "super_aggregate_oos": super_agg,
        "direction_aggregate_oos": dir_agg,
        "super_feature_importance": super_fi[:10] if super_fi else [],
        "direction_feature_importance": dir_fi[:10] if dir_fi else [],
        "fold_details": fold_details,
        "final_model_metrics": {
            "super_entry": {k: v for k, v in final_super_metrics.items()
                           if isinstance(v, (int, float, str, bool))},
            "super_dir": {k: v for k, v in final_dir_metrics.items()
                         if isinstance(v, (int, float, str, bool))},
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    report_path = os.path.join(output_dir, f"wfo_report_tf{tf}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📊 WFO report saved: {report_path}")
    print(f"  ⏱️  Elapsed: {elapsed:.1f}s")

    return report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk-Forward Optimization Training for Super Entry Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full WFO training (all TFs, CPU)
  python trainer/src/train_super_entry_wfo.py

  # GPU training, specific timeframes
  python trainer/src/train_super_entry_wfo.py --gpu --timeframes 15,60,240

  # Evaluate-only mode (no final model, just WFO metrics)
  python trainer/src/train_super_entry_wfo.py --evaluate-only --timeframes 60

  # Custom dataset path
  python trainer/src/train_super_entry_wfo.py --csv dataset/super_entry_dataset.csv --gpu
""",
    )
    parser.add_argument("--csv", default="dataset/super_entry_dataset.csv",
                        help="Path to super_entry_dataset.csv")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for models and reports")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for XGBoost training")
    parser.add_argument("--timeframes", default="all",
                        help="Comma-separated TFs to train (e.g., '15,60,240') or 'all'")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only compute WFO metrics, skip final model training")
    parser.add_argument("--no-fold-models", action="store_true",
                        help="Don't save per-fold models (saves disk space)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found!")
        print("Generate dataset first:")
        print("  cargo run --release -p ml_entry_strategy --bin super_entry_dataset")
        sys.exit(1)

    # Parse target timeframes
    if args.timeframes == "all":
        target_tfs = TIMEFRAMES
    else:
        target_tfs = sorted([int(x.strip()) for x in args.timeframes.split(",")])

    # ── Load data ──
    print(f"╔{'═' * 73}╗")
    print(f"║  Walk-Forward Optimization — Super Entry Strategy{' ' * 22}║")
    print(f"╚{'═' * 73}╝")
    print(f"\nLoading dataset from {args.csv}...")

    load_start = time_module.time()
    df = pd.read_csv(args.csv)
    load_elapsed = time_module.time() - load_start
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns in {load_elapsed:.1f}s")

    # Validate columns
    required = ALL_FEATURES + ["symbol", "tf_minutes", "direction",
                                "magnitude_pct", "is_super", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ Missing columns (filled with 0): {missing}")
        for c in missing:
            df[c] = 0.0

    # Dataset overview
    print(f"\n  Dataset overview:")
    for tf_val in sorted(df["tf_minutes"].unique()):
        sub = df[df["tf_minutes"] == tf_val]
        ts_range = ""
        if "timestamp" in sub.columns:
            ts_min = sub["timestamp"].min()
            ts_max = sub["timestamp"].max()
            ts_range = f"  [{str(ts_min)[:10]}..{str(ts_max)[:10]}]"
        marker = " ◄" if tf_val in target_tfs else ""
        print(f"    TF {tf_val:>5}m: {len(sub):>9,} rows, "
              f"{sub['symbol'].nunique():>3} symbols, "
              f"super={sub['is_super'].mean() * 100:.1f}%{ts_range}{marker}")

    # ── Run WFO for each TF ──
    all_reports: Dict[int, dict] = {}
    total_start = time_module.time()

    for tf in target_tfs:
        tf_df = df[df["tf_minutes"] == tf].copy()
        if len(tf_df) < 100:
            print(f"\n  Skipping TF {tf}m: only {len(tf_df)} rows (need ≥100)")
            continue

        report = train_wfo_for_tf(
            tf, tf_df,
            use_gpu=args.gpu,
            output_dir=args.output_dir,
            save_fold_models=not args.no_fold_models,
            evaluate_only=args.evaluate_only,
        )
        if report:
            all_reports[tf] = report

    total_elapsed = time_module.time() - total_start

    # ══ FINAL SUMMARY ══
    print(f"\n╔{'═' * 73}╗")
    print(f"║  WFO TRAINING COMPLETE — FINAL SUMMARY{' ' * 33}║")
    print(f"╚{'═' * 73}╝")

    for tf in sorted(all_reports.keys()):
        r = all_reports[tf]
        if not r:
            continue
        super_agg = r.get("super_aggregate_oos", {})
        dir_agg = r.get("direction_aggregate_oos", {})
        n_folds = super_agg.get("n_valid_folds", 0)

        print(f"\n  TF {tf}m ({r.get('total_rows', 0):,} rows, "
              f"{r.get('total_symbols', 0)} symbols, {n_folds} folds):")

        if super_agg:
            auc_m = super_agg.get("auc_mean", 0)
            auc_s = super_agg.get("auc_std", 0)
            f1_m = super_agg.get("f1_mean", 0)
            f1_s = super_agg.get("f1_std", 0)
            oos_n = super_agg.get("total_oos_samples", 0)
            print(f"    P(super):     AUC={auc_m:.4f}±{auc_s:.4f}  "
                  f"F1={f1_m:.4f}±{f1_s:.4f}  (OOS={oos_n:,})")

        if dir_agg:
            auc_m = dir_agg.get("auc_mean", 0)
            auc_s = dir_agg.get("auc_std", 0)
            f1_m = dir_agg.get("f1_mean", 0)
            f1_s = dir_agg.get("f1_std", 0)
            oos_n = dir_agg.get("total_oos_samples", 0)
            print(f"    P(direction): AUC={auc_m:.4f}±{auc_s:.4f}  "
                  f"F1={f1_m:.4f}±{f1_s:.4f}  (OOS={oos_n:,})")

    # Save combined report
    combined = {
        "training_method": "walk_forward_optimization",
        "lookahead_bars": LOOKAHEAD_BARS,
        "max_dynamic_lookback": MAX_DYNAMIC_LOOKBACK,
        "purge_embargo_bars": LOOKAHEAD_BARS + MAX_DYNAMIC_LOOKBACK,
        "evaluate_only": args.evaluate_only,
        "gpu": args.gpu,
        "total_elapsed_seconds": round(total_elapsed, 1),
        "per_tf": {str(k): v for k, v in all_reports.items()},
    }
    combined_path = os.path.join(args.output_dir, "wfo_combined_report.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    print(f"\n  📊 Combined report: {combined_path}")
    print(f"  🏷️  Models: {args.output_dir}/super_entry_v1_tf*.ubj")
    print(f"  ⏱️  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    mode = "EVALUATE-ONLY" if args.evaluate_only else "FULL TRAINING"
    print(f"  🏁 Done ({mode})")


if __name__ == "__main__":
    main()
