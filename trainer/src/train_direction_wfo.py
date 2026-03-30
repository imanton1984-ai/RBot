#!/usr/bin/env python3
"""
train_direction_wfo.py — Walk-Forward Optimization for Direction Model v4.

APPROACH (CNN-like Pattern Recognition on XGBoost):
  - Sliding window of W candles → flatten into wide row (W × features_per_candle)
  - All prices normalized relative to window[0].open → stationarity
  - No indicators! Only pure OHLCV price action patterns
  - XGBoost trees find combinations within the temporal window → mimics 1D-CNN

WHY THIS APPROACH:
  Previous v1-v3 with indicator features achieved ~0.50 accuracy (random).
  Indicators are lagging and noisy for short-term direction prediction.
  Raw normalized price patterns should contain more signal with less noise.

TRAINING:
  - Walk-Forward Optimization (expanding window):
    Train on past → Test on future → no lookahead bias
  - Purge + embargo between train/test splits
  - Regression on future return (sign = direction, magnitude = confidence)
  - Binary classification variant: UP vs DOWN (exclude FLAT)

Usage:
    python trainer/src/train_direction_wfo.py --csv dataset/direction_v4_dataset.csv --gpu
    python trainer/src/train_direction_wfo.py --timeframes 15,60 --evaluate-only
    python trainer/src/train_direction_wfo.py --csv dataset/direction_v4_dataset.csv --mode binary --gpu
"""

import argparse
import gc
import json
import logging
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
        accuracy_score, roc_auc_score, mean_squared_error,
        classification_report, confusion_matrix,
    )
except ImportError:
    print("ERROR: Required packages not installed.")
    print("  pip install xgboost scikit-learn pandas numpy")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging(log_path: str = "logs/direction_wfo_train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("direction_wfo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

log = setup_logging()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

TIMEFRAMES = [5, 15, 60, 240, 1440]

# WFO fold config per TF — same structure as v3, proven to work
WFO_CONFIG: Dict[int, dict] = {
    1:    {"n_splits": 2, "test_days": 0.75,  "min_train_days": 1.5},
    5:    {"n_splits": 3, "test_days": 3.5,   "min_train_days": 6.0},
    15:   {"n_splits": 5, "test_days": 14.0,  "min_train_days": 30.0},
    60:   {"n_splits": 5, "test_days": 60.0,  "min_train_days": 120.0},
    240:  {"n_splits": 6, "test_days": 120.0, "min_train_days": 365.0},
    1440: {"n_splits": 5, "test_days": 180.0, "min_train_days": 365.0},
}

# Default prediction horizon — MUST match Rust DirectionConfig (25 bars)
# and the dataset labels (built with horizon=25).
DEFAULT_HORIZON = 25

# Direction accuracy target
DIRECTION_ACC_TARGET = 0.55


# ═════════════════════════════════════════════════════════════════════════════
# XGBoost HYPERPARAMETERS — SINGLE SOURCE OF TRUTH
# ═════════════════════════════════════════════════════════════════════════════
#
# ROOT CAUSE ANALYSIS (from WFO logs):
# ─────────────────────────────────────
# 1. FINAL model (trained on ALL data, eval on last 10%) → 58-62% accuracy
#    OOS WFO folds → 50-52% accuracy → CLASSIC OVERFIT
#
# 2. Many folds stop at 1-5 trees because early stopping uses the OOS
#    test set, and distribution shift across time causes immediate logloss
#    degradation → model has ~50% accuracy with 1 tree = pure random.
#
# 3. The regression params (eta=0.555 etc.) were being changed but the
#    model runs in BINARY mode → regression params had zero effect.
#
# FIX STRATEGY:
# ─────────────
# A) LOW LEARNING RATE + MANY ROUNDS → gradual learning, smooth ensemble
# B) AGGRESSIVE REGULARIZATION → only the most robust patterns survive
# C) INTERNAL VALIDATION for early stopping (NOT OOS test!)
#    This prevents 1-tree models caused by train/test distribution shift.
# D) VERY SHALLOW TREES → prevents memorizing noise per-tree
# E) HEAVY SUBSAMPLING → decorrelated trees, better bagging effect

XGBOOST_PARAMS = {
    "eta": 0.1,                    # Very low LR → each tree adjusts by at most 1%
                                    # → needs 1000+ trees → ensemble averages out noise
    "max_depth": 5,                 # Max 8 leaves per tree → only broadest patterns
                                    # (depth 4 = 16 leaves was still overfitting)
    "min_child_weight": 50,        # Each leaf needs ≥200 samples → statistically stable
                                    # (on 300K+ datasets, this is <0.1% of data per leaf)
    "lambda": 5.0,                  # Strong L2 → predictions shrink toward 0.5 when unsure
                                    # Prevents confident wrong predictions
    "alpha": 1.0,                   # Moderate L1 → prunes weak features from splits
                                    # Not too strong (1.0 was killing useful features)
    "gamma": 0.5,                   # Each split must gain ≥0.5 → no trivial splits
                                    # (1.0 was too aggressive, killed most trees)
    "subsample": 0.7,              # Each tree trains on 50% of rows → decorrelated
    "colsample_bytree": 0.8,       # Each tree sees only 30% of 114 features → ~34 features
                                    # → massive diversity, prevents reliance on single feature
    "max_bin": 64,                  # Very coarse histograms → can't memorize exact price values
    "tree_method": "hist",
    "verbosity": 0,
}

NUM_BOOST_ROUND = 5000              # With eta=0.01, need many rounds to gradually converge
EARLY_STOPPING_ROUNDS = 200         # Very patient: low eta means slow improvement per round
                                    # Previous 100 was too impatient → premature stopping
VALIDATION_FRACTION = 0.2          # Hold out 20% of TRAIN data for early stopping
                                    # ╔═══════════════════════════════════════════════╗
                                    # ║  CRITICAL: Do NOT use OOS test for ES!        ║
                                    # ║  Using OOS for ES = data leakage + 1-tree bug ║
                                    # ╚═══════════════════════════════════════════════╝


# ═════════════════════════════════════════════════════════════════════════════
# WFO FOLD COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_wfo_folds(timestamps: pd.Series, tf_minutes: int, config: dict) -> List[Dict]:
    """Expanding-window WFO fold boundaries with purge + embargo."""
    n_splits = config["n_splits"]
    test_days = config["test_days"]
    min_train_days = config["min_train_days"]

    test_delta = timedelta(days=test_days)
    min_train_delta = timedelta(days=min_train_days)

    t_min = timestamps.min()
    t_max = timestamps.max()
    total_span = t_max - t_min

    total_test_span = n_splits * test_delta
    initial_train_end = t_max - total_test_span

    if (initial_train_end - t_min) < min_train_delta:
        avail_test = total_span - min_train_delta
        possible_splits = int(avail_test / test_delta)
        if possible_splits < 1:
            log.warning(f"Not enough data for WFO. Need {min_train_days}d train + "
                        f"{test_days}d test, have {total_span.days:.1f}d total.")
            return []
        log.info(f"Reducing n_splits from {n_splits} to {possible_splits}")
        n_splits = possible_splits
        total_test_span = n_splits * test_delta
        initial_train_end = t_max - total_test_span

    folds = []
    for i in range(n_splits):
        test_start_raw = initial_train_end + i * test_delta
        test_end_raw = initial_train_end + (i + 1) * test_delta
        train_start = t_min
        train_end_raw = test_start_raw

        if (train_end_raw - train_start) < min_train_delta:
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
    df: pd.DataFrame, fold: Dict, tf_minutes: int,
    prediction_horizon: int, ts_col: str = "ts",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split with purge & embargo. No lookahead bias."""
    # Purge: remove examples whose prediction window overlaps with test start
    purge_delta = timedelta(minutes=prediction_horizon * tf_minutes)
    # Embargo: skip a gap between train and test to avoid any leakage
    embargo_delta = timedelta(minutes=50 * tf_minutes)  # 50 bars buffer

    effective_train_end = fold["train_end_raw"] - purge_delta
    effective_test_start = fold["test_start_raw"] + embargo_delta

    train_mask = (df[ts_col] >= fold["train_start"]) & (df[ts_col] <= effective_train_end)
    test_mask = (df[ts_col] >= effective_test_start) & (df[ts_col] <= fold["test_end_raw"])

    return df[train_mask].copy(), df[test_mask].copy()


# ═════════════════════════════════════════════════════════════════════════════
# DETECT FEATURE COLUMNS from CSV header
# ═════════════════════════════════════════════════════════════════════════════

def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect pattern feature columns (w{i}_{type}, summary_*, htf_*)."""
    window_cols = [c for c in df.columns if c.startswith("w") and "_" in c]
    summary_cols = sorted([c for c in df.columns if c.startswith("summary_")])
    htf_cols = sorted([c for c in df.columns if c.startswith("htf_")])

    # Sort window cols by window index then feature name
    def sort_key(col):
        parts = col.split("_", 1)
        try:
            idx = int(parts[0][1:])
            return (idx, parts[1] if len(parts) > 1 else "")
        except ValueError:
            return (9999, col)
    window_cols.sort(key=sort_key)

    # Combine: window features + summary features + HTF features
    feat_cols = window_cols + summary_cols + htf_cols
    return feat_cols


# ═════════════════════════════════════════════════════════════════════════════
# HELPER: INTERNAL VALIDATION SPLIT (prevents early-stopping data leakage)
# ═════════════════════════════════════════════════════════════════════════════

def split_train_validation(
    X_train: np.ndarray, y_train: np.ndarray,
    val_fraction: float = VALIDATION_FRACTION,
    min_val: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split training data into fit + internal validation for early stopping.

    WHY: Using OOS test set for early stopping causes:
      1) Data leakage → OOS metrics are optimistically biased
      2) 1-tree models → when train/test distributions differ (common in finance),
         logloss on OOS degrades immediately → early stopping kills the model at tree 1.

    Solution: hold out 20% of training data (SAME temporal distribution as train)
    for early stopping. The OOS test set is only used for unbiased evaluation.
    """
    n_val = max(int(len(X_train) * val_fraction), min_val)
    n_val = min(n_val, len(X_train) // 2)  # Never use more than 50%

    X_fit = X_train[:-n_val]
    y_fit = y_train[:-n_val]
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]

    return X_fit, y_fit, X_val, y_val


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING: REGRESSION MODE
# ═════════════════════════════════════════════════════════════════════════════

def train_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_gpu: bool = False,
    fold_idx: int = 0,
    exclude_flat: bool = False,
) -> Tuple[Optional[xgb.Booster], dict]:
    """
    Regression on future_return_pct.
    Direction = sign(prediction), confidence = abs(prediction).
    """
    if exclude_flat:
        train_df = train_df[train_df["label"] != 0].copy()

    if len(train_df) < 200 or len(test_df) < 50:
        log.warning(f"  Fold {fold_idx}: Not enough data (train={len(train_df)}, test={len(test_df)}). Skipping.")
        return None, {"error": "not enough data"}

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["future_return_pct"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["future_return_pct"].values.astype(np.float32)
    labels_test = test_df["label"].values

    # Clean NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    y_test = np.nan_to_num(y_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Internal validation split for early stopping ──
    X_fit, y_fit, X_val, y_val = split_train_validation(X_train, y_train)

    dtrain = xgb.DMatrix(X_fit, label=y_fit, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # ── Regression params (shared base + regression-specific) ──
    params = {**XGBOOST_PARAMS}
    params["objective"] = "reg:squarederror"
    params["eval_metric"] = "rmse"
    params["device"] = "cuda" if use_gpu else "cpu"

    n_long = int((train_df["label"] == 1).sum())
    n_short = int((train_df["label"] == -1).sum())
    n_flat = int((train_df["label"] == 0).sum())
    log.info(f"  Fold {fold_idx}: Training regression on {len(y_train)} examples "
             f"(UP={n_long}, FLAT={n_flat}, DOWN={n_short}, fit={len(y_fit)}, val_es={len(y_val)})")

    model = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "valid")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=0,
    )

    # ── Evaluate on PURE OOS test (NOT used for early stopping) ──
    pred = model.predict(dtest)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    # Direction accuracy: sign(prediction) vs sign(actual return)
    pred_dir = np.sign(pred)
    actual_dir = np.sign(y_test)
    # Exclude zero returns for accuracy
    nonzero = actual_dir != 0
    if nonzero.sum() > 0:
        dir_acc = float((pred_dir[nonzero] == actual_dir[nonzero]).mean())
    else:
        dir_acc = 0.5

    # AUC
    dir_auc = 0.5
    if nonzero.sum() > 20:
        y_binary = (actual_dir[nonzero] > 0).astype(int)
        pred_scores = pred[nonzero]
        try:
            if len(np.unique(y_binary)) > 1:
                dir_auc = float(roc_auc_score(y_binary, pred_scores))
        except Exception:
            pass

    # Confidence gate analysis
    confidence_thresholds = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    gate_analysis = {}
    for thresh in confidence_thresholds:
        mask = np.abs(pred) >= thresh
        if mask.sum() > 20:
            gated_pred_dir = pred_dir[mask & nonzero]
            gated_actual_dir = actual_dir[mask & nonzero]
            if len(gated_pred_dir) > 0:
                gated_acc = float((gated_pred_dir == gated_actual_dir).mean())
                gate_analysis[f"gate_{thresh:.2f}"] = {
                    "accuracy": gated_acc,
                    "coverage": float(mask.sum()) / float(len(pred)),
                    "n_samples": int(mask.sum()),
                }

    # Top features
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top15 = [(name, round(gain, 2)) for name, gain in sorted_imp[:15]]

    metrics = {
        "mode": "regression",
        "dir_accuracy": dir_acc,
        "dir_auc": dir_auc,
        "rmse": rmse,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "best_iteration": int(model.best_iteration),
        "top15_features": top15,
        "confidence_gates": gate_analysis,
    }

    target_met = "✅" if dir_acc >= DIRECTION_ACC_TARGET else "❌"
    log.info(f"  Fold {fold_idx}: dir_acc={dir_acc:.4f} {target_met}  "
             f"AUC={dir_auc:.4f}  RMSE={rmse:.4f}  trees={model.best_iteration + 1}")

    for thresh in confidence_thresholds:
        key = f"gate_{thresh:.2f}"
        if key in gate_analysis:
            g = gate_analysis[key]
            log.info(f"    gate≥{thresh:.2f}: acc={g['accuracy']:.4f}  "
                     f"coverage={g['coverage']:.1%}  n={g['n_samples']}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING: BINARY CLASSIFICATION MODE
# ═════════════════════════════════════════════════════════════════════════════

def train_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_gpu: bool = False,
    fold_idx: int = 0,
) -> Tuple[Optional[xgb.Booster], dict]:
    """
    Binary classification: UP (1) vs DOWN (0).
    Excludes FLAT (label=0) from both train and test.
    """
    # Filter to only UP and DOWN
    train_bin = train_df[train_df["label"] != 0].copy()
    test_bin = test_df[test_df["label"] != 0].copy()

    # Convert labels: 1 → 1 (UP), -1 → 0 (DOWN)
    train_bin["y"] = (train_bin["label"] == 1).astype(int)
    test_bin["y"] = (test_bin["label"] == 1).astype(int)

    if len(train_bin) < 200 or len(test_bin) < 50:
        log.warning(f"  Fold {fold_idx}: Not enough UP/DOWN data "
                    f"(train={len(train_bin)}, test={len(test_bin)}). Skipping.")
        return None, {"error": "not enough data"}

    X_train = train_bin[feature_cols].values.astype(np.float32)
    y_train = train_bin["y"].values.astype(np.float32)
    X_test = test_bin[feature_cols].values.astype(np.float32)
    y_test = test_bin["y"].values.astype(np.float32)

    # Free intermediate DataFrames ASAP
    del train_bin, test_bin
    gc.collect()

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Internal validation split for early stopping ──
    # CRITICAL FIX: Previously used OOS test for early stopping, causing:
    #   1) Data leakage (model "sees" future data during training)
    #   2) 1-tree models (distribution shift → logloss degrades immediately)
    # Now: 20% of TRAIN data held out for ES. OOS test = pure evaluation only.
    X_fit, y_fit, X_val, y_val = split_train_validation(X_train, y_train)

    # Class balance weight (computed on fit portion)
    n_up_fit = int(y_fit.sum())
    n_down_fit = len(y_fit) - n_up_fit
    scale_pos = n_down_fit / max(n_up_fit, 1)

    # Total stats (for logging)
    n_up = int(y_train.sum())
    n_down = len(y_train) - n_up
    n_train_total = len(y_train)

    dtrain = xgb.DMatrix(X_fit, label=y_fit, feature_names=feature_cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Free numpy arrays — XGBoost DMatrix has its own copy
    del X_train, y_train, X_fit, y_fit, X_val, y_val, X_test
    gc.collect()

    # ── Binary params (shared base + binary-specific) ──
    params = {**XGBOOST_PARAMS}
    params["objective"] = "binary:logistic"
    params["eval_metric"] = "logloss"
    params["scale_pos_weight"] = scale_pos
    params["device"] = "cuda" if use_gpu else "cpu"

    log.info(f"  Fold {fold_idx}: Training binary (UP vs DOWN) on {n_train_total} examples "
             f"(UP={n_up}, DOWN={n_down}, scale_pos={scale_pos:.2f})")

    model = xgb.train(
        params, dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dval, "valid")],  # ← valid = INTERNAL, not OOS
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=0,
    )

    # ── Evaluate on PURE OOS test (NOT used for early stopping) ──
    pred_proba = model.predict(dtest)
    pred_class = (pred_proba >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, pred_class))
    try:
        auc = float(roc_auc_score(y_test, pred_proba))
    except Exception:
        auc = 0.5

    # Per-class analysis — critical for detecting UP/DOWN imbalance
    n_pred_up = int(pred_class.sum())
    n_pred_down = int(len(pred_class) - pred_class.sum())
    up_mask = y_test == 1
    down_mask = y_test == 0
    up_recall = float(pred_class[up_mask].mean()) if up_mask.sum() > 0 else 0.0
    down_recall = float((1 - pred_class[down_mask]).mean()) if down_mask.sum() > 0 else 0.0

    # Confidence gates — separately for UP and DOWN
    gate_analysis = {}
    confidence_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for thresh in confidence_thresholds:
        # Confident UP = P(UP) >= thresh
        confident_up = pred_proba >= thresh
        # Confident DOWN = P(DOWN) >= thresh ↔ P(UP) <= (1-thresh)
        confident_down = pred_proba <= (1 - thresh)
        confident = confident_up | confident_down

        gate_info = {}

        # Combined gate
        if confident.sum() > 10:
            gated_pred = pred_class[confident]
            gated_actual = y_test[confident]
            gated_acc = float(accuracy_score(gated_actual, gated_pred))
            gate_info["accuracy"] = gated_acc
            gate_info["coverage"] = float(confident.sum()) / float(len(pred_proba))
            gate_info["n_samples"] = int(confident.sum())

        # UP-only gate
        if confident_up.sum() > 5:
            up_pred = pred_class[confident_up]
            up_actual = y_test[confident_up]
            gate_info["up_accuracy"] = float(accuracy_score(up_actual, up_pred))
            gate_info["up_n"] = int(confident_up.sum())
            gate_info["up_coverage"] = float(confident_up.sum()) / float(len(pred_proba))

        # DOWN-only gate
        if confident_down.sum() > 5:
            down_pred = pred_class[confident_down]
            down_actual = y_test[confident_down]
            gate_info["down_accuracy"] = float(accuracy_score(down_actual, down_pred))
            gate_info["down_n"] = int(confident_down.sum())
            gate_info["down_coverage"] = float(confident_down.sum()) / float(len(pred_proba))

        if gate_info:
            gate_analysis[f"gate_{thresh:.2f}"] = gate_info

    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top15 = [(name, round(gain, 2)) for name, gain in sorted_imp[:15]]

    metrics = {
        "mode": "binary",
        "accuracy": accuracy,
        "auc": auc,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "n_pred_up": n_pred_up,
        "n_pred_down": n_pred_down,
        "train_size": n_train_total,
        "test_size": int(len(y_test)),
        "best_iteration": int(model.best_iteration),
        "top15_features": top15,
        "confidence_gates": gate_analysis,
    }

    target_met = "✅" if accuracy >= DIRECTION_ACC_TARGET else "❌"
    log.info(f"  Fold {fold_idx}: accuracy={accuracy:.4f} {target_met}  "
             f"AUC={auc:.4f}  trees={model.best_iteration + 1}")
    log.info(f"    Predictions: UP={n_pred_up} DOWN={n_pred_down}  "
             f"UP_recall={up_recall:.3f}  DOWN_recall={down_recall:.3f}")
    if n_pred_down == 0:
        log.warning(f"    ⚠️  Model predicts ZERO DOWN signals! Possible class collapse.")

    for thresh in confidence_thresholds:
        key = f"gate_{thresh:.2f}"
        if key in gate_analysis:
            g = gate_analysis[key]
            parts = []
            if "accuracy" in g:
                parts.append(f"all={g['accuracy']:.4f}({g['n_samples']})")
            if "up_accuracy" in g:
                parts.append(f"UP={g['up_accuracy']:.4f}({g['up_n']})")
            if "down_accuracy" in g:
                parts.append(f"DOWN={g['down_accuracy']:.4f}({g['down_n']})")
            cov = g.get("coverage", 0)
            log.info(f"    gate≥{thresh:.2f}: {' | '.join(parts)}  cov={cov:.1%}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# WFO PIPELINE (per TF)
# ═════════════════════════════════════════════════════════════════════════════

def train_wfo_for_tf(
    tf: int,
    tf_df: pd.DataFrame,
    feature_cols: List[str],
    mode: str = "regression",
    use_gpu: bool = False,
    output_dir: str = "models",
    evaluate_only: bool = False,
    prediction_horizon: int = DEFAULT_HORIZON,
    exclude_flat: bool = False,
) -> dict:
    """Run Walk-Forward Optimization for one timeframe."""
    config = WFO_CONFIG.get(tf)
    if config is None:
        log.warning(f"No WFO config for TF {tf}m. Skipping.")
        return {}

    t_start = time_module.time()
    n_features = len(feature_cols)

    log.info(f"\n{'═' * 75}")
    log.info(f"  DIRECTION v4 WFO — TF {tf}m — {mode.upper()} mode")
    log.info(f"  Data: {len(tf_df)} rows, {tf_df['symbol'].nunique()} symbols")
    log.info(f"  Features: {n_features} pattern features")
    n_up = int((tf_df["label"] == 1).sum())
    n_flat = int((tf_df["label"] == 0).sum())
    n_down = int((tf_df["label"] == -1).sum())
    log.info(f"  Labels: UP={n_up} ({n_up/len(tf_df)*100:.1f}%), "
             f"FLAT={n_flat} ({n_flat/len(tf_df)*100:.1f}%), "
             f"DOWN={n_down} ({n_down/len(tf_df)*100:.1f}%)")
    log.info(f"  Target: accuracy ≥ {DIRECTION_ACC_TARGET:.0%}")
    log.info(f"{'═' * 75}")

    tf_df = tf_df.copy()
    tf_df["ts"] = pd.to_datetime(tf_df["timestamp"], utc=True)

    folds = compute_wfo_folds(tf_df["ts"], tf, config)
    if not folds:
        log.error(f"No valid folds for TF {tf}m!")
        return {}

    log.info(f"  {len(folds)} WFO folds:")
    for fold in folds:
        train_days = (fold["train_end_raw"] - fold["train_start"]).total_seconds() / 86400
        test_days = (fold["test_end_raw"] - fold["test_start_raw"]).total_seconds() / 86400
        log.info(f"    Fold {fold['fold_idx']}: Train {train_days:.0f}d | Test {test_days:.0f}d")

    # Log hyperparameters
    log.info(f"  XGBoost: eta={XGBOOST_PARAMS['eta']}, depth={XGBOOST_PARAMS['max_depth']}, "
             f"mcw={XGBOOST_PARAMS['min_child_weight']}, lambda={XGBOOST_PARAMS['lambda']}, "
             f"alpha={XGBOOST_PARAMS['alpha']}, gamma={XGBOOST_PARAMS['gamma']}")
    log.info(f"  Sampling: sub={XGBOOST_PARAMS['subsample']}, col={XGBOOST_PARAMS['colsample_bytree']}, "
             f"bins={XGBOOST_PARAMS['max_bin']}")
    log.info(f"  Training: rounds={NUM_BOOST_ROUND}, es={EARLY_STOPPING_ROUNDS}, "
             f"val_frac={VALIDATION_FRACTION}")

    # ── Per-fold training ──
    fold_metrics: List[dict] = []
    fold_details: List[dict] = []

    for fold in folds:
        fi = fold["fold_idx"]
        log.info(f"\n  ── Fold {fi}/{len(folds) - 1} ──")

        train_df, test_df = apply_fold_split(tf_df, fold, tf, prediction_horizon, ts_col="ts")

        if len(train_df) < 200 or len(test_df) < 50:
            log.warning(f"  Too few samples (train={len(train_df)}, test={len(test_df)}). Skipping.")
            continue

        log.info(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        # Baseline: majority class accuracy
        test_labels = test_df["label"].values
        nonzero_test = test_labels != 0
        if nonzero_test.sum() > 0:
            majority_up = (test_labels[nonzero_test] == 1).mean()
            baseline = max(majority_up, 1 - majority_up)
        else:
            baseline = 0.5
        log.info(f"  Baseline (majority class): {baseline:.4f}")

        # Train model
        if mode == "binary":
            model, metrics = train_binary(
                train_df, test_df, feature_cols,
                use_gpu=use_gpu, fold_idx=fi,
            )
        else:
            model, metrics = train_regression(
                train_df, test_df, feature_cols,
                use_gpu=use_gpu, fold_idx=fi,
                exclude_flat=exclude_flat,
            )

        metrics["baseline"] = baseline
        fold_metrics.append(metrics)
        fold_details.append({
            "fold_idx": fi,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "baseline": baseline,
            "metrics": metrics,
        })

        # ── Memory cleanup between folds ──
        del train_df, test_df, model
        gc.collect()

    # ══ AGGREGATE ══
    log.info(f"\n  {'─' * 60}")
    log.info(f"  AGGREGATE OOS RESULTS — TF {tf}m ({mode})")
    log.info(f"  {'─' * 60}")

    valid_metrics = [m for m in fold_metrics if "error" not in m]
    if not valid_metrics:
        log.error("  No valid folds completed!")
        return {}

    if mode == "binary":
        accs = [m["accuracy"] for m in valid_metrics]
        aucs = [m["auc"] for m in valid_metrics]
        acc_key = "accuracy"
    else:
        accs = [m["dir_accuracy"] for m in valid_metrics]
        aucs = [m["dir_auc"] for m in valid_metrics]
        acc_key = "dir_accuracy"

    baselines = [m.get("baseline", 0.5) for m in valid_metrics]

    agg = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "acc_min": float(np.min(accs)),
        "acc_max": float(np.max(accs)),
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "baseline_mean": float(np.mean(baselines)),
    }

    met = "✅" if agg["acc_mean"] >= DIRECTION_ACC_TARGET else "❌"
    log.info(f"  Accuracy: {agg['acc_mean']:.4f}±{agg['acc_std']:.4f} {met}  "
             f"[{agg['acc_min']:.4f}..{agg['acc_max']:.4f}]")
    log.info(f"  AUC: {agg['auc_mean']:.4f}±{agg['auc_std']:.4f}")
    log.info(f"  Baseline: {agg['baseline_mean']:.4f}")
    lift = agg['acc_mean'] - agg['baseline_mean']
    log.info(f"  Lift over baseline: {lift:+.4f} {'📈' if lift > 0 else '📉'}")

    # Per-class recall analysis (binary mode)
    if mode == "binary":
        up_recalls = [m.get("up_recall", 0) for m in valid_metrics if "up_recall" in m]
        down_recalls = [m.get("down_recall", 0) for m in valid_metrics if "down_recall" in m]
        n_pred_ups = [m.get("n_pred_up", 0) for m in valid_metrics if "n_pred_up" in m]
        n_pred_downs = [m.get("n_pred_down", 0) for m in valid_metrics if "n_pred_down" in m]
        if up_recalls:
            log.info(f"  UP recall:   {np.mean(up_recalls):.4f}±{np.std(up_recalls):.4f}")
        if down_recalls:
            log.info(f"  DOWN recall: {np.mean(down_recalls):.4f}±{np.std(down_recalls):.4f}")
        total_up = sum(n_pred_ups)
        total_down = sum(n_pred_downs)
        log.info(f"  Predictions: UP={total_up} DOWN={total_down} "
                 f"(UP%={total_up/(total_up+total_down)*100:.1f}% if total_up+total_down > 0)")
        if total_down == 0:
            log.warning(f"  ⚠️  ALL predictions are UP! Model has class collapse → no SHORT signals.")
            log.warning(f"     Possible causes: class imbalance, features with directional bias,")
            log.warning(f"     or scale_pos_weight not compensating enough.")

    # Feature importance stability
    feat_counts: Dict[str, int] = {}
    for m in valid_metrics:
        for feat, _ in m.get("top15_features", []):
            feat_counts[feat] = feat_counts.get(feat, 0) + 1
    if feat_counts:
        stable = sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)
        log.info(f"  Stable top features: "
                 + ", ".join(f"{f}({c}/{len(valid_metrics)})" for f, c in stable[:10]))

    # Aggregate confidence gates
    gate_accs: Dict[str, List[float]] = {}
    for m in valid_metrics:
        for key, val in m.get("confidence_gates", {}).items():
            if "accuracy" not in val:
                continue
            if key not in gate_accs:
                gate_accs[key] = []
            gate_accs[key].append(val["accuracy"])
    if gate_accs:
        log.info(f"  Confidence gate OOS accuracy:")
        for key in sorted(gate_accs.keys()):
            vals = gate_accs[key]
            log.info(f"    {key}: {np.mean(vals):.4f}±{np.std(vals):.4f}")

    # ══ SAVE MODEL ══
    if not evaluate_only and valid_metrics:
        os.makedirs(output_dir, exist_ok=True)

        log.info(f"\n  ── FINAL model (ALL data) ──")
        n_val = max(int(len(tf_df) * 0.10), 100)
        val_df = tf_df.tail(n_val)
        train_all_df = tf_df.head(len(tf_df) - n_val)

        if mode == "binary":
            final_model, final_metrics = train_binary(
                train_all_df, val_df, feature_cols,
                use_gpu=use_gpu, fold_idx=-1,
            )
        else:
            final_model, final_metrics = train_regression(
                train_all_df, val_df, feature_cols,
                use_gpu=use_gpu, fold_idx=-1,
                exclude_flat=exclude_flat,
            )

        if final_model:
            path = os.path.join(output_dir, f"direction_v4_tf{tf}.ubj")
            final_model.save_model(path)
            log.info(f"  ✅ Saved: {path}")

            # Save schema
            schema = {
                "features": feature_cols,
                "feature_count": len(feature_cols),
                "task": f"direction_v4_{mode}",
                "objective": "reg:squarederror" if mode == "regression" else "binary:logistic",
                "tf_minutes": tf,
                "version": "v4_pattern",
                "training_approach": f"CNN-like pattern ({mode})",
                "wfo_acc_mean": agg["acc_mean"],
                "wfo_auc_mean": agg["auc_mean"],
                "wfo_baseline": agg["baseline_mean"],
                "wfo_folds": len(valid_metrics),
                "confidence_gates": {k: float(np.mean(v)) for k, v in gate_accs.items()},
                "hyperparams": {
                    "eta": XGBOOST_PARAMS["eta"],
                    "max_depth": XGBOOST_PARAMS["max_depth"],
                    "min_child_weight": XGBOOST_PARAMS["min_child_weight"],
                    "lambda": XGBOOST_PARAMS["lambda"],
                    "alpha": XGBOOST_PARAMS["alpha"],
                    "gamma": XGBOOST_PARAMS["gamma"],
                    "subsample": XGBOOST_PARAMS["subsample"],
                    "colsample_bytree": XGBOOST_PARAMS["colsample_bytree"],
                    "num_boost_round": NUM_BOOST_ROUND,
                    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
                    "validation_fraction": VALIDATION_FRACTION,
                },
            }
            schema_path = path.replace(".ubj", ".schema.json")
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            log.info(f"  ✅ Schema: {schema_path}")

    # ══ SAVE REPORT ══
    elapsed = time_module.time() - t_start
    report = {
        "timeframe_minutes": tf,
        "training_method": "walk_forward_optimization",
        "model_version": "direction_v4_pattern",
        "mode": mode,
        "feature_count": n_features,
        "target_accuracy": DIRECTION_ACC_TARGET,
        "total_rows": len(tf_df),
        "total_symbols": int(tf_df["symbol"].nunique()),
        "aggregate_oos": agg,
        "fold_details": fold_details,
        "elapsed_seconds": round(elapsed, 1),
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"direction_v4_wfo_report_tf{tf}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"\n  📊 Report: {report_path}")
    log.info(f"  ⏱️  Elapsed: {elapsed:.1f}s")

    return report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Direction Model v4 — CNN-like Pattern WFO Training",
    )
    parser.add_argument("--csv", default="dataset/direction_v4_dataset.csv",
                        help="Path to direction_v4_dataset.csv")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for models")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for XGBoost")
    parser.add_argument("--timeframes", default="all",
                        help="Comma-separated TFs (e.g., '15,60') or 'all'")
    parser.add_argument("--mode", default="binary",
                        choices=["regression", "binary"],
                        help="Training mode: regression or binary classification (default: binary)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only WFO evaluation, no final model save")
    parser.add_argument("--exclude-flat", action="store_true",
                        help="Exclude FLAT examples from training (regression mode)")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Prediction horizon in bars (default: {DEFAULT_HORIZON})")
    args = parser.parse_args()

    log.info("╔═══════════════════════════════════════════════════════════╗")
    log.info("║  Direction Model v6 — Internal Validation + Low LR      ║")
    log.info("║  OHLCV window + summary + HTF features                  ║")
    log.info("║  depth=3, eta=0.01, 5000 rounds, internal val for ES    ║")
    log.info("╚═══════════════════════════════════════════════════════════╝")
    log.info(f"Mode: {args.mode}")
    log.info(f"Prediction horizon: {args.horizon} bars")
    log.info(f"Exclude FLAT: {args.exclude_flat}")

    if not os.path.exists(args.csv):
        log.error(f"Dataset not found: {args.csv}")
        log.info("Generate it first:")
        log.info("  cargo build --release -p ml_entry_strategy --bin direction_dataset")
        log.info("  ./target/release/direction_dataset")
        sys.exit(1)

    # Parse TFs
    if args.timeframes == "all":
        target_tfs = TIMEFRAMES
    else:
        target_tfs = sorted([int(x.strip()) for x in args.timeframes.split(",")])

    # Load data — use float32 for feature columns to save ~50% memory
    log.info(f"Loading {args.csv}...")
    load_start = time_module.time()
    df = pd.read_csv(args.csv)
    load_t = time_module.time() - load_start
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns in {load_t:.1f}s")

    # Auto-detect feature columns
    feature_cols = detect_feature_columns(df)
    log.info(f"  Detected {len(feature_cols)} pattern feature columns")
    if len(feature_cols) == 0:
        log.error("No pattern feature columns found! Expected columns like w0_open_rel, w0_high_rel, ...")
        sys.exit(1)

    # Validate required columns
    required = ["symbol", "tf_minutes", "timestamp", "label", "future_return_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Missing required columns: {missing}")
        sys.exit(1)

    # ── Memory optimization: convert features to float32 (saves ~50% RAM) ──
    mem_before = df.memory_usage(deep=True).sum() / 1e9
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
    df["future_return_pct"] = df["future_return_pct"].astype(np.float32)
    if "max_up_pct" in df.columns:
        df["max_up_pct"] = df["max_up_pct"].astype(np.float32)
    if "max_down_pct" in df.columns:
        df["max_down_pct"] = df["max_down_pct"].astype(np.float32)
    mem_after = df.memory_usage(deep=True).sum() / 1e9
    log.info(f"  Memory: {mem_before:.1f} GB → {mem_after:.1f} GB (float32 conversion)")
    gc.collect()

    # Dataset overview
    log.info(f"\n  Dataset overview:")
    for tf_val in sorted(df["tf_minutes"].unique()):
        sub = df[df["tf_minutes"] == tf_val]
        n_up = int((sub["label"] == 1).sum())
        n_flat = int((sub["label"] == 0).sum())
        n_down = int((sub["label"] == -1).sum())
        marker = " ◄" if tf_val in target_tfs else ""
        log.info(f"    TF {tf_val:>5}m: {len(sub):>9,} rows, {sub['symbol'].nunique():>3} symbols, "
                 f"UP={n_up}({n_up/len(sub)*100:.0f}%) "
                 f"FLAT={n_flat}({n_flat/len(sub)*100:.0f}%) "
                 f"DOWN={n_down}({n_down/len(sub)*100:.0f}%){marker}")

    # Run WFO — process each TF separately to control memory
    all_reports: Dict[int, dict] = {}
    total_start = time_module.time()

    for tf in target_tfs:
        tf_df = df[df["tf_minutes"] == tf].copy()
        if len(tf_df) < 200:
            log.warning(f"Skipping TF {tf}m: only {len(tf_df)} rows")
            continue
        report = train_wfo_for_tf(
            tf, tf_df, feature_cols,
            mode=args.mode,
            use_gpu=args.gpu,
            output_dir=args.output_dir,
            evaluate_only=args.evaluate_only,
            prediction_horizon=args.horizon,
            exclude_flat=args.exclude_flat,
        )
        if report:
            all_reports[tf] = report

        # Free TF-specific data between timeframes
        del tf_df
        gc.collect()
        log.info(f"  [Memory] gc.collect() after TF {tf}m")

    total_elapsed = time_module.time() - total_start

    # ══ FINAL SUMMARY ══
    log.info(f"\n╔═══════════════════════════════════════════════════════════╗")
    log.info(f"║  DIRECTION v4 — FINAL SUMMARY ({args.mode.upper()})        ║")
    log.info(f"╚═══════════════════════════════════════════════════════════╝")

    for tf in sorted(all_reports.keys()):
        r = all_reports[tf]
        agg = r.get("aggregate_oos", {})
        acc = agg.get("acc_mean", 0)
        auc = agg.get("auc_mean", 0)
        base = agg.get("baseline_mean", 0.5)
        lift = acc - base
        met = "✅" if acc >= DIRECTION_ACC_TARGET else "❌"

        log.info(f"  TF {tf:>5}m: acc={acc:.4f}  AUC={auc:.4f}  "
                 f"base={base:.4f}  lift={lift:+.4f}  {met}")

    log.info(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    log.info(f"  Target: accuracy ≥ {DIRECTION_ACC_TARGET:.0%}")
    log.info(f"  Model files: models/direction_v4_tf{{X}}.ubj")
    log.info(f"  Done ✅")


if __name__ == "__main__":
    main()
