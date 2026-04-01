#!/usr/bin/env python3
"""
train_super_entry_nodir.py — No-Direction-Model approach for Super Entry Strategy.

Instead of training separate P(super) + P(direction) models, we train:
  - P(super_long):  "Is this a strong upward move?"   label = is_super_long
  - P(super_short): "Is this a strong downward move?"  label = is_super_short

Both models use the same 128 features (ALL_FEATURES).

Backtest logic:
  - If P(super_long) >= threshold  → LONG signal
  - If P(super_short) >= threshold → SHORT signal
  - If both >= threshold → pick higher probability
  - If neither → skip

This eliminates the need for a separate direction model entirely.
Direction information is embedded into the label via future_close vs entry_close check.

=== TRAINING ===
Uses Walk-Forward Optimization (same as train_super_entry_wfo.py).

=== OUTPUT ===
  - models/super_long_v1_tf{X}.ubj   — P(super_long) model
  - models/super_short_v1_tf{X}.ubj  — P(super_short) model
  - models/wfo_nodir_report_tf{X}.json — WFO report

Usage:
    python trainer/src/train_super_entry_nodir.py [--csv dataset/super_entry_dataset.csv] [--gpu]
    python trainer/src/train_super_entry_nodir.py --timeframes 15,60,240 --gpu
"""

import argparse
import gc
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
    "price_roc_lb1", "price_roc_lb2", "price_accel_1bar", "price_accel_3bar",
    "volume_roc_lb1", "volume_roc_lb3", "volume_roc_lb5", "volume_roc_lb10", "volume_accel",
    "bb_squeeze_pctl",
    "obv_price_divergence",
    "macd_hist_roc_lb1", "macd_hist_roc_lb3", "macd_hist_accel",
    # v4: HTF features
    "htf_trend", "htf_supertrend_dir", "htf_ema20_slope",
    # v4: Killer features
    "dist_to_low_50", "dist_to_high_50",
    "acute_wick_rejection_2bar", "bb_squeeze_x_vwap",
    "volume_up_vs_down_lb10",
]

STATIC_FEATURES = INDICATOR_FEATURES + DERIVED_FEATURES
ALL_FEATURES = STATIC_FEATURES + DYNAMIC_FEATURES

TIMEFRAMES = [1, 5, 15, 60, 240, 1440]

TF_TARGET_MOVE_PCT = {
    1: 1.2, 5: 2.8, 15: 3.5, 60: 5.0, 240: 7.5, 1440: 10.0,
}


# ═════════════════════════════════════════════════════════════════════════════
# WFO CONFIGURATION (same as train_super_entry_wfo.py)
# ═════════════════════════════════════════════════════════════════════════════

LOOKAHEAD_BARS = 25
MAX_DYNAMIC_LOOKBACK = 50

WFO_CONFIG: Dict[int, dict] = {
    1: {"n_splits": 2, "test_days": 0.75, "min_train_days": 1.5,
        "desc": "1m: very short history, 2 folds"},
    5: {"n_splits": 3, "test_days": 3.5, "min_train_days": 6.0,
        "desc": "5m: 18 days, 3 folds"},
    15: {"n_splits": 5, "test_days": 14.0, "min_train_days": 30.0,
         "desc": "15m: 4 months, 5 folds"},
    60: {"n_splits": 5, "test_days": 60.0, "min_train_days": 120.0,
         "desc": "1h: 16 months, 5 folds"},
    240: {"n_splits": 6, "test_days": 120.0, "min_train_days": 365.0,
          "desc": "4h: 5.3 years, 6 folds"},
    1440: {"n_splits": 5, "test_days": 180.0, "min_train_days": 365.0,
           "desc": "1d: 5.6 years, 5 folds"},
}


# ═════════════════════════════════════════════════════════════════════════════
# WFO FOLD COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_wfo_folds(timestamps, tf_minutes, config):
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
            print(f"    ⚠️ Not enough data for any WFO fold.")
            return []
        print(f"    ⚠️ Reducing n_splits from {n_splits} to {possible_splits}")
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


def apply_fold_split(df, fold, tf_minutes, ts_col="ts"):
    purge_delta = timedelta(minutes=LOOKAHEAD_BARS * tf_minutes)
    embargo_delta = timedelta(minutes=MAX_DYNAMIC_LOOKBACK * tf_minutes)

    effective_train_end = fold["train_end_raw"] - purge_delta
    effective_test_start = fold["test_start_raw"] + embargo_delta

    train_mask = (df[ts_col] >= fold["train_start"]) & (df[ts_col] <= effective_train_end)
    test_mask = (df[ts_col] >= effective_test_start) & (df[ts_col] <= fold["test_end_raw"])

    return df[train_mask].copy(), df[test_mask].copy()


# ═════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═════════════════════════════════════════════════════════════════════════════

def train_binary(train_df, test_df, label_col, feature_cols, use_gpu=False,
                 model_name="", custom_params=None, num_boost_round=500,
                 early_stopping_rounds=30):
    """Train binary XGBoost classifier and evaluate OOS."""

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    n_train_total = len(y_train)
    n_pos = float(y_train.sum())
    n_neg = float(n_train_total - n_pos)
    del X_train, y_train, X_test
    gc.collect()

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
          f"(train={n_train_total}, test={len(y_test)}, pos={n_pos:.0f}/{n_train_total} "
          f"= {n_pos/n_train_total*100:.1f}%)")

    evals = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=0,
    )

    del dtrain
    gc.collect()

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

    metrics["train_size"] = n_train_total
    metrics["test_size"] = int(len(y_test))
    metrics["train_pos_rate"] = float(n_pos / n_train_total)
    metrics["test_pos_rate"] = float(y_test.mean())
    metrics["best_iteration"] = int(model.best_iteration)

    print(f"      → AUC={metrics['auc']:.4f}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
          f"LogLoss={metrics['logloss']:.4f}  (trees={model.best_iteration + 1})")

    del dtest, y_test, y_pred_prob, y_pred
    gc.collect()

    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    metrics["top5_features"] = [name for name, _ in sorted_imp[:5]]

    return model, metrics


def save_model(model, feature_names, tf, task_name, metrics, output_dir, suffix=""):
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
        "training_method": "walk_forward_optimization_nodir",
    }
    schema_path = os.path.join(output_dir, f"{model_name}.schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

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
        "strategy": "ml_entry_strategy_nodir",
        "training_method": "walk_forward_optimization_nodir",
        "target_move_pct": TF_TARGET_MOVE_PCT.get(tf, 0),
    }
    meta_path = os.path.join(output_dir, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return ubj_path


# ═════════════════════════════════════════════════════════════════════════════
# WFO AGGREGATION
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_fold_metrics(fold_metrics, name):
    if not fold_metrics:
        return {}

    metric_names = ["auc", "precision", "recall", "f1", "logloss"]
    agg = {}

    for m in metric_names:
        values = [f[m] for f in fold_metrics if m in f and f.get("test_size", 0) > 0]
        if not values:
            continue
        arr = np.array(values, dtype=np.float64)
        agg[f"{m}_mean"] = float(arr.mean())
        agg[f"{m}_std"] = float(arr.std())
        agg[f"{m}_min"] = float(arr.min())
        agg[f"{m}_max"] = float(arr.max())
        if arr.mean() > 1e-8:
            agg[f"{m}_cv"] = float(arr.std() / arr.mean())
        if len(values) >= 3:
            x = np.arange(len(values), dtype=np.float64)
            slope = np.polyfit(x, arr, 1)[0]
            agg[f"{m}_trend_per_fold"] = float(slope)

    total_test = sum(f.get("test_size", 0) for f in fold_metrics)
    agg["total_oos_samples"] = total_test
    agg["n_valid_folds"] = len(fold_metrics)

    print(f"    {name} (across {len(fold_metrics)} folds, {total_test} OOS samples):")
    for m in metric_names:
        if f"{m}_mean" in agg:
            trend_str = ""
            if f"{m}_trend_per_fold" in agg:
                t = agg[f"{m}_trend_per_fold"]
                arrow = "↑" if t > 0.005 else ("↓" if t < -0.005 else "→")
                trend_str = f"  trend={arrow}{abs(t):.4f}/fold"
            print(f"      {m:12s}  {agg[f'{m}_mean']:.4f} ± {agg[f'{m}_std']:.4f}  "
                  f"[{agg[f'{m}_min']:.4f} .. {agg[f'{m}_max']:.4f}]"
                  f"{trend_str}")

    return agg


def collect_feature_importance(fold_metrics):
    counts = {}
    for fm in fold_metrics:
        for feat in fm.get("top5_features", []):
            counts[feat] = counts.get(feat, 0) + 1
    return sorted(counts.items(), key=lambda x: x[1], reverse=True)


# ═════════════════════════════════════════════════════════════════════════════
# WFO TRAINING PIPELINE (per TF)
# ═════════════════════════════════════════════════════════════════════════════

def train_wfo_nodir_for_tf(tf, tf_df, use_gpu=False, output_dir="models",
                            save_fold_models=True, evaluate_only=False):
    """
    Train P(super_long) and P(super_short) models for one TF using WFO.
    No direction model needed — direction is embedded in the label.
    """
    config = WFO_CONFIG.get(tf)
    if config is None:
        print(f"  No WFO config for TF {tf}m. Skipping.")
        return {}

    t_start = time_module.time()

    print(f"\n{'═' * 75}")
    print(f"  WFO NODIR TRAINING — TF {tf}m")
    print(f"  {config['desc']}")
    print(f"  Data: {len(tf_df)} rows, {tf_df['symbol'].nunique()} symbols")
    n_super_long = int(tf_df['is_super_long'].sum())
    n_super_short = int(tf_df['is_super_short'].sum())
    n_super = int(tf_df['is_super'].sum())
    print(f"  Labels: is_super={n_super} ({n_super/len(tf_df)*100:.1f}%), "
          f"super_long={n_super_long} ({n_super_long/len(tf_df)*100:.1f}%), "
          f"super_short={n_super_short} ({n_super_short/len(tf_df)*100:.1f}%)")
    print(f"{'═' * 75}")

    tf_df["ts"] = pd.to_datetime(tf_df["timestamp"], utc=True)

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

    # Per-fold training
    long_fold_metrics = []
    short_fold_metrics = []
    fold_details = []

    fold_models_dir = os.path.join(output_dir, "wfo_nodir_folds")
    if save_fold_models:
        os.makedirs(fold_models_dir, exist_ok=True)

    for fold in folds:
        fi = fold["fold_idx"]
        print(f"\n  ── Fold {fi}/{len(folds) - 1} ──")

        train_df, test_df = apply_fold_split(tf_df, fold, tf, ts_col="ts")

        if len(train_df) < 100 or len(test_df) < 30:
            print(f"    ⚠️ Too few samples (train={len(train_df)}, test={len(test_df)}). Skipping.")
            del train_df, test_df
            gc.collect()
            continue

        train_rows = len(train_df)
        test_rows = len(test_df)
        print(f"    Train: {train_rows} rows, super_long={int(train_df['is_super_long'].sum())}, "
              f"super_short={int(train_df['is_super_short'].sum())}")
        print(f"    Test:  {test_rows} rows, super_long={int(test_df['is_super_long'].sum())}, "
              f"super_short={int(test_df['is_super_short'].sum())}")

        # === Train P(super_long) ===
        long_model, long_metrics = train_binary(
            train_df, test_df,
            label_col="is_super_long",
            feature_cols=ALL_FEATURES,
            use_gpu=use_gpu,
            model_name=f"P(super_long) fold {fi}",
        )
        long_fold_metrics.append(long_metrics)

        if save_fold_models:
            save_model(long_model, ALL_FEATURES, tf, "super_long",
                       long_metrics, fold_models_dir, suffix=f"_fold{fi}")
        del long_model
        gc.collect()

        # === Train P(super_short) ===
        short_model, short_metrics = train_binary(
            train_df, test_df,
            label_col="is_super_short",
            feature_cols=ALL_FEATURES,
            use_gpu=use_gpu,
            model_name=f"P(super_short) fold {fi}",
        )
        short_fold_metrics.append(short_metrics)

        if save_fold_models:
            save_model(short_model, ALL_FEATURES, tf, "super_short",
                       short_metrics, fold_models_dir, suffix=f"_fold{fi}")
        del short_model
        gc.collect()

        fold_details.append({
            "fold_idx": fi,
            "train_start": fold["train_start"].isoformat(),
            "train_end_raw": fold["train_end_raw"].isoformat(),
            "test_start_raw": fold["test_start_raw"].isoformat(),
            "test_end_raw": fold["test_end_raw"].isoformat(),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "long_metrics": long_metrics,
            "short_metrics": short_metrics,
        })

        del train_df, test_df
        gc.collect()

    # Aggregate OOS Metrics
    print(f"\n  {'─' * 60}")
    print(f"  WFO NoDir Aggregate OOS Metrics — TF {tf}m")
    print(f"  {'─' * 60}")

    long_agg = aggregate_fold_metrics(long_fold_metrics, "P(super_long)")
    short_agg = aggregate_fold_metrics(short_fold_metrics, "P(super_short)")

    long_fi = collect_feature_importance(long_fold_metrics)
    short_fi = collect_feature_importance(short_fold_metrics)

    if long_fi:
        n_folds = len(long_fold_metrics)
        print(f"\n    Stable features P(super_long) (top-5 across {n_folds} folds):")
        for feat, cnt in long_fi[:8]:
            print(f"      {feat:35s}  {cnt}/{n_folds} folds")
    if short_fi:
        n_folds = len(short_fold_metrics)
        print(f"    Stable features P(super_short) (top-5 across {n_folds} folds):")
        for feat, cnt in short_fi[:8]:
            print(f"      {feat:35s}  {cnt}/{n_folds} folds")

    # Train FINAL production models
    final_long_metrics = {}
    final_short_metrics = {}

    if not evaluate_only:
        print(f"\n  ── FINAL models (trained on ALL data) ──")

        n_val = max(int(len(tf_df) * 0.10), 100)
        val_df = tf_df.tail(n_val)
        train_all_df = tf_df.head(len(tf_df) - n_val)

        # Final P(super_long)
        final_long_model, final_long_metrics = train_binary(
            train_all_df, val_df,
            label_col="is_super_long",
            feature_cols=ALL_FEATURES,
            use_gpu=use_gpu,
            model_name="FINAL P(super_long)",
        )
        final_long_metrics["wfo_oos_auc_mean"] = long_agg.get("auc_mean", 0)
        save_model(final_long_model, ALL_FEATURES, tf, "super_long",
                   final_long_metrics, output_dir)
        print(f"    ✅ Saved: {output_dir}/super_long_v1_tf{tf}.ubj")
        del final_long_model
        gc.collect()

        # Final P(super_short)
        final_short_model, final_short_metrics = train_binary(
            train_all_df, val_df,
            label_col="is_super_short",
            feature_cols=ALL_FEATURES,
            use_gpu=use_gpu,
            model_name="FINAL P(super_short)",
        )
        final_short_metrics["wfo_oos_auc_mean"] = short_agg.get("auc_mean", 0)
        save_model(final_short_model, ALL_FEATURES, tf, "super_short",
                   final_short_metrics, output_dir)
        print(f"    ✅ Saved: {output_dir}/super_short_v1_tf{tf}.ubj")
        del final_short_model, train_all_df, val_df
        gc.collect()

    # Save report
    elapsed = time_module.time() - t_start
    report = {
        "timeframe_minutes": tf,
        "training_method": "walk_forward_optimization_nodir",
        "approach": "P(super_long) + P(super_short) — no separate direction model",
        "wfo_config": config,
        "total_rows": len(tf_df),
        "total_symbols": int(tf_df["symbol"].nunique()),
        "label_stats": {
            "is_super": int(tf_df["is_super"].sum()),
            "is_super_long": int(tf_df["is_super_long"].sum()),
            "is_super_short": int(tf_df["is_super_short"].sum()),
        },
        "long_aggregate_oos": long_agg,
        "short_aggregate_oos": short_agg,
        "long_feature_importance": long_fi[:10] if long_fi else [],
        "short_feature_importance": short_fi[:10] if short_fi else [],
        "fold_details": fold_details,
        "final_model_metrics": {
            "super_long": {k: v for k, v in final_long_metrics.items()
                          if isinstance(v, (int, float, str, bool))},
            "super_short": {k: v for k, v in final_short_metrics.items()
                           if isinstance(v, (int, float, str, bool))},
        },
        "elapsed_seconds": round(elapsed, 1),
    }

    report_path = os.path.join(output_dir, f"wfo_nodir_report_tf{tf}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  📊 WFO NoDir report: {report_path}")
    print(f"  ⏱️  Elapsed: {elapsed:.1f}s")

    return report


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="No-Direction-Model Training for Super Entry Strategy",
    )
    parser.add_argument("--csv", default="dataset/super_entry_dataset.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--timeframes", default="all")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--no-fold-models", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found!")
        print("Generate dataset first:")
        print("  cargo run --release -p ml_entry_strategy --bin super_entry_dataset")
        sys.exit(1)

    if args.timeframes == "all":
        target_tfs = TIMEFRAMES
    else:
        target_tfs = sorted([int(x.strip()) for x in args.timeframes.split(",")])

    print(f"╔{'═' * 73}╗")
    print(f"║  NoDir WFO Training — P(super_long) + P(super_short){' ' * 19}║")
    print(f"╚{'═' * 73}╝")
    print(f"\nLoading dataset from {args.csv}...")

    load_start = time_module.time()
    df = pd.read_csv(args.csv)
    print(f"  Loaded {len(df):,} rows in {time_module.time() - load_start:.1f}s")

    # Check for new columns
    required_labels = ["is_super_long", "is_super_short"]
    missing_labels = [c for c in required_labels if c not in df.columns]
    if missing_labels:
        print(f"  ⚠️ Missing columns: {missing_labels}")
        print("  Regenerate dataset with updated dataset builder!")
        sys.exit(1)

    # Validate feature columns
    required = ALL_FEATURES + ["symbol", "tf_minutes", "is_super",
                                "is_super_long", "is_super_short", "timestamp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ⚠️ Missing columns (filled with 0): {missing}")
        for c in missing:
            df[c] = 0.0

    # Memory optimization
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    # Dataset overview
    print(f"\n  Dataset overview:")
    for tf_val in sorted(df["tf_minutes"].unique()):
        sub = df[df["tf_minutes"] == tf_val]
        marker = " ◄" if tf_val in target_tfs else ""
        n_sl = int(sub["is_super_long"].sum())
        n_ss = int(sub["is_super_short"].sum())
        print(f"    TF {tf_val:>5}m: {len(sub):>9,} rows, "
              f"super_long={n_sl} ({n_sl/len(sub)*100:.1f}%), "
              f"super_short={n_ss} ({n_ss/len(sub)*100:.1f}%){marker}")

    # Train
    all_reports = {}
    total_start = time_module.time()

    for tf in target_tfs:
        tf_df = df[df["tf_minutes"] == tf].copy()
        if len(tf_df) < 100:
            print(f"\n  Skipping TF {tf}m: only {len(tf_df)} rows")
            del tf_df
            continue

        report = train_wfo_nodir_for_tf(
            tf, tf_df,
            use_gpu=args.gpu,
            output_dir=args.output_dir,
            save_fold_models=not args.no_fold_models,
            evaluate_only=args.evaluate_only,
        )
        if report:
            all_reports[tf] = report
        del tf_df
        gc.collect()

    del df
    gc.collect()

    total_elapsed = time_module.time() - total_start

    # Summary
    print(f"\n╔{'═' * 73}╗")
    print(f"║  NODIR TRAINING COMPLETE{' ' * 48}║")
    print(f"╚{'═' * 73}╝")

    for tf in sorted(all_reports.keys()):
        r = all_reports[tf]
        long_agg = r.get("long_aggregate_oos", {})
        short_agg = r.get("short_aggregate_oos", {})

        print(f"\n  TF {tf}m ({r.get('total_rows', 0):,} rows):")
        if long_agg:
            print(f"    P(super_long):  AUC={long_agg.get('auc_mean', 0):.4f}"
                  f"±{long_agg.get('auc_std', 0):.4f}  "
                  f"F1={long_agg.get('f1_mean', 0):.4f}")
        if short_agg:
            print(f"    P(super_short): AUC={short_agg.get('auc_mean', 0):.4f}"
                  f"±{short_agg.get('auc_std', 0):.4f}  "
                  f"F1={short_agg.get('f1_mean', 0):.4f}")

    # Save combined report
    combined_path = os.path.join(args.output_dir, "wfo_nodir_combined_report.json")
    with open(combined_path, "w") as f:
        json.dump({
            "training_method": "walk_forward_optimization_nodir",
            "total_elapsed_seconds": round(total_elapsed, 1),
            "per_tf": {str(k): v for k, v in all_reports.items()},
        }, f, indent=2, default=str)

    print(f"\n  📊 Combined report: {combined_path}")
    print(f"  🏷️  Models: {args.output_dir}/super_long_v1_tf*.ubj + super_short_v1_tf*.ubj")
    print(f"  ⏱️  Total: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    print(f"  🏁 Done")


if __name__ == "__main__":
    main()
