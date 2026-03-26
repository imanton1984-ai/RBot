#!/usr/bin/env python3
"""
train_direction_wfo.py — Walk-Forward Optimization for Direction Model v3.

APPROACH:
  - 32 curated features from 6 domains (trend, momentum, structure, volume, BTC, volatility)
  - Regression on direction_quality = direction * (1/bars_to_tp)
  - Fast TP = high |quality| = clean signal, slow/no TP ≈ 0 = noise
  - Inference: sign(prediction) = direction, abs(prediction) = confidence

WHY REGRESSION (not binary classification):
  Binary classification on LONG/SHORT gives ~50% because short-TF returns
  are random walk. The signal is in the SPEED of move, not just direction.
  Regression on direction_quality creates a natural confidence gate:
  predictions near 0 = uncertain → skip.

CHANGES FROM v2:
  - Removed Oracle variant B (complexity without clear gain)
  - Removed temporal features (hour/dow sin/cos — noise)
  - Removed funding_rate/SR features (placeholder zeros)
  - Added 12 features from super_entry 128-set that had signal
  - Added HTF supertrend (was only in heuristic filter, now in model)
  - Target: direction accuracy ≥ 0.60 on WFO OOS (was 0.65)

Usage:
    python trainer/src/train_direction_wfo.py --csv dataset/direction_v3_dataset.csv --gpu
    python trainer/src/train_direction_wfo.py --timeframes 15,60 --evaluate-only
"""

import argparse
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
        roc_auc_score, mean_squared_error,
    )
except ImportError:
    print("ERROR: Required packages not installed.")
    print("  pip install xgboost scikit-learn pandas numpy")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging(log_path: str = "logs/direction_wfo_train.log"):
    """Setup dual logging: stdout + file."""
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
# FEATURE DEFINITIONS — must match Rust direction/features.rs EXACTLY
# ═════════════════════════════════════════════════════════════════════════════

DIRECTION_V3_FEATURES = [
    # GROUP 1: Trend alignment (6)
    "supertrend_dir", "htf_supertrend_dir",
    "trend", "trend_short", "trend_alignment", "ema_convergence_change",
    # GROUP 2: Momentum (6)
    "price_roc_lb1", "price_roc_lb2", "price_accel_1bar",
    "macd_hist", "macd_hist_roc_lb1", "rsi_slope_lb3",
    # GROUP 3: Market structure (6)
    "dist_to_low_50", "dist_to_high_50", "bb_position",
    "price_vs_vwap", "price_vs_ema20", "high_low_pressure",
    # GROUP 4: Volume/Pressure (4)
    "volume_up_vs_down_lb10", "obv_price_divergence",
    "acute_wick_rejection_2bar", "cmf",
    # GROUP 5: BTC relative (5)
    "btc_return_lb5", "btc_return_lb25", "btc_supertrend_dir",
    "alt_vs_btc_return_lb5", "alt_vs_btc_return_lb25",
    # GROUP 6: Volatility context (5)
    "bb_squeeze_pctl", "atr_ratio_lb5", "bb_width_pct",
    "supertrend_consistency", "volume_trend_ratio",
]

assert len(DIRECTION_V3_FEATURES) == 32, f"Expected 32 features, got {len(DIRECTION_V3_FEATURES)}"

TIMEFRAMES = [5, 15, 60, 240, 1440]

TF_TARGET_MOVE_PCT = {
    1: 1.2, 5: 2.8, 15: 3.5, 60: 5.0, 240: 7.5, 1440: 10.0,
}

# WFO parameters
LOOKAHEAD_BARS = 25
MAX_LOOKBACK = 50  # for embargo

# Direction accuracy target (honest, no overfitting)
DIRECTION_ACC_TARGET = 0.60


# ═════════════════════════════════════════════════════════════════════════════
# WFO CONFIG
# ═════════════════════════════════════════════════════════════════════════════

WFO_CONFIG: Dict[int, dict] = {
    1: {"n_splits": 2, "test_days": 0.75, "min_train_days": 1.5},
    5: {"n_splits": 3, "test_days": 3.5, "min_train_days": 6.0},
    15: {"n_splits": 5, "test_days": 14.0, "min_train_days": 30.0},
    60: {"n_splits": 5, "test_days": 60.0, "min_train_days": 120.0},
    240: {"n_splits": 6, "test_days": 120.0, "min_train_days": 365.0},
    1440: {"n_splits": 5, "test_days": 180.0, "min_train_days": 365.0},
}


# ═════════════════════════════════════════════════════════════════════════════
# WFO FOLD COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_wfo_folds(timestamps: pd.Series, tf_minutes: int, config: dict) -> List[Dict]:
    """Expanding-window WFO fold boundaries."""
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
    df: pd.DataFrame, fold: Dict, tf_minutes: int, ts_col: str = "ts",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split with purge & embargo."""
    purge_delta = timedelta(minutes=LOOKAHEAD_BARS * tf_minutes)
    embargo_delta = timedelta(minutes=MAX_LOOKBACK * tf_minutes)

    effective_train_end = fold["train_end_raw"] - purge_delta
    effective_test_start = fold["test_start_raw"] + embargo_delta

    train_mask = (df[ts_col] >= fold["train_start"]) & (df[ts_col] <= effective_train_end)
    test_mask = (df[ts_col] >= effective_test_start) & (df[ts_col] <= fold["test_end_raw"])

    return df[train_mask].copy(), df[test_mask].copy()


# ═════════════════════════════════════════════════════════════════════════════
# DIRECTION MODEL v3: REGRESSION ON direction_quality
# ═════════════════════════════════════════════════════════════════════════════

def train_direction_v3(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    use_gpu: bool = False,
    fold_idx: int = 0,
) -> Tuple[Optional[xgb.Booster], dict]:
    """
    Direction v3: Regression on direction_quality.

    Training only on is_super=True AND direction != 0 (exclude whipsaw).
    Target: direction_quality = direction * (1/bars_to_tp)

    Inference:
      - sign(prediction) = direction (LONG/SHORT)
      - abs(prediction) = confidence (higher = model more certain)
      - If abs(prediction) < threshold → skip (uncertain)
    """
    # Filter: train only on super examples with clear direction
    train_super = train_df[
        (train_df["is_super"] == 1) & (train_df["direction"] != 0)
    ].copy()
    test_super = test_df[
        (test_df["is_super"] == 1) & (test_df["direction"] != 0)
    ].copy()

    # But also evaluate on ALL test data (for comparison)
    test_all = test_df[test_df["direction"] != 0].copy()

    if len(train_super) < 100 or len(test_super) < 30:
        log.warning(f"  Fold {fold_idx}: Not enough super examples "
                    f"(train={len(train_super)}, test_super={len(test_super)}). Skipping.")
        return None, {
            "dir_accuracy": 0.5,
            "dir_auc": 0.5,
            "error": f"not enough super data (train={len(train_super)}, test={len(test_super)})",
        }

    X_train = train_super[feature_cols].values.astype(np.float32)
    y_train = train_super["direction_quality"].values.astype(np.float32)

    X_test_super = test_super[feature_cols].values.astype(np.float32)
    y_test_super = test_super["direction_quality"].values.astype(np.float32)

    X_test_all = test_all[feature_cols].values.astype(np.float32)

    # Clean NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_super = np.nan_to_num(X_test_super, nan=0.0, posinf=0.0, neginf=0.0)
    y_test_super = np.nan_to_num(y_test_super, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_all = np.nan_to_num(X_test_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Use sample_weight: fast TP = high weight (via 1/bars_to_tp)
    # Direction quality already encodes this, but we want the model
    # to focus more on high-quality examples
    bars = train_super["bars_to_tp"].fillna(LOOKAHEAD_BARS).values.astype(np.float32)
    bars = np.clip(bars, 1, LOOKAHEAD_BARS)
    sample_weights = 1.0 / bars  # fast TP = higher weight
    sample_weights = sample_weights / sample_weights.mean()  # normalize to mean=1

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights,
                         feature_names=feature_cols)
    dtest_super = xgb.DMatrix(X_test_super, label=y_test_super,
                              feature_names=feature_cols)
    dtest_all = xgb.DMatrix(X_test_all, feature_names=feature_cols)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.03,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 10,
        "lambda": 1.5,        # L2 regularization — prevent overfitting
        "alpha": 0.5,         # L1 regularization — feature selection
        "gamma": 0.1,         # Min loss reduction for split
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "verbosity": 0,
    }

    log.info(f"  Fold {fold_idx}: Training on {len(y_train)} super examples "
             f"(LONG={int((train_super['direction'] == 1).sum())}, "
             f"SHORT={int((train_super['direction'] == -1).sum())})")

    model = xgb.train(
        params, dtrain,
        num_boost_round=1200,
        evals=[(dtrain, "train"), (dtest_super, "test")],
        early_stopping_rounds=80,
        verbose_eval=0,
    )

    # ── Evaluate on SUPER test set ──
    pred_super = model.predict(dtest_super)
    rmse_super = float(np.sqrt(mean_squared_error(y_test_super, pred_super)))

    # Direction accuracy on super examples
    pred_dir_super = np.sign(pred_super)
    actual_dir_super = np.sign(y_test_super)
    nonzero_super = actual_dir_super != 0
    if nonzero_super.sum() > 0:
        dir_acc_super = float((pred_dir_super[nonzero_super] == actual_dir_super[nonzero_super]).mean())
    else:
        dir_acc_super = 0.5

    # Direction AUC on super examples
    dir_auc_super = 0.5
    if nonzero_super.sum() > 20:
        y_binary = (actual_dir_super[nonzero_super] > 0).astype(int)
        pred_scores = pred_super[nonzero_super]
        try:
            if len(np.unique(y_binary)) > 1:
                dir_auc_super = float(roc_auc_score(y_binary, pred_scores))
        except Exception:
            pass

    # ── Evaluate on ALL test set ──
    pred_all = model.predict(dtest_all)
    actual_dir_all = test_all["direction"].values
    pred_dir_all = np.sign(pred_all)
    nonzero_all = actual_dir_all != 0
    if nonzero_all.sum() > 0:
        dir_acc_all = float((pred_dir_all[nonzero_all] == actual_dir_all[nonzero_all]).mean())
    else:
        dir_acc_all = 0.5

    # ── Confidence gate analysis ──
    # How much does filtering by abs(prediction) improve accuracy?
    confidence_thresholds = [0.01, 0.03, 0.05, 0.10, 0.15]
    gate_analysis = {}
    for thresh in confidence_thresholds:
        mask = np.abs(pred_all) >= thresh
        if mask.sum() > 20:
            gated_dir = pred_dir_all[mask & nonzero_all]
            gated_actual = actual_dir_all[mask & nonzero_all]
            if len(gated_dir) > 0:
                gated_acc = float((gated_dir == gated_actual).mean())
                gate_analysis[f"gate_{thresh:.2f}"] = {
                    "accuracy": gated_acc,
                    "coverage": float(mask.sum()) / float(len(pred_all)),
                    "n_samples": int(mask.sum()),
                }

    # Feature importance
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top10 = [(name, round(gain, 2)) for name, gain in sorted_imp[:10]]

    metrics = {
        "dir_accuracy_super": dir_acc_super,
        "dir_auc_super": dir_auc_super,
        "dir_accuracy_all": dir_acc_all,
        "rmse_super": rmse_super,
        "train_size": int(len(y_train)),
        "test_super_size": int(len(y_test_super)),
        "test_all_size": int(len(test_all)),
        "best_iteration": int(model.best_iteration),
        "top10_features": top10,
        "confidence_gates": gate_analysis,
    }

    target_met = "✅" if dir_acc_super >= DIRECTION_ACC_TARGET else "❌"
    log.info(f"  Fold {fold_idx}: dir_acc(super)={dir_acc_super:.4f} {target_met}  "
             f"dir_AUC(super)={dir_auc_super:.4f}  "
             f"dir_acc(all)={dir_acc_all:.4f}  "
             f"RMSE={rmse_super:.4f}  trees={model.best_iteration + 1}")

    # Log confidence gate results
    for thresh in confidence_thresholds:
        key = f"gate_{thresh:.2f}"
        if key in gate_analysis:
            g = gate_analysis[key]
            log.info(f"    gate≥{thresh:.2f}: acc={g['accuracy']:.4f}  "
                     f"coverage={g['coverage']:.1%}  n={g['n_samples']}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# BASELINE
# ═════════════════════════════════════════════════════════════════════════════

def compute_baseline(test_df: pd.DataFrame) -> dict:
    """Baseline: random direction prediction → ~50% accuracy."""
    y_actual = test_df["direction"].values
    nonzero = y_actual != 0
    if nonzero.sum() == 0:
        return {"dir_accuracy": 0.5, "baseline_dir_accuracy": 0.5}

    long_pct = float((y_actual[nonzero] == 1).mean())
    return {
        "long_pct_in_test": long_pct,
        "baseline_dir_accuracy": max(long_pct, 1 - long_pct),
    }


# ═════════════════════════════════════════════════════════════════════════════
# WFO TRAINING PIPELINE (per TF)
# ═════════════════════════════════════════════════════════════════════════════

def train_wfo_for_tf(
    tf: int,
    tf_df: pd.DataFrame,
    use_gpu: bool = False,
    output_dir: str = "models",
    evaluate_only: bool = False,
) -> dict:
    """Run WFO for one timeframe."""
    config = WFO_CONFIG.get(tf)
    if config is None:
        log.warning(f"No WFO config for TF {tf}m. Skipping.")
        return {}

    t_start = time_module.time()

    log.info(f"\n{'═' * 75}")
    log.info(f"  DIRECTION v3 WFO — TF {tf}m")
    log.info(f"  Data: {len(tf_df)} rows, {tf_df['symbol'].nunique()} symbols")
    n_super = int(tf_df["is_super"].sum())
    n_super_dir = int(((tf_df["is_super"] == 1) & (tf_df["direction"] != 0)).sum())
    log.info(f"  Super: {n_super}, Super+direction: {n_super_dir}")
    log.info(f"  Target: direction accuracy ≥ {DIRECTION_ACC_TARGET:.0%}")
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

    # ── Per-fold training ──
    fold_metrics: List[dict] = []
    baseline_metrics: List[dict] = []
    fold_details: List[dict] = []

    for fold in folds:
        fi = fold["fold_idx"]
        log.info(f"\n  ── Fold {fi}/{len(folds) - 1} ──")

        train_df, test_df = apply_fold_split(tf_df, fold, tf, ts_col="ts")

        if len(train_df) < 200 or len(test_df) < 50:
            log.warning(f"  Too few samples (train={len(train_df)}, test={len(test_df)}). Skipping.")
            continue

        log.info(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        # Baseline
        base = compute_baseline(test_df)
        baseline_metrics.append(base)
        log.info(f"  [BASE] Fold {fi}: naive best="
                 f"{base.get('baseline_dir_accuracy', 0.5):.4f} "
                 f"(long_pct={base.get('long_pct_in_test', 0.5):.3f})")

        # Train direction v3
        model, metrics = train_direction_v3(
            train_df, test_df, DIRECTION_V3_FEATURES,
            use_gpu=use_gpu, fold_idx=fi,
        )
        fold_metrics.append(metrics)

        fold_details.append({
            "fold_idx": fi,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "baseline": base,
            "metrics": metrics,
        })

    # ══ AGGREGATE ══
    log.info(f"\n  {'─' * 60}")
    log.info(f"  AGGREGATE OOS RESULTS — TF {tf}m")
    log.info(f"  {'─' * 60}")

    def aggregate(metrics_list, name):
        if not metrics_list:
            return {}
        accs = [m.get("dir_accuracy_super", 0.5) for m in metrics_list
                if "error" not in m]
        aucs = [m.get("dir_auc_super", 0.5) for m in metrics_list
                if "error" not in m]
        accs_all = [m.get("dir_accuracy_all", 0.5) for m in metrics_list
                    if "error" not in m]
        agg = {}
        if accs:
            agg["dir_acc_super_mean"] = float(np.mean(accs))
            agg["dir_acc_super_std"] = float(np.std(accs))
            agg["dir_acc_super_min"] = float(np.min(accs))
            agg["dir_acc_super_max"] = float(np.max(accs))
        if aucs:
            agg["dir_auc_super_mean"] = float(np.mean(aucs))
            agg["dir_auc_super_std"] = float(np.std(aucs))
        if accs_all:
            agg["dir_acc_all_mean"] = float(np.mean(accs_all))

        met = "✅" if agg.get("dir_acc_super_mean", 0) >= DIRECTION_ACC_TARGET else "❌"
        log.info(f"  {name}: "
                 f"dir_acc(super)={agg.get('dir_acc_super_mean', 0):.4f}"
                 f"±{agg.get('dir_acc_super_std', 0):.4f} {met}  "
                 f"dir_AUC(super)={agg.get('dir_auc_super_mean', 0):.4f}"
                 f"±{agg.get('dir_auc_super_std', 0):.4f}  "
                 f"dir_acc(all)={agg.get('dir_acc_all_mean', 0):.4f}  "
                 f"[{agg.get('dir_acc_super_min', 0):.4f}..{agg.get('dir_acc_super_max', 0):.4f}]")
        return agg

    agg = aggregate(fold_metrics, "Direction v3")

    base_accs = [m.get("baseline_dir_accuracy", 0.5) for m in baseline_metrics]
    if base_accs:
        log.info(f"  Baseline (naive):  dir_acc={np.mean(base_accs):.4f}")

    # Feature importance stability
    counts: Dict[str, int] = {}
    for m in fold_metrics:
        if "error" in m:
            continue
        for feat, _ in m.get("top10_features", []):
            counts[feat] = counts.get(feat, 0) + 1
    if counts:
        stable = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        log.info(f"  Stable top features: "
                 + ", ".join(f"{f}({c}/{len(fold_metrics)})" for f, c in stable[:8]))

    # Aggregate confidence gate analysis
    gate_accs: Dict[str, List[float]] = {}
    for m in fold_metrics:
        if "error" in m:
            continue
        for key, val in m.get("confidence_gates", {}).items():
            if key not in gate_accs:
                gate_accs[key] = []
            gate_accs[key].append(val["accuracy"])
    if gate_accs:
        log.info(f"  Confidence gate OOS accuracy:")
        for key in sorted(gate_accs.keys()):
            accs = gate_accs[key]
            log.info(f"    {key}: {np.mean(accs):.4f}±{np.std(accs):.4f}")

    # ══ SAVE MODELS (if not evaluate-only) ══
    if not evaluate_only and fold_metrics:
        os.makedirs(output_dir, exist_ok=True)

        # Train final model on ALL data
        log.info(f"\n  ── FINAL Direction v3 model (ALL data) ──")
        n_val = max(int(len(tf_df) * 0.10), 100)
        val_df = tf_df.tail(n_val)
        train_all_df = tf_df.head(len(tf_df) - n_val)

        final_model, final_metrics = train_direction_v3(
            train_all_df, val_df, DIRECTION_V3_FEATURES,
            use_gpu=use_gpu, fold_idx=-1,
        )
        if final_model:
            # Save as direction_v3_tf{X}.ubj
            path = os.path.join(output_dir, f"direction_v3_tf{tf}.ubj")
            final_model.save_model(path)
            log.info(f"  ✅ Saved: {path}")

            # Save schema
            schema = {
                "features": DIRECTION_V3_FEATURES,
                "feature_count": len(DIRECTION_V3_FEATURES),
                "task": "direction_v3_regression",
                "objective": "reg:squarederror",
                "tf_minutes": tf,
                "version": "v3",
                "training_approach": "regression on direction_quality",
                "wfo_dir_acc_super_mean": agg.get("dir_acc_super_mean", 0),
                "wfo_dir_auc_super_mean": agg.get("dir_auc_super_mean", 0),
                "wfo_folds": len(fold_metrics),
                "confidence_gates": {k: float(np.mean(v)) for k, v in gate_accs.items()},
            }
            schema_path = path.replace(".ubj", ".schema.json")
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            log.info(f"  ✅ Schema: {schema_path}")

    # ══ SAVE WFO REPORT ══
    elapsed = time_module.time() - t_start
    report = {
        "timeframe_minutes": tf,
        "training_method": "walk_forward_optimization",
        "model_version": "direction_v3",
        "feature_count": len(DIRECTION_V3_FEATURES),
        "target_dir_accuracy": DIRECTION_ACC_TARGET,
        "total_rows": len(tf_df),
        "total_symbols": int(tf_df["symbol"].nunique()),
        "aggregate_oos": agg,
        "baseline_mean_acc": float(np.mean(base_accs)) if base_accs else 0.5,
        "fold_details": fold_details,
        "elapsed_seconds": round(elapsed, 1),
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"direction_v3_wfo_report_tf{tf}.json")
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
        description="Direction Model v3 — WFO Training (Regression on direction_quality)",
    )
    parser.add_argument("--csv", default="dataset/direction_v3_dataset.csv",
                        help="Path to direction_v3_dataset.csv")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for models")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for XGBoost")
    parser.add_argument("--timeframes", default="all",
                        help="Comma-separated TFs (e.g., '15,60') or 'all'")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only WFO metrics, no final model")
    args = parser.parse_args()

    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║  Direction Model v3 — WFO Training                   ║")
    log.info("║  32 curated features, 6 domains                      ║")
    log.info("║  Regression on direction_quality                     ║")
    log.info("║  Target: direction accuracy ≥ 0.60 (honest)          ║")
    log.info("╚═══════════════════════════════════════════════════════╝")

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

    # Load data
    log.info(f"Loading {args.csv}...")
    load_start = time_module.time()
    df = pd.read_csv(args.csv)
    load_t = time_module.time() - load_start
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns in {load_t:.1f}s")

    # Validate columns
    required = DIRECTION_V3_FEATURES + [
        "symbol", "tf_minutes", "timestamp",
        "future_return_25", "direction", "is_super",
        "bars_to_tp", "direction_quality",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.warning(f"Missing columns (filled with 0): {missing}")
        for c in missing:
            df[c] = 0.0

    # Dataset overview
    log.info(f"\n  Dataset overview:")
    for tf_val in sorted(df["tf_minutes"].unique()):
        sub = df[df["tf_minutes"] == tf_val]
        n_super = int(sub["is_super"].sum())
        n_super_dir = int(((sub["is_super"] == 1) & (sub["direction"] != 0)).sum())
        marker = " ◄" if tf_val in target_tfs else ""
        log.info(f"    TF {tf_val:>5}m: {len(sub):>9,} rows, "
                 f"{sub['symbol'].nunique():>3} symbols, "
                 f"super={n_super} ({n_super / len(sub) * 100:.1f}%), "
                 f"super+dir={n_super_dir}{marker}")

    # Run WFO
    all_reports: Dict[int, dict] = {}
    total_start = time_module.time()

    for tf in target_tfs:
        tf_df = df[df["tf_minutes"] == tf].copy()
        if len(tf_df) < 200:
            log.warning(f"Skipping TF {tf}m: only {len(tf_df)} rows")
            continue
        report = train_wfo_for_tf(
            tf, tf_df,
            use_gpu=args.gpu,
            output_dir=args.output_dir,
            evaluate_only=args.evaluate_only,
        )
        if report:
            all_reports[tf] = report

    total_elapsed = time_module.time() - total_start

    # ══ FINAL SUMMARY ══
    log.info(f"\n╔═══════════════════════════════════════════════════════════╗")
    log.info(f"║  DIRECTION v3 — FINAL SUMMARY                            ║")
    log.info(f"╚═══════════════════════════════════════════════════════════╝")

    for tf in sorted(all_reports.keys()):
        r = all_reports[tf]
        agg = r.get("aggregate_oos", {})
        base = r.get("baseline_mean_acc", 0.5)

        acc_super = agg.get("dir_acc_super_mean", 0)
        auc_super = agg.get("dir_auc_super_mean", 0)
        acc_all = agg.get("dir_acc_all_mean", 0)
        met = "✅" if acc_super >= DIRECTION_ACC_TARGET else "❌"

        log.info(f"  TF {tf:>5}m: acc(super)={acc_super:.4f}  "
                 f"AUC(super)={auc_super:.4f}  "
                 f"acc(all)={acc_all:.4f}  "
                 f"baseline={base:.4f}  {met}")

    log.info(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    log.info(f"  Target: direction accuracy ≥ {DIRECTION_ACC_TARGET:.0%}")
    log.info(f"  Model files: models/direction_v3_tf{{X}}.ubj")
    log.info(f"  Done ✅")


if __name__ == "__main__":
    main()
