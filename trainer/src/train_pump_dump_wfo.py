#!/usr/bin/env python3
"""
train_pump_dump_wfo.py — Walk-Forward Optimization for Pump/Dump Detection Model.

APPROACH:
  Multi-timeframe indicator snapshots (6 TFs × 10 candles × 82 features = 4920 features)
  captured BEFORE sharp pump/dump events. XGBoost learns pre-pump/dump patterns.

WHY BINARY (not regression):
  Pump/dump is a rare event detection problem. Binary classification
  (event vs no-event) is more natural than regression. We train two
  separate models: one for pumps, one for dumps.

TRAINING:
  Walk-Forward Optimization: train on past → test on future.
  Each fold uses an expanding training window.
  Purge + embargo to avoid lookahead bias.

MEMORY PROTECTION:
  - All features loaded as float32 (not float64) → ~50% less RAM
  - gc.collect() between folds and TFs
  - max ~4920 features × ~10K examples ≈ ~200 MB per DataFrame
  - Target: < 15 GB total RAM usage

Usage:
    python trainer/src/train_pump_dump_wfo.py --csv dataset/pump_dump_dataset.csv --gpu
    python trainer/src/train_pump_dump_wfo.py --csv dataset/pump_dump_dataset.csv --model-type pump --gpu
    python trainer/src/train_pump_dump_wfo.py --csv dataset/pump_dump_dataset.csv --evaluate-only
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
        accuracy_score, roc_auc_score, precision_score, recall_score,
        f1_score, classification_report, confusion_matrix,
    )
except ImportError:
    print("ERROR: Required packages not installed.")
    print("  pip install xgboost scikit-learn pandas numpy")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING — writes to both stdout and logs/pump_dump_wfo_train.log
# ═════════════════════════════════════════════════════════════════════════════

def setup_logging(log_path: str = "logs/pump_dump_wfo_train.log"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("pump_dump_wfo")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger

log = setup_logging()


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════

# WFO config — pump/dump events are rare, so use fewer folds with larger windows
WFO_CONFIG = {
    "n_splits": 4,
    "test_days": 90.0,     # 3 months test per fold
    "min_train_days": 180.0,  # 6 months minimum training
}

# Accuracy target for the classifier
ACC_TARGET = 0.65


# ═════════════════════════════════════════════════════════════════════════════
# WFO FOLD COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def compute_wfo_folds(timestamps: pd.Series, config: dict) -> List[Dict]:
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
    df: pd.DataFrame, fold: Dict,
    embargo_days: int = 7, ts_col: str = "ts",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split with purge & embargo."""
    embargo_delta = timedelta(days=embargo_days)

    effective_train_end = fold["train_end_raw"] - timedelta(days=3)
    effective_test_start = fold["test_start_raw"] + embargo_delta

    train_mask = (df[ts_col] >= fold["train_start"]) & (df[ts_col] <= effective_train_end)
    test_mask = (df[ts_col] >= effective_test_start) & (df[ts_col] <= fold["test_end_raw"])

    return df[train_mask].copy(), df[test_mask].copy()


# ═════════════════════════════════════════════════════════════════════════════
# DETECT FEATURE COLUMNS
# ═════════════════════════════════════════════════════════════════════════════

def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect multi-TF feature columns (tf{N}_c{M}_{feature})."""
    feat_cols = [c for c in df.columns if c.startswith("tf") and "_c" in c]

    def sort_key(col):
        try:
            parts = col.split("_", 2)
            tf = int(parts[0][2:])
            ci = int(parts[1][1:])
            return (tf, ci, parts[2] if len(parts) > 2 else "")
        except (ValueError, IndexError):
            return (9999, 9999, col)

    feat_cols.sort(key=sort_key)
    return feat_cols


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING: BINARY CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════════

def train_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    event_type: str,
    use_gpu: bool = False,
    fold_idx: int = 0,
) -> Tuple[Optional[xgb.Booster], dict]:
    """
    Binary classification: event (1) vs no-event (0).

    Handles class imbalance with scale_pos_weight since pump/dump events
    are typically 20-30% of the dataset.
    """
    if len(train_df) < 100 or len(test_df) < 20:
        log.warning(f"  Fold {fold_idx}: Not enough data (train={len(train_df)}, test={len(test_df)}). Skipping.")
        return None, {"error": "not enough data"}

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values.astype(np.float32)

    # Clean NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    # Class balance
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / max(n_pos, 1)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Free numpy arrays
    del X_train, X_test
    gc.collect()

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "eta": 0.03,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.5,      # More aggressive subsampling for 4920 features
        "min_child_weight": 15,
        "lambda": 3.0,                  # Strong L2 reg for high-dim features
        "alpha": 1.0,                   # Strong L1 reg (feature selection)
        "gamma": 0.3,                   # Prevent overfitting
        "max_bin": 128,
        "scale_pos_weight": scale_pos,
        "tree_method": "hist",
        "device": "cuda" if use_gpu else "cpu",
        "verbosity": 0,
    }

    log.info(f"  Fold {fold_idx}: Training {event_type} binary on {len(y_train)} examples "
             f"(POS={n_pos}, NEG={n_neg}, scale_pos={scale_pos:.2f})")

    model = xgb.train(
        params, dtrain,
        num_boost_round=1500,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=100,
        verbose_eval=0,
    )

    # Free DMatrix
    del dtrain
    gc.collect()

    # ── Evaluate ──
    pred_proba = model.predict(dtest)
    pred_class = (pred_proba >= 0.5).astype(int)

    accuracy = float(accuracy_score(y_test, pred_class))
    try:
        auc = float(roc_auc_score(y_test, pred_proba))
    except Exception:
        auc = 0.5

    precision = float(precision_score(y_test, pred_class, zero_division=0))
    recall = float(recall_score(y_test, pred_class, zero_division=0))
    f1 = float(f1_score(y_test, pred_class, zero_division=0))

    n_test_pos = int(y_test.sum())
    n_test_neg = len(y_test) - n_test_pos
    n_pred_pos = int(pred_class.sum())
    n_pred_neg = len(pred_class) - n_pred_pos

    # Confidence thresholds
    gate_analysis = {}
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        mask = pred_proba >= thresh
        if mask.sum() > 5:
            gated_pred = pred_class[mask]
            gated_actual = y_test[mask]
            gated_prec = float(precision_score(gated_actual, gated_pred, zero_division=0))
            gated_rec = float(recall_score(gated_actual, gated_pred, zero_division=0))
            gate_analysis[f"gate_{thresh:.2f}"] = {
                "precision": gated_prec,
                "recall": gated_rec,
                "n_samples": int(mask.sum()),
                "coverage": float(mask.sum()) / float(len(pred_proba)),
            }

    # Top features
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    top20 = [(name, round(gain, 2)) for name, gain in sorted_imp[:20]]

    metrics = {
        "mode": f"binary_{event_type}",
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_test_pos": n_test_pos,
        "n_test_neg": n_test_neg,
        "n_pred_pos": n_pred_pos,
        "n_pred_neg": n_pred_neg,
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "best_iteration": int(model.best_iteration),
        "top20_features": top20,
        "confidence_gates": gate_analysis,
    }

    met = "✅" if accuracy >= ACC_TARGET else "❌"
    log.info(f"  Fold {fold_idx}: acc={accuracy:.4f} AUC={auc:.4f} "
             f"prec={precision:.4f} recall={recall:.4f} F1={f1:.4f} {met}")
    log.info(f"    Test: POS={n_test_pos} NEG={n_test_neg} | "
             f"Pred: POS={n_pred_pos} NEG={n_pred_neg} | trees={model.best_iteration + 1}")

    for gk, gv in gate_analysis.items():
        log.info(f"    {gk}: prec={gv['precision']:.3f} rec={gv['recall']:.3f} "
                 f"n={gv['n_samples']} cov={gv['coverage']:.1%}")

    return model, metrics


# ═════════════════════════════════════════════════════════════════════════════
# WFO PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def train_wfo(
    df: pd.DataFrame,
    feature_cols: List[str],
    event_type: str,
    use_gpu: bool = False,
    output_dir: str = "models",
    evaluate_only: bool = False,
) -> dict:
    """Run Walk-Forward Optimization for pump or dump model."""
    t_start = time_module.time()
    n_features = len(feature_cols)

    log.info(f"\n{'═' * 75}")
    log.info(f"  PUMP/DUMP WFO — {event_type.upper()} model")
    log.info(f"  Data: {len(df)} rows, {df['symbol'].nunique()} symbols")
    log.info(f"  Features: {n_features}")
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    log.info(f"  Labels: POS={n_pos} ({n_pos/len(df)*100:.1f}%), NEG={n_neg} ({n_neg/len(df)*100:.1f}%)")
    log.info(f"  Target: accuracy ≥ {ACC_TARGET:.0%}")
    log.info(f"{'═' * 75}")

    df = df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)

    folds = compute_wfo_folds(df["ts"], WFO_CONFIG)
    if not folds:
        log.error("No valid folds!")
        return {}

    log.info(f"  {len(folds)} WFO folds:")
    for fold in folds:
        train_days = (fold["train_end_raw"] - fold["train_start"]).total_seconds() / 86400
        test_days = (fold["test_end_raw"] - fold["test_start_raw"]).total_seconds() / 86400
        log.info(f"    Fold {fold['fold_idx']}: Train {train_days:.0f}d | Test {test_days:.0f}d")

    fold_metrics: List[dict] = []
    fold_details: List[dict] = []

    for fold in folds:
        fi = fold["fold_idx"]
        log.info(f"\n  ── Fold {fi}/{len(folds) - 1} ──")

        train_df, test_df = apply_fold_split(df, fold, embargo_days=7, ts_col="ts")

        if len(train_df) < 100 or len(test_df) < 20:
            log.warning(f"  Too few samples (train={len(train_df)}, test={len(test_df)}). Skipping.")
            continue

        log.info(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

        # Baseline
        baseline = max(test_df["label"].mean(), 1 - test_df["label"].mean())
        log.info(f"  Baseline (majority class): {baseline:.4f}")

        model, metrics = train_binary(
            train_df, test_df, feature_cols,
            event_type=event_type, use_gpu=use_gpu, fold_idx=fi,
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

        # Memory cleanup
        del train_df, test_df, model
        gc.collect()

    # ══ AGGREGATE ══
    log.info(f"\n  {'─' * 60}")
    log.info(f"  AGGREGATE OOS — {event_type.upper()} model")
    log.info(f"  {'─' * 60}")

    valid = [m for m in fold_metrics if "error" not in m]
    if not valid:
        log.error("  No valid folds!")
        return {}

    accs = [m["accuracy"] for m in valid]
    aucs = [m["auc"] for m in valid]
    precs = [m["precision"] for m in valid]
    recalls = [m["recall"] for m in valid]
    f1s = [m["f1"] for m in valid]
    baselines = [m.get("baseline", 0.5) for m in valid]

    agg = {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "auc_mean": float(np.mean(aucs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recalls)),
        "f1_mean": float(np.mean(f1s)),
        "baseline_mean": float(np.mean(baselines)),
    }

    met = "✅" if agg["acc_mean"] >= ACC_TARGET else "❌"
    log.info(f"  Accuracy:  {agg['acc_mean']:.4f}±{np.std(accs):.4f} {met}")
    log.info(f"  AUC:       {agg['auc_mean']:.4f}±{np.std(aucs):.4f}")
    log.info(f"  Precision: {agg['precision_mean']:.4f}")
    log.info(f"  Recall:    {agg['recall_mean']:.4f}")
    log.info(f"  F1:        {agg['f1_mean']:.4f}")
    log.info(f"  Baseline:  {agg['baseline_mean']:.4f}")
    lift = agg['acc_mean'] - agg['baseline_mean']
    log.info(f"  Lift:      {lift:+.4f} {'📈' if lift > 0 else '📉'}")

    # Feature importance stability
    feat_counts: Dict[str, int] = {}
    for m in valid:
        for feat, _ in m.get("top20_features", []):
            feat_counts[feat] = feat_counts.get(feat, 0) + 1
    if feat_counts:
        stable = sorted(feat_counts.items(), key=lambda x: x[1], reverse=True)
        log.info(f"  Stable top features: "
                 + ", ".join(f"{f}({c}/{len(valid)})" for f, c in stable[:15]))

    # ══ SAVE MODEL ══
    if not evaluate_only and valid:
        os.makedirs(output_dir, exist_ok=True)

        log.info(f"\n  ── FINAL model (ALL data) ──")
        n_val = max(int(len(df) * 0.10), 50)
        val_df = df.tail(n_val)
        train_all_df = df.head(len(df) - n_val)

        final_model, final_metrics = train_binary(
            train_all_df, val_df, feature_cols,
            event_type=event_type, use_gpu=use_gpu, fold_idx=-1,
        )

        if final_model:
            path = os.path.join(output_dir, f"pump_dump_{event_type}_v1.ubj")
            final_model.save_model(path)
            log.info(f"  ✅ Saved: {path}")

            schema = {
                "features": feature_cols,
                "feature_count": len(feature_cols),
                "task": f"pump_dump_{event_type}_binary",
                "objective": "binary:logistic",
                "version": "v1",
                "wfo_acc_mean": agg["acc_mean"],
                "wfo_auc_mean": agg["auc_mean"],
                "wfo_f1_mean": agg["f1_mean"],
                "wfo_folds": len(valid),
            }
            schema_path = path.replace(".ubj", ".schema.json")
            with open(schema_path, "w") as f:
                json.dump(schema, f, indent=2)
            log.info(f"  ✅ Schema: {schema_path}")

        del train_all_df, val_df, final_model
        gc.collect()

    elapsed = time_module.time() - t_start

    report = {
        "event_type": event_type,
        "training_method": "walk_forward_optimization",
        "model_version": "pump_dump_v1",
        "feature_count": n_features,
        "target_accuracy": ACC_TARGET,
        "total_rows": len(df),
        "total_symbols": int(df["symbol"].nunique()),
        "aggregate_oos": agg,
        "fold_details": fold_details,
        "elapsed_seconds": round(elapsed, 1),
    }

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"pump_dump_{event_type}_wfo_report.json")
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
        description="Pump/Dump Detection — WFO Training",
    )
    parser.add_argument("--csv", default="dataset/pump_dump_dataset.csv",
                        help="Path to pump_dump_dataset.csv")
    parser.add_argument("--output-dir", default="models",
                        help="Output directory for models")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for XGBoost")
    parser.add_argument("--model-type", default="both",
                        choices=["pump", "dump", "both"],
                        help="Which model to train (default: both)")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Only WFO evaluation, no final model save")
    args = parser.parse_args()

    log.info("╔═══════════════════════════════════════════════════════╗")
    log.info("║  Pump/Dump Detection — WFO Training                   ║")
    log.info("║  Multi-TF indicator patterns → XGBoost                ║")
    log.info("║  Sharp move detection (concentrated in 1-2 hours)     ║")
    log.info("╚═══════════════════════════════════════════════════════╝")
    log.info(f"Model type: {args.model_type}")

    if not os.path.exists(args.csv):
        log.error(f"Dataset not found: {args.csv}")
        log.info("Generate it first:")
        log.info("  cargo build --release -p ml_pump_dump --bin pump_dump_dataset")
        log.info("  ./target/release/pump_dump_dataset")
        sys.exit(1)

    # Load data with float32 for memory efficiency
    log.info(f"Loading {args.csv}...")
    load_start = time_module.time()
    df = pd.read_csv(args.csv)
    load_t = time_module.time() - load_start
    log.info(f"  Loaded {len(df):,} rows, {len(df.columns)} columns in {load_t:.1f}s")

    # Detect feature columns
    feature_cols = detect_feature_columns(df)
    log.info(f"  Detected {len(feature_cols)} feature columns")
    if len(feature_cols) == 0:
        log.error("No feature columns found! Expected columns like tf1440_c0_rsi, ...")
        sys.exit(1)

    # Validate required columns
    required = ["symbol", "timestamp", "event_type", "label", "move_pct"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error(f"Missing required columns: {missing}")
        sys.exit(1)

    # ── Memory optimization: float32 ──
    mem_before = df.memory_usage(deep=True).sum() / 1e9
    for col in feature_cols:
        df[col] = df[col].astype(np.float32)
    df["move_pct"] = df["move_pct"].astype(np.float32)
    mem_after = df.memory_usage(deep=True).sum() / 1e9
    log.info(f"  Memory: {mem_before:.2f} GB → {mem_after:.2f} GB (float32)")
    gc.collect()

    # Dataset overview
    log.info(f"\n  Dataset overview:")
    for et in sorted(df["event_type"].unique()):
        sub = df[df["event_type"] == et]
        log.info(f"    {et:>6}: {len(sub):>8,} rows ({sub['symbol'].nunique()} symbols)")
    n_pos = int((df["label"] == 1).sum())
    n_neg = int((df["label"] == 0).sum())
    log.info(f"    Total POS={n_pos} NEG={n_neg} (POS ratio={n_pos/(n_pos+n_neg)*100:.1f}%)")

    all_reports: Dict[str, dict] = {}
    total_start = time_module.time()

    # ── Train PUMP model ──
    if args.model_type in ("pump", "both"):
        log.info("\n" + "=" * 75)
        log.info("  TRAINING PUMP MODEL")
        log.info("=" * 75)

        # Filter: keep PUMP events (label=1, event_type=PUMP) + all negatives (label=0)
        pump_df = df[(df["event_type"] == "PUMP") | (df["label"] == 0)].copy()
        log.info(f"  Pump dataset: {len(pump_df)} rows")

        if len(pump_df) >= 200:
            report = train_wfo(
                pump_df, feature_cols, event_type="pump",
                use_gpu=args.gpu, output_dir=args.output_dir,
                evaluate_only=args.evaluate_only,
            )
            if report:
                all_reports["pump"] = report
        else:
            log.warning(f"  Not enough pump data ({len(pump_df)} rows). Skipping.")

        del pump_df
        gc.collect()

    # ── Train DUMP model ──
    if args.model_type in ("dump", "both"):
        log.info("\n" + "=" * 75)
        log.info("  TRAINING DUMP MODEL")
        log.info("=" * 75)

        dump_df = df[(df["event_type"] == "DUMP") | (df["label"] == 0)].copy()
        log.info(f"  Dump dataset: {len(dump_df)} rows")

        if len(dump_df) >= 200:
            report = train_wfo(
                dump_df, feature_cols, event_type="dump",
                use_gpu=args.gpu, output_dir=args.output_dir,
                evaluate_only=args.evaluate_only,
            )
            if report:
                all_reports["dump"] = report
        else:
            log.warning(f"  Not enough dump data ({len(dump_df)} rows). Skipping.")

        del dump_df
        gc.collect()

    total_elapsed = time_module.time() - total_start

    # ══ FINAL SUMMARY ══
    log.info(f"\n╔═══════════════════════════════════════════════════════════╗")
    log.info(f"║  PUMP/DUMP — FINAL SUMMARY                                ║")
    log.info(f"╚═══════════════════════════════════════════════════════════╝")

    for et in sorted(all_reports.keys()):
        r = all_reports[et]
        agg = r.get("aggregate_oos", {})
        acc = agg.get("acc_mean", 0)
        auc = agg.get("auc_mean", 0)
        f1 = agg.get("f1_mean", 0)
        base = agg.get("baseline_mean", 0.5)
        met = "✅" if acc >= ACC_TARGET else "❌"
        log.info(f"  {et.upper():>5}: acc={acc:.4f} AUC={auc:.4f} F1={f1:.4f} base={base:.4f} {met}")

    log.info(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")
    log.info(f"  Models: models/pump_dump_{{pump,dump}}_v1.ubj")
    log.info(f"  Done ✅")


if __name__ == "__main__":
    main()
