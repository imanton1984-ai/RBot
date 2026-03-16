#!/usr/bin/env python3
"""
train_super_level.py — Train 5 XGBoost models for Super Level Strategy.

Input:  dataset/super_level_dataset.csv (from Rust dataset builder)
Output: 5 models per TF:
  1. models/slvl_level_v1_tf{X}.ubj   — P(strong_level)
  2. models/slvl_entry_v1_tf{X}.ubj   — P(good_entry)
  3. models/slvl_dir_v1_tf{X}.ubj     — P(LONG direction)
  4. models/slvl_bb_v1_tf{X}.ubj      — P(bounce)
  5. models/slvl_eval_v1_tf{X}.ubj    — P(win) evaluator

KEY DIFFERENCES from train_super_entry.py:
  - TIME-BASED split (not pair-based): first train_pct% = train, rest = test
  - 5 separate models instead of 2
  - Level features (18 extra: nearest_sup/res_dist, touches, strength, etc.)
  - Detailed per-model quality report
  - Percentage-based split from current candle count (handles different TFs)

Usage:
    python trainer/src/train_super_level.py [--csv dataset/super_level_dataset.csv] [--gpu]
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    from sklearn.metrics import (
        roc_auc_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, accuracy_score
    )
except ImportError:
    print("ERROR: pip install xgboost scikit-learn pandas numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# FEATURE NAMES (must match Rust config.rs)
# ═══════════════════════════════════════════════════════════════

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

LEVEL_FEATURES = [
    "nearest_sup_dist_atr", "nearest_sup_dist_pct",
    "nearest_sup_touches", "nearest_sup_strength",
    "nearest_sup_bars_since_touch", "nearest_sup_age_bars",
    "nearest_res_dist_atr", "nearest_res_dist_pct",
    "nearest_res_touches", "nearest_res_strength",
    "nearest_res_bars_since_touch", "nearest_res_age_bars",
    "in_level_zone", "channel_width_atr", "channel_position",
    "total_levels_nearby", "approach_velocity_3", "approach_velocity_5",
]

DYNAMIC_FEATURES = [
    "price_return_lb3", "atr_ratio_lb3", "rsi_slope_lb3",
    "trend_persist_lb3", "adx_slope_lb3", "macd_hist_slope_lb3",
    "price_return_lb5", "atr_ratio_lb5", "rsi_slope_lb5",
    "trend_persist_lb5", "adx_slope_lb5", "macd_hist_slope_lb5",
    "price_return_lb10", "atr_ratio_lb10", "rsi_slope_lb10",
    "trend_persist_lb10", "adx_slope_lb10", "macd_hist_slope_lb10",
    "price_return_lb15", "atr_ratio_lb15", "rsi_slope_lb15",
    "trend_persist_lb15", "adx_slope_lb15", "macd_hist_slope_lb15",
    "price_return_lb25", "atr_ratio_lb25", "rsi_slope_lb25",
    "trend_persist_lb25", "adx_slope_lb25", "macd_hist_slope_lb25",
    "supertrend_consistency", "trend_alignment",
    "price_accel", "volume_trend_ratio",
]

ALL_FEATURES = INDICATOR_FEATURES + DERIVED_FEATURES + LEVEL_FEATURES + DYNAMIC_FEATURES

TF_TARGET_MOVE_PCT = {1: 1.2, 5: 2.8, 15: 3.5, 60: 5.0, 240: 7.5, 1440: 10.0}
TIMEFRAMES = [1, 5, 15, 60, 240, 1440]


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_data(csv_path: str) -> pd.DataFrame:
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    required = ALL_FEATURES + [
        "symbol", "tf_minutes", "direction", "is_win",
        "is_good_entry", "is_bounce", "level_strength", "level_touches",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        for c in missing:
            df[c] = 0.0

    n_feats = len(ALL_FEATURES)
    print(f"  Features: {n_feats} (ind={len(INDICATOR_FEATURES)}, "
          f"der={len(DERIVED_FEATURES)}, lvl={len(LEVEL_FEATURES)}, dyn={len(DYNAMIC_FEATURES)})")

    # Stats per TF
    for tf in sorted(df["tf_minutes"].unique()):
        tf_df = df[df["tf_minutes"] == tf]
        n_win = (tf_df["is_win"] == 1).sum()
        n_entry = (tf_df["is_good_entry"] == 1).sum()
        n_bounce = (tf_df["is_bounce"] == 1).sum()
        n_strong = (tf_df["level_strength"] >= 3).sum()
        print(f"    TF {tf}m: {len(tf_df)} rows | "
              f"win={n_win}({n_win/len(tf_df)*100:.1f}%) | "
              f"entry={n_entry}({n_entry/len(tf_df)*100:.1f}%) | "
              f"bounce={n_bounce}({n_bounce/len(tf_df)*100:.1f}%) | "
              f"strong_lvl={n_strong}({n_strong/len(tf_df)*100:.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════
# TIME-BASED SPLIT
# ═══════════════════════════════════════════════════════════════

def split_time_based(df: pd.DataFrame, train_pct: float = 65.0, prep_pct: float = 5.0):
    """
    Time-based split PER SYMBOL:
      - First prep_pct% — prep (skipped, used for warmup)
      - Next train_pct% — train
      - Remaining — test
    This preserves temporal order and avoids look-ahead bias.
    """
    train_frames = []
    test_frames = []

    for symbol in df["symbol"].unique():
        sym_df = df[df["symbol"] == symbol].sort_values("timestamp").reset_index(drop=True)
        n = len(sym_df)

        prep_end = int(n * prep_pct / 100.0)
        train_end = prep_end + int(n * train_pct / 100.0)

        # Skip prep rows (warmup), split rest into train/test
        if train_end > prep_end:
            train_frames.append(sym_df.iloc[prep_end:train_end])
        if train_end < n:
            test_frames.append(sym_df.iloc[train_end:])

    train_df = pd.concat(train_frames, ignore_index=True) if train_frames else pd.DataFrame()
    test_df = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    print(f"\n  Time-based split (prep={prep_pct}%, train={train_pct}%, test={100-prep_pct-train_pct:.0f}%):")
    print(f"    Train: {len(train_df)} rows ({df['symbol'].nunique()} symbols)")
    print(f"    Test:  {len(test_df)} rows")

    return train_df, test_df


# ═══════════════════════════════════════════════════════════════
# TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════

def train_binary(train_df, test_df, label_col, feature_cols,
                 use_gpu=False, model_name="", custom_params=None,
                 num_boost_round=500, early_stopping_rounds=30):
    """Train one XGBoost binary classifier with detailed report."""

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df[label_col].values.astype(np.float32)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

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
        "verbosity": 1,
    }

    if custom_params:
        params.update(custom_params)

    print(f"\n  ┌─── Training: {model_name} ───")
    print(f"  │ pos/neg: {int(n_pos)}/{int(n_neg)} (weight={scale_pos_weight:.2f})")
    print(f"  │ eta={params['eta']}, depth={params['max_depth']}, "
          f"rounds={num_boost_round}, early_stop={early_stopping_rounds}")

    evals = [(dtrain, "train"), (dtest, "test")]
    model = xgb.train(
        params, dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=100,
    )

    # ── Detailed evaluation ──
    y_prob_train = model.predict(xgb.DMatrix(X_train, feature_names=feature_cols))
    y_prob_test = model.predict(dtest)
    y_pred_train = (y_prob_train >= 0.5).astype(int)
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    try:
        train_auc = roc_auc_score(y_train, y_prob_train)
    except:
        train_auc = 0.0
    try:
        test_auc = roc_auc_score(y_test, y_prob_test)
    except:
        test_auc = 0.0

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_prec = precision_score(y_test, y_pred_test, zero_division=0)
    test_rec = recall_score(y_test, y_pred_test, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

    print(f"  │")
    print(f"  │ ══════ {model_name} RESULTS ══════")
    print(f"  │   TRAIN: AUC={train_auc:.4f}  Acc={train_acc:.4f}")
    print(f"  │   TEST:  AUC={test_auc:.4f}  Acc={test_acc:.4f}")
    print(f"  │          Precision={test_prec:.4f}  Recall={test_rec:.4f}  F1={test_f1:.4f}")
    print(f"  │   Overfit gap: AUC={train_auc-test_auc:.4f}  Acc={train_acc-test_acc:.4f}")

    if len(np.unique(y_test)) > 1:
        cm = confusion_matrix(y_test, y_pred_test)
        tn, fp, fn, tp = cm.ravel()
        print(f"  │   Confusion: TP={tp} FP={fp} FN={fn} TN={tn}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"  │")
    print(f"  │   Top 10 Features (by Gain):")
    for name, gain in sorted_imp[:10]:
        print(f"  │     {name:35s} {gain:.1f}")
    print(f"  └─────────────────────────────")

    metrics = {
        "train_auc": train_auc, "test_auc": test_auc,
        "train_acc": train_acc, "test_acc": test_acc,
        "precision": test_prec, "recall": test_rec, "f1": test_f1,
        "overfit_auc_gap": train_auc - test_auc,
    }
    return model, metrics


def save_model(model, feature_names, tf, task_name, metrics, output_dir):
    """Save model as .ubj + schema + metadata."""
    os.makedirs(output_dir, exist_ok=True)
    model_name = f"{task_name}_v1_tf{tf}"

    ubj_path = os.path.join(output_dir, f"{model_name}.ubj")
    model.save_model(ubj_path)
    print(f"  Model: {ubj_path}")

    json_path = os.path.join(output_dir, f"{model_name}.json")
    model.save_model(json_path)

    schema = {
        "schema_id": hashlib.sha256(",".join(feature_names).encode()).hexdigest() + f"_tf{tf}",
        "features": feature_names,
        "task": task_name,
        "tf_minutes": tf,
        "outputs": 1,
        "objective": "binary:logistic",
    }
    schema_path = os.path.join(output_dir, f"{model_name}.schema.json")
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    meta = {
        "model_type": "xgboost_binary_classifier",
        "strategy": "super_level_strategy",
        "target": task_name,
        "n_features": len(feature_names),
        "metrics": metrics,
        "best_iteration": model.best_iteration,
        "n_trees": model.best_iteration + 1,
        "trained_at": datetime.utcnow().isoformat(),
    }
    meta_path = os.path.join(output_dir, f"{model_name}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train Super Level 5-Model Ensemble")
    parser.add_argument("--csv", default="dataset/super_level_dataset.csv")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--train-pct", type=float, default=65.0,
                        help="% of candles for training (time-based)")
    parser.add_argument("--prep-pct", type=float, default=5.0,
                        help="% of candles to skip (warmup)")
    args = parser.parse_args()

    # Setup logging
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"super_level_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║     SUPER LEVEL STRATEGY — 5 MODEL TRAINING                 ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"  CSV:       {args.csv}")
    print(f"  Output:    {args.output_dir}")
    print(f"  GPU:       {args.gpu}")
    print(f"  Split:     prep={args.prep_pct}%, train={args.train_pct}%, "
          f"test={100-args.prep_pct-args.train_pct:.0f}%")
    print(f"  Log:       {log_file}")
    print()

    if not os.path.exists(args.csv):
        print(f"ERROR: {args.csv} not found!")
        print("Generate dataset first:")
        print("  cargo run --release -p super_level_strategy --bin super_level_dataset")
        sys.exit(1)

    df = load_data(args.csv)
    if len(df) < 100:
        print(f"Too few samples ({len(df)})")
        sys.exit(1)

    all_model_results = {}

    for tf in TIMEFRAMES:
        tf_df = df[df["tf_minutes"] == tf].copy()
        if len(tf_df) < 50:
            print(f"\nSkipping TF {tf}m: only {len(tf_df)} samples")
            continue

        print(f"\n{'═' * 70}")
        print(f"  TRAINING MODELS FOR TF {tf}m ({len(tf_df)} samples)")
        print(f"  Target move: {TF_TARGET_MOVE_PCT.get(tf, 0)}%")
        print(f"{'═' * 70}")

        # Time-based split
        train_df, test_df = split_time_based(tf_df, args.train_pct, args.prep_pct)

        if len(train_df) < 30 or len(test_df) < 10:
            print(f"  Not enough data after split. Skipping.")
            continue

        tf_results = {}

        # ═══════════════════════════════════════════════════
        # MODEL 1: Level Quality (P(strong_level))
        # Label: level_strength >= 3 (Strong)
        # ═══════════════════════════════════════════════════
        train_df["label_level"] = (train_df["level_strength"] >= 3).astype(int)
        test_df["label_level"] = (test_df["level_strength"] >= 3).astype(int)

        level_model, level_metrics = train_binary(
            train_df, test_df,
            label_col="label_level",
            feature_cols=ALL_FEATURES,
            use_gpu=args.gpu,
            model_name=f"LEVEL TF{tf}m",
        )
        save_model(level_model, ALL_FEATURES, tf, "slvl_level", level_metrics, args.output_dir)
        tf_results["level"] = level_metrics

        # ═══════════════════════════════════════════════════
        # MODEL 2: Entry Quality (P(good_entry))
        # Label: is_good_entry (TP hit AND near level)
        # ═══════════════════════════════════════════════════
        entry_model, entry_metrics = train_binary(
            train_df, test_df,
            label_col="is_good_entry",
            feature_cols=ALL_FEATURES,
            use_gpu=args.gpu,
            model_name=f"ENTRY TF{tf}m",
        )
        save_model(entry_model, ALL_FEATURES, tf, "slvl_entry", entry_metrics, args.output_dir)
        tf_results["entry"] = entry_metrics

        # ═══════════════════════════════════════════════════
        # MODEL 3: Direction (P(LONG))
        # Train only on is_win=True examples (clear directional outcome)
        # Label: direction == 1 (LONG)
        # ═══════════════════════════════════════════════════
        train_dir = train_df[train_df["is_win"] == 1].copy()
        test_dir = test_df[test_df["is_win"] == 1].copy()

        if len(train_dir) >= 50 and len(test_dir) >= 20:
            train_dir["label_long"] = (train_dir["direction"] == 1).astype(int)
            test_dir["label_long"] = (test_dir["direction"] == 1).astype(int)

            print(f"\n  Direction model: {len(train_dir)} win-only train, {len(test_dir)} test")

            dir_params = {
                "eta": 0.02,
                "max_depth": 5,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 30,
                "lambda": 3.0,
                "alpha": 0.5,
                "gamma": 0.3,
            }

            dir_model, dir_metrics = train_binary(
                train_dir, test_dir,
                label_col="label_long",
                feature_cols=ALL_FEATURES,
                use_gpu=args.gpu,
                model_name=f"DIRECTION TF{tf}m",
                custom_params=dir_params,
                num_boost_round=1500,
                early_stopping_rounds=60,
            )
            save_model(dir_model, ALL_FEATURES, tf, "slvl_dir", dir_metrics, args.output_dir)
            tf_results["direction"] = dir_metrics
        else:
            print(f"  ⚠️ Not enough win examples for direction model "
                  f"(train={len(train_dir)}, test={len(test_dir)}). Skipping.")

        # ═══════════════════════════════════════════════════
        # MODEL 4: Bounce/Break (P(bounce))
        # Train on examples near levels (dist_atr < 1.0)
        # Label: is_bounce
        # ═══════════════════════════════════════════════════
        train_bb = train_df[train_df["nearest_level_dist_atr"] < 1.5].copy()
        test_bb = test_df[test_df["nearest_level_dist_atr"] < 1.5].copy()

        if len(train_bb) >= 50 and len(test_bb) >= 20:
            print(f"\n  Bounce/Break model: {len(train_bb)} near-level train, {len(test_bb)} test")

            bb_model, bb_metrics = train_binary(
                train_bb, test_bb,
                label_col="is_bounce",
                feature_cols=ALL_FEATURES,
                use_gpu=args.gpu,
                model_name=f"BOUNCE/BREAK TF{tf}m",
            )
            save_model(bb_model, ALL_FEATURES, tf, "slvl_bb", bb_metrics, args.output_dir)
            tf_results["bounce_break"] = bb_metrics
        else:
            print(f"  ⚠️ Not enough near-level examples for BB model. Skipping.")

        # ═══════════════════════════════════════════════════
        # MODEL 5: Evaluator (P(win))
        # Final judge: uses ALL features to predict win/loss
        # ═══════════════════════════════════════════════════
        eval_params = {
            "eta": 0.03,
            "max_depth": 5,
            "subsample": 0.75,
            "colsample_bytree": 0.75,
            "min_child_weight": 10,
            "lambda": 2.0,
            "gamma": 0.2,
        }

        eval_model, eval_metrics = train_binary(
            train_df, test_df,
            label_col="is_win",
            feature_cols=ALL_FEATURES,
            use_gpu=args.gpu,
            model_name=f"EVALUATOR TF{tf}m",
            custom_params=eval_params,
            num_boost_round=1000,
            early_stopping_rounds=50,
        )
        save_model(eval_model, ALL_FEATURES, tf, "slvl_eval", eval_metrics, args.output_dir)
        tf_results["evaluator"] = eval_metrics

        all_model_results[tf] = tf_results

    # ═══════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ═══════════════════════════════════════════════════════════

    print(f"\n{'═' * 70}")
    print(f"  ✅ SUPER LEVEL TRAINING COMPLETE — SUMMARY")
    print(f"{'═' * 70}\n")

    # Table header
    print(f"{'TF':>5s} {'Model':>12s} {'Train AUC':>10s} {'Test AUC':>10s} "
          f"{'Precision':>10s} {'Recall':>8s} {'F1':>6s} {'Overfit':>8s}")
    print("-" * 75)

    for tf in sorted(all_model_results.keys()):
        tf_res = all_model_results[tf]
        for model_name, metrics in tf_res.items():
            print(f"{tf:>5}m {model_name:>12s} "
                  f"{metrics.get('train_auc', 0):.4f}     "
                  f"{metrics.get('test_auc', 0):.4f}     "
                  f"{metrics.get('precision', 0):.4f}     "
                  f"{metrics.get('recall', 0):.4f}  "
                  f"{metrics.get('f1', 0):.4f} "
                  f"{metrics.get('overfit_auc_gap', 0):+.4f}")

    print(f"\n  Models saved to: {args.output_dir}/")
    print(f"  Log:             {log_file}")
    print(f"\n  Next steps:")
    print(f"  1. Run backtest: cargo run --release -p super_level_strategy --bin super_level_backtest")
    print(f"{'═' * 70}")

    # Save summary to log
    with open(log_file, "w") as f:
        f.write(f"Super Level Training Log\n")
        f.write(f"Date: {datetime.utcnow().isoformat()}\n")
        f.write(f"CSV: {args.csv}\n\n")
        for tf in sorted(all_model_results.keys()):
            f.write(f"\nTF {tf}m:\n")
            for model_name, metrics in all_model_results[tf].items():
                f.write(f"  {model_name}: {json.dumps(metrics, indent=2)}\n")


if __name__ == "__main__":
    main()
