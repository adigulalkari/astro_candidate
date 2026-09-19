"""Reports on the currently deployed exoplanet ranking model.

Prints the top-K ranking metrics against the domain-defined target label, plus
whatever validation the training run recorded in models/model_meta.json
(cross-validated average precision, out-of-fold metrics, feature correlation
and permutation-importance audit). See data_pipeline/train_model.py for how
those numbers are produced.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_pipeline"))
from feature_engineering import FEATURE_COLUMNS  # noqa: E402

PROCESSED_DIR = "data/processed"
PARQUET_PATH = os.path.join(PROCESSED_DIR, "planets.parquet")
MODEL_DIR = "models"
SKL_MODEL_PATH = os.path.join(MODEL_DIR, "exoplanet_ranker.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "exoplanet_ranker.json")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

RADIUS_RANGE_EARTH = (0.5, 1.5)
EQUILIBRIUM_TEMP_RANGE_K = (200.0, 320.0)
INSOLATION_RANGE_EARTH = (0.2, 2.2)
MAX_FOLLOWUP_DISTANCE_PC = 150.0


def _define_target(df: pd.DataFrame) -> pd.Series:
    radius_ok = df["pl_rade"].between(*RADIUS_RANGE_EARTH)
    temp_ok = df["pl_eqt"].between(*EQUILIBRIUM_TEMP_RANGE_K)
    insol_ok = df["pl_insol"].between(*INSOLATION_RANGE_EARTH)
    distance_ok = df["sy_dist"].isna() | (df["sy_dist"] <= MAX_FOLLOWUP_DISTANCE_PC)
    return (radius_ok & temp_ok & insol_ok & distance_ok).astype(int)


def _print_ranking_metrics(df: pd.DataFrame) -> None:
    df_sorted = df.sort_values(by="ml_score", ascending=False).reset_index(drop=True)
    total_positives = df["target"].sum()

    for k in (5, 10, 25):
        top_k = df_sorted.head(k)
        hits = top_k["target"].sum()
        precision = hits / k
        recall = hits / total_positives if total_positives > 0 else 0.0
        ndcg = ndcg_score(np.array([df["target"].values]), np.array([df["ml_score"].values]), k=k)

        print(f"Metrics @ K={k}:")
        print(f"  Precision@{k}: {precision:.4f} (Hits: {hits})")
        print(f"  Recall@{k}: {recall:.4f}")
        print(f"  NDCG@{k}: {ndcg:.4f}")

    print("\nTop 10 ranked candidates:")
    cols_show = ["pl_name", "data_source", "sy_dist", "pl_rade", "pl_eqt", "pl_insol", "ml_score", "target"]
    cols_show = [c for c in cols_show if c in df_sorted.columns]
    print(df_sorted[cols_show].head(10).to_string(index=False))


def _print_training_validation(meta: dict) -> None:
    print(f"\nTraining source: {meta.get('trained_in', 'unknown')}")
    print(f"Rows: {meta.get('num_rows')}, positive labels: {meta.get('num_positive_labels')}")

    cv_ap = meta.get("cv_average_precision")
    if cv_ap is not None:
        print(f"Hyperparameter-search CV average precision: {cv_ap:.4f}")

    oof = meta.get("out_of_fold_metrics")
    if oof:
        print("Out-of-fold metrics (honest full-dataset estimate, not a single split):")
        for name, value in oof.items():
            print(f"  {name}: {value:.4f}")

    test_metrics = meta.get("test_metrics")
    if test_metrics:
        print("Held-out test split metrics:")
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")

    high_corr = meta.get("high_correlation_feature_pairs")
    if high_corr:
        print(f"\nFeature pairs with |correlation| >= 0.9 ({len(high_corr)}):")
        for a, b, r in high_corr:
            print(f"  {a} <-> {b}: {r}")

    perm_importance = meta.get("permutation_importance")
    if perm_importance:
        ranked = sorted(perm_importance.items(), key=lambda kv: -kv[1])
        print("\nPermutation importance (mean drop in average precision when shuffled):")
        for name, value in ranked[:10]:
            print(f"  {name}: {value:.5f}")
        zero_importance = [name for name, value in perm_importance.items() if value == 0.0]
        if zero_importance:
            print(
                f"\n{len(zero_importance)} feature(s) contribute ~0 importance and are "
                f"candidates for removal in a future retrain: {', '.join(sorted(zero_importance))}"
            )


def _print_gain_based_feature_importance(active_model: str) -> None:
    print(f"\nModel gain-based feature importance ({active_model}):")
    try:
        if active_model == "xgboost" and os.path.exists(XGB_MODEL_PATH):
            import xgboost as xgb

            model = xgb.XGBClassifier()
            model.load_model(XGB_MODEL_PATH)
            importances = model.feature_importances_
        elif os.path.exists(SKL_MODEL_PATH):
            import joblib

            model = joblib.load(SKL_MODEL_PATH)
            importances = model.feature_importances_
        else:
            print("Model file not found.")
            return

        feat_imp = pd.Series(importances, index=FEATURE_COLUMNS).sort_values(ascending=False)
        print(feat_imp.head(10).to_string())
    except Exception as e:
        print("Could not compute feature importances:", e)


def evaluate_ml_model() -> None:
    if not os.path.exists(PARQUET_PATH):
        print(f"[ML Evaluation] Error: Processed file does not exist at {PARQUET_PATH}.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    if "target" not in df.columns:
        df["target"] = _define_target(df)

    print(f"[ML Evaluation] Total positive targets: {df['target'].sum()} out of {len(df)}")
    _print_ranking_metrics(df)

    active_model = "sklearn"
    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)
        active_model = meta.get("active_model", "sklearn")
        _print_training_validation(meta)

    _print_gain_based_feature_importance(active_model)


if __name__ == "__main__":
    evaluate_ml_model()
