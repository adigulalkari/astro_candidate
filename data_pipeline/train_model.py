"""Trains the exoplanet habitability ranking model.

The label is a domain-defined "Earth-like reference profile" (radius, equilibrium
temperature, insolation and distance close to Earth's), not a human-annotated
class, so there is no ambiguity about ground truth. The hard part is that
positive examples are rare (a few dozen out of ~14k rows after data_pipeline's
KOI/TOI catalog merge), so this script leans on stratified cross-validation and
a randomized hyperparameter search scored on average precision (better suited
to rare-positive problems than plain accuracy or ROC-AUC) instead of a single
train/test split, which would be too noisy to trust with so few positives.
"""
import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from feature_engineering import FEATURE_COLUMNS

PROCESSED_DIR = "data/processed"
PARQUET_PATH = os.path.join(PROCESSED_DIR, "planets.parquet")
DB_PATH = os.path.join(PROCESSED_DIR, "planets.db")
MODEL_DIR = "models"
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "exoplanet_ranker.json")
SKL_MODEL_PATH = os.path.join(MODEL_DIR, "exoplanet_ranker.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# Physical constraints for a temperate, Earth-sized planet around an M/K/G star.
# Distance is treated as "pass" when unknown (candidate catalogs like KOI don't
# report it) rather than excluding the row, since intrinsic habitability doesn't
# depend on how far away we happen to be from the planet.
RADIUS_RANGE_EARTH = (0.5, 1.5)
EQUILIBRIUM_TEMP_RANGE_K = (200.0, 320.0)
INSOLATION_RANGE_EARTH = (0.2, 2.2)
MAX_FOLLOWUP_DISTANCE_PC = 150.0

RANDOM_STATE = 42
CV_FOLDS = 5
SEARCH_ITERATIONS = 40

try:
    import xgboost as xgb

    xgb.XGBClassifier()  # confirms the shared library actually loads
    USE_XGB = True
    print("[ML Training] Successfully loaded XGBoost. Will use XGBoost for candidate ranking.")
except Exception as e:
    USE_XGB = False
    print(f"[ML Training] XGBoost load failed ({e}). Falling back to scikit-learn GradientBoostingClassifier.")
    from sklearn.ensemble import GradientBoostingClassifier


def _define_target(df: pd.DataFrame) -> pd.Series:
    radius_ok = df["pl_rade"].between(*RADIUS_RANGE_EARTH)
    temp_ok = df["pl_eqt"].between(*EQUILIBRIUM_TEMP_RANGE_K)
    insol_ok = df["pl_insol"].between(*INSOLATION_RANGE_EARTH)
    distance_ok = df["sy_dist"].isna() | (df["sy_dist"] <= MAX_FOLLOWUP_DISTANCE_PC)
    return (radius_ok & temp_ok & insol_ok & distance_ok).astype(int)


def _build_xgb_search(scale_pos_weight: float) -> RandomizedSearchCV:
    base_model = xgb.XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
    )
    param_distributions = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 7),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 6),
        "gamma": uniform(0.0, 0.5),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _build_sklearn_search() -> RandomizedSearchCV:
    base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    param_distributions = {
        "n_estimators": randint(100, 400),
        "max_depth": randint(3, 6),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return RandomizedSearchCV(
        base_model,
        param_distributions,
        n_iter=SEARCH_ITERATIONS,
        scoring="average_precision",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def _fit_best_model(X_train: pd.DataFrame, y_train: pd.Series):
    """Runs the cross-validated hyperparameter search and returns the best estimator."""
    if USE_XGB:
        scale_pos_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-5)
        search = _build_xgb_search(scale_pos_weight)
        search.fit(X_train, y_train)
    else:
        search = _build_sklearn_search()
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
        search.fit(X_train, y_train, sample_weight=sample_weight)

    print(f"[ML Training] Best CV average precision: {search.best_score_:.4f}")
    print(f"[ML Training] Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, search.best_score_


def _save_model(model) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    if USE_XGB:
        model.save_model(XGB_MODEL_PATH)
        print(f"[ML Training] Saved XGBoost model to {XGB_MODEL_PATH}")
    else:
        joblib.dump(model, SKL_MODEL_PATH)
        print(f"[ML Training] Saved scikit-learn model to {SKL_MODEL_PATH}")


def _evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
        "test_average_precision": average_precision_score(y_test, y_pred_proba),
        "test_precision": precision_score(y_test, y_pred_binary, zero_division=0),
        "test_recall": recall_score(y_test, y_pred_binary, zero_division=0),
    }
    for name, value in metrics.items():
        print(f"[ML Training] {name}: {value:.4f}")
    return metrics


def _score_full_dataset(model, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    df["ml_score"] = model.predict_proba(X)[:, 1]
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"[ML Training] Updated ml_score in Parquet: {PARQUET_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executemany(
            "UPDATE planets SET ml_score = ? WHERE pl_name = ?",
            df[["ml_score", "pl_name"]].values.tolist(),
        )
        conn.commit()
    finally:
        conn.close()
    print(f"[ML Training] SQLite database updated: {DB_PATH}")
    return df


def train_exoplanet_ranker() -> None:
    if not os.path.exists(PARQUET_PATH):
        print(f"[ML Training] Error: Processed file does not exist at {PARQUET_PATH}. Run clean_data.py first.")
        return

    df = pd.read_parquet(PARQUET_PATH)
    print(f"[ML Training] Loaded {len(df)} planets from parquet.")

    df["target"] = _define_target(df)
    print(f"[ML Training] Defined target label. Positive instances: {df['target'].sum()} out of {len(df)}")

    X = df[FEATURE_COLUMNS].copy()
    y = df["target"]

    if not USE_XGB:
        # scikit-learn's GradientBoostingClassifier can't handle NaNs natively.
        for col in FEATURE_COLUMNS:
            median = X[col].median()
            X[col] = X[col].fillna(median if not pd.isna(median) else 0.0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model, best_params, best_cv_score = _fit_best_model(X_train, y_train)
    _save_model(model)
    test_metrics = _evaluate(model, X_test, y_test)

    with open(META_PATH, "w") as f:
        json.dump(
            {
                "active_model": "xgboost" if USE_XGB else "sklearn",
                "cv_folds": CV_FOLDS,
                "cv_average_precision": best_cv_score,
                "best_params": best_params,
                "test_metrics": test_metrics,
                "num_positive_labels": int(y.sum()),
                "num_rows": int(len(df)),
            },
            f,
            indent=2,
        )

    # Refit on the full dataset (train + test) with the tuned hyperparameters so the
    # scores served by the API benefit from every labeled example, then score everyone.
    model.fit(X, y)
    _score_full_dataset(model, df, X)


if __name__ == "__main__":
    train_exoplanet_ranker()
