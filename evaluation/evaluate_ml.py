import os
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import joblib

def evaluate_ml_model():
    processed_dir = "data/processed"
    parquet_path = os.path.join(processed_dir, "planets.parquet")
    model_dir = "models"
    skl_model_path = os.path.join(model_dir, "exoplanet_ranker.joblib")
    xgb_model_path = os.path.join(model_dir, "exoplanet_ranker.json")
    meta_path = os.path.join(model_dir, "model_meta.json")

    if not os.path.exists(parquet_path):
        print(f"[ML Evaluation] Error: Processed file does not exist at {parquet_path}.")
        return

    # Load data
    df = pd.read_parquet(parquet_path)
    
    # Check if target exists
    if "target" not in df.columns:
        print("[ML Evaluation] Target column not found. Re-defining target labels for evaluation...")
        radius_cond = (df["pl_rade"] >= 0.5) & (df["pl_rade"] <= 1.5)
        temp_cond = (df["pl_eqt"] >= 200.0) & (df["pl_eqt"] <= 320.0)
        insol_cond = (df["pl_insol"] >= 0.2) & (df["pl_insol"] <= 2.2)
        dist_cond = df["sy_dist"] <= 150.0
        df["target"] = 0
        df.loc[radius_cond & temp_cond & insol_cond & dist_cond, "target"] = 1

    print(f"[ML Evaluation] Evaluating model ranking performance (Total test targets: {df['target'].sum()})...")

    # Sort by ml_score descending
    df_sorted = df.sort_values(by="ml_score", ascending=False).reset_index(drop=True)

    # Compute ranking metrics
    for K in [5, 10]:
        top_k = df_sorted.head(K)
        hits = top_k["target"].sum()
        precision = hits / K
        recall = hits / df["target"].sum() if df["target"].sum() > 0 else 0
        
        # Calculate NDCG@K
        # y_true needs to be shape (1, n_samples) and y_score needs to be shape (1, n_samples)
        y_true = np.array([df["target"].values])
        y_score = np.array([df["ml_score"].values])
        ndcg_val = ndcg_score(y_true, y_score, k=K)

        print(f"Metrics @ K={K}:")
        print(f"  Precision@{K}: {precision:.4f} (Hits: {hits})")
        print(f"  Recall@{K}: {recall:.4f}")
        print(f"  NDCG@{K}: {ndcg_val:.4f}")

    # Top 10 planets ranked by model
    print("\nTop 10 Ranked Planet Candidates by ML Model:")
    cols_show = ["pl_name", "sy_dist", "pl_rade", "pl_eqt", "pl_insol", "ml_score", "target"]
    print(df_sorted[cols_show].head(10).to_string(index=False))

    # Feature Importance
    feature_cols = [
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol", "pl_orbeccen",
        "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "sy_dist", "sy_vmag", "sy_kmag",
        "radius_ratio", "temp_similarity", "insol_similarity", "radius_similarity", "distance_score",
        "stellar_temperature_normalized", "planet_star_radius_ratio"
    ]

    active_model = "sklearn"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            active_model = json.load(f).get("active_model", "sklearn")

    print(f"\nModel Explainability - Top Feature Importances ({active_model}):")
    try:
        if active_model == "xgboost" and os.path.exists(xgb_model_path):
            import xgboost as xgb
            model = xgb.XGBClassifier()
            model.load_model(xgb_model_path)
            importances = model.feature_importances_
        elif os.path.exists(skl_model_path):
            model = joblib.load(skl_model_path)
            importances = model.feature_importances_
        else:
            print("Model files not found.")
            return

        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        print(feat_imp.head(10).to_string())
    except Exception as e:
        print("Could not compute feature importances:", e)

if __name__ == "__main__":
    evaluate_ml_model()
