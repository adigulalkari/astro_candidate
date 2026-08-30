import os
import json
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import joblib

# Attempt to import and load XGBoost, falling back to scikit-learn if OpenMP is missing
USE_XGB = False
try:
    import xgboost as xgb
    # Test if we can initialize a model (checks if shared libraries load successfully)
    test_model = xgb.XGBClassifier()
    USE_XGB = True
    print("[ML Training] Successfully loaded XGBoost. Will use XGBoost for candidate ranking.")
except Exception as e:
    print(f"[ML Training] XGBoost load failed ({e}). Falling back to scikit-learn GradientBoostingClassifier.")
    from sklearn.ensemble import GradientBoostingClassifier

def train_exoplanet_ranker():
    processed_dir = "data/processed"
    parquet_path = os.path.join(processed_dir, "planets.parquet")
    db_path = os.path.join(processed_dir, "planets.db")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    xgb_model_path = os.path.join(model_dir, "exoplanet_ranker.json")
    skl_model_path = os.path.join(model_dir, "exoplanet_ranker.joblib")
    meta_path = os.path.join(model_dir, "model_meta.json")

    if not os.path.exists(parquet_path):
        print(f"[ML Training] Error: Processed file does not exist at {parquet_path}. Run clean_data.py first.")
        return

    # Load clean data
    df = pd.read_parquet(parquet_path)
    print(f"[ML Training] Loaded {len(df)} planets from parquet.")

    # 1. Define Target Labels (Earth-like reference profile)
    # Physical constraints for a temperate Earth-sized planet around M/K/G stars
    radius_cond = (df["pl_rade"] >= 0.5) & (df["pl_rade"] <= 1.5)
    temp_cond = (df["pl_eqt"] >= 200.0) & (df["pl_eqt"] <= 320.0)
    insol_cond = (df["pl_insol"] >= 0.2) & (df["pl_insol"] <= 2.2)
    dist_cond = df["sy_dist"] <= 150.0  # limit distance for follow-up feasibility

    # Combine conditions to define the target reference set
    df["target"] = 0
    df.loc[radius_cond & temp_cond & insol_cond & dist_cond, "target"] = 1
    num_positives = df["target"].sum()
    print(f"[ML Training] Defined target label. Positive instances: {num_positives} out of {len(df)}")

    # 2. Select Features
    feature_cols = [
        "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol", "pl_orbeccen",
        "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "sy_dist", "sy_vmag", "sy_kmag",
        "radius_ratio", "temp_similarity", "insol_similarity", "radius_similarity", "distance_score",
        "stellar_temperature_normalized", "planet_star_radius_ratio"
    ]

    # Pre-process features: Scikit-learn cannot handle NaNs out-of-the-box like XGBoost.
    # We will use a simple median imputation for scikit-learn fallback (XGBoost can keep NaNs).
    X = df[feature_cols].copy()
    y = df["target"]

    if not USE_XGB:
        # Fill missing values with median for train/test split in sklearn
        for col in feature_cols:
            if X[col].isnull().any():
                X[col] = X[col].fillna(X[col].median() if not pd.isna(X[col].median()) else 0.0)

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4. Train Model
    if USE_XGB:
        scale_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-5)
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        )
        print("[ML Training] Training XGBoost model...")
        model.fit(X_train, y_train)
        
        # Save XGBoost
        model.save_model(xgb_model_path)
        print(f"[ML Training] Saved XGBoost model to {xgb_model_path}")
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        print("[ML Training] Training Scikit-learn GradientBoosting model...")
        model.fit(X_train, y_train)
        
        # Save Scikit-learn model
        joblib.dump(model, skl_model_path)
        print(f"[ML Training] Saved Scikit-learn model to {skl_model_path}")

    # Save metadata indicating which model is active
    with open(meta_path, "w") as f:
        json.dump({"active_model": "xgboost" if USE_XGB else "sklearn"}, f)

    # 5. Evaluate Model
    y_pred_train = model.predict_proba(X_train)[:, 1]
    y_pred_test = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    
    y_pred_binary = (y_pred_test >= 0.5).astype(int)
    test_precision = precision_score(y_test, y_pred_binary, zero_division=0)
    test_recall = recall_score(y_test, y_pred_binary, zero_division=0)

    print(f"[ML Training] Train ROC-AUC: {train_auc:.4f}")
    print(f"[ML Training] Test ROC-AUC: {test_auc:.4f}")
    print(f"[ML Training] Test Precision: {test_precision:.4f}")
    print(f"[ML Training] Test Recall: {test_recall:.4f}")

    # 6. Predict ML Scores for the entire dataset
    df["ml_score"] = model.predict_proba(X)[:, 1]

    # Save updated dataframe to parquet
    df.to_parquet(parquet_path, index=False)
    print(f"[ML Training] Updated ml_score in Parquet: {parquet_path}")

    # 7. Update SQLite Database with predicted ml_scores
    print(f"[ML Training] Updating SQLite database {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Update the ml_score column
    update_data = df[["ml_score", "pl_name"]].values.tolist()
    cursor.executemany("UPDATE planets SET ml_score = ? WHERE pl_name = ?", update_data)
    conn.commit()
    conn.close()
    print("[ML Training] SQLite database updated successfully.")

if __name__ == "__main__":
    train_exoplanet_ranker()
