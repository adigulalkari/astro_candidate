import os
import sqlite3
import sys
import json
import joblib
import pandas as pd
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(PROJECT_DIR, "data_pipeline"))
from feature_engineering import FEATURE_COLUMNS, engineer_features  # noqa: E402

# Defaults used when a hypothetical planet omits a primary field. These mirror
# Earth's own values so an otherwise-empty request scores as "Earth-like".
CUSTOM_PLANET_DEFAULTS = {
    "pl_rade": 1.0,
    "pl_eqt": 288.0,
    "pl_insol": 1.0,
    "sy_dist": 10.0,
    "st_teff": 5778.0,
    "st_rad": 1.0,
}


def rank_planets(
    planet_names: list = None,
    custom_planet_features: dict = None
) -> list:
    """
    Ranks exoplanets using the trained machine learning model.
    Can rank existing planets by name or perform inference on a custom hypothetical planet.
    """
    db_path = os.path.join(PROJECT_DIR, "data/processed/planets.db")
    model_dir = os.path.join(PROJECT_DIR, "models")

    # 1. Handle Custom Planet Inference
    if custom_planet_features:
        try:
            inputs = {**CUSTOM_PLANET_DEFAULTS, **custom_planet_features}
            df_inf = pd.DataFrame([inputs])
            df_inf = engineer_features(df_inf)
            df_inf = df_inf.reindex(columns=FEATURE_COLUMNS)

            meta_path = os.path.join(model_dir, "model_meta.json")
            active_model = "sklearn"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    active_model = json.load(f).get("active_model", "sklearn")

            if active_model == "xgboost":
                import xgboost as xgb
                xgb_path = os.path.join(model_dir, "exoplanet_ranker.json")
                model = xgb.XGBClassifier()
                model.load_model(xgb_path)
            else:
                skl_path = os.path.join(model_dir, "exoplanet_ranker.joblib")
                model = joblib.load(skl_path)
                # Sklearn needs NaNs filled; this is a single row, so 0.0 is a safe default.
                df_inf = df_inf.fillna(0.0)

            score = float(model.predict_proba(df_inf)[:, 1][0])
            return [{
                "planet": "Hypothetical Candidate",
                "ml_score": score,
                "input_features": custom_planet_features
            }]
        except Exception as e:
            print(f"[Tool: Rank Planets] Inference error: {e}")
            return []

    # 2. Handle Existing Planets Ranking
    if not planet_names:
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # Fetch precomputed ML scores
        placeholders = ",".join(["?"] * len(planet_names))
        query = f"SELECT pl_name, hostname, sy_dist, pl_rade, pl_eqt, pl_insol, ml_score FROM planets WHERE pl_name IN ({placeholders})"
        cursor.execute(query, planet_names)
        rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        # Sort by score descending
        results = sorted(results, key=lambda x: x["ml_score"], reverse=True)
        return results
    except Exception as e:
        print(f"[Tool: Rank Planets] Query error: {e}")
        return []
    finally:
        conn.close()
