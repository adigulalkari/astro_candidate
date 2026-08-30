import os
import sqlite3
import json
import joblib
import pandas as pd
import numpy as np

def rank_planets(
    planet_names: list = None,
    custom_planet_features: dict = None
) -> list:
    """
    Ranks exoplanets using the trained machine learning model.
    Can rank existing planets by name or perform inference on a custom hypothetical planet.
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    db_path = os.path.join(project_dir, "data/processed/planets.db")
    model_dir = os.path.join(project_dir, "models")
    
    # 1. Handle Custom Planet Inference
    if custom_planet_features:
        try:
            # Reconstruct the feature list in the exact order required by the model
            feature_cols = [
                "pl_rade", "pl_bmasse", "pl_orbper", "pl_eqt", "pl_insol", "pl_orbeccen",
                "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "sy_dist", "sy_vmag", "sy_kmag",
                "radius_ratio", "temp_similarity", "insol_similarity", "radius_similarity", "distance_score",
                "stellar_temperature_normalized", "planet_star_radius_ratio"
            ]

            # Parse input and calculate derived features
            inputs = {col: custom_planet_features.get(col, np.nan) for col in feature_cols}
            
            # Fill derived features if not provided
            pl_rade = inputs["pl_rade"] = custom_planet_features.get("pl_rade", 1.0)
            pl_eqt = inputs["pl_eqt"] = custom_planet_features.get("pl_eqt", 288.0)
            pl_insol = inputs["pl_insol"] = custom_planet_features.get("pl_insol", 1.0)
            sy_dist = inputs["sy_dist"] = custom_planet_features.get("sy_dist", 10.0)
            st_teff = inputs["st_teff"] = custom_planet_features.get("st_teff", 5778.0)
            st_rad = inputs["st_rad"] = custom_planet_features.get("st_rad", 1.0)

            inputs["radius_ratio"] = pl_rade
            inputs["temp_similarity"] = np.exp(-np.abs(pl_eqt - 288.0) / 100.0)
            inputs["insol_similarity"] = np.exp(-np.abs(np.log(pl_insol + 1e-5)) / 2.0) if pl_insol > 0 else np.nan
            inputs["radius_similarity"] = np.exp(-np.abs(np.log(pl_rade + 1e-5)) / 0.5) if pl_rade > 0 else np.nan
            inputs["distance_score"] = np.exp(-sy_dist / 100.0) if sy_dist > 0 else np.nan
            inputs["stellar_temperature_normalized"] = st_teff / 5778.0
            inputs["planet_star_radius_ratio"] = pl_rade / (st_rad * 109.2 + 1e-5)

            # Check model type
            meta_path = os.path.join(model_dir, "model_meta.json")
            active_model = "sklearn"
            if os.path.exists(meta_path):
                with open(meta_path, "r") as f:
                    active_model = json.load(f).get("active_model", "sklearn")

            # Convert to DataFrame
            df_inf = pd.DataFrame([inputs])

            if active_model == "xgboost":
                import xgboost as xgb
                xgb_path = os.path.join(model_dir, "exoplanet_ranker.json")
                model = xgb.XGBClassifier()
                model.load_model(xgb_path)
            else:
                skl_path = os.path.join(model_dir, "exoplanet_ranker.joblib")
                model = joblib.load(skl_path)
                # Sklearn needs NaNs filled (use standard medians or defaults)
                # Since this is a single row, fill any remaining NaNs with 0.0
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
