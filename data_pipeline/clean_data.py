import os
import sqlite3
import pandas as pd
import numpy as np

def clean_data():
    raw_path = "data/raw/nasa_exoplanets.csv"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    parquet_path = os.path.join(processed_dir, "planets.parquet")
    db_path = os.path.join(processed_dir, "planets.db")

    if not os.path.exists(raw_path):
        print(f"[Data Cleaning] Error: Raw file does not exist at {raw_path}")
        return

    print(f"[Data Cleaning] Loading raw data from {raw_path}...")
    # Read the NASA CSV (skipping comments starting with #)
    df = pd.read_csv(raw_path, comment="#")
    print(f"[Data Cleaning] Loaded {len(df)} rows.")

    # Remove duplicates of the same planet (pl_name should be unique)
    # The TAP query should have returned consolidated params, but double check.
    df = df.drop_duplicates(subset=["pl_name"])
    print(f"[Data Cleaning] Kept {len(df)} unique planets after removing duplicates.")

    # Calculate derived astronomical features
    # 1. Earth radius ratio
    df["radius_ratio"] = df["pl_rade"]

    # 2. Temperature similarity (scale = 100)
    # Earth eq temp is ~255-288 K. Let us use 288 K.
    df["temp_similarity"] = np.exp(-np.abs(df["pl_eqt"] - 288.0) / 100.0)

    # 3. Insolation similarity (scale = 2.0)
    # Earth insol is 1.0.
    # We must guard against <= 0 or missing values for log.
    valid_insol = (df["pl_insol"] > 0)
    df["insol_similarity"] = np.nan
    df.loc[valid_insol, "insol_similarity"] = np.exp(-np.abs(np.log(df.loc[valid_insol, "pl_insol"])) / 2.0)

    # 4. Radius similarity (scale = 0.5)
    # Earth radius is 1.0 pl_rade.
    valid_rade = (df["pl_rade"] > 0)
    df["radius_similarity"] = np.nan
    df.loc[valid_rade, "radius_similarity"] = np.exp(-np.abs(np.log(df.loc[valid_rade, "pl_rade"])) / 0.5)

    # 5. Distance score (scale = 100.0)
    valid_dist = (df["sy_dist"] > 0)
    df["distance_score"] = np.nan
    df.loc[valid_dist, "distance_score"] = np.exp(-df.loc[valid_dist, "sy_dist"] / 100.0)

    # Additional ML derived features
    df["stellar_temperature_normalized"] = df["st_teff"] / 5778.0 # Solar temperature
    df["planet_star_radius_ratio"] = df["pl_rade"] / (df["st_rad"] * 109.2 + 1e-5)

    # Initialize ML score column (to be updated after training)
    df["ml_score"] = 0.0

    # Save to Parquet
    df.to_parquet(parquet_path, index=False)
    print(f"[Data Cleaning] Saved cleaned dataset to {parquet_path}")

    # Build SQLite DB
    print(f"[Data Cleaning] Storing data in SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Drop table if exists
    cursor.execute("DROP TABLE IF EXISTS planets")

    # Create table
    cursor.execute("""
    CREATE TABLE planets (
        pl_name TEXT PRIMARY KEY,
        hostname TEXT,
        sy_snum INTEGER,
        sy_pnum INTEGER,
        discoverymethod TEXT,
        disc_year INTEGER,
        pl_orbper REAL,
        pl_rade REAL,
        pl_radj REAL,
        pl_bmasse REAL,
        pl_bmassj REAL,
        pl_eqt REAL,
        pl_insol REAL,
        pl_orbeccen REAL,
        pl_orbincl REAL,
        tran_flag INTEGER,
        rv_flag INTEGER,
        ima_flag INTEGER,
        pl_controv_flag INTEGER,
        pl_pubdate TEXT,
        st_teff REAL,
        st_rad REAL,
        st_mass REAL,
        st_lum REAL,
        st_met REAL,
        st_logg REAL,
        st_age REAL,
        sy_dist REAL,
        sy_gaiamag REAL,
        sy_vmag REAL,
        sy_kmag REAL,
        radius_ratio REAL,
        temp_similarity REAL,
        insol_similarity REAL,
        radius_similarity REAL,
        distance_score REAL,
        stellar_temperature_normalized REAL,
        planet_star_radius_ratio REAL,
        ml_score REAL
    )
    """)

    # Create indices
    cursor.execute("CREATE INDEX idx_distance ON planets(sy_dist)")
    cursor.execute("CREATE INDEX idx_radius ON planets(pl_rade)")
    cursor.execute("CREATE INDEX idx_host ON planets(hostname)")
    cursor.execute("CREATE INDEX idx_ml_score ON planets(ml_score)")

    # Prepare values for insertion, replacing NaN/None with SQLite NULL
    df_db = df.copy()
    df_db = df_db.replace({np.nan: None})

    cols = [
        "pl_name", "hostname", "sy_snum", "sy_pnum", "discoverymethod", "disc_year",
        "pl_orbper", "pl_rade", "pl_radj", "pl_bmasse", "pl_bmassj", "pl_eqt", "pl_insol",
        "pl_orbeccen", "pl_orbincl", "tran_flag", "rv_flag", "ima_flag", "pl_controv_flag",
        "pl_pubdate", "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "st_logg", "st_age",
        "sy_dist", "sy_gaiamag", "sy_vmag", "sy_kmag", "radius_ratio", "temp_similarity",
        "insol_similarity", "radius_similarity", "distance_score",
        "stellar_temperature_normalized", "planet_star_radius_ratio", "ml_score"
    ]

    records = df_db[cols].values.tolist()

    placeholders = ",".join(["?"] * len(cols))
    cursor.executemany(f"INSERT OR REPLACE INTO planets ({','.join(cols)}) VALUES ({placeholders})", records)

    conn.commit()
    conn.close()
    print(f"[Data Cleaning] SQLite database initialized. Inserted {len(records)} records.")

if __name__ == "__main__":
    clean_data()
