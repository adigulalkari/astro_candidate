"""Builds the processed exoplanet dataset used for ML training and the app's API.

Combines three NASA Exoplanet Archive catalogs into a single unified table so the
habitability model has far more than the ~6k confirmed planets to learn from:

  * pscomppars  - confirmed planets with fully vetted parameters (most reliable).
  * cumulative  - Kepler Objects of Interest, including still-open candidates.
  * toi         - TESS Objects of Interest, including still-open candidates.

Candidate catalogs (KOI/TOI) use different column names and lack some fields
(e.g. planet mass, stellar distance for KOI). Rows are mapped onto the same
schema as the confirmed-planet table; anything not measured for a given
candidate is left as NaN, which the downstream XGBoost model handles natively.

Output: data/processed/planets.parquet and data/processed/planets.db.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

from feature_engineering import FEATURE_COLUMNS, engineer_features

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
PARQUET_PATH = os.path.join(PROCESSED_DIR, "planets.parquet")
DB_PATH = os.path.join(PROCESSED_DIR, "planets.db")

# Columns shared by every row in the unified table, in DB insertion order.
SCHEMA_COLUMNS = [
    "pl_name", "hostname", "sy_snum", "sy_pnum", "discoverymethod", "disc_year",
    "pl_orbper", "pl_orbsmax", "pl_rade", "pl_radj", "pl_bmasse", "pl_bmassj", "pl_dens",
    "pl_eqt", "pl_insol", "pl_orbeccen", "pl_orbincl", "tran_flag", "rv_flag", "ima_flag",
    "pl_controv_flag", "pl_pubdate", "st_teff", "st_rad", "st_mass", "st_lum", "st_met",
    "st_logg", "st_age", "sy_dist", "sy_gaiamag", "sy_vmag", "sy_kmag",
]

# Engineered columns are those in FEATURE_COLUMNS that aren't already raw archive
# columns (i.e. everything engineer_features() adds).
ENGINEERED_COLUMNS = [c for c in FEATURE_COLUMNS if c not in SCHEMA_COLUMNS]


def _load_confirmed_planets() -> pd.DataFrame:
    """Loads the confirmed-planet composite-parameters catalog."""
    path = os.path.join(RAW_DIR, "nasa_exoplanets.csv")
    df = pd.read_csv(path, comment="#")
    df = df.drop_duplicates(subset=["pl_name"])
    df["data_source"] = "confirmed"
    return df.reindex(columns=SCHEMA_COLUMNS + ["data_source"])


def _load_koi_candidates() -> pd.DataFrame:
    """Loads Kepler Objects of Interest and maps them onto the unified schema.

    Confirmed KOIs are already present via the confirmed-planet catalog, so this
    mainly contributes still-open CANDIDATE rows plus their physical parameters.
    """
    path = os.path.join(RAW_DIR, "koi_candidates.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=SCHEMA_COLUMNS + ["data_source"])

    raw = pd.read_csv(path, comment="#")
    df = pd.DataFrame(index=raw.index)
    df["pl_name"] = raw["kepler_name"].fillna(raw["kepoi_name"])
    df["hostname"] = raw["kepoi_name"].str.split(".").str[0]
    df["sy_snum"] = np.nan
    df["sy_pnum"] = np.nan
    df["discoverymethod"] = "Transit"
    df["disc_year"] = np.nan
    df["pl_orbper"] = raw["koi_period"]
    df["pl_orbsmax"] = raw["koi_sma"]
    df["pl_rade"] = raw["koi_prad"]
    df["pl_radj"] = np.nan
    df["pl_bmasse"] = np.nan  # transit-only detections have no mass measurement
    df["pl_bmassj"] = np.nan
    df["pl_dens"] = np.nan
    df["pl_eqt"] = raw["koi_teq"]
    df["pl_insol"] = raw["koi_insol"]
    df["pl_orbeccen"] = raw["koi_eccen"]
    df["pl_orbincl"] = np.nan
    df["tran_flag"] = 1
    df["rv_flag"] = 0
    df["ima_flag"] = 0
    df["pl_controv_flag"] = 0
    df["pl_pubdate"] = None
    df["st_teff"] = raw["koi_steff"]
    df["st_rad"] = raw["koi_srad"]
    df["st_mass"] = raw["koi_smass"]
    df["st_lum"] = np.nan
    df["st_met"] = raw["koi_smet"]
    df["st_logg"] = raw["koi_slogg"]
    df["st_age"] = raw["koi_sage"]
    df["sy_dist"] = np.nan  # not available in the Kepler cumulative table
    df["sy_gaiamag"] = np.nan
    df["sy_vmag"] = np.nan
    df["sy_kmag"] = raw["koi_kepmag"]
    df["data_source"] = "koi_" + raw["koi_disposition"].str.lower().str.replace(" ", "_")
    return df.drop_duplicates(subset=["pl_name"])


def _load_toi_candidates() -> pd.DataFrame:
    """Loads TESS Objects of Interest and maps them onto the unified schema.

    Unlike KOI, TESS targets are nearby bright stars and this table reports a
    real stellar distance (st_dist), so it is the main source of new, plausibly
    close Earth-like candidates for the habitability model.
    """
    path = os.path.join(RAW_DIR, "toi_candidates.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=SCHEMA_COLUMNS + ["data_source"])

    raw = pd.read_csv(path, comment="#")
    df = pd.DataFrame(index=raw.index)
    df["pl_name"] = "TOI-" + raw["toi"].astype(str)
    df["hostname"] = "TOI-" + raw["toi"].astype(str).str.split(".").str[0]
    df["sy_snum"] = np.nan
    df["sy_pnum"] = np.nan
    df["discoverymethod"] = "Transit"
    df["disc_year"] = np.nan
    df["pl_orbper"] = raw["pl_orbper"]
    df["pl_orbsmax"] = np.nan
    df["pl_rade"] = raw["pl_rade"]
    df["pl_radj"] = np.nan
    df["pl_bmasse"] = np.nan
    df["pl_bmassj"] = np.nan
    df["pl_dens"] = np.nan
    df["pl_eqt"] = raw["pl_eqt"]
    df["pl_insol"] = raw["pl_insol"]
    df["pl_orbeccen"] = np.nan
    df["pl_orbincl"] = np.nan
    df["tran_flag"] = 1
    df["rv_flag"] = 0
    df["ima_flag"] = 0
    df["pl_controv_flag"] = 0
    df["pl_pubdate"] = None
    df["st_teff"] = raw["st_teff"]
    df["st_rad"] = raw["st_rad"]
    df["st_mass"] = np.nan
    df["st_lum"] = np.nan
    df["st_met"] = np.nan
    df["st_logg"] = raw["st_logg"]
    df["st_age"] = np.nan
    df["sy_dist"] = raw["st_dist"]
    df["sy_gaiamag"] = np.nan
    df["sy_vmag"] = np.nan
    df["sy_kmag"] = np.nan
    df["data_source"] = "toi_" + raw["tfopwg_disp"].str.lower()
    return df.drop_duplicates(subset=["pl_name"])


def _merge_catalogs() -> pd.DataFrame:
    """Concatenates the confirmed-planet and candidate catalogs into one table.

    A planet already present in the confirmed catalog is kept as "confirmed"
    (its parameters are the most reliable); candidate rows are only added for
    names not already covered, so the same object is never counted twice.
    """
    confirmed = _load_confirmed_planets()
    koi = _load_koi_candidates()
    toi = _load_toi_candidates()

    known_names = set(confirmed["pl_name"])
    koi = koi[~koi["pl_name"].isin(known_names)]
    known_names |= set(koi["pl_name"])
    toi = toi[~toi["pl_name"].isin(known_names)]

    combined = pd.concat([confirmed, koi, toi], ignore_index=True)
    combined = combined.drop_duplicates(subset=["pl_name"])
    print(
        f"[Data Cleaning] Combined catalog: {len(confirmed)} confirmed + "
        f"{len(koi)} KOI candidates + {len(toi)} TOI candidates = {len(combined)} rows."
    )
    return combined


def _write_parquet(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"[Data Cleaning] Saved cleaned dataset to {PARQUET_PATH}")


def _write_sqlite(df: pd.DataFrame) -> None:
    print(f"[Data Cleaning] Storing data in SQLite database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS planets")
        cursor.execute(f"""
        CREATE TABLE planets (
            pl_name TEXT PRIMARY KEY,
            hostname TEXT,
            sy_snum INTEGER,
            sy_pnum INTEGER,
            discoverymethod TEXT,
            disc_year INTEGER,
            pl_orbper REAL,
            pl_orbsmax REAL,
            pl_rade REAL,
            pl_radj REAL,
            pl_bmasse REAL,
            pl_bmassj REAL,
            pl_dens REAL,
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
            density_relative REAL,
            escape_velocity_relative REAL,
            earth_similarity_index REAL,
            hz_distance_au REAL,
            habitable_zone_position REAL,
            data_source TEXT,
            ml_score REAL
        )
        """)
        cursor.execute("CREATE INDEX idx_distance ON planets(sy_dist)")
        cursor.execute("CREATE INDEX idx_radius ON planets(pl_rade)")
        cursor.execute("CREATE INDEX idx_host ON planets(hostname)")
        cursor.execute("CREATE INDEX idx_ml_score ON planets(ml_score)")

        insert_columns = SCHEMA_COLUMNS + ENGINEERED_COLUMNS + ["data_source", "ml_score"]
        df_db = df[insert_columns].replace({np.nan: None})
        placeholders = ",".join(["?"] * len(insert_columns))
        cursor.executemany(
            f"INSERT OR REPLACE INTO planets ({','.join(insert_columns)}) VALUES ({placeholders})",
            df_db.values.tolist(),
        )
        conn.commit()
        print(f"[Data Cleaning] SQLite database initialized. Inserted {len(df_db)} records.")
    finally:
        conn.close()


def clean_data() -> None:
    confirmed_path = os.path.join(RAW_DIR, "nasa_exoplanets.csv")
    if not os.path.exists(confirmed_path):
        print(f"[Data Cleaning] Error: Raw file does not exist at {confirmed_path}")
        return

    df = _merge_catalogs()
    df = engineer_features(df)
    df["ml_score"] = 0.0  # populated by train_model.py
    _write_parquet(df)
    _write_sqlite(df)


if __name__ == "__main__":
    clean_data()
