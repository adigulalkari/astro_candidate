"""Earth-similarity feature engineering shared by training and inference.

This is the single source of truth for how raw NASA Exoplanet Archive columns
become the model's input features. Both `clean_data.py` (batch, many rows) and
the backend's `rank_planets` tool (single hypothetical planet) call
`engineer_features` so the two paths can never drift apart — a model trained on
one feature definition and served with another (training/serving skew) would
silently produce wrong scores.
"""
import numpy as np
import pandas as pd

EARTH_DENSITY_G_CM3 = 5.51
SOLAR_TEMPERATURE_K = 5778.0
SOLAR_RADII_PER_EARTH_RADII = 109.2
EARTH_EQUILIBRIUM_TEMP_K = 288.0

# Columns the ranking model is trained and served on, in a stable order.
FEATURE_COLUMNS = [
    "pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "pl_dens", "pl_eqt", "pl_insol",
    "pl_orbeccen", "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "sy_dist", "sy_vmag",
    "sy_kmag", "radius_ratio", "temp_similarity", "insol_similarity", "radius_similarity",
    "distance_score", "stellar_temperature_normalized", "planet_star_radius_ratio",
    "density_relative", "escape_velocity_relative", "earth_similarity_index",
    "hz_distance_au", "habitable_zone_position",
]

# Raw columns engineer_features() reads; anything not present in the input frame
# is treated as entirely missing (NaN) for that feature.
REQUIRED_RAW_COLUMNS = [
    "pl_rade", "pl_bmasse", "pl_orbper", "pl_orbsmax", "pl_dens", "pl_eqt", "pl_insol",
    "st_teff", "st_rad", "st_lum", "sy_dist",
]


def _esi_component(value: pd.Series, earth_reference: float, weight: float) -> pd.Series:
    """One factor of the Earth Similarity Index (Schulze-Makuch et al. 2011)."""
    ratio = np.abs((value - earth_reference) / (value + earth_reference))
    return (1.0 - ratio) ** (weight / 4.0)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds every column in FEATURE_COLUMNS that isn't already a raw archive column.

    Mutates and returns `df`. Missing raw inputs propagate to NaN outputs, which
    XGBoost handles natively at both train and inference time.
    """
    for col in REQUIRED_RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df["radius_ratio"] = df["pl_rade"]
    df["temp_similarity"] = np.exp(-np.abs(df["pl_eqt"] - EARTH_EQUILIBRIUM_TEMP_K) / 100.0)

    valid_insol = df["pl_insol"] > 0
    df["insol_similarity"] = np.nan
    df.loc[valid_insol, "insol_similarity"] = np.exp(
        -np.abs(np.log(df.loc[valid_insol, "pl_insol"])) / 2.0
    )

    valid_rade = df["pl_rade"] > 0
    df["radius_similarity"] = np.nan
    df.loc[valid_rade, "radius_similarity"] = np.exp(
        -np.abs(np.log(df.loc[valid_rade, "pl_rade"])) / 0.5
    )

    valid_dist = df["sy_dist"] > 0
    df["distance_score"] = np.nan
    df.loc[valid_dist, "distance_score"] = np.exp(-df.loc[valid_dist, "sy_dist"] / 100.0)

    df["stellar_temperature_normalized"] = df["st_teff"] / SOLAR_TEMPERATURE_K
    df["planet_star_radius_ratio"] = df["pl_rade"] / (
        df["st_rad"] * SOLAR_RADII_PER_EARTH_RADII + 1e-5
    )

    # Bulk density relative to Earth; fall back to mass/radius^3 when pl_dens is missing.
    df["density_relative"] = df["pl_dens"] / EARTH_DENSITY_G_CM3
    derived_density = df["pl_bmasse"] / (df["pl_rade"] ** 3).replace(0, np.nan)
    df["density_relative"] = df["density_relative"].fillna(derived_density)

    # Escape velocity relative to Earth: v_esc is proportional to sqrt(mass / radius).
    valid_mr = (df["pl_bmasse"] > 0) & (df["pl_rade"] > 0)
    df["escape_velocity_relative"] = np.nan
    df.loc[valid_mr, "escape_velocity_relative"] = np.sqrt(
        df.loc[valid_mr, "pl_bmasse"] / df.loc[valid_mr, "pl_rade"]
    )

    esi_radius = _esi_component(df["pl_rade"].clip(lower=1e-3), 1.0, 0.57)
    esi_density = _esi_component(df["density_relative"].clip(lower=1e-3), 1.0, 1.07)
    esi_escape_velocity = _esi_component(df["escape_velocity_relative"].clip(lower=1e-3), 1.0, 0.70)
    esi_temperature = _esi_component(df["pl_eqt"].clip(lower=1e-3), EARTH_EQUILIBRIUM_TEMP_K, 5.58)
    df["earth_similarity_index"] = (
        esi_radius * esi_density * esi_escape_velocity * esi_temperature
    ).clip(0, 1)

    # Habitable-zone position: orbital distance relative to the estimated habitable-zone
    # midpoint, which scales as sqrt(stellar luminosity). st_lum is log10(L / L_sun).
    valid_hz = df["st_lum"].notna() & (df["pl_orbsmax"] > 0)
    df["hz_distance_au"] = np.nan
    df.loc[valid_hz, "hz_distance_au"] = np.sqrt(10 ** df.loc[valid_hz, "st_lum"])
    valid_hz_position = valid_hz & (df["hz_distance_au"] > 0)
    df["habitable_zone_position"] = np.nan
    df.loc[valid_hz_position, "habitable_zone_position"] = (
        df.loc[valid_hz_position, "pl_orbsmax"] / df.loc[valid_hz_position, "hz_distance_au"]
    )

    return df
