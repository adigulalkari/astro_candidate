import os
import urllib.parse
import urllib.request


def _fetch(query, target_path, label):
    if os.path.exists(target_path):
        print(f"[Candidate Ingestion] {label} data already exists at {target_path}")
        return

    print(f"[Candidate Ingestion] Querying NASA Exoplanet Archive TAP API for {label}...")
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=csv"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            data = response.read()
            with open(target_path, "wb") as f:
                f.write(data)
        print(f"[Candidate Ingestion] Saved {label} data to {target_path} (size: {len(data) / 1024:.2f} KB)")
    except Exception as e:
        print(f"[Candidate Ingestion] Error downloading {label} data: {e}")
        raise


def download_koi_candidates():
    """Kepler Objects of Interest (cumulative table): confirmed + still-open candidates,
    excluding known false positives. Adds thousands of transiting-planet candidates with
    the same physical parameters used for habitability scoring."""
    columns = [
        "kepoi_name", "kepler_name", "koi_disposition", "koi_period", "koi_sma", "koi_prad",
        "koi_teq", "koi_insol", "koi_eccen", "koi_steff", "koi_srad", "koi_smass", "koi_smet",
        "koi_slogg", "koi_sage", "koi_kepmag"
    ]
    query = (
        f"select {','.join(columns)} from cumulative "
        "where koi_disposition != 'FALSE POSITIVE'"
    )
    _fetch(query, os.path.join("data", "raw", "koi_candidates.csv"), "KOI cumulative")


def download_toi_candidates():
    """TESS Objects of Interest: planet candidates (PC), confirmed planets (CP) and known
    planets (KP), excluding false positives (FP) and false alarms (FA). TESS targets are
    nearby bright stars, so this table includes real stellar distances (st_dist) unlike the
    Kepler KOI table, making it especially useful for finding nearby Earth-like candidates."""
    columns = [
        "toi", "tfopwg_disp", "pl_orbper", "pl_rade", "pl_eqt", "pl_insol", "st_dist",
        "st_teff", "st_rad", "st_logg"
    ]
    query = (
        f"select {','.join(columns)} from toi "
        "where tfopwg_disp in ('PC','CP','KP')"
    )
    _fetch(query, os.path.join("data", "raw", "toi_candidates.csv"), "TESS TOI")


if __name__ == "__main__":
    download_koi_candidates()
    download_toi_candidates()
