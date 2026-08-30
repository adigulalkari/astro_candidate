import os
import urllib.request

def download_nasa_data():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    target_path = os.path.join(raw_dir, "nasa_exoplanets.csv")

    if os.path.exists(target_path):
        print(f"[NASA Ingestion] Raw exoplanet data already exists at {target_path}")
        return

    print("[NASA Ingestion] Querying NASA Exoplanet Archive TAP API...")
    columns = [
        "pl_name", "hostname", "sy_snum", "sy_pnum", "discoverymethod", "disc_year",
        "pl_orbper", "pl_rade", "pl_radj", "pl_bmasse", "pl_bmassj", "pl_eqt", "pl_insol",
        "pl_orbeccen", "pl_orbincl", "tran_flag", "rv_flag", "ima_flag", "pl_controv_flag",
        "pl_pubdate", "st_teff", "st_rad", "st_mass", "st_lum", "st_met", "st_logg", "st_age",
        "sy_dist", "sy_gaiamag", "sy_vmag", "sy_kmag"
    ]
    query = f"select {','.join(columns)} from pscomppars"
    # URL encode query spaces to +
    encoded_query = query.replace(" ", "+")
    url = f"https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query={encoded_query}&format=csv"

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            with open(target_path, "wb") as f:
                f.write(data)
        print(f"[NASA Ingestion] Successfully downloaded raw exoplanet data to {target_path} (size: {len(data) / 1024:.2f} KB)")
    except Exception as e:
        print(f"[NASA Ingestion] Error downloading data: {e}")
        raise

if __name__ == "__main__":
    download_nasa_data()
