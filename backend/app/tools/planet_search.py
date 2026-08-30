import os
import sqlite3

def search_planets(
    query_str: str = None,
    max_distance: float = None,
    min_radius: float = None,
    max_radius: float = None,
    min_temp: float = None,
    max_temp: float = None,
    discovery_method: str = None,
    limit: int = 20
) -> list:
    """
    Search and filter planets in the SQLite database.
    """
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "data/processed/planets.db"
    )
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    query = "SELECT * FROM planets WHERE 1=1"
    params = []

    if query_str:
        query += " AND (pl_name LIKE ? OR hostname LIKE ?)"
        params.extend([f"%{query_str}%", f"%{query_str}%"])

    if max_distance is not None:
        query += " AND sy_dist <= ?"
        params.append(max_distance)

    if min_radius is not None:
        query += " AND pl_rade >= ?"
        params.append(min_radius)

    if max_radius is not None:
        query += " AND pl_rade <= ?"
        params.append(max_radius)

    if min_temp is not None:
        query += " AND pl_eqt >= ?"
        params.append(min_temp)

    if max_temp is not None:
        query += " AND pl_eqt <= ?"
        params.append(max_temp)

    if discovery_method:
        query += " AND discoverymethod LIKE ?"
        params.append(f"%{discovery_method}%")

    # Order by ML score or distance by default
    query += " ORDER BY ml_score DESC, sy_dist ASC LIMIT ?"
    params.append(limit)

    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append(dict(row))
        return results
    except Exception as e:
        print(f"[Tool: Planet Search] Error: {e}")
        return []
    finally:
        conn.close()
