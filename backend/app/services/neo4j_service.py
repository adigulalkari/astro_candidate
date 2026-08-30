import os
import sqlite3
from neo4j import GraphDatabase
import json

class Neo4jService:
    def __init__(self):
        self.uri = os.environ.get("NEO4J_URI")
        self.user = os.environ.get("NEO4J_USERNAME")
        self.pwd = os.environ.get("NEO4J_PASSWORD")
        
        self.use_neo4j = bool(self.uri and self.user and self.pwd)
        self.driver = None

        if self.use_neo4j:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
                print("[Neo4j Service] Connected to Neo4j AuraDB successfully.")
            except Exception as e:
                print(f"[Neo4j Service] Connection to Neo4j AuraDB failed: {e}. Falling back to SQLite.")
                self.use_neo4j = False

        if not self.use_neo4j:
            self.db_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "data/processed/planets.db"
            )
            print(f"[Neo4j Service] Running in local SQLite fallback mode (DB: {self.db_path}).")

    def close(self):
        if self.driver:
            self.driver.close()

    def get_planet_subgraph(self, planet_name: str) -> dict:
        """
        Retrieves a 1-hop subgraph around a planet including the host star,
        discovery method, and papers mentioning it.
        Returns:
            {"nodes": [{"id": ..., "label": ..., "properties": ...}], "edges": [{"source": ..., "target": ..., "type": ...}]}
        """
        if self.use_neo4j:
            return self._query_neo4j_subgraph(planet_name)
        else:
            return self._query_sqlite_subgraph(planet_name)

    def _query_neo4j_subgraph(self, planet_name: str) -> dict:
        nodes = []
        edges = []
        seen_nodes = set()

        query = """
        MATCH (p:Planet {name: $name})
        OPTIONAL MATCH (p)-[r]->(t)
        RETURN p, r, t
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, name=planet_name)
                for record in result:
                    p_node = record["p"]
                    r_rel = record["r"]
                    t_node = record["t"]

                    # Add Planet node
                    p_id = p_node.get("name")
                    if p_id not in seen_nodes:
                        seen_nodes.add(p_id)
                        nodes.append({
                            "id": p_id,
                            "label": "Planet",
                            "properties": {
                                "name": p_id,
                                "ml_score": p_node.get("ml_score", 0.0)
                            }
                        })

                    # Add target node and relationship
                    if t_node:
                        t_label = list(t_node.labels)[0]
                        t_id = t_node.get("id") if t_label == "Paper" else t_node.get("name")
                        
                        if t_id not in seen_nodes:
                            seen_nodes.add(t_id)
                            props = dict(t_node)
                            nodes.append({
                                "id": t_id,
                                "label": t_label,
                                "properties": props
                            })

                        edges.append({
                            "source": p_id,
                            "target": t_id,
                            "type": r_rel.type
                        })

            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            print(f"[Neo4j Service] Error querying Neo4j: {e}")
            return {"nodes": [], "edges": []}

    def _query_sqlite_subgraph(self, planet_name: str) -> dict:
        nodes = []
        edges = []
        seen_nodes = set()

        # Connect to SQLite
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Retrieve planet info
            cursor.execute("SELECT pl_name, ml_score, sy_dist, pl_rade, pl_eqt FROM planets WHERE pl_name = ?", (planet_name,))
            planet_row = cursor.fetchone()
            if not planet_row:
                conn.close()
                return {"nodes": [], "edges": []}

            # Add planet node
            p_name, ml_score, dist, rade, eqt = planet_row
            nodes.append({
                "id": p_name,
                "label": "Planet",
                "properties": {
                    "name": p_name,
                    "ml_score": ml_score,
                    "sy_dist": dist,
                    "pl_rade": rade,
                    "pl_eqt": eqt
                }
            })
            seen_nodes.add(p_name)

            # Query relationships
            cursor.execute("""
                SELECT source_type, source_name, rel_type, target_type, target_name
                FROM graph_relationships
                WHERE source_name = ? OR target_name = ?
            """, (planet_name, planet_name))
            
            rows = cursor.fetchall()
            for src_type, src_name, rel_type, tgt_type, tgt_name in rows:
                # Add source node if not seen
                if src_name not in seen_nodes:
                    seen_nodes.add(src_name)
                    props = {"name": src_name}
                    if src_type == "Star":
                        # Fetch star details
                        cursor.execute("SELECT st_teff, st_rad, st_mass FROM planets WHERE hostname = ? LIMIT 1", (src_name,))
                        star_row = cursor.fetchone()
                        if star_row:
                            props.update({"st_teff": star_row[0], "st_rad": star_row[1], "st_mass": star_row[2]})
                    nodes.append({
                        "id": src_name,
                        "label": src_type,
                        "properties": props
                    })

                # Add target node if not seen
                if tgt_name not in seen_nodes:
                    seen_nodes.add(tgt_name)
                    props = {"name": tgt_name}
                    if tgt_type == "Paper":
                        # Fetch paper details from models/papers_metadata.json (or load from SQLite if we stored it, but since papers are in data/raw/papers.jsonl we can search papers.jsonl or keep a light lookup cache)
                        # We will fetch title/year if possible, or fallback to simple paper_id
                        props = {"id": tgt_name}
                        # We will try to search for the title/url dynamically or from metadata
                    nodes.append({
                        "id": tgt_name,
                        "label": tgt_type,
                        "properties": props
                    })

                edges.append({
                    "source": src_name,
                    "target": tgt_name,
                    "type": rel_type
                })

            conn.close()

            # Let's populate paper details if present
            # Load metadata from models/papers_metadata.json for enriching Paper nodes
            metadata_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "models/papers_metadata.json"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, "r", encoding="utf-8") as f:
                    paper_meta = {p["paper_id"]: p for p in json.load(f)}
                for node in nodes:
                    if node["label"] == "Paper" and node["id"] in paper_meta:
                        node["properties"].update({
                            "title": paper_meta[node["id"]]["title"],
                            "year": paper_meta[node["id"]]["year"],
                            "url": paper_meta[node["id"]]["url"]
                        })

            return {"nodes": nodes, "edges": edges}

        except Exception as e:
            print(f"[Neo4j Service] Error querying SQLite graph: {e}")
            return {"nodes": [], "edges": []}

import json # Import here for local import in class definition
