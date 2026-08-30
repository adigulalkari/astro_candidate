import os
import json
import sqlite3
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd

# Load variables from project-root .env
load_dotenv()

def build_knowledge_graph():
    db_path = "data/processed/planets.db"
    papers_path = "data/raw/papers.jsonl"

    if not os.path.exists(db_path):
        print(f"[KG Builder] Error: SQLite DB does not exist at {db_path}.")
        return

    # 1. Load Data
    conn = sqlite3.connect(db_path)
    df_planets = pd.read_sql_query("SELECT pl_name, hostname, discoverymethod, disc_year, ml_score FROM planets", conn)
    
    papers = []
    if os.path.exists(papers_path):
        with open(papers_path, "r", encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
    print(f"[KG Builder] Loaded {len(df_planets)} planets and {len(papers)} papers.")

    # 2. Extract Relationships
    # Relationships list: (source_type, source_name, rel_type, target_type, target_name)
    relationships = []
    
    # Discovery methods set
    discovery_methods = set()

    print("[KG Builder] Processing planet and star nodes/relationships...")
    for idx, row in df_planets.iterrows():
        pl_name = row["pl_name"]
        hostname = row["hostname"]
        method = row["discoverymethod"]
        
        # 1. Planet ORBITS Star
        relationships.append(("Planet", pl_name, "ORBITS", "Star", hostname))
        
        # 2. Planet DISCOVERED_BY Method
        if pd.notna(method):
            relationships.append(("Planet", pl_name, "DISCOVERED_BY", "DiscoveryMethod", method))
            discovery_methods.add(method)

    # 3. Match Papers to Planets and Stars
    print("[KG Builder] Matching papers to planets/stars via text mentions...")
    updated_papers = []
    for paper in papers:
        title_abs = (paper["title"] + " " + paper["abstract"]).lower()
        mentioned_planets = []
        mentioned_stars = []

        # Simple case-insensitive matching
        for idx, row in df_planets.iterrows():
            pl_name = row["pl_name"]
            hostname = row["hostname"]

            # Match planet names like "TRAPPIST-1 e" or "TRAPPIST-1e"
            pl_variants = [pl_name.lower(), pl_name.lower().replace(" ", "")]
            if any(var in title_abs for var in pl_variants):
                mentioned_planets.append(pl_name)
                relationships.append(("Planet", pl_name, "MENTIONED_IN", "Paper", paper["paper_id"]))

            # Match host stars (avoid matching short star names like "HD" directly, require full hostname)
            if len(hostname) > 2 and hostname.lower() in title_abs:
                if hostname not in mentioned_stars:
                    mentioned_stars.append(hostname)
                    relationships.append(("Star", hostname, "MENTIONED_IN", "Paper", paper["paper_id"]))

        paper["mentioned_planets"] = list(set(mentioned_planets))
        paper["mentioned_stars"] = list(set(mentioned_stars))
        updated_papers.append(paper)

    # Rewrite papers.jsonl with mentioned entities updated
    with open(papers_path, "w", encoding="utf-8") as f:
        for p in updated_papers:
            f.write(json.dumps(p) + "\n")

    # 4. Save Relationships to SQLite
    print("[KG Builder] Saving graph relationships to SQLite...")
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS graph_relationships")
    cursor.execute("""
    CREATE TABLE graph_relationships (
        source_type TEXT,
        source_name TEXT,
        rel_type TEXT,
        target_type TEXT,
        target_name TEXT
    )
    """)
    cursor.execute("CREATE INDEX idx_source ON graph_relationships(source_name)")
    cursor.execute("CREATE INDEX idx_target ON graph_relationships(target_name)")

    cursor.executemany(
        "INSERT INTO graph_relationships VALUES (?, ?, ?, ?, ?)",
        relationships
    )
    conn.commit()
    conn.close()
    print(f"[KG Builder] Successfully stored {len(relationships)} edges in SQLite.")

    # 5. Optional Upload to Neo4j AuraDB
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USERNAME")
    neo4j_pwd = os.environ.get("NEO4J_PASSWORD")

    if neo4j_uri and neo4j_user and neo4j_pwd:
        print("[KG Builder] Neo4j credentials found. Uploading graph to AuraDB...")
        try:
            driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pwd))
            with driver.session() as session:
                # Clear Database
                print("[KG Builder] Clearing existing Neo4j graph...")
                session.run("MATCH (n) DETACH DELETE n")
                
                # Create Constraints
                session.run("CREATE CONSTRAINT UNIQUE_PLANET IF NOT EXISTS FOR (p:Planet) REQUIRE p.name IS UNIQUE")
                session.run("CREATE CONSTRAINT UNIQUE_STAR IF NOT EXISTS FOR (s:Star) REQUIRE s.name IS UNIQUE")
                session.run("CREATE CONSTRAINT UNIQUE_PAPER IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
                session.run("CREATE CONSTRAINT UNIQUE_METHOD IF NOT EXISTS FOR (m:DiscoveryMethod) REQUIRE m.name IS UNIQUE")

                # Insert Planets
                print("[KG Builder] Uploading Planet nodes...")
                for idx, row in df_planets.iterrows():
                    session.run(
                        "MERGE (p:Planet {name: $name}) SET p.ml_score = $ml_score",
                        name=row["pl_name"], ml_score=float(row["ml_score"])
                    )

                # Insert Stars
                print("[KG Builder] Uploading Star nodes...")
                stars = df_planets["hostname"].unique()
                for star in stars:
                    session.run("MERGE (s:Star {name: $name})", name=star)

                # Insert Discovery Methods
                print("[KG Builder] Uploading DiscoveryMethod nodes...")
                for dm in discovery_methods:
                    session.run("MERGE (m:DiscoveryMethod {name: $name})", name=dm)

                # Insert Papers
                print("[KG Builder] Uploading Paper nodes...")
                for paper in updated_papers:
                    session.run(
                        "MERGE (p:Paper {id: $id}) SET p.title = $title, p.year = $year, p.url = $url",
                        id=paper["paper_id"], title=paper["title"], year=paper["year"], url=paper["url"]
                    )

                # Insert Relationships
                print("[KG Builder] Uploading relationships...")
                for source_type, source_name, rel_type, target_type, target_name in relationships:
                    query = f"""
                    MATCH (s:{source_type} {{name: $source_name}})
                    MATCH (t:{target_type} {{name: $target_name}})
                    MERGE (s)-[:{rel_type}]->(t)
                    """
                    # Adjust query for Paper which uses id instead of name
                    s_key = "id" if source_type == "Paper" else "name"
                    t_key = "id" if target_type == "Paper" else "name"
                    
                    query = f"""
                    MATCH (s:{source_type} {{{s_key}: $source_name}})
                    MATCH (t:{target_type} {{{t_key}: $target_name}})
                    MERGE (s)-[:{rel_type}]->(t)
                    """
                    session.run(query, source_name=source_name, target_name=target_name)

            driver.close()
            print("[KG Builder] Neo4j upload completed successfully.")
        except Exception as e:
            print(f"[KG Builder] Neo4j upload failed: {e}")
    else:
        print("[KG Builder] No Neo4j credentials found in environment. Skipping upload. Graph runs in local SQLite fallback mode.")

if __name__ == "__main__":
    build_knowledge_graph()
