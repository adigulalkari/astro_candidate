from backend.app.services.neo4j_service import Neo4jService

def query_knowledge_graph(planet_name: str) -> dict:
    """
    Traverse the exoplanet knowledge graph around a target planet.
    Returns:
        {"nodes": [...], "edges": [...]}
    """
    service = Neo4jService()
    try:
        subgraph = service.get_planet_subgraph(planet_name)
        return subgraph
    except Exception as e:
        print(f"[Tool: KG] Error querying graph: {e}")
        return {"nodes": [], "edges": []}
    finally:
        service.close()
