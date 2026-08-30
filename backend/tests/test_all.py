import os
import sys
import pytest

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from backend.app.tools.planet_search import search_planets
from backend.app.tools.ranking import rank_planets
from backend.app.tools.knowledge_graph import query_knowledge_graph
from backend.app.tools.literature import search_literature
from backend.app.agent.graph import agent_graph

def test_planet_search():
    # Test searching for a specific planet
    res = search_planets(query_str="TRAPPIST-1 e", limit=1)
    assert len(res) == 1
    assert res[0]["pl_name"] == "TRAPPIST-1 e"
    assert res[0]["hostname"] == "TRAPPIST-1"

    # Test filtering by distance
    res_dist = search_planets(max_distance=10, limit=5)
    for p in res_dist:
        assert p["sy_dist"] <= 10.0

def test_ml_ranking():
    # Test ranking precomputed planet
    res = rank_planets(planet_names=["Proxima Cen b", "TRAPPIST-1 e"])
    assert len(res) == 2
    assert res[0]["ml_score"] > 0.0

    # Test ranking hypothetical planet
    custom_feats = {
        "pl_rade": 1.0,
        "pl_eqt": 288.0,
        "pl_insol": 1.0,
        "sy_dist": 10.0
    }
    res_custom = rank_planets(custom_planet_features=custom_feats)
    assert len(res_custom) == 1
    assert "ml_score" in res_custom[0]
    assert res_custom[0]["ml_score"] >= 0.0

def test_kg_query():
    # Test local graph lookup
    res = query_knowledge_graph("TRAPPIST-1 e")
    assert "nodes" in res
    assert "edges" in res
    assert len(res["nodes"]) > 0
    # The first node should be the planet itself
    assert res["nodes"][0]["id"] == "TRAPPIST-1 e"

def test_rag_retrieval():
    # Test RAG search on exoplanet abstracts
    res = search_literature("transmission spectroscopy", top_k=2)
    assert len(res) <= 2
    for p in res:
        assert "title" in p
        assert "score" in p
        assert p["score"] > 0.0

def test_agent_execution():
    # Test agent state progression and tool orchestration
    query = "Find TRAPPIST-1 e, explain why it ranks, and show its graph and literature papers."
    res = agent_graph.invoke({"user_query": query})
    
    assert "steps_log" in res
    assert "final_answer" in res
    assert "answer" in res["final_answer"]
    assert "candidates" in res["final_answer"]
    assert "evidence" in res["final_answer"]
    assert "uncertainties" in res["final_answer"]
