from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from backend.app.agent.graph import agent_graph
from backend.app.tools.planet_search import search_planets
from backend.app.tools.ranking import rank_planets
from backend.app.tools.knowledge_graph import query_knowledge_graph

router = APIRouter()

# Pydantic schemas
class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    candidates: List[Dict[str, Any]]
    evidence: List[Dict[str, Any]]
    uncertainties: List[str]
    steps_log: List[str]
    graph: Dict[str, Any]

class CustomRankRequest(BaseModel):
    pl_rade: float
    pl_eqt: float
    pl_insol: float
    sy_dist: float
    st_teff: Optional[float] = 5778.0
    st_rad: Optional[float] = 1.0

# Endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        # Execute LangGraph agent
        res = agent_graph.invoke({"user_query": req.query})
        
        final_answer = res.get("final_answer", {})
        steps_log = res.get("steps_log", [])
        graph_results = res.get("graph_results", {"nodes": [], "edges": []})

        return ChatResponse(
            answer=final_answer.get("answer", "No answer could be generated."),
            candidates=final_answer.get("candidates", []),
            evidence=final_answer.get("evidence", []),
            uncertainties=final_answer.get("uncertainties", []),
            steps_log=steps_log,
            graph=graph_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e}")

@router.get("/planets")
async def get_planets(
    query: Optional[str] = None,
    max_distance: Optional[float] = None,
    min_radius: Optional[float] = None,
    max_radius: Optional[float] = None,
    min_temp: Optional[float] = None,
    max_temp: Optional[float] = None,
    discovery_method: Optional[str] = None,
    limit: Optional[int] = 20
):
    try:
        results = search_planets(
            query_str=query,
            max_distance=max_distance,
            min_radius=min_radius,
            max_radius=max_radius,
            min_temp=min_temp,
            max_temp=max_temp,
            discovery_method=discovery_method,
            limit=limit
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/planets/{name}")
async def get_planet_details(name: str):
    try:
        # Search exact name
        results = search_planets(query_str=name, limit=1)
        if not results:
            raise HTTPException(status_code=404, detail="Planet not found")
        
        planet_data = results[0]
        
        # If the planet name doesn't match exactly, check
        if planet_data["pl_name"].lower() != name.lower():
            raise HTTPException(status_code=404, detail="Planet not found")
            
        return planet_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/planets/{name}/graph")
async def get_planet_graph(name: str):
    try:
        subgraph = query_knowledge_graph(name)
        return subgraph
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rank")
async def rank_candidates(req: CustomRankRequest):
    try:
        custom_features = {
            "pl_rade": req.pl_rade,
            "pl_eqt": req.pl_eqt,
            "pl_insol": req.pl_insol,
            "sy_dist": req.sy_dist,
            "st_teff": req.st_teff,
            "st_rad": req.st_rad
        }
        # Run model inference on custom features
        rankings = rank_planets(custom_planet_features=custom_features)
        return rankings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
