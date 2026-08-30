from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from backend.app.services.llm_service import LLMService
from backend.app.tools.planet_search import search_planets
from backend.app.tools.ranking import rank_planets
from backend.app.tools.knowledge_graph import query_knowledge_graph
from backend.app.tools.literature import search_literature
import json

class AgentState(TypedDict):
    user_query: str
    selected_tools: List[str]
    current_step: int
    steps_log: List[str]
    planets: List[Dict[str, Any]]
    ml_results: List[Dict[str, Any]]
    graph_results: Dict[str, Any]
    retrieved_papers: List[Dict[str, Any]]
    final_answer: Dict[str, Any]

# Initialize LLM Service
llm = LLMService()

def intent_router(state: AgentState) -> Dict[str, Any]:
    """
    Evaluates the query and determines which tools are needed.
    """
    query = state["user_query"]
    
    # If LLM service is in mock mode, use a simple rule-based classifier
    if llm.provider == "mock":
        selected = []
        q_lower = query.lower()
        
        # Rule-based tool selection
        if any(w in q_lower for w in ["find", "search", "planet", "radius", "light year", "ly", "distance", "orbit"]):
            selected.append("planet_search")
        if any(w in q_lower for w in ["rank", "best", "most earth-like", "priority", "why"]):
            selected.append("ml_ranking")
        if any(w in q_lower for w in ["graph", "relation", "system", "orbit", "discovery"]):
            selected.append("kg_lookup")
        if any(w in q_lower for w in ["literature", "paper", "study", "evidence", "atmosphere", "read", "say"]):
            selected.append("rag_search")
            
        # Ensure at least planet_search is chosen if none matched
        if not selected:
            selected = ["planet_search"]
    else:
        # Ask LLM to determine tool routing
        sys_prompt = (
            "You are the Exoplanet Analyst Router. Analyze the user query and decide which tools are required to answer it.\n"
            "Available tools:\n"
            "1. 'planet_search': Search and filter planet catalogs.\n"
            "2. 'ml_ranking': Rank candidates or explain scores.\n"
            "3. 'kg_lookup': Fetch star-planet-paper relationship graphs.\n"
            "4. 'rag_search': Search scientific paper abstracts.\n\n"
            "Return a JSON array containing ONLY the string names of the tools needed, for example: ['planet_search', 'ml_ranking']."
        )
        try:
            res_str = llm.generate_response(sys_prompt, f"User Question: {query}", response_format_json=True)
            selected = json.loads(res_str)
        except Exception:
            selected = ["planet_search"]

    steps = [f"Route Query -> Selected Tools: {selected}"]
    return {
        "selected_tools": selected,
        "current_step": 0,
        "steps_log": steps,
        "planets": [],
        "ml_results": [],
        "graph_results": {"nodes": [], "edges": []},
        "retrieved_papers": [],
        "final_answer": {}
    }

def planet_search_node(state: AgentState) -> Dict[str, Any]:
    query = state["user_query"]
    print("[Agent Node] Executing planet_search...")
    
    # Extract radius/distance heuristics from query for filtering
    max_dist = None
    max_rad = None
    
    import re
    dist_match = re.search(r"within\s+(\d+)\s+(light year|ly)", query, re.IGNORECASE)
    if dist_match:
        # Convert light years to parsecs (1 pc = 3.26 ly)
        max_dist = float(dist_match.group(1)) / 3.26
        
    rad_match = re.search(r"(\d+(\.\d+)?)\s*earth\s*radi", query, re.IGNORECASE)
    if rad_match:
        max_rad = float(rad_match.group(1))

    # Perform search
    planet_name_extract = None
    # Check if a specific planet like "TRAPPIST-1 e" or "Proxima Cen b" is mentioned
    for name in ["TRAPPIST-1 e", "TRAPPIST-1 d", "TRAPPIST-1 f", "Proxima Cen b", "Barnard b", "GJ 1002 b"]:
        if name.lower() in query.lower() or name.lower().replace(" ", "") in query.lower().replace(" ", ""):
            planet_name_extract = name
            break

    results = search_planets(
        query_str=planet_name_extract,
        max_distance=max_dist,
        max_radius=max_rad
    )
    
    steps = state["steps_log"] + [f"Executed planet_search: retrieved {len(results)} candidate records."]
    return {"planets": results, "steps_log": steps, "current_step": state["current_step"] + 1}

def ml_ranking_node(state: AgentState) -> Dict[str, Any]:
    print("[Agent Node] Executing ml_ranking...")
    candidates = state["planets"]
    
    # If we already retrieved planets, rank them
    if candidates:
        names = [c["pl_name"] for c in candidates]
        results = rank_planets(planet_names=names)
    else:
        # Fallback search top ranked planets from SQLite
        results = search_planets(limit=10)

    steps = state["steps_log"] + [f"Executed ml_ranking: ranked {len(results)} candidates."]
    return {"ml_results": results, "steps_log": steps, "current_step": state["current_step"] + 1}

def kg_lookup_node(state: AgentState) -> Dict[str, Any]:
    print("[Agent Node] Executing kg_lookup...")
    
    # We find relationships for the top candidate
    target_planet = None
    if state["ml_results"]:
        target_planet = state["ml_results"][0]["pl_name"]
    elif state["planets"]:
        target_planet = state["planets"][0]["pl_name"]
    else:
        target_planet = "TRAPPIST-1 e" # Default fallback for KG demonstration
        
    results = query_knowledge_graph(target_planet)
    
    steps = state["steps_log"] + [f"Executed kg_lookup: fetched graph with {len(results.get('nodes', []))} nodes for planet {target_planet}."]
    return {"graph_results": results, "steps_log": steps, "current_step": state["current_step"] + 1}

def rag_search_node(state: AgentState) -> Dict[str, Any]:
    print("[Agent Node] Executing rag_search...")
    query = state["user_query"]
    
    # Extract planet filters if any
    target_planet = None
    if state["planets"]:
        target_planet = state["planets"][0]["pl_name"]
        
    results = search_literature(query=query, filter_planet=target_planet, top_k=5)
    if not results:
        # Try searching without hard planet filter if empty
        results = search_literature(query=query, top_k=5)

    steps = state["steps_log"] + [f"Executed rag_search: retrieved {len(results)} matching scientific papers."]
    return {"retrieved_papers": results, "steps_log": steps, "current_step": state["current_step"] + 1}

def synthesis_node(state: AgentState) -> Dict[str, Any]:
    print("[Agent Node] Executing synthesis...")
    
    # Prepare context for the LLM
    context = {
        "User Question": state["user_query"],
        "Planet Search Results": state["planets"][:5], # limit size to avoid prompt bloat
        "ML Ranking Results": state["ml_results"][:5],
        "Knowledge Graph Node Count": len(state["graph_results"].get("nodes", [])),
        "Retrieved Papers": state["retrieved_papers"][:3]
    }
    
    sys_prompt = (
        "You are the Exoplanet Candidate Analyst (ECA). Synthesize a research answer using the provided database search, ML rankings, graph metadata, and literature RAG context.\n"
        "You must output in JSON matching this schema:\n"
        "{\n"
        "  'answer': 'Concise response explaining the findings with citations (e.g. [1]).',\n"
        "  'candidates': [\n"
        "     {\n"
        "        'planet': 'Planet Name',\n"
        "        'score': 0.95,\n"
        "        'reasons': ['Reason 1', 'Reason 2']\n"
        "     }\n"
        "  ],\n"
        "  'evidence': [\n"
        "     {\n"
        "        'title': 'Paper title',\n"
        "        'year': 2023,\n"
        "        'url': 'Paper URL',\n"
        "        'claim_supported': 'What evidence does this paper provide?'\n"
        "     }\n"
        "  ],\n"
        "  'uncertainties': ['Uncertainty statements']\n"
        "}\n\n"
        "CRITICAL RULES:\n"
        "1. Never claim a planet is definitely inhabited, contains life, or is fully habitable.\n"
        "2. Do not fabricate papers. Only cite the papers provided in the 'Retrieved Papers' context.\n"
        "3. Emphasize astrophysical uncertainties (e.g. unconstrained atmospheric compositions, error bounds on equilibrium temp)."
    )

    try:
        response_str = llm.generate_response(sys_prompt, json.dumps(context, indent=2), response_format_json=True)
        final_ans = json.loads(response_str)
    except Exception as e:
        print(f"[Agent Node] Synthesis failed: {e}. Falling back to default JSON.")
        # Fallback to a structured answer constructed using our LLM Service mock fallback directly
        fallback_str = llm._generate_mock(sys_prompt, json.dumps(context))
        final_ans = json.loads(fallback_str)

    steps = state["steps_log"] + ["Synthesis complete. Outputting structured response."]
    return {"final_answer": final_ans, "steps_log": steps}

def route_next_node(state: AgentState) -> str:
    """
    Decides which tool node to execute next based on selected_tools list.
    When all tools are exhausted, transitions to synthesis.
    """
    tools = state["selected_tools"]
    curr_step = state["current_step"]
    
    if curr_step < len(tools):
        next_tool = tools[curr_step]
        if next_tool == "planet_search":
            return "planet_search"
        elif next_tool == "ml_ranking":
            return "ml_ranking"
        elif next_tool == "kg_lookup":
            return "kg_lookup"
        elif next_tool == "rag_search":
            return "rag_search"
            
    return "synthesis"

# Construct LangGraph State Graph
workflow = StateGraph(AgentState)

workflow.add_node("intent_router", intent_router)
workflow.add_node("planet_search", planet_search_node)
workflow.add_node("ml_ranking", ml_ranking_node)
workflow.add_node("kg_lookup", kg_lookup_node)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("synthesis", synthesis_node)

# Set entry point
workflow.set_entry_point("intent_router")

# Add conditional edges
workflow.add_conditional_edges(
    "intent_router",
    route_next_node,
    {
        "planet_search": "planet_search",
        "ml_ranking": "ml_ranking",
        "kg_lookup": "kg_lookup",
        "rag_search": "rag_search",
        "synthesis": "synthesis"
    }
)

workflow.add_conditional_edges("planet_search", route_next_node)
workflow.add_conditional_edges("ml_ranking", route_next_node)
workflow.add_conditional_edges("kg_lookup", route_next_node)
workflow.add_conditional_edges("rag_search", route_next_node)

# Synthesis transitions to END
workflow.add_edge("synthesis", END)

# Compile graph
agent_graph = workflow.compile()
