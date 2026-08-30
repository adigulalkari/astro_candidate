from backend.app.rag.retriever import LiteratureRetriever

def search_literature(
    query: str,
    top_k: int = 5,
    filter_planet: str = None,
    filter_star: str = None
) -> list:
    """
    Search indexed scientific paper abstracts for exoplanet-related evidence.
    """
    retriever = LiteratureRetriever()
    try:
        results = retriever.search_literature(
            query=query,
            top_k=top_k,
            filter_planet=filter_planet,
            filter_star=filter_star
        )
        return results
    except Exception as e:
        print(f"[Tool: Literature] Error searching index: {e}")
        return []
