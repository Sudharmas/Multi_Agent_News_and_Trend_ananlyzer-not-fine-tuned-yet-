from typing import List
from tools.memory import similarity_search


def rag_agent(state):
    """
    RAG Agent: performs similarity search over ChromaDB memory and returns
    text chunks as 'search_results'.
    """
    query = state['messages'][-1].content
    print(f"\n💾 RAG Agent: Retrieving from memory for '{query}'...")
    try:
        results = similarity_search(query, k=5)
        # Convert to a list of plain text snippets (include source metadata if available)
        search_results: List[str] = []
        for doc, dist, meta in results:
            prefix = "[MEMORY]"
            source = meta.get('source') if isinstance(meta, dict) else None
            if source:
                prefix += f" source={source}"
            search_results.append(f"{prefix} (score={1 - dist:.3f})\n{doc}")
    except Exception as e:
        print(f"❌ RAG Agent error: {e}")
        search_results = []

    state['search_results'] = search_results
    # Note: downstream routing decided by parallel merge node; default summarize
    state['next_node'] = 'summarize'
    print("✅ RAG retrieval complete.")
    return state
