import asyncio
from typing import List

from agents.rag_agent import rag_agent
from agents.search_agent import search_agent


async def _rag_task(state):
    # Make a shallow copy to avoid cross-writing of state during race
    local_state = {**state}
    result_state = rag_agent(local_state)
    return result_state.get("search_results", []), "rag"


async def _web_task(state):
    local_state = {**state}
    result_state = search_agent(local_state)
    return result_state.get("search_results", []), "web"


async def _race(state):
    t1 = asyncio.create_task(_rag_task(state))
    t2 = asyncio.create_task(_web_task(state))

    done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)

    # Get the first result
    first = done.pop()
    try:
        results, source = first.result()
    finally:
        # Cancel the loser
        for p in pending:
            p.cancel()
        # Optionally, drain cancellations
        await asyncio.gather(*pending, return_exceptions=True)

    return results, source


def parallel_retrieval_agent(state):
    """
    Executes RAG retrieval and Web search in parallel; first result wins.
    Places the winning results into state['search_results'] and routes to 'summarize'.
    """
    query = state['messages'][-1].content
    print(f"\n⚡ Parallel Retrieval: Launching RAG + Web for '{query}' (first result wins)...")

    try:
        results, source = asyncio.run(_race(state))
    except RuntimeError:
        # If already in an event loop (e.g., some environments), use a fallback
        results, source = asyncio.get_event_loop().run_until_complete(_race(state))  # type: ignore

    # If both failed, results may be empty
    if not results:
        print("⚠️ Parallel Retrieval: No results from either path.")
    else:
        print(f"✅ Parallel Retrieval: Using {source.upper()} results ({len(results)} items).")

    state['search_results'] = results
    state['next_node'] = 'summarize'
    return state
