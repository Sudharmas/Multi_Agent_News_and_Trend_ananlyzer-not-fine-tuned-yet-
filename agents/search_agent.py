from langchain_core.messages import HumanMessage
from tools.search_tool import get_search_tool

search_tool = get_search_tool()


def search_agent(state):
    """
    Agent that performs a web search based on the user's query.
    It updates the state with the search results.
    """
    query = state['messages'][-1].content
    print(f"\n🔍 Search Agent: Searching for '{query}'...")

    # The search tool takes a query string
    try:
        search_results = search_tool.invoke(query)
    except Exception as e:
        print(f"❌ Search Agent error: {e}")
        search_results = []  # Let summarizer handle empty results gracefully

    # We add the results to our state for the next agent to use
    state['search_results'] = search_results
    state['next_node'] = "summarize"

    print("✅ Search complete.")
    return state