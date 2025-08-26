import os
from langchain_community.tools.tavily_search import TavilySearchResults

def get_search_tool():
    """
    Initializes and returns a Tavily search tool.
    This tool allows agents to perform real-time web searches.
    """
    return TavilySearchResults(max_results=5)