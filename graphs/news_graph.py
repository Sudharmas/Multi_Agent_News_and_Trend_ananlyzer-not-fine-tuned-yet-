from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage

# Import our agents
from agents.supervisor import supervisor_agent
from agents.search_agent import search_agent
from agents.summarizer_agent import summarizer_agent
from agents.trend_agent import trend_agent
from agents.report_agent import report_agent
from agents.parallel_retrieval_agent import parallel_retrieval_agent


class NewsState(TypedDict):
    """
    Represents the state of our graph.
    """
    messages: Annotated[list[BaseMessage], "messages"]
    search_results: list
    summary: str
    trends: str
    report: str
    next_node: str


def create_news_graph():
    """
    Creates and compiles the LangGraph workflow.
    """
    # Define the graph state
    workflow = StateGraph(NewsState)

    # Add nodes (our agents) to the workflow
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("parallel_retrieve", parallel_retrieval_agent)
    workflow.add_node("search", search_agent)
    workflow.add_node("summarize", summarizer_agent)
    workflow.add_node("analyze_trends", trend_agent)
    workflow.add_node("report", report_agent)

    # Set the entry point of the graph
    workflow.set_entry_point("supervisor")

    # Define the conditional edges (the routing logic)
    workflow.add_conditional_edges(
        "supervisor",
        # The supervisor_agent returns a dictionary with the 'next_node' key
        lambda x: x["next_node"],
        {
            "parallel_retrieve": "parallel_retrieve",
            "search": "search",
            "summarize": "summarize",
            "analyze_trends": "analyze_trends",
            "report": "report",
            "end": END
        }
    )

    # Define the regular edges (the fixed flow)
    workflow.add_edge("parallel_retrieve", "supervisor")
    workflow.add_edge("search", "supervisor")
    workflow.add_edge("summarize", "supervisor")
    workflow.add_edge("analyze_trends", "supervisor")
    workflow.add_edge("report", END)

    # Compile the graph
    app = workflow.compile()

    return app