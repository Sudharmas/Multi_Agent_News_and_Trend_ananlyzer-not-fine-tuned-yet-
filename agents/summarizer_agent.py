from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def summarizer_agent(state):
    """
    Agent that summarizes search results.
    It takes the raw search results from the state and returns a summary.
    """
    search_results = state.get('search_results', [])
    if not search_results:
        print("❌ Summarizer Agent: No search results to summarize.")
        state['summary'] = "No relevant information found."
        state['next_node'] = "report"  # Go straight to reporting
        return state

    print("\n📝 Summarizer Agent: Summarizing search results...")

    # Combine search results into a single string for the LLM
    results_string = "\n\n".join([str(result) for result in search_results])

    prompt = f"""
    You are an expert summarizer. Your task is to provide a concise and clear summary
    of the key information found in the following search results.

    Search Results:
    {results_string}

    Provide a summary of the key points.
    """

    summary = llm.invoke(prompt).content

    state['summary'] = summary
    state['next_node'] = "analyze_trends"

    print("✅ Summarization complete.")
    return state