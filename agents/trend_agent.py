from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def trend_agent(state):
    """
    Agent that analyzes the summarized information to identify trends.
    It takes the summary from the state and returns an analysis of key trends.
    """
    summary = state.get('summary')
    if not summary or summary == "No relevant information found.":
        print("❌ Trend Agent: No summary to analyze.")
        state['trends'] = "No trends could be identified due to a lack of data."
        state['next_node'] = "report"
        return state

    print("\n📈 Trend Agent: Analyzing for key trends...")

    prompt = f"""
    You are a professional trend analyst. Based on the following summary of news,
    identify and list any emerging trends, key players, or future implications.
    Format your response clearly using bullet points.

    Summary:
    {summary}

    Trend Analysis:
    """

    trends = llm.invoke(prompt).content

    state['trends'] = trends
    state['next_node'] = "report"

    print("✅ Trend analysis complete.")
    return state