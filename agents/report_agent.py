from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def report_agent(state):
    """
    Agent that generates the final report.
    It uses the summary and trend analysis from the state to create a comprehensive report.
    """
    query = state['messages'][-1].content
    summary = state.get('summary', "No summary available.")
    trends = state.get('trends', "No trends identified.")

    print("\n📋 Report Agent: Generating final report...")

    prompt = f"""
    You are an expert report writer. Your task is to generate a comprehensive and
    well-structured report based on the following information. The report should
    be professional and easy to read.

    Topic: {query}

    Summary:
    {summary}

    Key Trends and Analysis:
    {trends}

    Final Report:
    """

    report = llm.invoke(prompt).content

    state['report'] = report
    state['next_node'] = "end"

    print("✅ Final report generated.")
    return state