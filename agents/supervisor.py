from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def supervisor_agent(state):
    """
    The supervisor agent, responsible for routing the workflow.
    It decides the next step based on the user's query and the current state.
    """
    # If a previous agent has explicitly set the next node, honor it.
    explicit_next = state.get("next_node")
    valid_nodes = ["search", "summarize", "analyze_trends", "report", "end"]
    if explicit_next in valid_nodes:
        return {"next_node": explicit_next}

    messages = state['messages']

    # We use a simple, explicit routing prompt. For a more complex system,
    # you could use a more dynamic LangChain router.
    prompt_template = PromptTemplate(
        input_variables=["messages"],
        template="""You are a supervisor agent. Your job is to decide the next action in a news analysis workflow.

        The user wants to analyze a topic. Here is the conversation so far:

        {messages}

        Based on the messages, what should be the next step?

        Choose one of the following actions:
        - `search`: The query requires a web search to find relevant information.
        - `summarize`: The search results have been gathered and need to be summarized.
        - `analyze_trends`: The summaries are ready and need to be analyzed for trends.
        - `report`: All analysis is complete, and the final report needs to be generated.
        - `end`: The process is finished.

        Your response must be a single word, exactly one of the actions listed above. Do not include any other text.
        """
    )

    # Format messages to be more readable for the LLM
    formatted_messages = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

    response = llm.invoke(prompt_template.format(messages=formatted_messages))
    next_node = response.content.strip().lower()

    # Simple validation to ensure the response is one of the valid options
    if next_node not in valid_nodes:
        print(f"Warning: Supervisor returned an invalid node: {next_node}. Defaulting to 'search'.")
        next_node = "search"

    return {"next_node": next_node}