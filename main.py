from langchain_core.messages import HumanMessage
from graphs.news_graph import create_news_graph
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """
    Main function to run the news analyzer graph.
    Now loops to accept multiple queries until the user types 'exit' or 'quit'.
    """
    # Ensure API keys are loaded
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        print("Error: API keys not found. Please set them in a .env file.")
        return

    # Create and compile the graph once
    app = create_news_graph()

    print("--- Multi-Agent News & Trend Analyzer ---")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        # Prompt the user for a query
        user_query = input("\nWhat do you want to research? ").strip()

        # Exit conditions
        if user_query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Skip empty input
        if not user_query:
            print("Please enter a topic or type 'exit' to quit.")
            continue

        # Define the initial state of the graph for this query
        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "next_node": "parallel_retrieve",
        }

        print("\n------------------------------------------")
        print(f"Query: {user_query}")
        print("------------------------------------------")

        # Run the graph and get the final state
        final_state = app.invoke(initial_state)

        # Display the final report
        print("\n--- Final Report ---")
        print(final_state.get('report', 'No report generated.'))
        print("--------------------")


if __name__ == "__main__":
    main()