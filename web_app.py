from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage
from graphs.news_graph import create_news_graph

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize the news analysis graph once per server process
news_app = create_news_graph()


def ensure_keys_present():
    """Return (ok, msg). ok=False if required API keys are missing."""
    missing = []
    if not os.getenv("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY")
    if not os.getenv("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY")
    if missing:
        return False, f"Missing API keys: {', '.join(missing)}. Please set them in .env."
    return True, "OK"


@app.get("/")
def index():
    ok, msg = ensure_keys_present()
    warning = None if ok else msg
    return render_template("index.html", warning=warning)


@app.post("/ask")
def ask():
    # Accept JSON or form-encoded input
    query = (request.json or {}).get("query") if request.is_json else request.form.get("query")
    if not query or not query.strip():
        return jsonify({"ok": False, "error": "Empty query. Please enter a topic."}), 400

    ok, msg = ensure_keys_present()
    if not ok:
        return jsonify({"ok": False, "error": msg}), 500

    try:
        initial_state = {"messages": [HumanMessage(content=query.strip())], "next_node": "parallel_retrieve"}
        final_state = news_app.invoke(initial_state)
        report = final_state.get("report", "No report generated.")
        return jsonify({"ok": True, "report": report})
    except Exception as e:
        # Provide a safe error message to UI
        return jsonify({"ok": False, "error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    # Run the Flask development server
    # Users can set FLASK_RUN_PORT or we default to 5000
    port = int(os.getenv("PORT", os.getenv("FLASK_RUN_PORT", 5050)))
    app.run(host="0.0.0.0", port=port, debug=True)
