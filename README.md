# Multi‑Agent News & Trend Analyzer

A small multi‑agent workflow that performs live web research and produces a final report:

search → summarize → analyze trends → report.

You can run it via:
- Command‑line (interactive loop)
- Web UI (Flask)

## 1) Requirements

- Python 3.10+
- API keys:
  - GOOGLE_API_KEY (for Google Gemini via langchain-google-genai)
  - TAVILY_API_KEY (for Tavily web search)

## 2) Setup

1. (Recommended) Create and activate a virtual environment
   - macOS/Linux:
     - python3 -m venv .venv
     - source .venv/bin/activate
   - Windows (PowerShell):
     - python -m venv .venv
     - .\.venv\Scripts\Activate.ps1

2. Install dependencies

   pip install -r requirements.txt

3. Create a .env file in the project root with your keys

   GOOGLE_API_KEY=your_google_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here

If you don’t have these yet:
- Google API key: https://aistudio.google.com/app/apikey
- Tavily API key: https://tavily.com/

## 3) Run the CLI

Runs an interactive loop. Type your research query; type "exit" or "quit" to stop.

   python main.py

Example session:

   --- Multi-Agent News & Trend Analyzer ---
   Type 'exit' or 'quit' to end the session.

   What do you want to research? AI investment trends 2025
   ...
   --- Final Report ---
   <report text>

## 4) Run the Web UI (Flask)

Start the Flask dev server:

   python web_app.py

Open your browser at:

   http://localhost:5000/

- Enter a query in the input box and press Send.
- The assistant will think and then display a report with a handy Copy button.

To change the port, set PORT or FLASK_RUN_PORT in your environment before running.

## 5) Project Structure

- agents/: individual agents (search, summarizer, trend, report, supervisor)
- graphs/: LangGraph workflow definition
- tools/: external tools (Tavily search)
- templates/: HTML templates for Flask (index.html)
- main.py: CLI entry point
- web_app.py: Flask server entry point

## 6) Troubleshooting

- ModuleNotFoundError: No module named 'flask'
  - Run: pip install -r requirements.txt

- Error: API keys not found. Please set them in a .env file.
  - Create .env with GOOGLE_API_KEY and TAVILY_API_KEY in the project root.
  - If running in Docker or a cloud environment, be sure the variables are exported into the process environment.

- Tavily errors / empty results
  - Ensure your TAVILY_API_KEY is correct and active.

- Google Generative AI errors
  - Ensure GOOGLE_API_KEY is valid and has quota.

## 7) Notes

- The workflow routes automatically, but agents explicitly set `next_node` to enforce the intended sequence.
- CLI supports multiple queries in one run.
- The Flask app initializes the graph once per process for performance.
# Multi_Agent_News_and_Trend_ananlyzer-not-fine-tuned-yet-
