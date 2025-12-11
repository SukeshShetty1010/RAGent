# rough.py
"""
Test harness for the new LLM-powered MinimalAgent (agent/core.py).

Updated to use MinimalAgent.decide_and_fetch_if_needed(...) from the refactored core.py.
- Creates Tool Registry
- Registers RetrieverTool + DataFetcherTool
- Instantiates MinimalAgent(registry)
- Executes a sample query (or one provided via --query)
- Prints full structured trace
- Saves all console output to 'agent_trace.txt'
"""

from __future__ import annotations
import json
import argparse
import sys
import traceback
import os  # Added for file handling if needed

from agent.tools.registry import ToolRegistry
from agent.tools.retriever_tool import RetrieverTool
from agent.tools.data_fetcher_tool import DataFetcherTool
from agent.core import MinimalAgent  # updated core.py provides MinimalAgent(registry)


# --- ADDED: Logger Class ---
class DualLogger(object):
    """
    Helper class to redirect standard output to both the terminal and a log file.
    """
    def __init__(self, filename="agent_trace.txt"):
        self.terminal = sys.stdout
        self.filename = filename
        # clear the file on new run, or use "a" to append
        self.log = open(filename, "w", encoding="utf-8") 

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed for python compatibility and to ensure data is written immediately
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
# ---------------------------


def pretty(obj):
    """Nicely print dicts/lists with indentation."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _check_modal_available() -> bool:
    try:
        import modal  # type: ignore
        return True
    except Exception:
        return False


def main():
    # --- ADDED: Initialize Logging ---
    # This redirects both print statements (stdout) and errors (stderr) to the file
    logger = DualLogger("agent_trace.txt")
    sys.stdout = logger
    sys.stderr = logger 
    # ---------------------------------

    parser = argparse.ArgumentParser(description="Run MinimalAgent (rough test harness).")
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default="What are the minimum requirements for Assassin's Creed Valhalla?",
        help="The user query to send to the agent.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of top results to return from retriever (and final results).",
    )
    args = parser.parse_args()
    query = args.query
    k = args.k

    print("\n=== Initializing Tools & Agent ===\n")
    print(f"[INFO] Logging all output to: {os.path.abspath(logger.filename)}")

    modal_available = _check_modal_available()
    if modal_available:
        print("[INFO] modal package found — LLM remote path should be available.")
    else:
        print("[WARN] modal package NOT found. Agent will fall back to score-threshold behavior.\n")

    # ------------------------------------
    # Setup Tool Registry
    # ------------------------------------
    registry = ToolRegistry()

    # Tools must define their own .name attribute and implement execute(...)
    retriever = RetrieverTool(weaviate_url="http://localhost:8080")
    data_fetcher = DataFetcherTool(weaviate_url="http://localhost:8080")

    # Register tools in the registry (the agent expects registry.get(name) to return a Tool)
    registry.register(retriever)
    registry.register(data_fetcher)

    # ------------------------------------
    # Create Agent
    # ------------------------------------
    agent = MinimalAgent(registry)

    # ------------------------------------
    # Run Query
    # ------------------------------------
    print(f"Running agent on query:\n → {query}\n")

    try:
        # Use the new decide_and_fetch_if_needed entrypoint which encapsulates:
        # - retrieval
        # - LLM relevance grading (Strict Information Auditor)
        # - conditional data fetching
        trace = agent.decide_and_fetch_if_needed(query, retriever, k=k)
    except ImportError as ie:
        print("[ERROR] ImportError during agent.decide_and_fetch_if_needed():")
        print(str(ie))
        sys.exit(2)
    except Exception as e:
        print("[ERROR] Unexpected exception while running agent:")
        traceback.print_exc()
        sys.exit(3)

    # ------------------------------------
    # Print Raw Trace
    # ------------------------------------
    print("\n=== TRACE (FULL OUTPUT) ===")
    print(pretty(trace))

    # ------------------------------------
    # SUMMARY
    # ------------------------------------
    print("\n=== SUMMARY ===")
    steps = trace.get("steps", []) if isinstance(trace, dict) else []
    if "data_fetch" in steps:
        print("✔ Agent triggered ingestion (retrieval insufficient).")
    else:
        print("✔ Agent used retrieval only (no ingestion triggered).")

    llm_judgement = trace.get("llm_judgement") if isinstance(trace, dict) else None
    if llm_judgement:
        print(f"✔ LLM Relevance Judgment: {llm_judgement}")

    results = trace.get("results") if isinstance(trace, dict) else None
    if results:
        print(f"\nTop {min(len(results), k)} result(s):")
        for i, r in enumerate(results[:k], start=1):
            title = r.get("title") or r.get("meta", {}).get("title") or r.get("id") or f"result_{i}"
            score = r.get("score", "N/A")
            snippet = r.get("content") or r.get("text") or r.get("meta", {}).get("text") or ""
            snippet_short = (snippet[:300] + "...") if len(snippet) > 300 else snippet
            print(f"\n[{i}] {title} (score={score})\n{snippet_short}")

    print("\nDone.\n")
    
    # Clean close of the log file
    logger.close()


if __name__ == "__main__":
    main()