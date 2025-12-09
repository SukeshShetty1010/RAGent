# agent/core.py
"""
Agent core: MinimalAgent

Provides a small rule-based planning loop that uses registered tools to:
  1) attempt retrieval
  2) if retrieval is insufficient, heuristically extract a game name and call the data_fetcher
  3) retry retrieval once and return a trace of actions taken.

This file expects the following tools to be registered in the provided ToolRegistry:
  - "retriever" -> a Tool implementing retrieval (expects execute({"query": ...}) -> List[Dict])
  - "data_fetcher" -> a Tool implementing ingestion (expects execute({"game_name": ...}) -> Dict)

Constraints satisfied:
  - Tools are accessed via registry.get(...)
  - Robust error handling around data_fetcher (we proceed to retry even if fetcher fails)
  - Logging prints "Agent's Thoughts" with prefix [AGENT]
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from agent.tools.registry import ToolRegistry
from agent.base import Tool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class MinimalAgent:
    """
    Minimal rule-based agent that coordinates retrieval and, when necessary,
    a data-fetching ingestion pipeline to populate the vector DB.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry instance")
        self.registry = registry
        # Pre-check available tools (not required, but helps early failure)
        # We do not instantiate tools here; they should be registered already.
        self._retriever_tool_name = "retriever"
        self._data_fetcher_tool_name = "data_fetcher"

    def _log(self, msg: str) -> None:
        logger.info("[AGENT] %s", msg)

    def _heuristically_extract_game_name(self, query: str) -> str:
        """
        Heuristic extraction of a game name from a question/query.
        Tries several patterns and fallbacks:
          - quoted phrase: "X"
          - patterns like "Who made X", "Who developed X"
          - "Tell me about X", "Info about X"
          - capitalized consecutive words (2+)
          - otherwise fallback to last 3 words
        """
        if not query or not isinstance(query, str):
            return ""

        q = query.strip()

        # 1) quoted phrase
        m = re.search(r'["“”\'\u2018\u2019](.+?)["“”\'\u2018\u2019]', q)
        if m:
            name = m.group(1).strip()
            if name:
                return name

        # 2) Who made / Who developed / Who created
        m = re.search(r'who (?:made|created|developed|published|built|produced)\s+(.+?)[\?\.]?$', q, flags=re.I)
        if m:
            return m.group(1).strip().strip('?"\'')

        # 3) tell me about / info about / details about
        m = re.search(r'(?:tell me about|info(?:rmation)? about|details about|what is|what\'s)\s+(.+?)[\?\.]?$', q, flags=re.I)
        if m:
            return m.group(1).strip().strip('?"\'')

        # 4) look for "for <Game>" or "about <Game>"
        m = re.search(r'(?:for|about)\s+([A-Z][\w\'\s:&-]{2,})', q)
        if m:
            cand = m.group(1).strip()
            # limit trailing words
            return cand.strip('?"\'')

        # 5) capitalized run of words (e.g., "Far Cry 5", "Assassin's Creed Valhalla")
        capitalized_runs = re.findall(r'(?:[A-Z][\w\'’-]+(?:\s+[A-Z][\w\'’-]+)+)', q)
        if capitalized_runs:
            # return the longest run
            capitalized_runs.sort(key=lambda s: len(s), reverse=True)
            return capitalized_runs[0].strip()

        # 6) fallback to last 3 words (strip common stop words)
        tokens = re.findall(r"\w+'?\w+|\w+", q)
        if len(tokens) >= 3:
            return " ".join(tokens[-3:]).strip()
        elif tokens:
            return " ".join(tokens).strip()

        return ""

    def run(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Execute the agent loop:
          1) call retriever
          2) if results empty -> call data_fetcher with heuristically extracted game name
          3) retry retriever once
        Returns a trace dict with steps and final results.
        """
        steps: List[str] = []
        final_results: List[Dict[str, Any]] = []
        status = "failure"

        # 1) Initial retrieval
        self._log("Retrieving initial results...")
        steps.append("retriever")
        try:
            retriever_tool: Tool = self.registry.get(self._retriever_tool_name)
        except Exception as e:
            msg = f"Retriever tool not found in registry: {e}"
            logger.exception(msg)
            return {"status": "error", "error": msg, "steps": steps, "results": []}

        try:
            results = retriever_tool.execute({"query": query, "k": k})
            if not isinstance(results, list):
                # defensive: if tool returns non-list, coerce or wrap
                logger.warning("Retriever returned non-list result; coercing to list")
                results = list(results) if results is not None else []
        except Exception as e:
            logger.exception("Retriever execution failed: %s", e)
            results = []

        # 2) Evaluate
        if results and len(results) > 0:
            self._log(f"Retriever returned {len(results)} hits. Returning results.")
            status = "success"
            final_results = results
            return {"status": status, "steps": steps, "results": final_results}
        else:
            # Failure branch: attempt to fetch external data
            self._log("Low retrieval hits. Attempting to fetch external data.")
            # Heuristically extract a game name
            extracted_name = self._heuristically_extract_game_name(query)
            if not extracted_name:
                self._log("Could not heuristically extract a game name from the query. Will retry retrieval once anyway.")
            else:
                self._log(f"Heuristically extracted game name: '{extracted_name}'")
            # Call data_fetcher tool if available
            try:
                data_fetcher_tool: Tool = self.registry.get(self._data_fetcher_tool_name)
                steps.append("data_fetcher")
                self._log("Invoking data_fetcher to ingest data for: " + (extracted_name or "<unknown>"))
                try:
                    fetch_args = {"game_name": extracted_name} if extracted_name else {"game_name": query}
                    fetch_res = data_fetcher_tool.execute(fetch_args)
                    # fetch_res is expected to be dict; but be tolerant
                    if isinstance(fetch_res, dict):
                        fetch_success = bool(fetch_res.get("success", False))
                        self._log(f"data_fetcher finished: success={fetch_success}")
                    else:
                        self._log("data_fetcher returned non-dict response; continuing to retry retrieval.")
                except Exception as fe:
                    # Robustness: log the fetcher exception and proceed to retry retrieval anyway
                    logger.exception("DataFetcherTool execution failed: %s", fe)
                    self._log("DataFetcherTool raised an exception; proceeding to retry retrieval anyway.")
            except KeyError:
                self._log("No data_fetcher tool registered; skipping ingestion step.")
            except Exception as e:
                # Any other unexpected error retrieving the tool
                logger.exception("Failed to access data_fetcher tool: %s", e)
                self._log("Proceeding to retry retrieval despite missing data_fetcher.")

            # Retry retrieval once more
            self._log("Retrying retrieval after ingestion attempt...")
            steps.append("retriever_retry")
            try:
                retry_results = retriever_tool.execute({"query": query, "k": k})
                if not isinstance(retry_results, list):
                    retry_results = list(retry_results) if retry_results is not None else []
            except Exception as e:
                logger.exception("Retry retriever execution failed: %s", e)
                retry_results = []

            if retry_results and len(retry_results) > 0:
                status = "success_after_fetch"
                final_results = retry_results
                self._log(f"Retry retrieved {len(retry_results)} hits. Returning results.")
            else:
                status = "failure"
                final_results = []
                self._log("Retry retrieval returned no results.")

        trace = {"status": status, "steps": steps, "results": final_results}
        return trace
