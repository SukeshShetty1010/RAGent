# agent/core.py
"""
Agent core: MinimalAgent

This version adds score-threshold based decisioning:
 - initial retrieval requests unthresholded results so the agent can inspect scores
 - if top_score_before < score_threshold the agent triggers ingestion via data_fetcher
 - retries retrieval once and returns results; if top_score_after still < score_threshold,
   the trace includes low_confidence=True and both scores are returned alongside the results.
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

    Parameters
    ----------
    registry : ToolRegistry
        Registry containing 'retriever' and 'data_fetcher' tools.
    score_threshold : float
        Minimum top score required to consider retrieval high-quality. Default 0.65
    top_k_for_score : int
        The k used when computing top score (default 10).
    """

    def __init__(self, registry: ToolRegistry, score_threshold: float = 0.65, top_k_for_score: int = 10) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry instance")
        self.registry = registry
        self._retriever_tool_name = "retriever"
        self._data_fetcher_tool_name = "data_fetcher"
        self.score_threshold = float(score_threshold)
        self.top_k_for_score = int(top_k_for_score)

    def _log(self, msg: str) -> None:
        logger.info("[AGENT] %s", msg)

    def _heuristically_extract_game_name(self, query: str) -> str:
        """
        Heuristic extraction of a game name from a question/query.
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

        # 3) tell me about / info about / details about / what is
        m = re.search(r'(?:tell me about|info(?:rmation)? about|details about|what is|what\'s)\s+(.+?)[\?\.]?$', q, flags=re.I)
        if m:
            return m.group(1).strip().strip('?"\'')

        # 4) look for "for <Game>" or "about <Game>"
        m = re.search(r'(?:for|about)\s+([A-Z][\w\'\s:&-]{2,})', q)
        if m:
            cand = m.group(1).strip()
            return cand.strip('?"\'')

        # 5) capitalized run of words (e.g., "Far Cry 5")
        capitalized_runs = re.findall(r'(?:[A-Z][\w\'’-]+(?:\s+[A-Z][\w\'’-]+)+)', q)
        if capitalized_runs:
            capitalized_runs.sort(key=lambda s: len(s), reverse=True)
            return capitalized_runs[0].strip()

        # 6) fallback to last 3 words
        tokens = re.findall(r"\w+'?\w+|\w+", q)
        if len(tokens) >= 3:
            return " ".join(tokens[-3:]).strip()
        elif tokens:
            return " ".join(tokens).strip()

        return ""

    def _top_score_from_results(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute the top per-chunk score from results defensively.
        """
        if not results:
            return 0.0
        top = 0.0
        for r in results:
            s = r.get("score")
            if s is None:
                # if no explicit score, attempt to read from _raw._additional.certainty/distance
                add = r.get("_raw", {}).get("_additional", {}) if isinstance(r.get("_raw"), dict) else {}
                s = None
                if isinstance(add, dict):
                    if add.get("certainty") is not None:
                        try:
                            s = float(add.get("certainty"))
                        except Exception:
                            s = None
                    elif add.get("distance") is not None:
                        try:
                            d = float(add.get("distance"))
                            if d and d > 0 and d != float("inf"):
                                s = 1.0 / (1.0 + d)
                        except Exception:
                            s = None
            try:
                sval = float(s) if s is not None else 0.0
            except Exception:
                sval = 0.0
            if sval > top:
                top = sval
        return float(top)

    def run(self, query: str, k: int = 10) -> Dict[str, Any]:
        """
        Execute the agent loop with score-threshold decisioning.

        Returns a trace dict with:
          - status: "success" | "success_after_fetch" | "low_confidence" | "failure" | "error"
          - steps: list of steps executed
          - results: final list of chunks (may be empty)
          - top_score_before, top_score_after
          - did_fetch: bool
          - low_confidence: bool
        """
        steps: List[str] = []
        final_results: List[Dict[str, Any]] = []
        status = "failure"
        top_score_before = 0.0
        top_score_after = 0.0
        did_fetch = False
        low_confidence = False

        self._log("Retrieving initial results (unthresholded) to evaluate top score...")
        steps.append("retriever")
        try:
            retriever_tool: Tool = self.registry.get(self._retriever_tool_name)
        except Exception as e:
            msg = f"Retriever tool not found in registry: {e}"
            logger.exception(msg)
            return {"status": "error", "error": msg, "steps": steps, "results": []}

        # Request unthresholded results so we can inspect score distribution
        try:
            results = retriever_tool.execute({"query": query, "k": self.top_k_for_score, "similarity_threshold": None})
            if not isinstance(results, list):
                logger.warning("Retriever returned non-list result; coercing to list")
                results = list(results) if results is not None else []
        except Exception as e:
            logger.exception("Retriever execution failed: %s", e)
            results = []

        top_score_before = self._top_score_from_results(results)
        self._log(f"Top score before decision: {top_score_before:.4f} (threshold={self.score_threshold:.4f})")

        # If we already have sufficiently confident results, return top-k (user-requested k)
        if results and top_score_before >= self.score_threshold:
            self._log(f"Retriever returned high-confidence top score {top_score_before:.4f}. Returning top {k} results.")
            # Get final top-k (may re-run with threshold if desired; here we return existing results trimmed to k)
            final_results = results[:k]
            status = "success"
            return {
                "status": status,
                "steps": steps,
                "results": final_results,
                "top_score_before": top_score_before,
                "top_score_after": top_score_after,
                "did_fetch": did_fetch,
                "low_confidence": low_confidence,
            }

        # Otherwise, we treat as a miss and attempt to fetch external data
        self._log("Low retrieval hits / low confidence. Attempting to fetch external data.")
        extracted_name = self._heuristically_extract_game_name(query)
        if not extracted_name:
            self._log("Could not heuristically extract a game name from the query. Will attempt fetch with the query string.")
            extracted_name = query

        # Call data_fetcher tool if available
        try:
            data_fetcher_tool: Tool = self.registry.get(self._data_fetcher_tool_name)
            steps.append("data_fetcher")
            did_fetch = True
            self._log(f"Invoking data_fetcher to ingest data for: {extracted_name}")
            try:
                # pass through optional hints (agent might pass min_char_length to reduce noise)
                fetch_args = {"game_name": extracted_name}
                # If we want, we could pass min_char_length here; keep minimal for now.
                fetch_res = data_fetcher_tool.execute(fetch_args)
                if isinstance(fetch_res, dict):
                    fetch_success = bool(fetch_res.get("success", False))
                    self._log(f"data_fetcher finished: success={fetch_success}")
                else:
                    self._log("data_fetcher returned non-dict response; continuing to retry retrieval.")
            except Exception as fe:
                logger.exception("DataFetcherTool execution failed: %s", fe)
                self._log("DataFetcherTool raised an exception; proceeding to retry retrieval anyway.")
        except KeyError:
            self._log("No data_fetcher tool registered; skipping ingestion step.")
            did_fetch = False
        except Exception as e:
            logger.exception("Failed to access data_fetcher tool: %s", e)
            self._log("Proceeding to retry retrieval despite missing data_fetcher.")
            did_fetch = False

        # Retry retrieval once more (unthresholded to measure score)
        self._log("Retrying retrieval after ingestion attempt (unthresholded)...")
        steps.append("retriever_retry")
        try:
            retry_results = retriever_tool.execute({"query": query, "k": self.top_k_for_score, "similarity_threshold": None})
            if not isinstance(retry_results, list):
                retry_results = list(retry_results) if retry_results is not None else []
        except Exception as e:
            logger.exception("Retry retriever execution failed: %s", e)
            retry_results = []

        top_score_after = self._top_score_from_results(retry_results)
        self._log(f"Top score after fetch: {top_score_after:.4f}")

        # Decide final outcome
        if retry_results and top_score_after >= self.score_threshold:
            status = "success_after_fetch"
            final_results = retry_results[:k]
            self._log(f"Retry retrieved high-confidence results (top_score={top_score_after:.4f}). Returning top {k}.")
        elif retry_results and len(retry_results) > 0:
            # Low confidence but return best hits anyway
            status = "low_confidence"
            final_results = retry_results[:k]
            low_confidence = True
            self._log("Retry returned results but top score remains below threshold; returning top hits with low_confidence flag.")
        else:
            status = "failure"
            final_results = []
            self._log("Retry retrieval returned no results.")

        trace = {
            "status": status,
            "steps": steps,
            "results": final_results,
            "top_score_before": top_score_before,
            "top_score_after": top_score_after,
            "did_fetch": did_fetch,
            "low_confidence": low_confidence,
        }
        return trace
