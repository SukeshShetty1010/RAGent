"""
agent/tools/web_search.py
=========================

Web Search Tool (Tavily Integration) — FULLY OBSERVABLE

Purpose:
- Strict fallback retrieval when local evidence is weak or empty
- Queries Tavily Search API
- Normalizes results into EditorialChunk-compatible schema
"""

from __future__ import annotations

import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from tavily import TavilyClient

from tests.observability import ProfileBlock, MetricsRegistry

load_dotenv()

# ============================================================
# Logging Setup
# ============================================================

logger = logging.getLogger("RAG_WEB_SEARCH")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Web Search Tool
# ============================================================

class WebSearchTool:
    """
    Tavily-powered web search fallback tool.

    This tool MUST ONLY be used when the RetrievalQualityGate
    reports QUALITY_WEAK or QUALITY_EMPTY.
    """

    def __init__(self) -> None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY environment variable is missing. "
                "WebSearchTool cannot be initialized."
            )

        self.client = TavilyClient(api_key=api_key)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Tavily web search and normalize results
        into local EditorialChunk-compatible dictionaries.
        """

        with ProfileBlock("WebSearch"):
            logger.info(f"🌍 Triggering Tavily Search for: '{query}'")

            # ------------------------------------------------
            # Tavily API call
            # ------------------------------------------------
            try:
                with ProfileBlock("TavilyAPICall"):
                    response = self.client.search(
                        query=query,
                        max_results=max_results,
                        search_depth="advanced",
                        include_answer=False,
                        include_raw_content=False,
                    )
            except Exception as exc:
                logger.error(f"❌ Tavily search failed: {exc}")
                MetricsRegistry.get().inc("web_search_failures")
                return []

            raw_results = response.get("results", [])

            MetricsRegistry.get().observe(
                "web_results_returned", len(raw_results)
            )

            # ------------------------------------------------
            # Result normalization
            # ------------------------------------------------
            with ProfileBlock("WebResultNormalization"):
                normalized: List[Dict[str, Any]] = []

                for r in raw_results:
                    normalized.append(
                        {
                            "content": r.get("content", ""),
                            "source_title": r.get("title", ""),
                            "source_type": "web",
                            "source_url": r.get("url", ""),
                            "score": r.get("score", 0.0),
                            "chunk_index": -1,  # Non-sequential web content
                            "retrieval_context": "fallback",
                        }
                    )

            return normalized


# ============================================================
# Test Harness
# ============================================================

if __name__ == "__main__":
    if not os.environ.get("TAVILY_API_KEY"):
        print(
            "❌ TAVILY_API_KEY not set. "
            "Please export the key before running this test."
        )
    else:
        tool = WebSearchTool()

        test_query = "Latest patch notes for Assassin's Creed Valhalla"

        with ProfileBlock("REQUEST_TOTAL"):
            results = tool.search(test_query, max_results=3)

        print("\n=== NORMALIZED WEB SEARCH OUTPUT ===\n")
        for idx, chunk in enumerate(results, start=1):
            print(f"[{idx}]")
            for k, v in chunk.items():
                print(f"  {k}: {v}")
            print("-" * 60)

        print("\n--- OBSERVABILITY REPORT ---")
        print(
            MetricsRegistry.get().generate_report()
        )
