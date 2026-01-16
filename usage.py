"""
usage.py
========
Standalone validation script for WebSearchTool (Tavily).

Purpose:
- Verify TAVILY_API_KEY is loaded
- Verify Tavily search executes
- Verify results are normalized into EditorialChunk schema
"""

from __future__ import annotations

import logging
import sys

from agent.tools.web_search import WebSearchTool


# ============================================================
# Logging Setup
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger("WEB_SEARCH_USAGE")


# ============================================================
# Main
# ============================================================

def main() -> None:
    logger.info("=== WebSearchTool Usage Test ===")

    try:
        tool = WebSearchTool()
    except RuntimeError as exc:
        logger.error("❌ WebSearchTool initialization failed")
        logger.error(str(exc))
        return

    test_query = "Latest patch notes for Assassin's Creed Valhalla"

    logger.info(f"\nRunning test query:\n{test_query}\n")

    results = tool.search(test_query, max_results=5)

    if not results:
        logger.warning("⚠️ No results returned from Tavily")
        return

    logger.info(f"✅ Retrieved {len(results)} normalized web chunks\n")

    # --------------------------------------------------------
    # Print normalized results
    # --------------------------------------------------------
    for idx, chunk in enumerate(results, start=1):
        logger.info(f"--- Result {idx} ---")
        logger.info(f"source_title      : {chunk.get('source_title')}")
        logger.info(f"source_type       : {chunk.get('source_type')}")
        logger.info(f"source_url        : {chunk.get('source_url')}")
        logger.info(f"score             : {chunk.get('score')}")
        logger.info(f"chunk_index       : {chunk.get('chunk_index')}")
        logger.info(f"retrieval_context : {chunk.get('retrieval_context')}")
        logger.info(f"content (preview) : {chunk.get('content', '')[:200]}...")
        logger.info("-" * 80)


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    main()
