# ============================================================
# retriever/orchestrator.py
# Step 3: Retrieval Orchestrator (Execution Brain)
# ============================================================

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Iterable

from retriever.strategy_selector import RetrievalConfiguration
from retriever.rag_retriever import RAGRetriever


# ============================================================
# Logging Setup
# ============================================================

logger = logging.getLogger("RAG_ORCHESTRATOR")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Retrieval Orchestrator
# ============================================================

class RetrievalOrchestrator:
    """
    Executes retrieval strategies based on RetrievalConfiguration.

    This class:
    - Coordinates multiple retrieval calls
    - Merges and deduplicates results
    - Handles fail-soft fallback logic
    """

    def __init__(self) -> None:
        self.retriever = RAGRetriever()
        # Future hook for web search
        self.web_tool = None

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(self, query: str, config: RetrievalConfiguration) -> List[Dict[str, Any]]:
        """
        Execute retrieval based on configuration.
        """

        if config.use_query_decomposition:
            return self._execute_comparison(query, config)

        if config.use_window_expansion:
            return self._execute_listicle(query, config)

        if config.allow_web_fallback:
            return self._execute_open(query, config)

        return self._execute_factual(query, config)

    # --------------------------------------------------------
    # Execution Modes
    # --------------------------------------------------------

    def _execute_factual(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        """
        High-precision, single retrieval.
        """
        logger.info("Executing FACTUAL retrieval")

        return self.retriever.retrieve(query, limit=config.limit)

    def _execute_comparison(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        """
        Query decomposition for comparison tasks.
        """
        logger.info("Executing COMPARISON retrieval (query decomposition)")

        sub_queries = self._decompose_query(query)

        all_results: List[List[Dict[str, Any]]] = []

        for sub_query in sub_queries:
            logger.info(f"Retrieving for entity: '{sub_query}'")
            try:
                chunks = self.retriever.retrieve(
                    sub_query,
                    limit=config.limit,
                )
            except Exception as exc:
                logger.warning(f"Retrieval failed for '{sub_query}': {exc}")
                continue

            # Tag chunks with retrieval context
            for c in chunks:
                c["retrieval_context"] = sub_query

            all_results.append(chunks)

        return self._merge_and_dedupe(all_results)

    def _execute_listicle(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        """
        Window expansion simulation for listicle tasks.
        """
        logger.info("Executing LISTICLE retrieval (window expansion simulation)")

        chunks = self.retriever.retrieve(query, limit=config.limit)

        chunk_ids = [
            c.get("chunk_index")
            for c in chunks
            if c.get("chunk_index") is not None
        ]

        logger.info(
            f"Simulating window expansion for chunk IDs: {chunk_ids}"
        )
        # Future:
        # for chunk_id in chunk_ids:
        #     retriever.fetch_adjacent(chunk_id)

        return sorted(
            chunks,
            key=lambda c: c.get("chunk_index", 0),
        )

    def _execute_open(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        """
        Open retrieval with quality check and web fallback stub.
        """
        logger.info("Executing OPEN retrieval (web fallback enabled)")

        chunks = self.retriever.retrieve(query, limit=config.limit)

        if not chunks:
            avg_score = 0.0
        else:
            top_scores = [
                c.get("score", 0.0)
                for c in chunks[:3]
                if isinstance(c.get("score"), (int, float))
            ]
            avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        if not chunks or avg_score < 0.4:
            logger.warning(
                f"⚠️ Local retrieval quality low (Score: {avg_score:.2f}). "
                "Triggering Web Search (Stub)..."
            )
            for c in chunks:
                c["low_confidence"] = True
        else:
            logger.info("✅ Local retrieval quality sufficient.")

        return chunks

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def _decompose_query(self, query: str) -> List[str]:
        """
        Simple heuristic-based query decomposition.
        Fail-soft: returns original query if split fails.
        """
        parts = re.split(
            r"\bvs\b|\bversus\b|\bcompare\b|\band\b",
            query,
            flags=re.IGNORECASE,
        )

        cleaned = [p.strip() for p in parts if p.strip()]

        if len(cleaned) < 2:
            logger.warning("Query decomposition failed; using full query.")
            return [query]

        return cleaned

    def _merge_and_dedupe(
        self,
        results: Iterable[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate chunks using content hash.
        """
        seen = set()
        merged: List[Dict[str, Any]] = []

        for group in results:
            for chunk in group:
                content = chunk.get("content", "")
                key = hash(content)

                if key in seen:
                    continue

                seen.add(key)
                merged.append(chunk)

        return merged

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    def close(self) -> None:
        self.retriever.close()


# ============================================================
# Test Harness
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    orchestrator = RetrievalOrchestrator()

    test_configs = [
        RetrievalConfiguration(
            limit=3,
            use_window_expansion=False,
            use_query_decomposition=False,
            allow_web_fallback=False,
        ),
        RetrievalConfiguration(
            limit=5,
            use_window_expansion=False,
            use_query_decomposition=True,
            allow_web_fallback=False,
        ),
        RetrievalConfiguration(
            limit=10,
            use_window_expansion=True,
            use_query_decomposition=False,
            allow_web_fallback=False,
        ),
        RetrievalConfiguration(
            limit=5,
            use_window_expansion=False,
            use_query_decomposition=False,
            allow_web_fallback=True,
        ),
    ]

    test_query = "Compare Assassin's Creed Valhalla vs Far Cry 5"

    for idx, cfg in enumerate(test_configs, start=1):
        logger.info("\n" + "=" * 60)
        logger.info(f"TEST MODE {idx}")
        logger.info("=" * 60)

        results = orchestrator.run(test_query, cfg)
        logger.info(f"Result count: {len(results)}")

    orchestrator.close()
