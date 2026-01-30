# ============================================================
# retriever/orchestrator.py
# Intent-Aware Retrieval Orchestrator (QUALITY REPORT FIXED)
# ============================================================

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Iterable, Tuple

from retriever.strategy_selector import RetrievalConfiguration
from retriever.rag_retriever import RAGRetriever
from retriever.quality_gate import RetrievalQualityGate, QualityStatus
from agent.tools.web_search import WebSearchTool
from agent.task_router import TaskType, RouterDecision

from tests.observability import ProfileBlock, MetricsRegistry


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_ORCHESTRATOR")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Orchestrator
# ============================================================

class RetrievalOrchestrator:
    """
    Intent-aware, local-first retrieval orchestrator.

    Responsibilities:
    - Execute retrieval based on RetrievalConfiguration
    - Evaluate retrieval quality
    - Optionally augment with web data
    - NEVER judge answer feasibility (CapabilityAssessor owns that)
    """

    MAX_WEB_CHUNKS = 3

    def __init__(self) -> None:
        self.retriever = RAGRetriever()
        self.quality_gate = RetrievalQualityGate()

        try:
            self.web_tool = WebSearchTool()
        except Exception as exc:
            logger.warning(f"WebSearchTool unavailable: {exc}")
            self.web_tool = None

    # =========================================================
    # Public API
    # =========================================================

    def run(
        self,
        *,
        query: str,
        decision: RouterDecision,
        config: RetrievalConfiguration,
    ) -> Tuple[List[Dict[str, Any]], str, Any]:
        """
        Execute retrieval according to routing + strategy.

        Returns:
            chunks
            merge_state
            quality_report (QualityReport object)
        """

        task = decision.task

        with ProfileBlock("Retrieval"):

            # -------------------------------------------------
            # STEP 1: LOCAL RETRIEVAL
            # -------------------------------------------------

            if config.use_query_decomposition:
                with ProfileBlock("QueryDecomposition"):
                    local_chunks = self._execute_comparison(query, config)

            elif config.use_window_expansion:
                with ProfileBlock("LocalVectorSearch"):
                    local_chunks = self._execute_listicle(query, config)

            else:
                with ProfileBlock("LocalVectorSearch"):
                    local_chunks = self._execute_factual(query, config)

            # -------------------------------------------------
            # STEP 2: QUALITY EVALUATION
            # -------------------------------------------------

            with ProfileBlock("QualityGate"):
                quality_report = self.quality_gate.evaluate(
                    query=query,
                    task=task,
                    chunks=local_chunks,
                )

            merge_state = "LOCAL_ONLY"
            web_chunks: List[Dict[str, Any]] = []

            # -------------------------------------------------
            # STEP 3: SHOULD WE ATTEMPT WEB?
            # -------------------------------------------------

            should_attempt_web = False

            if quality_report.status in {
                QualityStatus.QUALITY_EMPTY,
                QualityStatus.QUALITY_WEAK,
            }:
                should_attempt_web = True

            elif config.allow_web_fallback and quality_report.has_temporal_signal:
                should_attempt_web = True

            # -------------------------------------------------
            # STEP 4: WEB AUGMENTATION (OBSERVABLE)
            # -------------------------------------------------

            if should_attempt_web and self.web_tool:

                trigger_reason = None

                if quality_report.status == QualityStatus.QUALITY_EMPTY:
                    trigger_reason = "quality_empty"

                elif quality_report.status == QualityStatus.QUALITY_WEAK:
                    trigger_reason = "quality_weak"

                elif quality_report.has_temporal_signal:
                    trigger_reason = "temporal_signal"

                if trigger_reason:
                    MetricsRegistry.get().record(
                        "web_trigger_reason",
                        trigger_reason,
                    )

                MetricsRegistry.get().inc("web_search_triggers")

                with ProfileBlock("WebSearch"):
                    logger.info("🌍 Attempting Web augmentation")
                    merge_state = "LOCAL_WEB_ATTEMPTED"

                    raw_web_chunks = self.web_tool.search(
                        query=query,
                        max_results=self.MAX_WEB_CHUNKS,
                    )

                    refined_web_chunks = self._refine_web_results(
                        raw_web_chunks
                    )

                    if refined_web_chunks:
                        web_chunks = refined_web_chunks[: self.MAX_WEB_CHUNKS]
                        merge_state = "LOCAL_PLUS_WEB"

            logger.info(f"[ORCHESTRATOR] Merge State: {merge_state}")

            # -------------------------------------------------
            # STEP 5: FINAL MERGE
            # -------------------------------------------------

            with ProfileBlock("ChunkMerge"):

                if not local_chunks and web_chunks:
                    return web_chunks, merge_state, quality_report

                if web_chunks:
                    return (
                        local_chunks + web_chunks,
                        merge_state,
                        quality_report,
                    )

                return local_chunks, merge_state, quality_report

    # =========================================================
    # Web refinement
    # =========================================================

    def _refine_web_results(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        refined: List[Dict[str, Any]] = []

        for c in chunks:
            score = float(c.get("score", 0.0))
            title = (c.get("source_title") or "").lower()
            content = (c.get("content") or "").lower()
            url = (c.get("source_url") or "").lower()

            if score < 0.5:
                continue

            if self.quality_gate.is_noise(title, content):
                continue

            if any(
                d in url
                for d in (
                    "steamcommunity.com",
                    "reddit.com",
                    "twitter.com",
                )
            ):
                if score <= 0.8:
                    continue

            refined.append(c)

        return refined

    # =========================================================
    # Local execution modes
    # =========================================================

    def _execute_factual(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        logger.info("Executing FACTUAL retrieval")
        return self.retriever.retrieve(query, limit=config.limit)

    def _execute_comparison(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        logger.info("Executing COMPARISON retrieval (query decomposition)")

        sub_queries = self._decompose_query(query)
        results: List[List[Dict[str, Any]]] = []

        for sq in sub_queries:
            with ProfileBlock("LocalVectorSearch"):
                chunks = self.retriever.retrieve(
                    sq, limit=config.limit
                )

            for c in chunks:
                c["retrieval_context"] = sq

            results.append(chunks)

        return self._merge_and_dedupe(results)

    def _execute_listicle(
        self,
        query: str,
        config: RetrievalConfiguration,
    ) -> List[Dict[str, Any]]:
        logger.info("Executing LISTICLE retrieval")
        chunks = self.retriever.retrieve(query, limit=config.limit)
        return sorted(chunks, key=lambda c: c.get("chunk_index", 0))

    # =========================================================
    # Helpers
    # =========================================================

    def _decompose_query(self, query: str) -> List[str]:
        parts = re.split(
            r"\bvs\b|\bversus\b|\band\b|\bcompare\b",
            query,
            flags=re.IGNORECASE,
        )
        cleaned = [p.strip() for p in parts if p.strip()]
        return cleaned if len(cleaned) >= 2 else [query]

    def _merge_and_dedupe(
        self,
        groups: Iterable[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        seen = set()
        merged: List[Dict[str, Any]] = []

        for g in groups:
            for c in g:
                key = hash(c.get("content", ""))
                if key not in seen:
                    seen.add(key)
                    merged.append(c)

        return merged

    # =========================================================
    # Cleanup
    # =========================================================

    def close(self) -> None:
        self.retriever.close()
