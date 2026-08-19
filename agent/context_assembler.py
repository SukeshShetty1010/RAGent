# FILE: agent/context_assembler.py
# Context Assembly & Ranking (SAFE CONTEXT CAP — FULLY OBSERVABLE)

from __future__ import annotations

import logging
from typing import List, Dict, Any

from agent.task_router import TaskType
from agent.context_algorithms import (
    deduplicate_chunks,
    order_comparison,
    order_listicle,
    order_factual,
    apply_character_budget,
    is_fully_reranked,
)

from utils.observability import ProfileBlock, MetricsRegistry

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_CONTEXT_ASSEMBLER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# Context Assembler
# ============================================================

class ContextAssembler:
    """
    Final structural processing layer before prompt injection.

    Orchestration-only responsibilities:
    - Observability & metrics
    - Task-aware ordering selection
    - Delegation to pure context algorithms

    Guarantees:
    - HARD character budget enforced
    - Atomic chunk inclusion (no truncation)
    - Ordering BEFORE budget enforcement
    - Redundancy filtering DURING budget assembly
    """

    # HARD safety cap (prompt-manager independent)
    MAX_CONTEXT_CHARS = 4000

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def assemble(
        self,
        chunks: List[Dict[str, Any]],
        task: TaskType,
    ) -> List[Dict[str, Any]]:

        with ProfileBlock("ContextAssembly"):

            if not chunks:
                return []

            MetricsRegistry.get().observe(
                "context_input_chunks", len(chunks)
            )

            # ------------------------------------------------
            # Step A: Coarse Deduplication
            # ------------------------------------------------
            with ProfileBlock("Deduplication"):
                deduped = deduplicate_chunks(chunks)

            MetricsRegistry.get().observe(
                "context_deduped_chunks", len(deduped)
            )

            MetricsRegistry.get().record(
                "context_order_key",
                "rerank_score" if is_fully_reranked(deduped) else "score",
            )

            # ------------------------------------------------
            # Step B: Task-aware Ordering
            # ------------------------------------------------
            with ProfileBlock("Ordering"):
                if task == TaskType.COMPARISON:
                    ordered = order_comparison(deduped)
                elif task == TaskType.LISTICLE:
                    ordered = order_listicle(deduped)
                else:
                    ordered = order_factual(deduped)

            # ------------------------------------------------
            # Step C: Source Labeling (Safety)
            # ------------------------------------------------
            for c in ordered:
                if not c.get("source_type"):
                    c["source_type"] = "local"

            # ------------------------------------------------
            # Step D: HARD Context Budget Enforcement
            # ------------------------------------------------
            oversized = sum(
                1 for c in ordered
                if len(c.get("content") or "") > self.MAX_CONTEXT_CHARS
            )
            if oversized:
                logger.warning(
                    f"{oversized} chunk(s) exceed MAX_CONTEXT_CHARS="
                    f"{self.MAX_CONTEXT_CHARS} and were dropped whole"
                )

            with ProfileBlock("SafeContextCap"):
                final_chunks = apply_character_budget(
                    ordered_chunks=ordered,
                    char_cap=self.MAX_CONTEXT_CHARS,
                )

            MetricsRegistry.get().observe(
                "context_final_chunks", len(final_chunks)
            )
            MetricsRegistry.get().observe(
                "context_final_chars",
                sum(len(c.get("content", "")) for c in final_chunks),
            )

            dropped = len(ordered) - len(final_chunks)
            if dropped:
                MetricsRegistry.get().observe(
                    "context_chunks_dropped_by_budget", dropped
                )

            return final_chunks
