# ============================================================
# agent/context_assembler.py
# Context Assembly & Ranking (SAFE CONTEXT CAP — FULLY OBSERVABLE)
# ============================================================

from __future__ import annotations

import logging
import hashlib
import re
from typing import List, Dict, Any, DefaultDict, Set
from collections import defaultdict

from agent.task_router import TaskType
from tests.observability import ProfileBlock, MetricsRegistry

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_CONTEXT_ASSEMBLER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# Constants (MODULE-LEVEL)
# ============================================================

# Jaccard similarity threshold for redundancy rejection
JACCARD_REDUNDANCY_THRESHOLD = 0.85


# ============================================================
# Context Assembler
# ============================================================

class ContextAssembler:
    """
    Final structural processing layer before prompt injection.

    Responsibilities:
    - Coarse deduplication
    - Task-aware ordering
    - Fine-grained redundancy filtering (Jaccard)
    - Strict character-budget enforcement (atomic inclusion only)

    Fully instrumented for observability.
    """

    # --------------------------------------------------------
    # 🔑 CLASS-OWNED RUNTIME-PATCHABLE BUDGET
    # --------------------------------------------------------

    # HARD safety cap — may be temporarily expanded by E2E fallback
    MAX_CONTEXT_CHARS = 4000

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def assemble(
        self,
        chunks: List[Dict[str, Any]],
        task: TaskType,
    ) -> List[Dict[str, Any]]:
        """
        Assemble a coherent, ordered context for the given task.

        DESIGN GUARANTEES:
        - HARD character budget enforced
        - Chunks are indivisible (no truncation)
        - Ordering BEFORE budget enforcement
        - Redundancy filtering DURING budget assembly
        """

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
                deduped = self._deduplicate(chunks)

            MetricsRegistry.get().observe(
                "context_deduped_chunks", len(deduped)
            )

            # ------------------------------------------------
            # Step B: Task-aware Ordering
            # ------------------------------------------------
            with ProfileBlock("Ordering"):
                if task == TaskType.COMPARISON:
                    with ProfileBlock("OrderComparison"):
                        ordered = self._order_comparison(deduped)
                elif task == TaskType.LISTICLE:
                    with ProfileBlock("OrderListicle"):
                        ordered = self._order_listicle(deduped)
                else:
                    with ProfileBlock("OrderFactual"):
                        ordered = self._order_factual(deduped)

            # ------------------------------------------------
            # Step C: Source Labeling (Safety)
            # ------------------------------------------------
            for c in ordered:
                if not c.get("source_type"):
                    c["source_type"] = "local"

            # ------------------------------------------------
            # Step D: Safe Context Cap (CHAR BUDGET)
            # ------------------------------------------------
            with ProfileBlock("SafeContextCap"):
                final_chunks = self._apply_character_budget(ordered)

            MetricsRegistry.get().observe(
                "context_final_chunks", len(final_chunks)
            )

            MetricsRegistry.get().observe(
                "context_final_chars",
                sum(len(c.get("content", "")) for c in final_chunks),
            )

            return final_chunks

    # ========================================================
    # Deduplication
    # ========================================================

    def _deduplicate(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        normalized_map: Dict[str, Dict[str, Any]] = {}

        for c in chunks:
            content = (c.get("content") or "").strip()
            if not content:
                continue

            source_title = (c.get("source_title") or "").strip()
            norm = self._normalize_text(content)

            key_material = f"{source_title}::{norm}"
            key = self._hash_text(key_material)

            existing = normalized_map.get(key)
            if not existing or len(content) > len(existing.get("content", "")):
                normalized_map[key] = c

        candidates = list(normalized_map.values())
        survivors: List[Dict[str, Any]] = []

        for c in candidates:
            content = c.get("content", "")
            source = c.get("source_type", "local")

            is_substring = False
            for other in candidates:
                if other is c:
                    continue
                if (
                    other.get("source_type", "local") == source
                    and content in other.get("content", "")
                    and len(other.get("content", "")) > len(content)
                ):
                    is_substring = True
                    break

            if not is_substring:
                survivors.append(c)

        removed = len(chunks) - len(survivors)
        if removed > 0:
            logger.info(f"🧹 Deduplicated context: removed {removed} chunks")

        return survivors

    # ========================================================
    # Ordering Strategies
    # ========================================================

    def _order_comparison(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        general: List[Dict[str, Any]] = []

        for c in chunks:
            ctx = c.get("retrieval_context")
            if not ctx or ctx == "fallback":
                general.append(c)
            else:
                grouped[ctx].append(c)

        for items in grouped.values():
            items.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        general.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        ordered: List[Dict[str, Any]] = []

        for items in grouped.values():
            if items:
                ordered.append(items.pop(0))

        remaining: List[Dict[str, Any]] = []
        for items in grouped.values():
            remaining.extend(items)

        remaining.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        ordered.extend(general)
        ordered.extend(remaining)

        logger.info(
            f"🧩 Assembling context for COMPARISON: "
            f"{len(grouped)} entities, fairness enforced"
        )

        return ordered

    def _order_listicle(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        ordered, unordered = [], []

        for c in chunks:
            idx = c.get("chunk_index", -1)
            if isinstance(idx, int) and idx >= 0:
                ordered.append(c)
            else:
                unordered.append(c)

        ordered.sort(key=lambda c: c.get("chunk_index", 0))
        unordered.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        logger.info(
            f"📚 Assembling context for LISTICLE: "
            f"{len(ordered)} ordered + {len(unordered)} unordered chunks"
        )

        return ordered + unordered

    def _order_factual(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        logger.info(
            f"📌 Assembling context for FACTUAL: "
            f"{len(chunks)} chunks sorted by score"
        )

        return sorted(
            chunks,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )

    # ========================================================
    # Safe Context Cap (CORE FIX)
    # ========================================================

    def _apply_character_budget(
        self,
        ordered_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        final_chunks: List[Dict[str, Any]] = []
        used_chars = 0
        accepted_token_sets: List[Set[str]] = []

        char_cap = self.__class__.MAX_CONTEXT_CHARS

        for c in ordered_chunks:
            content = (c.get("content") or "").strip()
            if not content:
                continue

            content_len = len(content)

            if used_chars + content_len > char_cap:
                MetricsRegistry.get().inc("context_budget_rejections")
                continue

            if self._is_redundant(content, accepted_token_sets):
                MetricsRegistry.get().inc("context_redundant_rejections")
                continue

            final_chunks.append(c)
            used_chars += content_len
            accepted_token_sets.append(self._tokenize(content))

        return final_chunks

    # ========================================================
    # Redundancy Detection (Jaccard)
    # ========================================================

    def _is_redundant(
        self,
        content: str,
        existing_token_sets: List[Set[str]],
    ) -> bool:

        tokens = self._tokenize(content)
        if not tokens:
            return True

        for existing in existing_token_sets:
            intersection = tokens & existing
            union = tokens | existing
            if not union:
                continue

            similarity = len(intersection) / len(union)
            if similarity >= JACCARD_REDUNDANCY_THRESHOLD:
                return True

        return False

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
