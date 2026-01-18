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
# Constants
# ============================================================

# HARD safety cap — never exceed this
MAX_CONTEXT_CHARS = 4000

# Jaccard similarity threshold for redundancy rejection
JACCARD_REDUNDANCY_THRESHOLD = 0.85

# ============================================================
# Context Assembler
# ============================================================

class ContextAssembler:
    """
    Final structural processing layer before prompt injection.

    Responsibilities:
    - Coarse deduplication (existing logic)
    - Task-aware ordering
    - Fine-grained redundancy filtering (Jaccard)
    - Strict character-budget enforcement (atomic inclusion only)

    Fully instrumented for observability.
    """

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

        IMPORTANT DESIGN NOTES:
        - We enforce a HARD character budget (MAX_CONTEXT_CHARS).
        - Chunks are indivisible units — no truncation allowed.
        - Ordering happens BEFORE budget enforcement.
        - Redundancy filtering happens DURING budget assembly.
        """

        with ProfileBlock("ContextAssembly"):

            if not chunks:
                return []

            MetricsRegistry.get().observe(
                "context_input_chunks", len(chunks)
            )

            # ------------------------------------------------
            # Step A: Coarse Deduplication (UNCHANGED)
            # ------------------------------------------------
            with ProfileBlock("Deduplication"):
                deduped = self._deduplicate(chunks)

            MetricsRegistry.get().observe(
                "context_deduped_chunks", len(deduped)
            )

            # ------------------------------------------------
            # Step B: Task-aware ordering
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
            # Step C: Source labeling (safety)
            # ------------------------------------------------
            for c in ordered:
                if "source_type" not in c or not c["source_type"]:
                    c["source_type"] = "local"

            # ------------------------------------------------
            # Step D: Safe Context Cap (CHAR BUDGET + REDUNDANCY)
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
    # Step A — Deduplication (UNCHANGED)
    # ========================================================

    def _deduplicate(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Content-based deduplication with substring suppression.

        This is a COARSE filter and must remain intact.
        """

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
            if not existing:
                normalized_map[key] = c
            else:
                if len(content) > len(existing.get("content", "")):
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
    # Step B — Ordering Strategies
    # ========================================================

    def _order_comparison(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        FAIR comparison ordering.

        Guarantees:
        1. Each entity contributes at least ONE chunk (if available).
        2. Remaining chunks are globally score-sorted.

        This prevents one entity from dominating early context.
        """

        grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        general: List[Dict[str, Any]] = []

        for c in chunks:
            ctx = c.get("retrieval_context")
            if not ctx or ctx == "fallback":
                general.append(c)
            else:
                grouped[ctx].append(c)

        # Sort each group by score
        for items in grouped.values():
            items.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        general.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        ordered: List[Dict[str, Any]] = []

        # --- Fairness pass: one per entity ---
        for entity, items in grouped.items():
            if items:
                ordered.append(items.pop(0))

        # --- Remaining chunks by global score ---
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
        """
        Ordered editorial chunks first, then unordered/web.
        """

        ordered: List[Dict[str, Any]] = []
        unordered: List[Dict[str, Any]] = []

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
        """
        Simple best-evidence-first ordering.
        """

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
    # Step D — Safe Context Cap (CORE LOGIC)
    # ========================================================

    def _apply_character_budget(
        self,
        ordered_chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Incrementally build context until MAX_CONTEXT_CHARS is reached.

        Rules:
        - Atomic inclusion only (no truncation).
        - Redundant chunks rejected via Jaccard similarity.
        - First-fit, order-preserving strategy.
        """

        final_chunks: List[Dict[str, Any]] = []
        used_chars = 0

        accepted_token_sets: List[Set[str]] = []

        for c in ordered_chunks:
            content = (c.get("content") or "").strip()
            if not content:
                continue

            content_len = len(content)

            # --- Hard safety gate ---
            if used_chars + content_len > MAX_CONTEXT_CHARS:
                MetricsRegistry.get().inc("context_budget_rejections")
                continue

            # --- Redundancy check ---
            if self._is_redundant(content, accepted_token_sets):
                MetricsRegistry.get().inc("context_redundant_rejections")
                continue

            # --- Accept chunk ---
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
        """
        Fine-grained redundancy detection using Jaccard similarity.

        This complements (not replaces) coarse deduplication.
        """

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
        """
        Lightweight tokenization:
        - lowercase
        - alphanumeric words only
        """
        return set(re.findall(r"[a-z0-9]+", text.lower()))
