# ============================================================
# agent/context_assembler.py
# Step 6: Context Assembly & Ranking
# ============================================================

from __future__ import annotations

import logging
import hashlib
from typing import List, Dict, Any, DefaultDict
from collections import defaultdict

from agent.task_router import TaskType

# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_CONTEXT_ASSEMBLER")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

# ============================================================
# Constants
# ============================================================

MAX_CONTEXT_CHUNKS = 15

# ============================================================
# Context Assembler
# ============================================================

class ContextAssembler:
    """
    Final structural processing layer before prompt injection.

    Responsibilities:
    - Content-based deduplication
    - Task-aware ordering
    - Source normalization
    - Budget control
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
        """

        if not chunks:
            return []

        # ----------------------------------------------------
        # Step A: Deduplication
        # ----------------------------------------------------
        deduped = self._deduplicate(chunks)

        # ----------------------------------------------------
        # Step B: Task-aware ordering
        # ----------------------------------------------------
        if task == TaskType.COMPARISON:
            ordered = self._order_comparison(deduped)
        elif task == TaskType.LISTICLE:
            ordered = self._order_listicle(deduped)
        else:
            ordered = self._order_factual(deduped)

        # ----------------------------------------------------
        # Step C: Source labeling
        # ----------------------------------------------------
        for c in ordered:
            if "source_type" not in c or not c["source_type"]:
                c["source_type"] = "local"

        # ----------------------------------------------------
        # Step D: Budget control
        # ----------------------------------------------------
        return ordered[:MAX_CONTEXT_CHUNKS]

    # ========================================================
    # Step A — Deduplication (FIXED)
    # ========================================================

    def _deduplicate(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Content-based deduplication with substring suppression.

        Dedup key:
        - source_title + normalized content
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
                # Keep richer (longer) version
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
        Group by retrieval_context:
        [General] → [Entity A] → [Entity B]
        """

        grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        general: List[Dict[str, Any]] = []

        for c in chunks:
            ctx = c.get("retrieval_context")
            if not ctx or ctx == "fallback":
                general.append(c)
            else:
                grouped[ctx].append(c)

        general.sort(key=lambda c: c.get("score", 0.0), reverse=True)

        ordered: List[Dict[str, Any]] = []
        ordered.extend(general)

        for entity, items in grouped.items():
            items.sort(key=lambda c: c.get("score", 0.0), reverse=True)
            ordered.extend(items)

        logger.info(
            f"🧩 Assembling context for COMPARISON: "
            f"Grouped {len(chunks)} chunks into {len(grouped)} entities"
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
            f"📌 Assembling context for {TaskType.FACTUAL.value.upper()}: "
            f"{len(chunks)} chunks sorted by score"
        )

        return sorted(
            chunks,
            key=lambda c: c.get("score", 0.0),
            reverse=True,
        )

    # ========================================================
    # Utilities
    # ========================================================

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
