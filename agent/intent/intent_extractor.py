"""
agent/intent/intent_extractor.py

This module performs feature extraction using a configurable pattern registry.

It translates raw user queries into a set of canonical intent signals defined
in `agent.intent.intent_signals.IntentSignal`.

This module performs feature extraction, not decision making.
"""

from __future__ import annotations

import re
from typing import Set, Dict, List

from agent.intent.intent_signals import IntentSignal


class IntentSignalExtractor:
    """
    Deterministic intent signal extractor based on a pattern registry.

    Responsibilities:
    - Detect ALL intent signals present in a query (monotonic accumulation)
    - Use declarative regex-based patterns
    - Remain stateless, fail-soft, and O(N)

    Non-responsibilities:
    - No prioritization or routing
    - No task or retrieval decisions
    """

    # ------------------------------------------------------------------
    # Declarative Intent → Regex Pattern Registry
    # ------------------------------------------------------------------
    # NOTE:
    # - Patterns are compiled once at class load time
    # - All signals are treated uniformly by the extraction loop
    # - Word boundaries and IGNORECASE are enforced to reduce false positives
    # ------------------------------------------------------------------

    _SIGNAL_PATTERNS: Dict[IntentSignal, List[re.Pattern]] = {

        # --------------------------------------------------------------
        # COMPARISON (FIXED: semantic + structural comparison)
        # --------------------------------------------------------------
        IntentSignal.COMPARISON: [
            # explicit comparison terms
            re.compile(r"\bvs\b", re.IGNORECASE),
            re.compile(r"\bversus\b", re.IGNORECASE),
            re.compile(r"\bcompare\b", re.IGNORECASE),
            re.compile(r"\bcomparison\b", re.IGNORECASE),
            re.compile(r"\bdifference(s)?\b", re.IGNORECASE),
            re.compile(r"\bsimilarit(y|ies)\b", re.IGNORECASE),
            re.compile(r"\bcontrast\b", re.IGNORECASE),
            re.compile(r"\bbetter\s+than\b", re.IGNORECASE),

            # 🔑 CRITICAL FIX:
            # semantic relational structure: "between X and Y"
            re.compile(
                r"\bbetween\s+.+?\s+and\s+.+",
                re.IGNORECASE,
            ),
        ],

        # --------------------------------------------------------------
        # LISTICLE
        # --------------------------------------------------------------
        IntentSignal.LISTICLE: [
            re.compile(r"\btop\s+\d+\b", re.IGNORECASE),
            re.compile(r"\bbest\b", re.IGNORECASE),
            re.compile(r"\bworst\b", re.IGNORECASE),
            re.compile(r"\branking\b", re.IGNORECASE),
            re.compile(r"\brecommendations\b", re.IGNORECASE),
            re.compile(r"\blist\s+of\b", re.IGNORECASE),
        ],

        # --------------------------------------------------------------
        # FACTUAL
        # --------------------------------------------------------------
        IntentSignal.FACTUAL: [
            re.compile(r"\bwhat\s+is\b", re.IGNORECASE),
            re.compile(r"\bwho\s+is\b", re.IGNORECASE),
            re.compile(r"\bwhen\s+was\b", re.IGNORECASE),
            re.compile(r"\brelease\s+date\b", re.IGNORECASE),
            re.compile(r"\bspecs\b", re.IGNORECASE),
            re.compile(r"\bdeveloper\b", re.IGNORECASE),
        ],

        # --------------------------------------------------------------
        # TEMPORAL
        # --------------------------------------------------------------
        IntentSignal.TEMPORAL: [
            re.compile(r"\blatest\b", re.IGNORECASE),
            re.compile(r"\brecent\b", re.IGNORECASE),
            re.compile(r"\bcurrent\b", re.IGNORECASE),
            re.compile(r"\bhistory\s+of\b", re.IGNORECASE),
            re.compile(r"\btimeline\b", re.IGNORECASE),
            re.compile(r"\bupdate\b", re.IGNORECASE),
        ],

        # --------------------------------------------------------------
        # EXPLANATORY
        # --------------------------------------------------------------
        IntentSignal.EXPLANATORY: [
            re.compile(r"\bexplain\b", re.IGNORECASE),
            re.compile(r"\bhow\s+does\b", re.IGNORECASE),
            re.compile(r"\bwhy\s+is\b", re.IGNORECASE),
            re.compile(r"\breason\s+for\b", re.IGNORECASE),
        ],
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, query: str) -> Set[IntentSignal]:
        """
        Extract all intent signals present in the query.

        Args:
            query: Raw user query string.

        Returns:
            A set of IntentSignal values. Returns an empty set on
            invalid input or no matches.
        """
        if not isinstance(query, str):
            return set()

        try:
            normalized = self._normalize(query)
            detected: Set[IntentSignal] = set()

            for signal, patterns in self._SIGNAL_PATTERNS.items():
                for pattern in patterns:
                    if pattern.search(normalized):
                        detected.add(signal)
                        break  # monotonic: move to next signal

            return detected

        except Exception:
            # Fail-soft guarantee
            return set()

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(query: str) -> str:
        """
        Baseline normalization:
        - Lowercasing
        - Whitespace collapsing

        Punctuation handling is intentionally minimal to preserve
        word-boundary semantics in regex matching.
        """
        text = query.lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip()
