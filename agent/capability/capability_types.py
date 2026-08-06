"""
agent/capability/capability_types.py

This module defines the **Safe Production Boundary** of the RAG system.

`AnswerCapability` is a pure data contract that expresses whether the system
can *honestly and safely* fulfill a user request after intent detection and
evidence retrieval, but before response generation.

It encodes **system honesty**, not system ambition.

Key principles:
- No logic, no heuristics, no policy
- No coupling to intent, routing, or retrieval layers
- Stable, observable states for UI, metrics, and audits

This enum answers only one question:
    "Given what we know, how safely can we respond?"
"""

from enum import Enum


class AnswerCapability(str, Enum):
    """
    Canonical feasibility states for answer generation.

    Traffic-light semantics:
    - FULL         → Safe to answer completely with high integrity
    - PARTIAL      → Safe to answer partially with explicit degradation
    - INSUFFICIENT → Not safe to answer; requires refusal or deferral
    """

    FULL = "full"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
