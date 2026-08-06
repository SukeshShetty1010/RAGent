"""
agent/intent/intent_signals.py

This module defines the canonical vocabulary for **user intent operations**
within the Intent Layer of the RAG system.

An `IntentSignal` represents *what the user is asking for* at a semantic level
(e.g., comparing options, requesting facts, seeking explanations), completely
decoupled from:

- routing decisions
- retrieval strategies
- system capabilities
- execution logic

This file is intentionally minimal and immutable. It serves as a shared
language between upstream intent extraction and downstream capability
assessment layers.
"""

from enum import Enum


class IntentSignal(str, Enum):
    """
    Canonical enumeration of user intent signals.

    These signals describe the *requested operation* expressed by the user,
    not how the system chooses to fulfill it.
    """

    COMPARISON = "comparison"
    LISTICLE = "listicle"
    FACTUAL = "factual"
    TEMPORAL = "temporal"
    EXPLANATORY = "explanatory"
