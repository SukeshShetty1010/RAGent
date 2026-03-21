# api/routes.py
"""
API routing endpoint for RAGent.

Provides a programmatic interface to the RAG engine,
separate from the Streamlit UI.

This is a placeholder for future REST/FastAPI integration.
"""

from __future__ import annotations

from typing import Dict, Any, Optional


def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "ragent"}


def query(
    user_query: str,
    limit: int = 5,
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute a RAG query and return structured results.

    Args:
        user_query: The user's natural language question
        limit: Number of evidence chunks to retrieve
        options: Additional execution options

    Returns:
        Dict with final_answer, evidence, kpis, agent_decisions
    """
    from engine.execution_engine_streaming import StreamingRageEngine

    engine = StreamingRageEngine()
    try:
        result = engine.run_streaming(user_query, options=options)
        return {
            "final_answer": result.final_answer,
            "evidence": result.evidence,
            "kpis": result.kpis,
            "agent_decisions": result.agent_decisions,
        }
    finally:
        engine.close()
