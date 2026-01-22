# ============================================================
# engine/execution_engine.py
# RAG Execution Engine (API-First, UI-Safe)
# ============================================================

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional, TypedDict

from agent.task_router import TaskRouter
from retriever.strategy_selector import StrategySelector
from retriever.orchestrator import RetrievalOrchestrator
from agent.context_assembler import ContextAssembler
from agent.prompt_manager import PromptManager

from tests.observability import MetricsRegistry, ProfileBlock


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_EXECUTION_ENGINE")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Execution Result Contract
# ============================================================

class ExecutionResult(TypedDict):
    final_answer: str
    agent_decisions: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    kpis: Dict[str, Any]
    raw_metrics: Dict[str, Any]


# ============================================================
# RAG Execution Engine
# ============================================================

class RageEngine:
    """
    Callable, API-first execution engine for the RAG pipeline.

    Guarantees:
    - Silent (no stdout prints)
    - Deterministic return schema
    - Per-request metric isolation
    - Explicit resource lifecycle (close())
    """

    def __init__(self) -> None:
        self.router = TaskRouter()
        self.strategy_selector = StrategySelector()
        self.orchestrator = RetrievalOrchestrator()
        self.context_assembler = ContextAssembler()
        self.prompt_manager = PromptManager()
        self._closed = False

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def run(
        self,
        query: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> ExecutionResult:
        """
        Execute the full RAG pipeline for a single query and
        return a structured, JSON-ready result.
        """

        if self._closed:
            raise RuntimeError(
                "RageEngine.run() called after engine was closed"
            )

        options = options or {}
        registry = MetricsRegistry.get()

        # ----------------------------------------------------
        # Session Boundary (hard reset)
        # ----------------------------------------------------
        self._reset_metrics(registry)

        engine_start = time.perf_counter()

        final_answer: str = ""
        llm_ran = False
        llm_latency_ms: Optional[float] = None

        agent_decisions: Dict[str, Any] = {}
        assembled_chunks: List[Dict[str, Any]] = []
        quality_status: str = "unknown"
        confidence_score: float = 0.0

        try:
            # =================================================
            # ROOT REQUEST SPAN
            # =================================================
            with ProfileBlock("REQUEST_TOTAL"):

                # ------------------------------------------------
                # STEP 1: TASK ROUTING
                # ------------------------------------------------
                decision = self.router.route(query)

                agent_decisions["task"] = decision.task.value
                agent_decisions["routing_reason"] = decision.reason

                # ------------------------------------------------
                # STEP 2: STRATEGY SELECTION
                # ------------------------------------------------
                config = self.strategy_selector.select(decision)

                agent_decisions["retrieval_strategy"] = {
                    "limit": config.limit,
                    "use_query_decomposition": config.use_query_decomposition,
                    "use_window_expansion": config.use_window_expansion,
                    "allow_web_fallback": config.allow_web_fallback,
                }

                # ------------------------------------------------
                # STEP 3: RETRIEVAL
                # ------------------------------------------------
                raw_chunks, merge_state, quality = self.orchestrator.run(
                    query=query,
                    task=decision.task,
                    config=config,
                )

                agent_decisions["merge_state"] = merge_state

                quality_status = quality.status.value
                confidence_score = quality.confidence_score

                agent_decisions["quality"] = {
                    "status": quality_status,
                    "confidence_score": confidence_score,
                    "has_temporal_signal": quality.has_temporal_signal,
                }

                # ------------------------------------------------
                # STEP 4: CONTEXT ASSEMBLY
                # ------------------------------------------------
                assembled_chunks = self.context_assembler.assemble(
                    raw_chunks,
                    decision.task,
                )

                # ------------------------------------------------
                # STEP 5: PROMPT CONSTRUCTION
                # ------------------------------------------------
                prompt = self.prompt_manager.generate_prompt(
                    query=query,
                    chunks=assembled_chunks,
                    task=decision.task,
                )

                # ------------------------------------------------
                # STEP 6: LLM GENERATION (FAIL-SOFT)
                # ------------------------------------------------
                try:
                    from llm.ragent_client import chat_completion_remote

                    llm_start = time.perf_counter()
                    response = chat_completion_remote(prompt)
                    llm_latency_ms = (
                        time.perf_counter() - llm_start
                    ) * 1000.0

                    final_answer = response.strip()
                    llm_ran = True

                except Exception as exc:
                    logger.warning(f"LLM unavailable (fail-soft): {exc}")
                    final_answer = (
                        "I could not generate a response at this time."
                    )

        except Exception:
            logger.exception("Fatal execution error")
            final_answer = (
                "An internal error occurred while processing the request."
            )

        # ----------------------------------------------------
        # KPI Aggregation (per-request)
        # ----------------------------------------------------
        engine_latency_ms = (
            time.perf_counter() - engine_start
        ) * 1000.0

        kpis: Dict[str, Any] = {
            "engine_latency_ms": round(engine_latency_ms, 2),
            "llm_ran": llm_ran,
            "llm_latency_ms": round(llm_latency_ms, 2)
            if llm_latency_ms is not None
            else None,
            "quality_status": quality_status,
            "confidence_score": round(confidence_score, 4),
            "retrieved_chunks": len(assembled_chunks),
            "task_success": bool(
                llm_ran and quality_status != "quality_empty"
            ),
        }

        # ----------------------------------------------------
        # Final Metrics Snapshot
        # ----------------------------------------------------
        raw_metrics = registry.generate_report()

        return ExecutionResult(
            final_answer=final_answer,
            agent_decisions=agent_decisions,
            evidence=assembled_chunks,
            kpis=kpis,
            raw_metrics=raw_metrics,
        )

    # --------------------------------------------------------
    # Resource Lifecycle
    # --------------------------------------------------------

    def close(self) -> None:
        """
        Explicit teardown of all owned resources.

        MUST be called in:
        - CLI tools
        - Streamlit reruns
        - FastAPI shutdown events
        - CI / test harnesses
        """
        if self._closed:
            return

        try:
            if hasattr(self, "orchestrator"):
                self.orchestrator.close()
        finally:
            self._closed = True

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _reset_metrics(registry: MetricsRegistry) -> None:
        """
        Hard reset metric state to enforce per-request isolation.
        """
        registry._counters.clear()
        registry._distributions.clear()
        registry._categoricals.clear()
