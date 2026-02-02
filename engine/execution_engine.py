# ============================================================
# engine/execution_engine.py
# Capability-Aware RAG Execution Engine (FINAL)
# ============================================================

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional, TypedDict

from agent.task_router import TaskRouter
from retriever.strategy_selector import StrategySelector
from retriever.orchestrator import RetrievalOrchestrator
from agent.capability.capability_assessor import CapabilityAssessor
from agent.capability.capability_types import AnswerCapability
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
    Capability-aware, API-first execution engine.

    Guarantees:
    - Deterministic control flow
    - Honest answer degradation
    - Correct intent routing
    - Full observability
    """

    def __init__(self) -> None:
        self.router = TaskRouter()
        self.strategy_selector = StrategySelector()
        self.orchestrator = RetrievalOrchestrator()
        self.capability_assessor = CapabilityAssessor()
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

        if self._closed:
            raise RuntimeError("Engine already closed")

        registry = MetricsRegistry.get()

        engine_start = time.perf_counter()

        final_answer = ""
        llm_ran = False
        llm_latency_ms: Optional[float] = None

        agent_decisions: Dict[str, Any] = {}
        assembled_chunks: List[Dict[str, Any]] = []

        capability: AnswerCapability = AnswerCapability.INSUFFICIENT
        quality_status = "unknown"
        confidence_score = 0.0

        try:
            # =================================================
            # ROOT REQUEST SPAN
            # =================================================
            with ProfileBlock("REQUEST_TOTAL"):

                # ------------------------------------------------
                # STEP 1: ROUTING (EXPLICIT VERSIONING)
                # ------------------------------------------------
                decision = self.router.route(
                    query,
                    intent_schema_version="v2",
                )

                agent_decisions["task"] = decision.task.value
                agent_decisions["intent_signals"] = [
                    s.value for s in decision.intent_signals
                ]
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
                    decision=decision,
                    config=config,
                )

                agent_decisions["merge_state"] = merge_state

                # QualityReport is an OBJECT (correct)
                quality_status = quality.status.value
                confidence_score = quality.confidence_score

                agent_decisions["quality"] = {
                    "status": quality.status.value,
                    "confidence_score": quality.confidence_score,
                    "has_temporal_signal": quality.has_temporal_signal,
                }

                # ------------------------------------------------
                # STEP 4: CAPABILITY ASSESSMENT
                # ------------------------------------------------
                capability = self.capability_assessor.assess(
                    intent_signals=decision.intent_signals,
                    evidence=raw_chunks,
                    quality=quality,
                )

                agent_decisions["answer_capability"] = capability.value

                # ------------------------------------------------
                # STEP 5: CONTEXT ASSEMBLY
                # ------------------------------------------------
                assembled_chunks = self.context_assembler.assemble(
                    raw_chunks,
                    decision.task,
                )

                # ------------------------------------------------
                # STEP 6: PROMPT CONSTRUCTION
                # ------------------------------------------------
                prompt = self.prompt_manager.generate_prompt(
                    query=query,
                    chunks=assembled_chunks,
                    task=decision.task,
                    capability=capability,
                )

                # ------------------------------------------------
                # STEP 7: LLM GENERATION (GUARDED)
                # ------------------------------------------------
                if capability != AnswerCapability.INSUFFICIENT:
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
                        logger.warning(
                            f"LLM unavailable (fail-soft): {exc}"
                        )
                        final_answer = (
                            "I could not generate a response at this time."
                        )
                else:
                    final_answer = (
                        "I don’t have enough reliable information "
                        "to answer this request safely."
                    )

        except Exception:
            logger.exception("Fatal execution error")
            final_answer = (
                "An internal error occurred while processing the request."
            )

        # ----------------------------------------------------
        # KPI AGGREGATION
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
            "answer_capability": capability.value,
            "retrieved_chunks": len(assembled_chunks),
            "task_success": bool(
                llm_ran and capability != AnswerCapability.INSUFFICIENT
            ),
        }

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
        if self._closed:
            return
        try:
            self.orchestrator.close()
        finally:
            self._closed = True

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    @staticmethod
    def _reset_metrics(registry: MetricsRegistry) -> None:
        registry._counters.clear()
        registry._distributions.clear()
        registry._categoricals.clear()
