# ============================================================
# engine/execution_engine_streaming.py
# Streaming-Capable RAG Execution Engine with Callbacks
# ============================================================
"""
Enhanced execution engine that supports streaming callbacks
for real-time UI updates during generation.

This engine provides the same guarantees as the base engine
but adds callback hooks for:
- Progress updates during pipeline stages
- Streaming token output during generation
- Real-time metric updates

Usage:
    engine = StreamingRageEngine()
    
    def on_token(token: str):
        print(token, end="", flush=True)
    
    def on_stage(stage: str, data: dict):
        print(f"Stage: {stage}")
    
    result = engine.run_streaming(
        query="...",
        on_token_callback=on_token,
        on_stage_callback=on_stage,
    )
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Any, Optional, Callable, Generator
from dataclasses import dataclass, field

from agent.task_router import TaskRouter
from retriever.strategy_selector import StrategySelector
from retriever.orchestrator import RetrievalOrchestrator
from agent.capability.capability_assessor import CapabilityAssessor
from agent.capability.capability_types import AnswerCapability
from agent.context_assembler import ContextAssembler
from agent.prompt_manager import PromptManager
from agent.output_validator import validate_answer

from utils.observability import MetricsRegistry, ProfileBlock
from utils import tracing


# ============================================================
# Logging
# ============================================================

logger = logging.getLogger("RAG_STREAMING_ENGINE")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


# ============================================================
# Streaming Result Types
# ============================================================

@dataclass
class StreamingStage:
    """Information about a pipeline stage."""
    name: str
    status: str  # "started", "completed", "failed"
    data: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


@dataclass
class StreamingResult:
    """Result object with streaming metadata."""
    final_answer: str
    agent_decisions: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    kpis: Dict[str, Any]
    raw_metrics: Dict[str, Any]
    stages: List[StreamingStage] = field(default_factory=list)


# ============================================================
# Streaming RAG Engine
# ============================================================

class StreamingRageEngine:
    """
    Streaming-capable RAG execution engine.
    
    Extends the base RageEngine with:
    - Stage progress callbacks
    - Token streaming callbacks
    - Real-time metric updates
    """

    def __init__(self, *, enable_web: bool = True) -> None:
        self.router = TaskRouter()
        self.strategy_selector = StrategySelector()
        self.orchestrator = RetrievalOrchestrator(enable_web=enable_web)
        self.capability_assessor = CapabilityAssessor()
        self.context_assembler = ContextAssembler()
        self.prompt_manager = PromptManager()
        self._closed = False

    # --------------------------------------------------------
    # Streaming API
    # --------------------------------------------------------

    def run_streaming(
        self,
        query: str,
        on_token_callback: Optional[Callable[[str], None]] = None,
        on_stage_callback: Optional[Callable[[StreamingStage], None]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> StreamingResult:
        """
        Execute RAG pipeline with streaming callbacks.
        
        Args:
            query: User query string
            on_token_callback: Called for each generated token/chunk
            on_stage_callback: Called when pipeline stages start/complete
            options: Additional execution options
        
        Returns:
            StreamingResult with full execution data
        """
        if self._closed:
            raise RuntimeError("Engine already closed")

        registry = MetricsRegistry.get()
        self._reset_metrics(registry)
        engine_start = time.perf_counter()

        final_answer = ""
        llm_ran = False
        llm_latency_ms: Optional[float] = None

        agent_decisions: Dict[str, Any] = {}
        assembled_chunks: List[Dict[str, Any]] = []
        stages: List[StreamingStage] = []

        capability: AnswerCapability = AnswerCapability.INSUFFICIENT
        quality_status = "unknown"
        confidence_score = 0.0

        def emit_stage(name: str, status: str, data: Dict = None, duration: float = None):
            stage = StreamingStage(
                name=name,
                status=status,
                data=data or {},
                duration_ms=duration
            )
            stages.append(stage)
            if on_stage_callback:
                on_stage_callback(stage)

        try:
            with tracing.trace_request(query), ProfileBlock("REQUEST_TOTAL"):

                # ------------------------------------------------
                # STEP 1: ROUTING
                # ------------------------------------------------
                emit_stage("routing", "started")
                step_start = time.perf_counter()
                
                decision = self.router.route(
                    query,
                    intent_schema_version="v2",
                )

                agent_decisions["task"] = decision.task.value
                agent_decisions["intent_signals"] = [
                    s.value for s in decision.intent_signals
                ]
                agent_decisions["routing_reason"] = decision.reason

                emit_stage("routing", "completed", {
                    "task": decision.task.value,
                    "signals": agent_decisions["intent_signals"],
                }, (time.perf_counter() - step_start) * 1000)

                # ------------------------------------------------
                # STEP 2: STRATEGY SELECTION
                # ------------------------------------------------
                emit_stage("strategy", "started")
                step_start = time.perf_counter()
                
                config = self.strategy_selector.select(decision)

                agent_decisions["retrieval_strategy"] = {
                    "limit": config.limit,
                    "use_query_decomposition": config.use_query_decomposition,
                    "use_window_expansion": config.use_window_expansion,
                    "allow_web_fallback": config.allow_web_fallback,
                }

                emit_stage("strategy", "completed", {
                    "config": agent_decisions["retrieval_strategy"]
                }, (time.perf_counter() - step_start) * 1000)

                # ------------------------------------------------
                # STEP 3: RETRIEVAL
                # ------------------------------------------------
                emit_stage("retrieval", "started")
                step_start = time.perf_counter()
                
                raw_chunks, merge_state, quality, web_decision, pre_web_quality = self.orchestrator.run(
                    query=query,
                    decision=decision,
                    config=config,
                )

                agent_decisions["merge_state"] = merge_state
                if web_decision is not None:
                    agent_decisions["web_search_decision"] = web_decision.model_dump()
                quality_status = quality.status.value
                confidence_score = quality.confidence_score

                def _report_dict(q):
                    return {
                        "status": q.status.value,
                        "confidence_score": q.confidence_score,
                        "has_temporal_signal": q.has_temporal_signal,
                        "max_relevance": q.max_relevance,
                        "entity_grounded": q.entity_grounded,
                        "evidence_count": q.evidence_count,
                    }

                agent_decisions["quality"] = _report_dict(quality)
                if pre_web_quality is not quality:
                    agent_decisions["quality_pre_web"] = _report_dict(pre_web_quality)

                emit_stage("retrieval", "completed", {
                    "chunks_found": len(raw_chunks),
                    "merge_state": merge_state,
                    "quality": quality_status,
                }, (time.perf_counter() - step_start) * 1000)

                # ------------------------------------------------
                # STEP 4: CAPABILITY ASSESSMENT
                # ------------------------------------------------
                emit_stage("capability", "started")
                step_start = time.perf_counter()
                
                capability = self.capability_assessor.assess(
                    intent_signals=decision.intent_signals,
                    evidence=raw_chunks,
                    quality=quality,
                )

                agent_decisions["answer_capability"] = capability.value

                tracing.set_trace_attributes(
                    task=decision.task.value,
                    answer_capability=capability.value,
                    quality_status=quality_status,
                    merge_state=merge_state,
                )

                emit_stage("capability", "completed", {
                    "capability": capability.value
                }, (time.perf_counter() - step_start) * 1000)

                # ------------------------------------------------
                # STEP 5: CONTEXT ASSEMBLY
                # ------------------------------------------------
                emit_stage("context_assembly", "started")
                step_start = time.perf_counter()
                
                assembled_chunks = self.context_assembler.assemble(
                    raw_chunks,
                    decision.task,
                )

                emit_stage("context_assembly", "completed", {
                    "chunks_assembled": len(assembled_chunks)
                }, (time.perf_counter() - step_start) * 1000)

                # ------------------------------------------------
                # STEP 6: PROMPT CONSTRUCTION
                # ------------------------------------------------
                emit_stage("prompt_construction", "started")
                step_start = time.perf_counter()
                
                prompt = self.prompt_manager.generate_prompt(
                    query=query,
                    chunks=assembled_chunks,
                    task=decision.task,
                    capability=capability,
                )

                emit_stage("prompt_construction", "completed", {
                    "prompt_length": len(prompt)
                }, (time.perf_counter() - step_start) * 1000)

                # ------------------------------------------------
                # STEP 7: LLM GENERATION (WITH STREAMING)
                # ------------------------------------------------
                if capability != AnswerCapability.INSUFFICIENT:
                    emit_stage("generation", "started")
                    step_start = time.perf_counter()
                    
                    try:
                        # Try streaming generation first
                        from llm.ragent_client_streaming import chat_completion_streaming
                        
                        accumulated = []
                        for chunk in chat_completion_streaming(
                            prompt,
                            on_chunk=on_token_callback
                        ):
                            accumulated.append(chunk)
                        
                        final_answer = "".join(accumulated).strip()
                        llm_ran = True

                    except ImportError:
                        # Fall back to blocking generation
                        from llm.ragent_client import chat_completion_remote, last_used_model
                        
                        llm_start = time.perf_counter()
                        response = chat_completion_remote(prompt)
                        llm_latency_ms = (time.perf_counter() - llm_start) * 1000.0
                        
                        final_answer = response.strip()
                        llm_ran = True
                        
                        # Simulate streaming for UI
                        if on_token_callback:
                            words = final_answer.split()
                            for word in words:
                                on_token_callback(word + " ")

                    except Exception as exc:
                        logger.warning(f"LLM unavailable (fail-soft): {exc}")
                        final_answer = (
                            "I could not generate a response at this time."
                        )

                    llm_latency_ms = (time.perf_counter() - step_start) * 1000.0

                    if llm_ran:
                        from llm.ragent_client import last_used_model

                        tracing.record_generation(
                            model=last_used_model(),
                            prompt=prompt,
                            output=final_answer,
                            prompt_tokens=int(MetricsRegistry.get().last("llm_prompt_tokens") or 0),
                            completion_tokens=int(MetricsRegistry.get().last("llm_completion_tokens") or 0),
                            cost_usd=MetricsRegistry.get().last("llm_cost_usd") or 0.0,
                            latency_ms=llm_latency_ms,
                        )

                    stage_data = {
                        "tokens_generated": len(final_answer.split()),
                        "llm_latency_ms": llm_latency_ms,
                    }

                    if llm_ran:
                        validation = validate_answer(
                            final_answer,
                            capability,
                            context_chunks=assembled_chunks,
                        )
                        agent_decisions["output_validation"] = (
                            validation.model_dump()
                        )
                        MetricsRegistry.get().record(
                            "output_validation",
                            "valid" if validation.is_valid else "invalid",
                        )
                        MetricsRegistry.get().observe(
                            "unmatched_citations",
                            len(validation.unmatched_citations),
                        )
                        stage_data["output_validation"] = (
                            validation.model_dump()
                        )

                    emit_stage(
                        "generation", "completed", stage_data, llm_latency_ms
                    )
                else:
                    final_answer = (
                        "I don't have enough reliable information "
                        "to answer this request safely."
                    )
                    emit_stage("generation", "skipped", {
                        "reason": "insufficient_capability"
                    })

        except Exception:
            logger.exception("Fatal execution error")
            final_answer = (
                "An internal error occurred while processing the request."
            )
            emit_stage("error", "failed")

        # ----------------------------------------------------
        # KPI AGGREGATION
        # ----------------------------------------------------
        engine_latency_ms = (time.perf_counter() - engine_start) * 1000.0

        kpis: Dict[str, Any] = {
            "engine_latency_ms": round(engine_latency_ms, 2),
            "llm_ran": llm_ran,
            "llm_latency_ms": round(llm_latency_ms, 2) if llm_latency_ms else None,
            "quality_status": quality_status,
            "confidence_score": round(confidence_score, 4),
            "answer_capability": capability.value,
            "retrieved_chunks": len(assembled_chunks),
            "task_success": bool(
                llm_ran and capability != AnswerCapability.INSUFFICIENT
            ),
        }

        raw_metrics = registry.generate_report()

        prompt_tokens_dist = raw_metrics["distributions"].get("llm_prompt_tokens")
        completion_tokens_dist = raw_metrics["distributions"].get("llm_completion_tokens")
        cost_dist = raw_metrics["distributions"].get("llm_cost_usd")
        kpis["prompt_tokens"] = int(prompt_tokens_dist["avg"] * prompt_tokens_dist["count"]) if prompt_tokens_dist else None
        kpis["completion_tokens"] = int(completion_tokens_dist["avg"] * completion_tokens_dist["count"]) if completion_tokens_dist else None
        kpis["cost_usd"] = round(cost_dist["avg"] * cost_dist["count"], 8) if cost_dist else None

        return StreamingResult(
            final_answer=final_answer,
            agent_decisions=agent_decisions,
            evidence=assembled_chunks,
            kpis=kpis,
            raw_metrics=raw_metrics,
            stages=stages,
        )

    # --------------------------------------------------------
    # Standard API (Compatible with base engine)
    # --------------------------------------------------------

    def run(
        self,
        query: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Standard execution (blocking, no streaming).
        
        Returns a dict compatible with the base engine.
        """
        result = self.run_streaming(query, options=options)
        
        return {
            "final_answer": result.final_answer,
            "agent_decisions": result.agent_decisions,
            "evidence": result.evidence,
            "kpis": result.kpis,
            "raw_metrics": result.raw_metrics,
        }

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
