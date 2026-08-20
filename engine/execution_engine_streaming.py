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
import threading
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
from agent.output_validator import validate_answer, is_refusal

from utils.observability import MetricsRegistry, ProfileBlock
from utils import tracing
from engine.contracts import (
    INSUFFICIENT_REFUSAL,
    GENERATION_FAILED,
    INTERNAL_ERROR,
    quality_report_dict,
)


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


class RequestCancelled(Exception):
    """Raised at a stage boundary when the client has gone away.

    Carries the name of the stage that was about to start. Caught by
    run_streaming() itself -- never propagates to the caller.
    """


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
        cancel_event: Optional[threading.Event] = None,
    ) -> StreamingResult:
        """
        Execute RAG pipeline with streaming callbacks.

        Args:
            query: User query string
            on_token_callback: Called for each generated token/chunk
            on_stage_callback: Called when pipeline stages start/complete
            options: Additional execution options
            cancel_event: When set, run_streaming stops at the next stage
                boundary (or the next generated chunk) instead of continuing.

        Returns:
            StreamingResult with full execution data
        """
        if self._closed:
            raise RuntimeError("Engine already closed")

        registry = MetricsRegistry.get()
        registry.reset()
        engine_start = time.perf_counter()

        final_answer = ""
        llm_ran = False
        llm_latency_ms: Optional[float] = None
        answer_model_used: Optional[str] = None
        cancelled = False
        refusal_mode: Optional[str] = None
        output_validation: Optional[Dict[str, Any]] = None
        capability_reason: Optional[str] = None

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

        def checkpoint(next_stage: str) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise RequestCancelled(next_stage)

        try:
            with tracing.trace_request(query), ProfileBlock("REQUEST_TOTAL"):
                try:
                    # ------------------------------------------------
                    # STEP 1: ROUTING
                    # ------------------------------------------------
                    checkpoint("routing")
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
                    checkpoint("strategy")
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
                    checkpoint("retrieval")
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

                    agent_decisions["quality"] = quality_report_dict(quality)
                    if pre_web_quality is not quality:
                        agent_decisions["quality_pre_web"] = quality_report_dict(pre_web_quality)

                    emit_stage("retrieval", "completed", {
                        "chunks_found": len(raw_chunks),
                        "merge_state": merge_state,
                        "quality": quality_status,
                    }, (time.perf_counter() - step_start) * 1000)

                    # ------------------------------------------------
                    # STEP 4: CAPABILITY ASSESSMENT
                    # ------------------------------------------------
                    checkpoint("capability")
                    emit_stage("capability", "started")
                    step_start = time.perf_counter()

                    capability = self.capability_assessor.assess(
                        intent_signals=decision.intent_signals,
                        evidence=raw_chunks,
                        quality=quality,
                    )

                    agent_decisions["answer_capability"] = capability.value
                    capability_reason = MetricsRegistry.get().last_label("capability_reason")

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
                    checkpoint("context_assembly")
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
                    checkpoint("prompt_construction")
                    emit_stage("prompt_construction", "started")
                    step_start = time.perf_counter()

                    prompt = self.prompt_manager.generate_prompt(
                        query=query,
                        chunks=assembled_chunks,
                        task=decision.task,
                        capability=capability,
                        capability_reason=capability_reason,
                    )

                    emit_stage("prompt_construction", "completed", {
                        "prompt_length": len(prompt)
                    }, (time.perf_counter() - step_start) * 1000)

                    # ------------------------------------------------
                    # STEP 7: LLM GENERATION (WITH STREAMING)
                    # ------------------------------------------------
                    if capability != AnswerCapability.INSUFFICIENT:
                        checkpoint("generation")
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
                                if cancel_event is not None and cancel_event.is_set():
                                    raise RequestCancelled("generation")

                            final_answer = "".join(accumulated).strip()
                            llm_ran = True
                            llm_latency_ms = (time.perf_counter() - step_start) * 1000.0

                        except RequestCancelled:
                            raise

                        except ImportError:
                            # Fall back to blocking generation
                            from llm.ragent_client import chat_completion_remote

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
                            final_answer = GENERATION_FAILED

                        if llm_ran:
                            from llm.ragent_client import answer_model

                            answer_model_used = answer_model()
                            tracing.record_generation(
                                model=answer_model_used,
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
                            try:
                                validation = validate_answer(
                                    final_answer,
                                    capability,
                                    context_chunks=assembled_chunks,
                                )
                                output_validation = validation.model_dump()
                                agent_decisions["output_validation"] = output_validation
                                MetricsRegistry.get().record(
                                    "output_validation",
                                    "valid" if validation.is_valid else "invalid",
                                )
                                MetricsRegistry.get().observe(
                                    "unmatched_citations",
                                    len(validation.unmatched_citations),
                                )
                                stage_data["output_validation"] = output_validation
                            except Exception as exc:
                                logger.warning(
                                    f"Answer validation failed (fail-soft, answer kept): {exc}"
                                )
                                MetricsRegistry.get().inc("output_validation_errors")

                        emit_stage(
                            "generation",
                            "completed" if llm_ran else "failed",
                            stage_data,
                            llm_latency_ms,
                        )
                    else:
                        # ------------------------------------------------
                        # STEP 7 (INSUFFICIENT): GENERATE THE REFUSAL
                        # ------------------------------------------------
                        # insufficient_prompt() (built above, in STEP 6) used
                        # to be discarded here in favor of a fixed string.
                        # Send it: the refusal names the query and why the
                        # evidence fell short, gated through is_refusal()
                        # before it ships. Any failure -- empty output, a
                        # citation slipping through, an exception -- falls
                        # back to the static constant.
                        checkpoint("generation")
                        emit_stage("generation", "started")
                        step_start = time.perf_counter()

                        refusal_mode = "static_fallback"

                        try:
                            from llm.ragent_client_streaming import chat_completion_streaming

                            accumulated = []
                            for chunk in chat_completion_streaming(
                                prompt,
                                max_tokens=200,
                                on_chunk=on_token_callback,
                            ):
                                accumulated.append(chunk)
                                if cancel_event is not None and cancel_event.is_set():
                                    raise RequestCancelled("generation")

                            candidate = "".join(accumulated).strip()
                            if is_refusal(candidate):
                                final_answer = candidate
                                refusal_mode = "generated"
                            else:
                                final_answer = INSUFFICIENT_REFUSAL

                        except RequestCancelled:
                            raise

                        except Exception as exc:
                            logger.warning(f"Refusal generation unavailable (fail-soft): {exc}")
                            final_answer = INSUFFICIENT_REFUSAL

                        emit_stage("generation", "skipped", {
                            "reason": "insufficient_capability",
                            "capability_reason": capability_reason,
                            "refusal_mode": refusal_mode,
                        }, (time.perf_counter() - step_start) * 1000)

                except RequestCancelled as exc:
                    cancelled = True
                    logger.info(f"Request cancelled by client before stage: {exc}")
                    emit_stage(str(exc), "cancelled", {"reason": "client_disconnected"})
                    MetricsRegistry.get().inc("requests_cancelled")

                finally:
                    # Inside the active trace window (unlike the old
                    # placement), so this is the one write that reaches
                    # Langfuse on every exit path -- normal, cancelled, or
                    # fatal (the outer except below re-raises through this
                    # finally before converting to INTERNAL_ERROR).
                    tracing.set_trace_attributes(
                        llm_ran=llm_ran,
                        output_validation=(output_validation or {}).get("is_valid"),
                        cancelled=cancelled,
                    )

        except Exception:
            logger.exception("Fatal execution error")
            final_answer = INTERNAL_ERROR
            emit_stage("error", "failed")

        # ----------------------------------------------------
        # KPI AGGREGATION
        # ----------------------------------------------------
        engine_latency_ms = (time.perf_counter() - engine_start) * 1000.0

        kpis: Dict[str, Any] = {
            "engine_latency_ms": round(engine_latency_ms, 2),
            "llm_ran": llm_ran,
            "llm_latency_ms": round(llm_latency_ms, 2) if llm_latency_ms is not None else None,
            "quality_status": quality_status,
            "confidence_score": round(confidence_score, 4),
            "answer_capability": capability.value,
            "retrieved_chunks": len(assembled_chunks),
            "task_success": bool(
                llm_ran and capability != AnswerCapability.INSUFFICIENT
            ),
            "answer_model": answer_model_used,
            "cancelled": cancelled,
            "refusal_mode": refusal_mode,
        }

        raw_metrics = registry.generate_report()

        prompt_tokens_dist = raw_metrics["distributions"].get("llm_prompt_tokens")
        completion_tokens_dist = raw_metrics["distributions"].get("llm_completion_tokens")
        cost_dist = raw_metrics["distributions"].get("llm_cost_usd")
        kpis["prompt_tokens"] = int(prompt_tokens_dist["avg"] * prompt_tokens_dist["count"]) if prompt_tokens_dist else None
        kpis["completion_tokens"] = int(completion_tokens_dist["avg"] * completion_tokens_dist["count"]) if completion_tokens_dist else None
        kpis["cost_usd"] = round(cost_dist["avg"] * cost_dist["count"], 8) if cost_dist else None

        # An answer is complete only if generation ran AND the provider
        # said why it stopped AND that reason was "stop". Two distinct
        # failures hide here, both observed live 2026-08-17:
        #   - "length": hit max_tokens, answer cut mid-sentence.
        #   - no finish_reason at all: the provider closed the SSE stream
        #     mid-generation without terminating it. Gemini's free tier
        #     does this when the request quota is exhausted, and because
        #     tokens had already been yielded the client cannot fall back
        #     to Groq without duplicating the prefix. The stream simply
        #     ends and a half-sentence reaches the UI.
        # Neither raises, so without this the pipeline reports
        # task_success on a severed answer.
        finish_reasons = raw_metrics["categoricals"].get("llm_finish_reason", {})
        finish_reason = (
            max(finish_reasons, key=finish_reasons.get) if finish_reasons else None
        )
        kpis["finish_reason"] = finish_reason
        kpis["answer_truncated"] = bool(llm_ran and finish_reason != "stop")

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
