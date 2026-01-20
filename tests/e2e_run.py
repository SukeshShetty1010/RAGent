# ============================================================
# tests/e2e_run.py
# End-to-End Black Box Recorder for RAG Pipeline
# (SAFE QUALITY FALLBACK — CHAR-BUDGET AWARE)
# ============================================================

from __future__ import annotations

import json
from typing import Dict, Any

from agent.task_router import TaskRouter, TaskType
from retriever.strategy_selector import StrategySelector
from retriever.orchestrator import RetrievalOrchestrator
from retriever.quality_gate import QualityStatus
from agent.context_assembler import ContextAssembler
from agent.prompt_manager import PromptManager
from llm.ragent_client import chat_completion_remote

from tests.observability import MetricsRegistry, ProfileBlock


# ------------------------------------------------------------
# ANSI Colors
# ------------------------------------------------------------

YELLOW = "\033[93m"
RESET = "\033[0m"


# ============================================================
# E2E Probe
# ============================================================

class E2EProbe:
    """
    Black-box probe that executes the full RAG pipeline
    and records behavioral + metric deltas.

    Includes:
    - Single-pass Safe Quality Fallback
    - Character-budget based auto-recovery (CURRENT ARCH)
    """

    def __init__(self) -> None:
        self.router = TaskRouter()
        self.strategy_selector = StrategySelector()
        self.orchestrator = RetrievalOrchestrator()
        self.context_assembler = ContextAssembler()
        self.prompt_manager = PromptManager()
        self.metrics = MetricsRegistry.get()

    # --------------------------------------------------------
    # Metrics Snapshotting
    # --------------------------------------------------------

    def _snapshot_metrics(self) -> Dict[str, Any]:
        report = self.metrics.generate_report()
        return {
            "counters": dict(report.get("counters", {})),
            "distributions": {
                k: v["count"]
                for k, v in report.get("distributions", {}).items()
            },
            "categoricals": {
                k: dict(v)
                for k, v in report.get("categoricals", {}).items()
            },
        }

    def _diff_metrics(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> Dict[str, Any]:
        def diff_dict(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
            return {
                k: b.get(k, 0) - a.get(k, 0)
                for k in set(a) | set(b)
                if b.get(k, 0) - a.get(k, 0) != 0
            }

        return {
            "counters_delta": diff_dict(
                before["counters"], after["counters"]
            ),
            "distributions_delta": diff_dict(
                before["distributions"], after["distributions"]
            ),
            "categoricals_delta": {
                name: diff_dict(
                    before["categoricals"].get(name, {}),
                    after["categoricals"].get(name, {}),
                )
                for name in set(before["categoricals"])
                | set(after["categoricals"])
                if diff_dict(
                    before["categoricals"].get(name, {}),
                    after["categoricals"].get(name, {}),
                )
            },
        }

    # --------------------------------------------------------
    # Suspicious Output Detection
    # --------------------------------------------------------

    def _is_suspiciously_short(
        self,
        task: TaskType,
        output: str,
    ) -> bool:
        """
        Detect silent under-answering for COMPARISON tasks.
        """

        if task != TaskType.COMPARISON:
            return False

        if not isinstance(output, str):
            return True

        if len(output) < 650:
            return True

        required_headers = [
            "**Overview**",
            "**Gameplay**",
        ]

        return any(h not in output for h in required_headers)

    # --------------------------------------------------------
    # Pipeline Execution
    # --------------------------------------------------------

    def run_pipeline(self, query: str) -> Dict[str, Any]:
        """
        Execute one full RAG pipeline run with auto-recovery.
        """

        metrics_before = self._snapshot_metrics()
        fallback_triggered = False

        with ProfileBlock("REQUEST_TOTAL"):

            # ---------------- Step 1: Routing ----------------
            decision = self.router.route(query)

            # ---------------- Step 2: Strategy ----------------
            retrieval_config = self.strategy_selector.select(decision)

            # ---------------- Step 3: Retrieval ----------------
            chunks, merge_state, quality_report = self.orchestrator.run(
                query=query,
                task=decision.task,
                config=retrieval_config,
            )

            # ---------------- Step 4: Context Assembly (Pass 1) ----------------
            assembled_chunks = self.context_assembler.assemble(
                chunks,
                decision.task,
            )

            final_context_chars = sum(
                len(c.get("content", "")) for c in assembled_chunks
            )

            # ---------------- Step 5: Prompt ----------------
            prompt = self.prompt_manager.generate_prompt(
                query=query,
                chunks=assembled_chunks,
                task=decision.task,
            )

            # ---------------- Step 6: LLM (Pass 1) ----------------
            response = chat_completion_remote(prompt)

            # ---------------- Step 7: Safe Quality Fallback ----------------
            if (
                self._is_suspiciously_short(decision.task, response)
                and quality_report.status == QualityStatus.QUALITY_OK
            ):
                print(
                    f"{YELLOW}⚠️ Fallback Triggered: Auto-Recovery engaged{RESET}"
                )
                fallback_triggered = True

                # 🔑 REAL control knob (current architecture)
                original_char_cap = (
                    self.context_assembler.__class__.MAX_CONTEXT_CHARS
                )

                try:
                    # Expand character budget (single retry)
                    self.context_assembler.__class__.MAX_CONTEXT_CHARS = int(
                        original_char_cap * 1.75
                    )

                    assembled_chunks = self.context_assembler.assemble(
                        chunks,
                        decision.task,
                    )

                    final_context_chars = sum(
                        len(c.get("content", "")) for c in assembled_chunks
                    )

                    prompt = self.prompt_manager.generate_prompt(
                        query=query,
                        chunks=assembled_chunks,
                        task=decision.task,
                    )

                    response = chat_completion_remote(prompt)

                finally:
                    # 🛡️ Restore global invariant
                    self.context_assembler.__class__.MAX_CONTEXT_CHARS = (
                        original_char_cap
                    )

        metrics_after = self._snapshot_metrics()
        metrics_delta = self._diff_metrics(metrics_before, metrics_after)

        # ------------------------------------------------
        # Structured Telemetry
        # ------------------------------------------------

        return {
            "query": query,
            "output_preview": response[:500],
            "behavior": {
                "task_type": decision.task.value,
                "retrieval_strategy": decision.retrieval_strategy,
                "merge_state": merge_state,
                "quality_status": quality_report.status.value,
                "confidence_score": quality_report.confidence_score,
                "temporal_signal": quality_report.has_temporal_signal,
                "fallback_triggered": fallback_triggered,
                "final_context_chars": final_context_chars,
            },
            "metrics_delta": metrics_delta,
            "final_metrics_snapshot": self.metrics.generate_report(),
        }


# ============================================================
# Manual Test Harness
# ============================================================

if __name__ == "__main__":
    probe = E2EProbe()
    QUERY = "Compare Assassin's Creed Valhalla vs Far Cry 5"

    print("\n🧪 E2E RUN WITH AUTO-RECOVERY\n" + "=" * 60)
    result = probe.run_pipeline(QUERY)
    print(json.dumps(result, indent=2))
