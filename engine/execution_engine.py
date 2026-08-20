# ============================================================
# engine/execution_engine.py
# Blocking facade over StreamingRageEngine
# ============================================================
"""
Historically a second, hand-maintained copy of the 7-step pipeline
(routing -> strategy -> retrieval -> capability -> context -> prompt ->
generation), kept "in sync" with engine/execution_engine_streaming.py
by hand. It drifted (AUDIT_TASKS.md T7): different refusal string,
different KPI shape, different exception handling around validation.

engine/execution_engine_streaming.py is now the single engine body.
This class exists only because evaluation/ and KPI/ import RageEngine
by this path -- and because evaluation should exercise the same code
production actually runs, not a stale duplicate. StreamingRageEngine.run()
already returns the 5-key dict this module's callers expect
(final_answer, agent_decisions, evidence, kpis, raw_metrics), so no
override is needed here.
"""

from __future__ import annotations

from engine.execution_engine_streaming import StreamingRageEngine


class RageEngine(StreamingRageEngine):
    """Blocking-style entry point: `run(query)` with no callbacks."""
