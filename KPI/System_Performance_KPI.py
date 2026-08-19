# ============================================================
# tests/KPI/System_Performance_KPI.py
# RESUME-GRADE KPI: SYSTEM PERFORMANCE (SUCCESS-ONLY)
# ============================================================

from __future__ import annotations

import logging
from typing import Dict

from engine.execution_engine import RageEngine
from utils.observability import MetricsRegistry, PROFILE_PATH_SEP
from tests.evaluation_metrics import analyze_latency_profile
from tests.regression_suite import REGRESSION_VAULT, RegressionRunner

# ------------------------------------------------------------
# ANSI formatting
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"


class SystemPerformanceKPI:
    """
    INCLUDED SUCCESS METRICS ONLY:
    C — Regression Guard Coverage
    D — Latency Attribution (correctly normalized)
    E — Context Efficiency
    """

    def __init__(self) -> None:
        self.engine = RageEngine()
        self.registry = MetricsRegistry.get()

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        # ----------------------------------------------------
        # Reset observability
        # ----------------------------------------------------
        self.registry.reset()

        # ----------------------------------------------------
        # Execute representative request
        # ----------------------------------------------------
        self.engine.run("Compare Far Cry 5 vs Assassin’s Creed Valhalla")

        # ====================================================
        # C — Regression Guard Coverage
        # ====================================================

        runner = RegressionRunner()
        passed = 0
        total = len(REGRESSION_VAULT)

        for case in REGRESSION_VAULT:
            result = runner._run_case(case)
            if result is None:
                continue
            ok, _ = runner._compare(case.test_case, result)
            if ok:
                passed += 1

        # ====================================================
        # D — Latency Attribution (FINAL, CORRECT)
        # ====================================================

        metrics = analyze_latency_profile(self.registry.generate_report())

        stage_times: Dict[str, float] = {}

        for path, avg_ms in metrics.items():
            if not path.startswith(f"latency::REQUEST_TOTAL{PROFILE_PATH_SEP}"):
                continue

            parts = path.replace("latency::", "").split(PROFILE_PATH_SEP)

            # Accept only REQUEST_TOTAL -> STAGE
            if len(parts) != 2:
                continue

            stage = parts[1]

            if stage == "REQUEST_TOTAL":
                continue

            stage_times[stage] = avg_ms

        total_attributed = sum(stage_times.values())

        attribution: Dict[str, float] = {}
        for stage, ms in stage_times.items():
            if total_attributed > 0:
                attribution[stage] = round(
                    (ms / total_attributed) * 100.0, 2
                )

        # ====================================================
        # E — Context Efficiency
        # ====================================================

        input_chunks = sum(
            self.registry._distributions
            .get("context_input_chunks", {})
            .values
        )

        final_chunks = sum(
            self.registry._distributions
            .get("context_final_chunks", {})
            .values
        )

        noise_reduction_ratio = (
            round(1.0 - (final_chunks / input_chunks), 4)
            if input_chunks > 0 else 0.0
        )

        # ====================================================
        # Output
        # ====================================================

        print(f"\n{BOLD}{CYAN}RESUME-GRADE KPI: SYSTEM PERFORMANCE{RESET}\n")

        print(f"{BOLD}C — Regression Guard Coverage{RESET}")
        print(f"  • Permanent regressions enforced: {GREEN}{passed}/{total}{RESET}\n")

        print(f"{BOLD}D — Latency Attribution (% of total time){RESET}")
        for stage, pct in sorted(attribution.items(), key=lambda x: -x[1]):
            print(f"  • {stage}: {pct:.2f}%")
        print()

        print(f"{BOLD}E — Context Efficiency{RESET}")
        print(
            f"  • Context noise reduction ratio: "
            f"{GREEN}{noise_reduction_ratio:.2%}{RESET}"
        )
        print(
            f"  • Input chunks: {input_chunks}, "
            f"Final chunks used: {final_chunks}"
        )
        print()

        self.engine.close()


if __name__ == "__main__":
    SystemPerformanceKPI().run()
