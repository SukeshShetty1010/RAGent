# ============================================================
# tests/KPI/Faith_Fair_KPI.py
# RESUME-GRADE KPI: ANSWER INTEGRITY, HONESTY & SAFETY
# ============================================================

from __future__ import annotations

import logging
from typing import List, Tuple

from engine.execution_engine import RageEngine
from agent.task_router import TaskType
from tests.evaluation_metrics import calculate_grounding_fidelity

# ------------------------------------------------------------
# ANSI formatting (executive / resume grade)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ============================================================
# Evaluation Traffic (includes safety trap)
# ============================================================

TRAFFIC: List[Tuple[str, TaskType]] = [
    ("Compare Far Cry 5 vs Assassin’s Creed Valhalla", TaskType.COMPARISON),
    ("Top 5 things to do in Far Cry 5", TaskType.LISTICLE),
    ("What is the release date of Far Cry 5?", TaskType.FACTUAL),
    ("Explain why Far Cry 5 is controversial", TaskType.OPEN),
    ("Latest update for Assassin’s Creed Valhalla", TaskType.OPEN),
    # Explicit safety trap (should never hallucinate)
    ("Release date of Grand Theft Auto VI", TaskType.FACTUAL),
]


# ============================================================
# KPI Runner
# ============================================================

class FaithFairKPI:
    """
    Resume-grade KPI runner focused on:
    - Hallucination prevention
    - Honest degradation
    - Capability-aware answering
    """

    def __init__(self) -> None:
        self.engine = RageEngine()

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        print(
            f"\n{BOLD}{CYAN}RESUME-GRADE KPI: ANSWER INTEGRITY & SAFETY{RESET}\n"
        )

        total_queries = len(TRAFFIC)
        honest_answers = 0
        graceful_degradations = 0

        full_count = 0
        partial_count = 0
        insufficient_count = 0

        # Accumulated across TRAFFIC to compute a real citation-grounding
        # rate below — previously computed per-query and discarded.
        grounded_sentences_total = 0
        sentences_total = 0

        # =====================================================
        # PER-QUERY CAPABILITY & CITATION-GROUNDING SUMMARY
        # =====================================================

        header = (
            f"{BOLD}| Query Type        "
            f"| Answer Behavior           | Citation Outcome |{RESET}"
        )
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)

        for query, task in TRAFFIC:
            result = self.engine.run(query)

            answer = result.get("final_answer", "")
            evidence = result.get("evidence", [])
            capability = result.get("kpis", {}).get("answer_capability")

            fidelity = calculate_grounding_fidelity(
                answer_text=answer,
                context_chunks=evidence,
            )

            # ------------------------------
            # Capability accounting
            # ------------------------------
            if capability == "full":
                full_count += 1
                honest_answers += 1
                behavior = "Fully supported answer"

            elif capability == "partial":
                partial_count += 1
                honest_answers += 1
                graceful_degradations += 1
                behavior = "Graceful degradation"

            else:
                insufficient_count += 1
                behavior = "Refused to answer"

            # ------------------------------
            # Citation grounding (real, can fail)
            # ------------------------------
            grounded_sentences_total += fidelity.grounded_sentences
            sentences_total += fidelity.total_sentences

            if fidelity.total_sentences == 0:
                outcome = f"{YELLOW}No claims{RESET}"
            elif fidelity.grounded_sentences == fidelity.total_sentences:
                outcome = f"{GREEN}Fully cited{RESET}"
            else:
                outcome = f"{YELLOW}Uncited claims{RESET}"

            query_label = task.value.upper().ljust(15)

            print(
                f"| {query_label} "
                f"| {behavior:<25} "
                f"| {outcome:<26} |"
            )

        print(divider)
        print()

        # =====================================================
        # EXECUTIVE KPI SUMMARY
        # =====================================================

        # NOTE: "capability_distribution" (not "honest_rate") — this is a
        # rate of FULL+PARTIAL vs INSUFFICIENT outcomes, not a hallucination
        # measurement. The CapabilityAssessor only ever returns one of
        # these three labels, so every non-refusal counts as "answered"
        # by construction; it says nothing about whether the answer was
        # actually grounded. Citation-Grounded Sentence Rate below is the
        # metric that can actually fail.
        capability_distribution = round(
            (honest_answers / total_queries) * 100, 2
        )
        graceful_rate = round((graceful_degradations / total_queries) * 100, 2)

        grounded_rate = (
            round((grounded_sentences_total / sentences_total) * 100, 2)
            if sentences_total > 0
            else 0.0
        )
        uncited_rate = round(100.0 - grounded_rate, 2) if sentences_total > 0 else 0.0

        print(f"{BOLD}{CYAN}EXECUTIVE INTEGRITY METRICS{RESET}\n")

        summary_header = (
            f"{BOLD}| Metric                          "
            f"| Value                  |{RESET}"
        )
        summary_divider = "-" * len(summary_header)

        print(summary_divider)
        print(summary_header)
        print(summary_divider)

        print(
            f"| Queries Evaluated               "
            f"| {total_queries:<22} |"
        )
        print(
            f"| Answered Rate (Full+Partial)    "
            f"| {capability_distribution:>6.2f}%               |"
        )
        print(
            f"| Graceful Degradation Coverage   "
            f"| {graceful_rate:>6.2f}%               |"
        )
        print(
            f"| Citation-Grounded Sentence Rate "
            f"| {grounded_rate:>6.2f}%               |"
        )
        print(
            f"| Uncited Claim Rate              "
            f"| {uncited_rate:>6.2f}%               |"
        )

        print(summary_divider)
        print()

        # =====================================================
        # CAPABILITY DISTRIBUTION (HONESTY PROFILE)
        # =====================================================

        print(f"{BOLD}{CYAN}CAPABILITY-AWARE ANSWER PROFILE{RESET}\n")

        cap_header = (
            f"{BOLD}| Capability Level | Count | Interpretation               |{RESET}"
        )
        cap_divider = "-" * len(cap_header)

        print(cap_divider)
        print(cap_header)
        print(cap_divider)

        print(
            f"| FULL             | {full_count:<5} "
            f"| Fully evidence-backed answers     |"
        )
        print(
            f"| PARTIAL          | {partial_count:<5} "
            f"| Transparent, non-hallucinating   |"
        )
        print(
            f"| INSUFFICIENT     | {insufficient_count:<5} "
            f"| Safe refusal                     |"
        )

        print(cap_divider)
        print()

        # ----------------------------------------------------
        # Shutdown
        # ----------------------------------------------------
        self.engine.close()


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    FaithFairKPI().run()
