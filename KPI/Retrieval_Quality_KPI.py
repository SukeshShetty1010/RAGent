# ============================================================
# tests/KPI/Retrieval_Quality_KPI.py
# RESUME-GRADE KPI: SEARCH RELEVANCE (FIXED)
# ============================================================

from __future__ import annotations

import logging
from typing import List, Tuple, Dict

from engine.execution_engine import RageEngine
from tests.evaluation_metrics import (
    calculate_evidence_hit_rate,
    calculate_entity_coverage,
)

# ------------------------------------------------------------
# ANSI formatting (resume-grade CLI tables)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ============================================================
# Traffic with Ground Truth (ENTITY-LEVEL)
# ============================================================
# Each entry:
# (query, expected_entities)
# ============================================================

TRAFFIC_WITH_TRUTH: List[Tuple[str, List[str]]] = [
    (
        "Compare Far Cry 5 vs Assassin’s Creed Valhalla",
        ["Far Cry 5", "Assassin’s Creed Valhalla"],
    ),
    (
        "Top 5 things to do in Far Cry 5",
        ["Far Cry 5"],
    ),
    (
        "What is the release date of Far Cry 5?",
        ["Far Cry 5"],
    ),
    (
        "Latest patch notes for Assassin’s Creed Valhalla",
        ["Assassin’s Creed Valhalla"],
    ),
]


# ============================================================
# KPI Runner
# ============================================================

class RetrievalQualityKPI:
    """
    Resume-grade KPI runner for Search Relevance.

    Primary Metrics:
    - Evidence Hit Rate
    - Entity Coverage

    Supporting Metrics:
    - Retrieval Quality Distribution
    - Avg Retrieval Confidence
    - Web Fallback Trigger Rate
    """

    def __init__(self) -> None:
        self.engine = RageEngine()

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        print(f"\n{BOLD}{CYAN}RESUME-GRADE KPI: SEARCH RELEVANCE{RESET}\n")

        # =====================================================
        # Evidence Hit + Entity Coverage Table
        # =====================================================

        header = (
            f"{BOLD}| Query Snippet                          "
            f"| Evidence Hit | Entity Coverage |{RESET}"
        )
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)

        # -----------------------------------------------------
        # Aggregation stores
        # -----------------------------------------------------

        hit_count = 0
        coverage_sum = 0.0
        total_runs = 0

        quality_counts: Dict[str, int] = {
            "QUALITY_OK": 0,
            "QUALITY_WEAK": 0,
            "QUALITY_EMPTY": 0,
        }

        confidence_scores: List[float] = []
        web_fallback_triggers = 0

        # -----------------------------------------------------
        # Run traffic
        # -----------------------------------------------------

        for query, expected_entities in TRAFFIC_WITH_TRUTH:
            total_runs += 1
            result = self.engine.run(query)

            retrieved_chunks = result.get("evidence", [])

            # -------------------------------
            # FIX 1 — Evidence Hit Rate
            # -------------------------------
            hit_result = calculate_evidence_hit_rate(
                retrieved_chunks=retrieved_chunks,
                expected_entities=expected_entities,
            )
            hit_count += hit_result.hit_queries

            # -------------------------------
            # FIX 2 — Entity Coverage
            # -------------------------------
            coverage_result = calculate_entity_coverage(
                retrieved_chunks=retrieved_chunks,
                expected_entities=expected_entities,
            )
            coverage_sum += coverage_result.coverage

            # -------------------------------
            # Display row
            # -------------------------------
            query_snippet = (
                query[:30] + "…" if len(query) > 30 else query
            )

            hit_str = (
                f"{GREEN}YES{RESET}"
                if hit_result.hit_queries == 1
                else f"{YELLOW}NO{RESET}"
            )

            coverage_pct = coverage_result.coverage * 100.0
            coverage_str = f"{coverage_pct:>6.2f}%"

            print(
                f"| {query_snippet:<35} "
                f"| {hit_str:^12} "
                f"| {coverage_str:^15} |"
            )

            # -------------------------------
            # Retrieval Quality Distribution
            # -------------------------------
            qs = result.get("kpis", {}).get("quality_status", "")
            if isinstance(qs, str):
                qs = qs.upper()
            if qs in quality_counts:
                quality_counts[qs] += 1

            # -------------------------------
            # Confidence aggregation
            # -------------------------------
            conf = result.get("kpis", {}).get("confidence_score")
            if isinstance(conf, (int, float)):
                confidence_scores.append(float(conf))

            # -------------------------------
            # Web fallback detection
            # -------------------------------
            merge_state = result.get("agent_decisions", {}).get("merge_state")
            if merge_state and merge_state != "LOCAL_ONLY":
                web_fallback_triggers += 1

        print(divider)
        print()

        # =====================================================
        # Aggregate Resume Metrics
        # =====================================================

        evidence_hit_rate = round((hit_count / total_runs) * 100, 2)
        avg_entity_coverage = round((coverage_sum / total_runs) * 100, 2)

        avg_confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 4)
            if confidence_scores else 0.0
        )

        web_fallback_rate = round(
            (web_fallback_triggers / total_runs) * 100, 2
        )

        print(f"{BOLD}{CYAN}AGGREGATE PERFORMANCE METRICS{RESET}\n")

        agg_header = (
            f"{BOLD}| Metric Name                    "
            f"| Value      |{RESET}"
        )
        agg_divider = "-" * len(agg_header)

        print(agg_divider)
        print(agg_header)
        print(agg_divider)

        print(
            f"| Evidence Hit Rate              "
            f"| {GREEN}{evidence_hit_rate:>7.2f}%{RESET} |"
        )
        print(
            f"| Avg Entity Coverage            "
            f"| {GREEN}{avg_entity_coverage:>7.2f}%{RESET} |"
        )
        print(
            f"| Avg Retrieval Confidence       "
            f"| {avg_confidence:<10.4f} |"
        )
        print(
            f"| Web Fallback Trigger Rate      "
            f"| {web_fallback_rate:>7.2f}% |"
        )

        print(agg_divider)
        print()


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    RetrievalQualityKPI().run()
