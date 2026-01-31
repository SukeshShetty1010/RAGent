# ============================================================
# tests/KPI/Retrieval_Quality_KPI.py
# RESUME-GRADE KPI: SEARCH RELEVANCE
# ============================================================

from __future__ import annotations

import logging
from typing import List, Tuple, Dict

from engine.execution_engine import RageEngine
from tests.evaluation_metrics import calculate_precision_at_k

# ------------------------------------------------------------
# ANSI formatting (resume-grade CLI tables)
# ------------------------------------------------------------

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# ============================================================
# Traffic with Ground Truth (CRITICAL)
# ============================================================
# Each entry:
# (query, expected_source_titles)
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
    # --- Guaranteed Web Fallback Case (Temporal Signal) ---
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
    Focused KPI runner for Search Relevance.

    Measures:
    - Retrieval Precision@5
    - Retrieval Quality Distribution
    - Avg Retrieval Confidence
    - Noise Rejection Rate (best-effort)
    - Web Fallback Trigger Rate

    Purpose:
    - Executive-friendly Hybrid Search observability
    - Resume-grade system health summary
    """

    def __init__(self, k: int = 5) -> None:
        self.engine = RageEngine()
        self.k = k

    # --------------------------------------------------------
    # Execution
    # --------------------------------------------------------

    def run(self) -> None:
        logging.getLogger().setLevel(logging.WARNING)

        print(f"\n{BOLD}{CYAN}RESUME-GRADE KPI: SEARCH RELEVANCE{RESET}\n")

        # =====================================================
        # Precision@K Table
        # =====================================================

        header = (
            f"{BOLD}| Query Snippet                          "
            f"| Expected Source                    "
            f"| Precision@{self.k} |{RESET}"
        )
        divider = "-" * len(header)

        print(divider)
        print(header)
        print(divider)

        # -----------------------------------------------------
        # Aggregation stores
        # -----------------------------------------------------
        quality_counts: Dict[str, int] = {
            "QUALITY_OK": 0,
            "QUALITY_WEAK": 0,
            "QUALITY_EMPTY": 0,
        }

        confidence_scores: List[float] = []
        total_retrieved_chunks = 0
        total_noise_rejected = 0  # simulated until instrumented
        web_fallback_triggers = 0

        total_runs = 0

        for query, expected_sources in TRAFFIC_WITH_TRUTH:
            total_runs += 1

            result = self.engine.run(query)

            # -------------------------------
            # Precision@K
            # -------------------------------
            retrieved_chunks = result.get("evidence", [])
            total_retrieved_chunks += len(retrieved_chunks)

            precision_result = calculate_precision_at_k(
                retrieved_chunks=retrieved_chunks,
                expected_source_titles=expected_sources,
                k=self.k,
            )

            precision_pct = precision_result.precision * 100.0

            query_snippet = (
                query[:30] + "…" if len(query) > 30 else query
            )
            expected_str = ", ".join(expected_sources)

            value_str = f"{GREEN}{precision_pct:.2f}%{RESET}"

            print(
                f"| {query_snippet:<35} "
                f"| {expected_str:<32} "
                f"| {value_str:<12} |"
            )

            # -------------------------------
            # Retrieval Quality Distribution
            # -------------------------------
            quality_status = (
                result.get("kpis", {}).get("quality_status", "")
            )
            if isinstance(quality_status, str):
                quality_status = quality_status.upper()

            if quality_status in quality_counts:
                quality_counts[quality_status] += 1

            # -------------------------------
            # Confidence aggregation
            # -------------------------------
            confidence = result.get("kpis", {}).get("confidence_score")
            if isinstance(confidence, (int, float)):
                confidence_scores.append(float(confidence))

            # -------------------------------
            # Noise rejection (best-effort)
            # -------------------------------
            try:
                total_noise_rejected += int(
                    result.get("kpis", {}).get("noise_filtered_count", 0)
                )
            except Exception:
                # TODO: Instrument quality_gate.py to expose noise_filtered_count
                pass

            # -------------------------------
            # Web Fallback Trigger Detection
            # -------------------------------
            merge_state = result.get("agent_decisions", {}).get("merge_state")
            if merge_state and merge_state != "LOCAL_ONLY":
                web_fallback_triggers += 1

        print(divider)
        print()

        # =====================================================
        # Retrieval Quality Distribution (SUMMARY)
        # =====================================================

        print(f"{BOLD}{CYAN}RETRIEVAL QUALITY DISTRIBUTION{RESET}\n")

        summary_header = (
            f"{BOLD}| Quality Bucket | Count | Distribution (%) |{RESET}"
        )
        summary_divider = "-" * len(summary_header)

        print(summary_divider)
        print(summary_header)
        print(summary_divider)

        for bucket, count in quality_counts.items():
            pct = (
                round((count / total_runs) * 100, 2)
                if total_runs else 0.0
            )
            color = GREEN if bucket == "QUALITY_OK" else YELLOW

            print(
                f"| {bucket:<14} "
                f"| {count:<5} "
                f"| {color}{pct:>7.2f}%{RESET:<3} |"
            )

        print(summary_divider)
        print()

        # =====================================================
        # Aggregate Performance Metrics (EXECUTIVE)
        # =====================================================

        avg_confidence = (
            round(sum(confidence_scores) / len(confidence_scores), 4)
            if confidence_scores else 0.0
        )

        noise_rejection_rate = (
            round(
                (total_noise_rejected / total_retrieved_chunks) * 100,
                2,
            )
            if total_retrieved_chunks else 0.0
        )

        web_fallback_rate = (
            round((web_fallback_triggers / total_runs) * 100, 2)
            if total_runs else 0.0
        )

        print(f"{BOLD}{CYAN}AGGREGATE PERFORMANCE METRICS{RESET}\n")

        agg_header = (
            f"{BOLD}| Metric Name                    "
            f"| Value      | Target  |{RESET}"
        )
        agg_divider = "-" * len(agg_header)

        print(agg_divider)
        print(agg_header)
        print(agg_divider)

        print(
            f"| Avg Retrieval Confidence       "
            f"| {avg_confidence:<10.4f} "
            f"| > 0.75  |"
        )
        print(
            f"| Noise Rejection Rate           "
            f"| {noise_rejection_rate:>7.2f}% "
            f"| < 20%   |"
        )
        print(
            f"| Web Fallback Trigger Rate      "
            f"| {web_fallback_rate:>7.2f}% "
            f"| < 10%   |"
        )

        print(agg_divider)
        print()


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    RetrievalQualityKPI(k=5).run()
