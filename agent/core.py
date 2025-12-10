# agent/core.py
"""
Refactored MinimalAgent — more robust LLM relevance grading with single-retry
for wrapper/boilerplate-only responses.

Key behavior:
 - Uses a short Chain-of-Thought style "Reasoning Validator" prompt requiring:
       Reasoning: <one-sentence explanation>
       Relevant: YES or NO
 - If the LLM returns a known wrapper/boilerplate message (e.g. "Do not provide additional context..."),
   the grader performs one immediate retry with an even stricter instruction.
 - Parsing strips common assistant boilerplate lines before searching for the 'Relevant:' line.
 - If still ambiguous, returns None → caller falls back to score-threshold logic.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class Tool:
    """Minimal Tool stub for typing when running the module outside the full project."""
    name: str
    description: str

    def execute(self, args: Dict[str, Any]) -> Any:
        raise NotImplementedError()


class MinimalAgent:
    """
    Minimal agent skeleton focused on the relevance grading improvements.
    """

    def __init__(self, registry=None):
        # configuration defaults
        self.registry = registry
        self.grade_top_n = 3
        self.top_k_for_score = 20
        self.score_threshold = 0.6
        self._retriever_tool_name = "retriever"
        self._data_fetcher_tool_name = "data_fetcher"

    # ----------------------------
    # Logging helper
    # ----------------------------
    def _log(self, msg: str) -> None:
        logger.info(f"[AGENT] {msg}")

    # ----------------------------
    # LLM call wrapper (Modal-backed)
    # ----------------------------
    def _call_llm(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """
        Call the remote Modal LLM function.

        Raises ImportError if 'modal' package is unavailable.
        """
        FUNCTION_APP_NAME = "rag-llama3-3b"
        FUNCTION_NAME = "chat_completion_remote"

        try:
            import modal  # type: ignore
        except Exception as e:
            raise ImportError("Modal client not available (import modal failed).") from e

        try:
            chat_fn = modal.Function.from_name(FUNCTION_APP_NAME, FUNCTION_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to lookup remote function {FUNCTION_APP_NAME}/{FUNCTION_NAME}: {e}") from e

        try:
            self._log("Invoking remote Modal function for LLM call...")
            result = chat_fn.remote(prompt, max_tokens=max_tokens, temperature=temperature)
            if result is None:
                return ""
            return str(result)
        except Exception as e:
            raise RuntimeError(f"Modal remote invocation failed: {e}") from e

    # ----------------------------
    # LLM-based entity extraction
    # ----------------------------
    def _llm_extract_entity(self, query: str) -> Optional[str]:
        """
        Extract single core subject from query (game title/entity) or None.
        """
        system = (
            "You are a concise information-extraction helper. "
            "Given a user's question, extract the single most likely core subject (game title, company, or entity). "
            "If there is no clear specific subject, respond with 'NONE'."
        )

        examples = (
            "Examples:\n"
            "- Question: \"What are the minimum requirements for Assassin's Creed Valhalla?\"\n"
            "  Subject: Assassin's Creed Valhalla\n"
            "- Question: \"Tell me about Ubisoft's recent layoffs\"\n"
            "  Subject: Ubisoft\n"
            "- Question: \"How do I fix shader compilation errors on PC?\"\n"
            "  Subject: NONE\n"
        )

        prompt = (
            f"{system}\n\n{examples}\n\nQuestion: \"{query.strip()}\"\n\n"
            "Answer with ONLY the subject on a single line, or EXACTLY the word NONE if there is no specific subject."
        )

        try:
            self._log("LLM extracting entity from query.")
            resp = self._call_llm(prompt, max_tokens=64, temperature=0.0)
            if not resp:
                return None
            ans = resp.strip().splitlines()[0].strip()
            if not ans:
                return None
            if ans.upper() in ("NONE", "NO", "N/A", "NONE FOUND"):
                return None
            for prefix in ("Subject:", "subject:", "Answer:", "answer:"):
                if ans.startswith(prefix):
                    ans = ans[len(prefix):].strip()
            if not ans:
                return None
            return ans
        except ImportError as ie:
            self._log(f"LLM extract failed (modal missing): {ie}")
            return None
        except Exception as e:
            self._log(f"LLM extract failed: {e}; falling back to None")
            return None

    # ----------------------------
    # Internal: detect wrapper/boilerplate responses
    # ----------------------------
    def _is_wrapper_only_response(self, text: str) -> bool:
        """
        Heuristic to detect common wrapper/boilerplate-only replies that are NOT the expected two-line output.
        Example observed: "Do not provide additional context or explanations beyond the requested format."
        """
        if not text:
            return False
        low = text.strip().lower()
        # common patterns that appear when a system wrapper is echoed
        patterns = [
            "do not provide additional context",
            "do not provide additional context or explanations",
            "do not include additional context",
            "only respond with",
            "do not provide any additional",
            "do not provide any other text",
            "please only respond with",
        ]
        for p in patterns:
            if p in low:
                return True
        return False

    # ----------------------------
    # LLM-based relevance grading (Chain-of-Thought style) with retry
    # ----------------------------
    def _llm_grade_relevance(self, query: str, chunks: List[Dict[str, Any]]) -> Optional[bool]:
        """
        Improved LLM-based relevance grading.

        Required LLM output (two-line):
            Reasoning: <one-sentence explanation>
            Relevant: YES or NO

        Robustness:
         - If LLM returns wrapper/boilerplate-only message, retry once with stricter prompt.
         - Strip common boilerplate lines before parsing.
         - If still ambiguous, return None to trigger fallback behavior.
        """
        top_chunks = (chunks or [])[: self.grade_top_n]
        if not top_chunks:
            return False

        # Build concise context
        ctx_parts = []
        for i, c in enumerate(top_chunks, start=1):
            tid = c.get("id") or c.get("doc_id") or f"chunk{i}"
            title = c.get("title") or ""
            content = (c.get("content") or c.get("text") or "")
            snippet = (content or "").strip().replace("\n", " ")
            if len(snippet) > 600:
                snippet = snippet[:600].rsplit(" ", 1)[0] + " ..."
            score = c.get("score")
            ctx_parts.append(f"CHUNK {i} ID:{tid} SCORE:{score}\nTITLE: {title}\nTEXT: {snippet}")

        context_combined = "\n\n---\n\n".join(ctx_parts)

        system = (
            "You are a Reasoning Validator. Step 1: Read the user's request precisely. "
            "Step 2: Check whether the provided text contains the exact information required to answer it "
            "(not just mentions, sales, or other loosely related facts). "
            "If the context contains the specific factual items needed (e.g. explicit minimum system requirements: OS, CPU, GPU, RAM), mark Relevant: YES. "
            "If the context lacks the specific facts and only mentions the title, sales, release dates, etc., mark Relevant: NO."
        )

        few_shot = (
            "Examples:\n"
            "- User: \"What are the specs for Game X?\"\n"
            "  Context: \"Game X is on sale for $20.\"\n"
            "  Assistant: Reasoning: The text mentions Game X and prices but does not list hardware specifications (CPU, GPU).\n"
            "             Relevant: NO\n\n"
            "- User: \"What are the minimum system requirements for AmazingGame?\"\n"
            "  Context: \"Minimum: OS Windows 10; CPU Intel i5-2400; GPU GTX 970; RAM 8GB\"\n"
            "  Assistant: Reasoning: The context lists explicit minimum OS, CPU, GPU and RAM values.\n"
            "             Relevant: YES\n"
        )

        base_prompt = (
            f"{system}\n\n{few_shot}\nUser question: {query.strip()}\n\n"
            f"Context (top {len(top_chunks)} chunks):\n{context_combined}\n\n"
            "Answer EXACTLY in the following two-line format (NO additional text):\n"
            "Reasoning: <one-sentence explanation>\n"
            "Relevant: YES or NO\n"
            "If unsure, respond with Relevant: NO."
        )

        # send prompt and possibly retry once
        try:
            self._log("LLM grading relevance of top chunks (Reasoning Validator).")
            resp = self._call_llm(base_prompt, max_tokens=180, temperature=0.0)
        except ImportError as ie:
            self._log(f"LLM grade call failed (modal missing): {ie}")
            return None
        except Exception as e:
            self._log(f"LLM grade call failed: {e}\n{traceback.format_exc()}")
            return None

        # If the model returned a wrapper-only string, attempt a single retry with even stricter instruction
        if self._is_wrapper_only_response(resp):
            self._log("Detected wrapper/boilerplate-only LLM response; retrying once with stricter prompt.")
            strict_prompt = (
                "STRICT INSTRUCTIONS: You MUST reply with EXACTLY two non-empty lines and nothing else.\n"
                "Line1: Reasoning: <one-sentence explanation>\n"
                "Line2: Relevant: YES or NO\n"
                "Do NOT include any other text, disclaimers, or meta-instructions. If you cannot comply, answer:\n"
                "Reasoning: Could not comply with formatting\nRelevant: NO\n\n"
            ) + base_prompt
            try:
                resp_retry = self._call_llm(strict_prompt, max_tokens=120, temperature=0.0)
                # prefer retry if it looks usable
                if resp_retry and not self._is_wrapper_only_response(resp_retry):
                    resp = resp_retry
                else:
                    self._log("Retry did not produce usable judgment; proceeding to parse what we have.")
            except Exception as e:
                self._log(f"Retry failed: {e}")

        # Normalize and filter out common assistant boilerplate lines before parsing
        def sanitize_lines(raw_text: str) -> List[str]:
            if not raw_text:
                return []
            lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
            filtered = []
            for ln in lines:
                low = ln.lower()
                # remove assistant wrapper-like sentences
                if (
                    "do not provide additional context" in low
                    or "do not include additional context" in low
                    or "only respond with" in low
                    or "please only respond with" in low
                    or low.startswith("assistant:")
                ):
                    # skip boilerplate
                    continue
                filtered.append(ln)
            return filtered

        lines = sanitize_lines(resp)

        # Primary parsing: look for explicit 'Relevant:' line
        relevant_val: Optional[bool] = None
        for ln in lines:
            up = ln.upper()
            if up.startswith("RELEVANT:"):
                try:
                    val = ln.split(":", 1)[1].strip().upper()
                    if val.startswith("YES"):
                        relevant_val = True
                        break
                    if val.startswith("NO"):
                        relevant_val = False
                        break
                except Exception:
                    continue

        # Fallback tolerant parsing if no explicit line
        if relevant_val is None:
            combined = " ".join(lines).upper()
            tokens = [t.strip(".,;:") for t in combined.split()]
            if "RELEVANT: YES" in combined:
                relevant_val = True
            elif "RELEVANT: NO" in combined:
                relevant_val = False
            else:
                # conservative rule: require explicit YES token and no NO token
                if "YES" in tokens and "NO" not in tokens:
                    relevant_val = True
                elif "NO" in tokens:
                    relevant_val = False

        if relevant_val is not None:
            return relevant_val

        # Still ambiguous: log and return None for fallback
        self._log(f"Ambiguous LLM relevance output after parsing; raw response: {resp!r}")
        return None

    # ----------------------------
    # Helper: compute top raw score (simple stub)
    # ----------------------------
    def _top_score_from_results(self, results: List[Dict[str, Any]]) -> float:
        top = 0.0
        if not results:
            return 0.0
        for r in results:
            s = r.get("score", 0.0)
            try:
                s = float(s)
            except Exception:
                s = 0.0
            if s > top:
                top = s
        return top

    # ----------------------------
    # MAIN AGENT LOOP (simplified)
    # ----------------------------
    def run(self, query: str, k: int = 10) -> Dict[str, Any]:
        steps: List[str] = []
        final_results: List[Dict[str, Any]] = []
        did_fetch = False

        self._log("Retrieving initial results (no similarity threshold)...")
        steps.append("retriever")

        try:
            retriever_tool: Tool = self.registry.get(self._retriever_tool_name)
        except Exception as e:
            return {"status": "error", "error": f"Missing retriever tool: {e}", "steps": steps}

        try:
            results = retriever_tool.execute(
                {"query": query, "k": self.top_k_for_score, "similarity_threshold": None}
            )
            if not isinstance(results, list):
                results = list(results) if results else []
        except Exception:
            logger.exception("Retriever failed")
            results = []

        top_score_before = self._top_score_from_results(results)
        self._log(f"Top score before: {top_score_before:.4f} (threshold={self.score_threshold})")

        graded_relevance: Optional[bool] = None
        try:
            graded_relevance = self._llm_grade_relevance(query, results)
            if graded_relevance is True:
                self._log("LLM judged retrieved chunks RELEVANT to the question.")
            elif graded_relevance is False:
                self._log("LLM judged retrieved chunks NOT relevant to the question.")
            else:
                self._log("LLM returned ambiguous/no judgement; will fallback to score threshold.")
        except Exception as e:
            self._log(f"Relevance grading failed with exception: {e}; will fallback to score threshold.")

        # If LLM explicitly says relevant -> return results
        if graded_relevance is True and results:
            self._log("Returning top results per LLM relevance judgment.")
            return {
                "status": "success",
                "steps": steps,
                "results": results[:k],
                "top_score_before": top_score_before,
                "top_score_after": top_score_before,
                "did_fetch": False,
                "low_confidence": False,
                "fetch_info": None,
                "llm_judgement": "relevant",
            }

        # If LLM explicitly said NOT relevant OR (ambiguous and top score low) -> fetch
        if graded_relevance is False or (graded_relevance is None and top_score_before < self.score_threshold):
            self._log("Determined that retrieval is insufficient — will attempt data fetching.")
            steps.append("data_fetch")
            try:
                fetch_tool: Tool = self.registry.get(self._data_fetcher_tool_name)
            except Exception as e:
                return {"status": "error", "error": f"Missing data_fetcher tool: {e}", "steps": steps}

            try:
                entity = self._llm_extract_entity(query)
                fetch_args = {"game_name": entity or query}
                fetch_res = fetch_tool.execute(fetch_args)
                did_fetch = True
                steps.append("retriever_after_fetch")
                try:
                    results = retriever_tool.execute({"query": query, "k": k})
                except Exception:
                    logger.exception("Retriever failed after fetch")
                    results = []
            except Exception as e:
                self._log(f"Data fetch failed: {e}")

        top_score_after = self._top_score_from_results(results)
        low_confidence = top_score_after < self.score_threshold
        return {
            "status": "success",
            "steps": steps,
            "results": results[:k],
            "top_score_before": top_score_before,
            "top_score_after": top_score_after,
            "did_fetch": did_fetch,
            "low_confidence": low_confidence,
            "fetch_info": None,
            "llm_judgement": "unknown" if graded_relevance is None else ("relevant" if graded_relevance else "not_relevant"),
        }
