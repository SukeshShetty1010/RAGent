# agent/core.py
"""
Refactored MinimalAgent — improved LLM relevance grading, robust entity extraction,
and integration with the DataFetcherTool's canonical name returned by the ingestion pipeline.

This full file replaces the previous core.py. Key behaviour changes:
  - When triggering the data fetcher, the agent now prefers the canonical/resolved name
    returned by the DataFetcherTool (e.g., "RAWG-corrected name") and uses that canonical
    name for follow-up retrievals, filenames, and logging.
  - Adds a conservative _sanitize_entity_for_fetch helper as a fallback if the fetcher does
    not provide a canonical name.
  - Logs both raw_entity (LLM output) and chosen canonical/sanitized name for auditing.

This change pairs with the updated DataFetcherTool implementation which returns
"canonical_name" in its execute(...) result — see agent/tools/data_fetcher_tool.py for details. :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import logging
import traceback
import re
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
    Minimal agent skeleton focused on robust relevance grading and canonical-name-aware fetching.
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
    # Helper: detect wrapper/boilerplate-only LLM responses
    # ----------------------------
    def _is_wrapper_only_response(self, resp: str) -> bool:
        """
        Heuristic to detect when an LLM returns a generic assistant wrapper or fails to follow strict output formatting.
        """
        if not resp or not isinstance(resp, str):
            return True
        boilerplate_signals = [
            "as an ai", "i'm sorry", "i cannot", "i'm unable to", "i do not have",
            "please note", "note:", "disclaimer", "i'm sorry but", "i cannot comply",
            "the answer is", "if you are asking", "it depends"
        ]
        low_content = len(resp.strip()) < 10
        if low_content:
            return True
        lower = resp.lower()
        for sig in boilerplate_signals:
            if sig in lower:
                # If 'Relevant:' exists, accept; otherwise treat as wrapper-only
                if "relevant:" not in lower and "relevance:" not in lower:
                    return True
        if "relevant:" not in lower and "relevance:" not in lower:
            return True
        return False

    # ----------------------------
    # Parse LLM grader response
    # ----------------------------
    def _parse_relevance_response(self, resp: str) -> Optional[bool]:
        """
        Parse the two-line response expected from the grader:

        Reasoning: <one-sentence explanation>
        Relevant: YES or NO

        Returns True for YES, False for NO, None for ambiguous/unparseable.
        """
        if not resp or not isinstance(resp, str):
            return None
        lines = [ln.strip() for ln in resp.splitlines() if ln.strip()]
        relevant_line = None
        for ln in lines:
            m = re.search(r"^\s*Relevant\s*[:\-]\s*(YES|NO)\b", ln, flags=re.IGNORECASE)
            if m:
                relevant_line = m.group(1).upper()
                break
            m2 = re.search(r"^\s*Relevance\s*[:\-]\s*(YES|NO)\b", ln, flags=re.IGNORECASE)
            if m2:
                relevant_line = m2.group(1).upper()
                break

        if relevant_line == "YES":
            return True
        if relevant_line == "NO":
            return False

        if lines:
            last = lines[-1]
            if re.search(r"\bYES\b", last, flags=re.IGNORECASE):
                return True
            if re.search(r"\bNO\b", last, flags=re.IGNORECASE):
                return False

        return None

    # ----------------------------
    # LLM-based entity extraction (unchanged core behaviour)
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
        prompt = f"{system}\n\nUser question: {query.strip()}\n\nRespond with the single subject or NONE."
        try:
            resp = self._call_llm(prompt, max_tokens=32, temperature=0.0)
            if not resp:
                return None
            candidate = resp.strip().splitlines()[0].strip()
            if not candidate:
                return None
            if candidate.upper() == "NONE":
                return None
            candidate = candidate.strip(' "\'')
            return candidate
        except Exception:
            return None

    # ----------------------------
    # Fallback sanitizer used only if fetcher doesn't provide canonical_name
    # ----------------------------
    def _sanitize_entity_for_fetch(self, raw_entity: Optional[str]) -> Optional[str]:
        """
        Clean LLM-extracted entity before passing to fetcher or using for logs.
        Conservative: strips assistant prefixes, trims punctuation, rejects noisy sentences.
        """
        if not raw_entity or not isinstance(raw_entity, str):
            return None
        ent = raw_entity.strip()

        # Remove common assistant prefixes (case-insensitive)
        ent = re.sub(r'^(the answer is|answer|result|the result is)\s*[:\-–—]?\s*', '', ent, flags=re.IGNORECASE)

        # Remove leading/trailing quotes/punctuation
        ent = ent.strip(" \t\n\"'`.:;,-")

        # Truncate after newline or long separators
        ent = re.split(r'[\n\r]|-{2,}|—|–', ent)[0].strip()

        # If it still starts with 'User:' or 'Context:' or similar, reject
        if re.match(r'^(user|context|assistant)\b', ent, flags=re.IGNORECASE):
            return None

        # If it looks like a long sentence containing verbs and >8 words, reject
        if len(ent.split()) > 8 and re.search(r'\b(is|are|was|were|will|should|can|could|have|has|had|do|does|did)\b', ent, flags=re.IGNORECASE):
            return None

        # Scrub excessive non-word chars but keep common punctuation used in titles (':', '&', '-', "'")
        ent = re.sub(r'[^A-Za-z0-9 \'\-\:&]', '', ent)

        # Normalize whitespace
        ent = re.sub(r'\s+', ' ', ent).strip()

        if not ent:
            return None
        return ent

    # ----------------------------
    # Core: LLM-based relevance grader (Strict Information Auditor)
    # ----------------------------
    def _llm_grade_relevance(self, query: str, top_chunks: List[Dict[str, Any]]) -> Optional[bool]:
        """
        Ask the LLM to judge if the provided top_chunks contain the exact information
        required to answer `query`. Returns True/False/None as before.

        Persona: Strict Information Auditor. Uses trap example and One-Fact Rule.
        """
        try:
            excerpt_lines = []
            for i, ch in enumerate(top_chunks[: self.grade_top_n]):
                title = ch.get("title") or ch.get("meta", {}).get("title") or f"chunk_{i}"
                text = (ch.get("content") or ch.get("text") or ch.get("meta", {}).get("text") or "")[:4000]
                excerpt_lines.append(f"--- CHUNK {i+1}: {title}\n{text}\n")
            context_combined = "\n\n".join(excerpt_lines).strip()

            system = (
                "You are a Strict Information Auditor. Your job is to determine whether the provided "
                "context contains the exact factual information required to answer the user's question. "
                "You MUST be conservative: only mark Relevant: YES if the context contains the explicit, "
                "verifiable facts needed to answer the question without guessing or inferring. "
            )

            negative_constraint = (
                "CRUCIAL: If the text mentions the entity (for example, the game's title or platforms) "
                "but DOES NOT contain the specific details requested (for example, hardware specifications, "
                "OS/CPU/GPU/RAM values, or other exact facts), you MUST mark Relevant: NO. "
            )

            one_fact_rule = (
                "ONE-FACT RULE: 'Does this text contain the exact answer?' If you have to infer, estimate, "
                "or guess any part of the answer, treat it as NOT present and answer Relevant: NO."
            )

            few_shot = (
                "Examples (follow these to learn the pattern):\n\n"
                "1) User: \"What are the minimum specs for Game X?\"\n"
                "   Context: \"Game X is available now! Buy it on sale for PC and PS5.\"\n"
                "   Assistant: Reasoning: The text mentions Game X and platforms/sales but lacks explicit hardware specs "
                "(CPU, GPU, RAM). It would require guessing to answer minimum specs.\n"
                "              Relevant: NO\n\n"
                "2) User: \"What are the minimum system requirements for AmazingGame?\"\n"
                "   Context: \"Minimum: OS Windows 10; CPU Intel i5-2400; GPU GTX 970; RAM 8GB\"\n"
                "   Assistant: Reasoning: The context lists explicit minimum OS, CPU, GPU and RAM values.\n"
                "              Relevant: YES\n\n"
            )

            output_instr = (
                "Now consider the following. Answer EXACTLY in this two-line format (NO additional text):\n"
                "Line 1: Reasoning: <one-sentence explanation>\n"
                "Line 2: Relevant: YES or NO\n"
                "Rules recap (MUST follow):\n"
                "- If the context only mentions the game title/platforms/releases/sales but does NOT list the specific facts needed, answer Relevant: NO.\n"
                "- If the context contains the explicit facts needed and you can extract them exactly, answer Relevant: YES.\n"
                "- If unsure or if you would need to infer/guess, answer Relevant: NO.\n"
            )

            base_prompt = (
                f"{system}\n\n{negative_constraint}\n\n{one_fact_rule}\n\n{few_shot}\n\nUser question: {query.strip()}\n\n"
                f"Context (top {min(len(top_chunks), self.grade_top_n)} chunks):\n{context_combined}\n\n"
                f"{output_instr}\n"
            )

            try:
                self._log("LLM grading relevance of top chunks (Strict Information Auditor).")
                resp = self._call_llm(base_prompt, max_tokens=200, temperature=0.0)
            except ImportError as ie:
                self._log(f"LLM grade call failed (modal missing): {ie}")
                return None
            except Exception as e:
                self._log(f"LLM grade call failed: {e}\n{traceback.format_exc()}")
                return None

            if self._is_wrapper_only_response(resp):
                self._log("Detected wrapper/boilerplate-only LLM response; retrying once with stricter instructions.")
                strict_prefix = (
                    "STRICT INSTRUCTIONS: You MUST reply with EXACTLY two non-empty lines and nothing else.\n"
                    "Line1: Reasoning: <one-sentence explanation>\n"
                    "Line2: Relevant: YES or NO\n"
                    "Do NOT include any other text, disclaimers, or meta-instructions. If you cannot comply, answer:\n"
                    "Reasoning: Could not comply with formatting\nRelevant: NO\n\n"
                )
                try:
                    resp_retry = self._call_llm(strict_prefix + base_prompt, max_tokens=140, temperature=0.0)
                    if resp_retry and not self._is_wrapper_only_response(resp_retry):
                        resp = resp_retry
                    else:
                        self._log("Retry did not produce usable judgment; proceeding to parse what we have.")
                except Exception as e:
                    self._log(f"Retry failed: {e}")

            try:
                judgement = self._parse_relevance_response(resp)
                if judgement is True:
                    self._log("LLM grader returned Relevant: YES")
                    return True
                if judgement is False:
                    self._log("LLM grader returned Relevant: NO")
                    return False
                self._log("LLM grader response ambiguous/unparseable; returning None to let caller fallback.")
                return None
            except Exception as e:
                self._log(f"Failed to parse LLM grader response: {e}\n{traceback.format_exc()}")
                return None

        except Exception as e:
            self._log(f"Unexpected exception in _llm_grade_relevance: {e}\n{traceback.format_exc()}")
            return None

    # ----------------------------
    # Main entrypoint: decide & fetch if needed (uses canonical_name from fetcher when available)
    # ----------------------------
    def decide_and_fetch_if_needed(self, query: str, retriever_tool: Tool, k: int = 5) -> Dict[str, Any]:
        """
        High-level usage example:
          - Use retriever_tool to get results
          - Ask LLM grader if results already answer the query
          - If NO or ambiguous with low score, call the data fetcher tool
          - Prefer fetcher's canonical_name for re-query / logging / filenames
        """
        steps = ["retriever_initial"]
        did_fetch = False

        # 1) initial retrieval (caller provides retriever tool)
        try:
            results = retriever_tool.execute({"query": query, "k": max(k, self.top_k_for_score)})
        except Exception as e:
            return {"status": "error", "error": f"Retriever failed: {e}", "steps": steps}

        # compute top score before (simple heuristic if results carry 'score')
        top_score_before = 0.0
        if results and isinstance(results, list):
            scores = [float(r.get("score") or 0.0) for r in results]
            top_score_before = max(scores) if scores else 0.0

        # 2) LLM grade
        graded_relevance: Optional[bool] = None
        try:
            graded_relevance = self._llm_grade_relevance(query, results or [])
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
                # Ask LLM for an entity as before (may be noisy)
                raw_entity = self._llm_extract_entity(query)
                self._log(f"Raw LLM entity before fetch: {raw_entity!r}")

                fetch_args = {"game_name": raw_entity or query}
                fetch_res = fetch_tool.execute(fetch_args)
                did_fetch = True

                # Prefer canonical_name from fetcher if available
                canonical_name = None
                if isinstance(fetch_res, dict):
                    canonical_name = fetch_res.get("canonical_name") or fetch_res.get("resolved_name") or None
                    # Also check nested meta
                    if canonical_name is None:
                        meta = fetch_res.get("meta") if isinstance(fetch_res.get("meta"), dict) else {}
                        canonical_name = meta.get("canonical_name") or meta.get("resolved_name")

                self._log(f"Data fetch result: canonical_name={canonical_name!r}; fetch_res_keys={list(fetch_res.keys()) if isinstance(fetch_res, dict) else 'n/a'}")

                # Decide what name to use for follow-up retrieval & filenames:
                chosen_name = canonical_name
                if not chosen_name:
                    chosen_name = self._sanitize_entity_for_fetch(raw_entity) if hasattr(self, "_sanitize_entity_for_fetch") else raw_entity

                if not chosen_name:
                    # last-resort heuristics from query
                    mq = re.search(r'["“”\']([^"“”\']{3,})["“”\']', query)
                    if mq:
                        chosen_name = mq.group(1).strip()
                    else:
                        tcs = re.findall(r"\b([A-Z][a-z0-9'’\-]+(?:\s+[A-Z][a-z0-9'’\-]+)+)\b", query)
                        chosen_name = max(tcs, key=lambda s: (len(s.split()), len(s))) if tcs else None

                self._log(f"Fetch canonicalization: raw_entity={raw_entity!r} -> chosen_name={chosen_name!r}")

                # Re-run retriever using the original query first to preserve semantics
                steps.append("retriever_after_fetch")
                try:
                    results = retriever_tool.execute({"query": query, "k": k})
                except Exception:
                    logger.exception("Retriever failed after fetch")
                    results = []

                # Optional: if retriever supports querying by entity/name, attempt that for better precision
                if chosen_name:
                    try:
                        # Try a richer call; if retriever doesn't support 'entity' or fails, keep prior results
                        alt_results = retriever_tool.execute({"query": query, "entity": chosen_name, "k": k})
                        if isinstance(alt_results, list) and alt_results:
                            self._log("Retriever returned results for entity-filtered query; using these.")
                            results = alt_results
                    except Exception:
                        # Ignore failures here (retriever might not accept 'entity')
                        pass

                fetch_info = fetch_res if isinstance(fetch_res, dict) else {"raw_fetch_result": fetch_res}

            except Exception as e:
                self._log(f"Data fetch failed: {e}\n{traceback.format_exc()}")

        else:
            fetch_info = None

        top_score_after = 0.0
        if results and isinstance(results, list):
            scores = [float(r.get("score") or 0.0) for r in results]
            top_score_after = max(scores) if scores else 0.0

        low_confidence = top_score_after < self.score_threshold
        return {
            "status": "success",
            "steps": steps,
            "results": results[:k] if results else [],
            "top_score_before": top_score_before,
            "top_score_after": top_score_after,
            "did_fetch": did_fetch,
            "low_confidence": low_confidence,
            "fetch_info": fetch_info if 'fetch_info' in locals() else None,
            "llm_judgement": "unknown" if graded_relevance is None else ("relevant" if graded_relevance else "not_relevant"),
        }
