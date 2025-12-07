# retriever.py (FULL updated file)
# ------------------------------------------------------------
# Refactor notes:
#  - format_rag_prompt uses Few-Shot prompting and inline citation token: <<SRC:ID>>.
#  - Context now forces SRC to equal the displayed chunk ID (so citations match ID: lines).
#  - generate_answer_via_modal: improved streaming fallback; handles remote_gen missing or failing.
#  - Keeps all argparse flags and Weaviate GraphQL retrieval logic.
# ------------------------------------------------------------

"""
python -m retriever.retriever --query "What are the problems with Far Cry 5" --k 10 --weaviate http://localhost:8080 --debug --show-meta --generate --show-prompt --stream
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_CLASS = "GameChunk"


# ---------------------------
# Helpers
# ---------------------------

class WeaviateConnectionError(Exception):
    pass


def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def _post_graphql(weaviate_url: str, query: str, timeout: int = 30) -> Dict[str, Any]:
    gw_url = weaviate_url.rstrip("/") + "/v1/graphql"
    headers = {"Content-Type": "application/json"}
    try:
        r = requests.post(gw_url, json={"query": query}, headers=headers, timeout=timeout)
    except RequestsConnectionError as e:
        raise WeaviateConnectionError(f"Failed to connect to Weaviate at {weaviate_url}: {e}")
    try:
        parsed = r.json()
    except ValueError:
        parsed = {"_raw_text": r.text}
    return {"status_code": r.status_code, "ok": r.ok, "json": parsed, "text": r.text}


def _weaviate_schema_properties(weaviate_url: str, class_name: str, timeout: int = 5) -> Set[str]:
    try:
        r = requests.get(weaviate_url.rstrip("/") + "/v1/schema", timeout=timeout)
        r.raise_for_status()
        schema = r.json()
        classes = schema.get("classes", []) if isinstance(schema, dict) else []
        for cls in classes:
            if cls.get("class") == class_name:
                props = cls.get("properties", []) or []
                return {p.get("name") for p in props if p.get("name")}
    except RequestsConnectionError as e:
        raise WeaviateConnectionError(f"Failed to connect to Weaviate at {weaviate_url}: {e}")
    except Exception:
        # swallow and return empty set so logic can continue (we'll be conservative)
        return set()
    return set()


def _parse_meta_prop(meta_prop: Optional[Any]) -> Dict[str, Any]:
    if not meta_prop:
        return {}
    if isinstance(meta_prop, dict):
        return meta_prop
    try:
        return json.loads(meta_prop)
    except Exception:
        return {"raw": str(meta_prop)}


def _score_from_additional(additional: Dict[str, Any]) -> Optional[float]:
    if not additional:
        return None
    # Weaviate often returns 'certainty' or 'distance' in _additional
    if "certainty" in additional and additional["certainty"] is not None:
        try:
            return float(additional["certainty"])
        except Exception:
            pass
    if "distance" in additional and additional["distance"] is not None:
        try:
            d = float(additional["distance"])
            # convert distance -> similarity-like score (0..1)
            if math.isfinite(d):
                return 1.0 / (1.0 + d)
        except Exception:
            pass
    return None


# ---------------------------
# Core retrieval: query -> filter -> return
# ---------------------------

def _build_near_vector_gql(class_name: str, vec: List[float], props_gql: str, limit: int, where_clause: str = "") -> str:
    vec_list_literal = ", ".join([repr(float(v)) for v in vec])
    return f"""
    {{
      Get {{
        {class_name}(nearVector: {{ vector: [{vec_list_literal}] }}{where_clause}, limit: {int(limit)}) {{
          {props_gql}
        }}
      }}
    }}
    """


def retrieve(
    query: str,
    k: int = 5,
    weaviate_url: str = DEFAULT_WEAVIATE_URL,
    class_name: str = DEFAULT_CLASS,
    model_name: str = DEFAULT_MODEL,
    min_char_length: int = 50,
    similarity_threshold: float = 0.6,
    unified_game_id: Optional[str] = None,
    fetch_multiplier: int = 2,
    debug: bool = False,
    show_meta: bool = False,
    use_hybrid: bool = False,
    hybrid_alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Performs retrieval from Weaviate using nearVector. Applies gates:
      - Drop chunks shorter than min_char_length
      - If unified_game_id provided and schema supports unified field, filter by it
      - Apply similarity threshold (converted from Weaviate's 'certainty' or distance)
      - Return top-k (post-filter) sorted by score desc
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be non-empty string")
    if k <= 0:
        raise ValueError("k must be > 0")

    model = _load_model(model_name)
    try:
        vec = model.encode([query], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        logger.exception("Failed to encode query: %s", e)
        raise

    # inspect schema to request only valid props
    class_props = _weaviate_schema_properties(weaviate_url, class_name)
    if debug:
        logger.info("Schema properties for %s: %s", class_name, sorted(list(class_props)))

    desired_props = ["title", "text", "meta", "char_length", "doc_id", "content_hash", "id"]
    requested_props = [p for p in desired_props if p in class_props]
    # Always ask for _additional
    prop_list = ["_additional { distance certainty id }"] + requested_props
    props_gql = "\n          ".join(prop_list)

    fetch_limit = max(k * int(fetch_multiplier), k, 10)

    # Determine unified prop if present
    unified_prop_name = None
    for candidate in ("unified_id", "unified_game_id", "unified"):
        if candidate in class_props:
            unified_prop_name = candidate
            break

    where_clause = ""
    if unified_game_id and unified_prop_name:
        safe_value = json.dumps(str(unified_game_id))
        where_clause = f", where: {{ path: [\"{unified_prop_name}\"], operator: Equal, valueString: {safe_value} }}"

    # Build and execute query
    gql = _build_near_vector_gql(class_name, vec, props_gql, fetch_limit, where_clause)
    if debug:
        logger.info("GraphQL query:\n%s", gql)

    response = _post_graphql(weaviate_url, gql)
    if not response.get("ok"):
        logger.warning("GraphQL request failed (status=%s). Response text may show error: %s", response.get("status_code"), response.get("text")[:1000])

    hits = []
    try:
        data = response.get("json", {})
        get_block = data.get("data", {}).get("Get", {}).get(class_name, [])
        if debug:
            logger.info("Raw hits len=%d", len(get_block))
        for item in get_block:
            additional = item.get("_additional", {}) or {}
            score = _score_from_additional(additional)
            parsed = {
                "id": additional.get("id") or item.get("id") or item.get("content_hash") or item.get("doc_id"),
                "title": item.get("title"),
                "content": item.get("text") or None,
                "meta": _parse_meta_prop(item.get("meta")),
                "char_length": item.get("char_length") or (len((item.get("text") or "") or "") if item.get("text") else None),
                "doc_id": item.get("doc_id"),
                "content_hash": item.get("content_hash"),
                "score": score,
                "_raw": item,
            }
            hits.append(parsed)
    except Exception as e:
        logger.exception("Failed to parse GraphQL response: %s", e)
        # proceed with whatever we have

    # Gate 1: drop very short chunks or missing content
    filtered = []
    for h in hits:
        clen = h.get("char_length")
        if clen is None:
            # Try to compute from content if available
            if h.get("content"):
                clen = len(h["content"])
                h["char_length"] = clen
            else:
                # If meta contains text-like fields try to derive a length
                meta_text = json.dumps(h.get("meta") or {})
                if meta_text:
                    h["char_length"] = len(meta_text)
                else:
                    h["char_length"] = 0
        if h["char_length"] < min_char_length:
            if debug:
                logger.debug("Dropping chunk %s due to short length %s", h.get("id"), h["char_length"])
            continue
        filtered.append(h)

    # Gate 2: similarity threshold
    if similarity_threshold is not None:
        post_thresh = []
        for h in filtered:
            s = h.get("score")
            if s is None:
                # if we don't have score try to conservatively include (or drop?) -> we drop
                if debug:
                    logger.debug("Dropping chunk %s due to missing score", h.get("id"))
                continue
            if s >= similarity_threshold:
                post_thresh.append(h)
            else:
                if debug:
                    logger.debug("Dropping chunk %s due to low similarity (%s < %s)", h.get("id"), s, similarity_threshold)
        filtered = post_thresh

    # Sort by score desc
    filtered.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)

    # Return top-k
    final_results = filtered[:k]

    if debug:
        logger.info("Returning %d results (requested k=%d) after gates", len(final_results), k)

    if show_meta:
        for r in final_results:
            logger.info("Row: id=%s score=%s char_length=%s meta=%s", r.get("id"), r.get("score"), r.get("char_length"), json.dumps(r.get("meta") or {})[:400])

    return final_results


# ---------------------------
# Prompt template builder (FEW-SHOT + clearer citation token)
# ---------------------------

def format_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build a templated RAG prompt using a 'Comprehensive Analyst' persona, and
    few-shot examples to teach the model the desired inline citation format.

    New inline citation token: <<SRC:ID>>
      - Place immediately after any factual claim, with no space:
        Example: "Bob is the CEO.<<SRC:123>>"

    Important: The model MUST use the chunk ID as shown in the "ID:" line
    in the Context block. To make this explicit and unambiguous, this
    function forces the Context 'SRC' line to equal the chunk ID.
    """
    system_instr = (
        "You are the Comprehensive Analyst — thorough, methodical, and synthesis-first. "
        "Produce clear, structured answers that reason over the provided Context. Prioritize "
        "synthesis and explanation rather than terse one-line replies. Use the Context "
        "sections exclusively as evidence; do not invent facts beyond what the Context supports.\n\n"
        "INLINE CITATION FORMAT (strict): Use the token sequence <<SRC:ID>> immediately "
        "after any factual claim that is supported by the Context. 'ID' must be exactly the chunk's "
        "ID shown in the Context's 'ID:' line (for example: dd7f60c5-... or chunk_12). "
        "There must be NO space between the claim and the citation token. If you cannot support a claim from the Context, write: "
        "'No verifiable information found in the provided context.'\n\n"
        "FEW-SHOT EXAMPLES:\n"
        "User: Who is CEO?\n"
        "Context:\n"
        "ID: 123\n"
        "Text: Bob is CEO of ExampleCorp.\n"
        "Assistant: Bob is the CEO of ExampleCorp.<<SRC:123>>\n\n"
        "User: When was the game released?\n"
        "Context:\n"
        "ID: abc\n"
        "Text: The game shipped on 2019-11-12.\n"
        "Assistant: The game was released on 2019-11-12.<<SRC:abc>>\n\n"
        "End examples. Now answer using only the provided Context and include inline citations."
    )

    # Build Context body
    ctx_parts = []
    for i, c in enumerate(chunks, start=1):
        cid = c.get("id") or f"chunk{i}"
        # CRITICAL: force src to equal the displayed chunk ID so the model must cite that ID
        src = cid
        score = c.get("score") if c.get("score") is not None else 0.0
        content = c.get("content") or ""
        snippet = (content.strip()[:1200] + "...") if len(content) > 1200 else content.strip()
        ctx_parts.append(
            f"---\nID: {cid}\nSRC: {src}\nSCORE: {float(score):.4f}\n\n{snippet}\n---"
        )

    context_block = "\n\n".join(ctx_parts) if ctx_parts else "<NO_CONTEXT>"

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_instr}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Context:\n"
        f"{context_block}\n\n"
        f"Question: {query}\n"
        "Answer with explicit inline citations using <<SRC:ID>> immediately after each factual claim.\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt


# ---------------------------
# Modal generation (IMPROVED streaming fallback)
# ---------------------------

def _consume_async_generator_and_collect(gen_async) -> str:
    """
    Helper to run an async generator synchronously and collect output,
    while printing chunks as they arrive.
    """
    async def _consumer(ag):
        accumulated = []
        async for chunk in ag:
            try:
                s = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            except Exception:
                s = str(chunk)
            sys.stdout.write(s)
            sys.stdout.flush()
            accumulated.append(s)
        return "".join(accumulated)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside an existing loop; create a new loop for this operation
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(_consumer(gen_async))
            finally:
                new_loop.close()
        else:
            return loop.run_until_complete(_consumer(gen_async))
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(_consumer(gen_async))
        finally:
            new_loop.close()


def generate_answer_via_modal(prompt: str, max_tokens: int = 512, temperature: float = 0.1, stream: bool = False) -> str:
    """
    Soft-import modal and call the remote function.

    If stream=True, attempt to call remote_gen (which returns an async or sync generator).
    If streaming is not supported or fails, fall back to blocking .remote(...) and return the full result.
    """
    try:
        import modal  # type: ignore
    except Exception as e:
        raise ImportError("Modal client not available (import modal failed). Install 'modal' to enable generation.") from e

    # NOTE: change these to match your actual Modal app/function identification
    FUNCTION_APP_NAME = "rag-llama3-3b"
    FUNCTION_NAME = "chat_completion_remote"

    try:
        chat_fn = modal.Function.from_name(FUNCTION_APP_NAME, FUNCTION_NAME)
    except Exception as e:
        raise RuntimeError(f"Failed to lookup remote function {FUNCTION_APP_NAME}/{FUNCTION_NAME}: {e}") from e

    logger.info("Waking up Agent (remote Modal app may be cold-starting)...")

    # Attempt streaming if requested
    if stream:
        remote_gen = getattr(chat_fn, "remote_gen", None)
        if callable(remote_gen):
            try:
                gen = remote_gen(prompt, max_tokens=max_tokens, temperature=temperature)
                if hasattr(gen, "__aiter__"):
                    accumulated = _consume_async_generator_and_collect(gen)
                    if not accumulated.endswith("\n"):
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    return accumulated.strip()
                else:
                    # synchronous generator or iterable
                    accumulated_parts = []
                    for chunk in gen:
                        try:
                            s = chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk)
                        except Exception:
                            s = str(chunk)
                        sys.stdout.write(s)
                        sys.stdout.flush()
                        accumulated_parts.append(s)
                    accumulated = "".join(accumulated_parts)
                    if not accumulated.endswith("\n"):
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                    return accumulated.strip()
            except Exception as e:
                logger.warning("Streaming (remote_gen) attempt failed; falling back to blocking remote() call: %s", e)
        else:
            logger.info("Modal remote_gen not available on function object; falling back to blocking remote().")

    # Blocking fallback
    try:
        result = chat_fn.remote(prompt, max_tokens=max_tokens, temperature=temperature)
        if result is None:
            return ""
        if stream:
            # parity with streaming: print the single result
            sys.stdout.write(str(result).strip() + "\n")
            sys.stdout.flush()
        return str(result).strip()
    except Exception as e:
        raise RuntimeError(f"Remote generation failed: {e}") from e


# ---------------------------
# CLI Entrypoint
# ---------------------------

def _build_arg_parser():
    p = argparse.ArgumentParser(prog="retriever", description="Schema-accurate Hardened Retriever for GameChunk (with optional generation)")
    p.add_argument("--query", "-q", required=True, help="Query text")
    p.add_argument("--k", type=int, default=5, help="Top-k to return (post-filter)")
    p.add_argument("--weaviate", default=DEFAULT_WEAVIATE_URL, help="Weaviate URL, e.g. http://localhost:8080")
    p.add_argument("--class-name", default=DEFAULT_CLASS, help="Weaviate class name (default GameChunk)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformers model name")
    p.add_argument("--min-char-length", type=int, default=50, help="Minimum chunk char length (Gate 2)")
    p.add_argument("--similarity-threshold", type=float, default=0.6, help="Minimum similarity score to keep (Gate 3)")
    p.add_argument("--unified-game-id", default=None, help="Optional unified_game_id - will map to schema's unified_id if present")
    p.add_argument("--fetch-multiplier", type=int, default=2, help="Multiplier for initial fetch limit (fetch_limit = k * fetch_multiplier)")
    p.add_argument("--debug", action="store_true", help="Print GraphQL query, full response, and run probes if needed")
    p.add_argument("--show-meta", action="store_true", help="Log parsed meta for each returned row (useful when text is null)")
    p.add_argument("--use-hybrid", action="store_true", help="Attempt Weaviate hybrid search (BM25 + vector) first, fallback to vector-only if unsupported")
    p.add_argument("--hybrid-alpha", type=float, default=0.5, help="Alpha for hybrid operator (0.1). Higher => more lexical emphasis")

    # Generation options
    p.add_argument("--generate", action="store_true", help="If set, synthesize a natural language answer using the remote Modal LLM (rag-llama3-3b).")
    p.add_argument("--gen-max-tokens", type=int, default=512, help="Max tokens for LLM generation")
    p.add_argument("--gen-temp", type=float, default=0.1, help="Temperature for generation")

    # Show assembled prompt
    p.add_argument("--show-prompt", action="store_true", help="Print the assembled RAG prompt to stdout before remote generation (debugging).")

    # Streaming flag
    p.add_argument("--stream", action="store_true", help="If set, attempt to stream the model response as it is generated (requires remote function supporting generator output).")

    return p


def _print_rows(rows: List[Dict[str, Any]]):
    print(f"\nReturned {len(rows)} items\n")
    for i, r in enumerate(rows, start=1):
        print(f"=== {i}. id={r.get('id')} score={(r.get('score') or 0.0):.4f} char_length={r.get('char_length')}")
        if r.get("title"):
            print("title:", r.get("title"))
        if r.get("content"):
            print("content (first 200 chars):")
            print((r.get("content") or "")[:200].replace("\n", " "))
        else:
            print("content: <none stored in 'text'>")
            print("doc_id:", r.get("doc_id"))
            print("content_hash:", r.get("content_hash"))
            if r.get("meta"):
                print("meta (snippet):", json.dumps(r.get("meta") or {}, indent=2)[:1000])


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        rows = retrieve(
            args.query,
            k=args.k,
            weaviate_url=args.weaviate,
            class_name=args.class_name,
            model_name=args.model,
            min_char_length=args.min_char_length,
            similarity_threshold=args.similarity_threshold,
            unified_game_id=args.unified_game_id,
            fetch_multiplier=args.fetch_multiplier,
            debug=args.debug,
            show_meta=args.show_meta,
            use_hybrid=args.use_hybrid,
            hybrid_alpha=args.hybrid_alpha,
        )

        _print_rows(rows)

        if args.generate:
            if not rows:
                print("\n[generate] No retrieved chunks to synthesize from. Aborting generation.")
            else:
                prompt = format_rag_prompt(args.query, rows)

                if args.show_prompt:
                    print("\n--- RAG Prompt (BEGIN) ---\n")
                    print(prompt)
                    print("\n--- RAG Prompt (END) ---\n")

                try:
                    answer = generate_answer_via_modal(
                        prompt,
                        max_tokens=args.gen_max_tokens,
                        temperature=args.gen_temp,
                        stream=args.stream,
                    )
                    if not args.stream:
                        print("\n--- Generated Answer (Modal LLM) ---\n")
                        print(answer)
                        print("\n--- End Generated Answer ---\n")
                    else:
                        # When streaming, the generator printed output live; still print end marker
                        print("\n--- End Generated Answer (stream) ---\n")
                except ImportError as ie:
                    logger.warning("Generation disabled: %s", ie)
                    print("\n[generate] Modal client missing. Install 'modal' if you want to enable remote generation.")
                except Exception as e:
                    logger.exception("Generation attempt failed: %s", e)
                    print(f"\n[generate] Generation failed: {e}")

    except WeaviateConnectionError as e:
        logger.exception("Weaviate unreachable: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        sys.exit(1)