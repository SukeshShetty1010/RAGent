# retriever.py (RAG Agent refactor with optional generation)
# Based on original hardened retriever and Modal LLM remote function.
# Keeps all prior behavior; new flag --generate will call remote LLM to synthesize answer.
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_CLASS = "GameChunk"

# ---------------------------
# Existing helper defs (unchanged, copied from original retriever)
# ---------------------------

class WeaviateConnectionError(Exception):
    """Raised when we cannot reach the configured Weaviate HTTP endpoint."""
    pass


def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


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
    except Exception as e:
        logger.debug("Failed to fetch schema: %s", e)
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
    if "certainty" in additional and additional["certainty"] is not None:
        try:
            return float(additional["certainty"])
        except Exception:
            pass
    if "distance" in additional and additional["distance"] is not None:
        try:
            d = float(additional["distance"])
            return 1.0 / (1.0 + d) if d >= 0 else None
        except Exception:
            pass
    return None


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


# ---------------------------
# Core retrieval (unchanged logic; preserved)
# ---------------------------

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
    Hardened retrieval with 3 gates + local fallback scoring.
    (This implementation is kept identical to your original logic.)
    """
    # --- validations & embedding of query ---
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be > 0")
    if fetch_multiplier < 1:
        raise ValueError("fetch_multiplier must be >= 1")
    if not (0.0 <= hybrid_alpha <= 1.0):
        raise ValueError("hybrid_alpha must be between 0.0 and 1.0")

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

    desired_props = ["title", "text", "meta", "char_length", "doc_id", "content_hash"]
    requested_props = [p for p in desired_props if p in class_props]
    if "char_length" in class_props and "char_length" not in requested_props:
        requested_props.append("char_length")

    prop_list = ["_additional { distance certainty id }"] + requested_props
    props_gql = "\n          ".join(prop_list)

    fetch_limit = max(k * int(fetch_multiplier), k)

    unified_prop_name = None
    for candidate in ("unified_id", "unified_game_id", "unified"):
        if candidate in class_props:
            unified_prop_name = candidate
            break

    where_clause = ""
    if unified_game_id and unified_prop_name:
        safe_value = json.dumps(str(unified_game_id))
        where_clause = f", where: {{ path: [\"{unified_prop_name}\"], operator: Equal, valueString: {safe_value} }}"

    vec_list_literal = ", ".join([repr(float(v)) for v in vec])

    response_bundle = None
    used_mode = "None"
    last_error = None

    def _near_vector_query(limit: int = fetch_limit) -> str:
        return f"""
        {{
          Get {{
            {class_name}(nearVector: {{ vector: [{vec_list_literal}] }}{where_clause}, limit: {int(limit)}) {{
              {props_gql}
            }}
          }}
        }}
        """

    def _hybrid_query(limit: int = fetch_limit) -> str:
        safe_q = json.dumps(str(query))
        return f"""
        {{
          Get {{
            {class_name}(hybrid: {{ query: {safe_q}, vector: [{vec_list_literal}], alpha: {float(hybrid_alpha)} }}{where_clause}, limit: {int(limit)}) {{
              {props_gql}
            }}
          }}
        }}
        """

    # Attempt hybrid (if requested) then fallback
    if use_hybrid:
        gql_h = _hybrid_query()
        if debug:
            logger.info("GraphQL query (hybrid):\n%s", gql_h)
        try:
            response_bundle = _post_graphql(weaviate_url, gql_h)
            resp_json = response_bundle["json"]
            if isinstance(resp_json, dict) and "errors" in resp_json and resp_json.get("errors"):
                logger.warning("Hybrid GraphQL returned errors: %s", json.dumps(resp_json.get("errors"), default=str)[:2000])
                last_error = resp_json.get("errors")
                response_bundle = None
            else:
                used_mode = "Hybrid"
        except WeaviateConnectionError:
            raise
        except Exception as e:
            logger.warning("Hybrid search attempt failed: %s. Falling back to vector-only.", e)
            last_error = str(e)
            response_bundle = None

    if response_bundle is None:
        gql_v = _near_vector_query()
        if debug:
            logger.info("GraphQL query (nearVector):\n%s", gql_v)
        try:
            response_bundle = _post_graphql(weaviate_url, gql_v)
            used_mode = "Vector-Fallback" if use_hybrid else "Vector"
        except WeaviateConnectionError:
            raise
        except Exception as e:
            logger.exception("Vector search failed: %s", e)
            raise

    logger.info("Search mode used: %s", used_mode)

    resp_json = response_bundle["json"]

    if debug:
        logger.info("HTTP status=%s ok=%s", response_bundle["status_code"], response_bundle["ok"])
        try:
            logger.info("Weaviate response JSON (truncated): %s", json.dumps(resp_json, indent=2)[:2000])
        except Exception:
            logger.info("Weaviate response text (truncated): %s", str(response_bundle["text"])[:2000])

    if isinstance(resp_json, dict) and "errors" in resp_json:
        logger.warning("Weaviate GraphQL returned errors (post-search): %s", json.dumps(resp_json.get("errors"), default=str)[:2000])

    # extract items
    items: List[Dict[str, Any]] = []
    try:
        get_block = resp_json.get("data", {}).get("Get", {}) if isinstance(resp_json, dict) else {}
        if isinstance(get_block, dict) and class_name in get_block:
            items = get_block[class_name] or []
        else:
            if isinstance(get_block, dict):
                for v in get_block.values():
                    if isinstance(v, list):
                        items = v
                        break
    except Exception:
        items = []

    fetched_count = len(items)
    logger.info("Gate1 (DB fetch) - requested limit=%d, fetched=%d (where on %s=%s)", fetch_limit, fetched_count, unified_prop_name, str(unified_game_id))

    # debug probes if zero fetched
    if fetched_count == 0 and debug:
        logger.info("Fetched 0 items — running debug probes...")
        probe1 = f"""
        {{
          Get {{
            {class_name}(limit:1) {{
              _additional {{ id }}
              char_length
            }}
          }}
        }}
        """
        try:
            p1 = _post_graphql(weaviate_url, probe1)
            try:
                logger.info("Probe1 json (truncated): %s", json.dumps(p1["json"], indent=2)[:2000])
            except Exception:
                logger.info("Probe1 text (truncated): %s", str(p1["text"])[:2000])
        except WeaviateConnectionError:
            raise

        probe2 = f"""
        {{
          Get {{
            {class_name}(nearVector: {{ vector: [{vec_list_literal}] }}, limit: 1) {{
              _additional {{ distance certainty id }}
            }}
          }}
        }}
        """
        try:
            p2 = _post_graphql(weaviate_url, probe2)
            try:
                logger.info("Probe2 json (truncated): %s", json.dumps(p2["json"], indent=2)[:2000])
            except Exception:
                logger.info("Probe2 text (truncated): %s", str(p2["text"])[:2000])
        except WeaviateConnectionError:
            raise

    # Gate2 & Gate3 processing (robust scoring fallback)
    parsed_results: List[Dict[str, Any]] = []
    dropped_gate2 = 0
    dropped_gate3 = 0
    dropped_parse_fail = 0
    sample_additional = None

    def _extract_content_from_meta(parsed_meta: Dict[str, Any]) -> Optional[str]:
        if not parsed_meta:
            return None
        candidate_keys = ["text", "content", "raw_text", "body", "description", "excerpt"]
        for k in candidate_keys:
            v = parsed_meta.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        nested_paths = [("source_data", "text"), ("source_data", "content"), ("source_data", "body")]
        for path in nested_paths:
            node = parsed_meta
            found = True
            for p in path:
                if isinstance(node, dict) and p in node:
                    node = node[p]
                else:
                    found = False
                    break
            if found and isinstance(node, str) and node.strip():
                return node.strip()
        return None

    for obj in items:
        try:
            props = obj or {}
            additional = props.get("_additional") or {}
            if sample_additional is None:
                sample_additional = additional

            title = props.get("title") or None
            text_field = props.get("text") if "text" in props else None
            meta_prop = props.get("meta") or None
            parsed_meta = _parse_meta_prop(meta_prop)

            content_value = None
            if isinstance(text_field, str) and text_field.strip():
                content_value = text_field
            else:
                extracted = _extract_content_from_meta(parsed_meta)
                if extracted:
                    content_value = extracted

            score = _score_from_additional(additional)
            numeric_score = None

            if score is not None:
                try:
                    numeric_score = float(score)
                except Exception:
                    numeric_score = None

            if numeric_score is None:
                candidate_text_for_embedding = None
                if isinstance(text_field, str) and text_field.strip():
                    candidate_text_for_embedding = text_field
                elif content_value:
                    candidate_text_for_embedding = content_value
                else:
                    candidate_text_for_embedding = parsed_meta.get("text") or parsed_meta.get("content") or parsed_meta.get("raw")

                if candidate_text_for_embedding and candidate_text_for_embedding.strip():
                    try:
                        emb = model.encode([candidate_text_for_embedding], convert_to_numpy=True)[0]
                        qvec = np.array(vec, dtype=float)
                        emb_norm = np.linalg.norm(emb)
                        qvec_norm = np.linalg.norm(qvec)
                        if emb_norm > 0 and qvec_norm > 0:
                            cos_sim = float(np.dot(qvec, emb) / (qvec_norm * emb_norm))
                            numeric_score = (cos_sim + 1.0) / 2.0
                        else:
                            numeric_score = 0.0
                    except Exception as e:
                        logger.debug("Fallback embedding similarity failed: %s", e)
                        numeric_score = 0.0
                else:
                    numeric_score = 0.0

            char_length = None
            if "char_length" in props and props.get("char_length") is not None:
                try:
                    char_length = int(props.get("char_length"))
                except Exception:
                    try:
                        char_length = int(float(props.get("char_length")))
                    except Exception:
                        char_length = None
            if char_length is None:
                try:
                    cl = parsed_meta.get("char_length")
                    if cl is not None:
                        char_length = int(cl)
                except Exception:
                    char_length = None

            # Gate2: quality
            if char_length is None or char_length < int(min_char_length):
                dropped_gate2 += 1
                continue

            # Gate3: relevance
            if numeric_score < float(similarity_threshold):
                dropped_gate3 += 1
                continue

            wid = additional.get("id") or props.get("doc_id") or props.get("docId") or None

            parsed_results.append({
                "id": wid,
                "title": title,
                "content": (content_value or "")[:500],
                "score": numeric_score,
                "char_length": char_length,
                "raw_additional": additional,
                "meta": parsed_meta,
                "doc_id": parsed_meta.get("doc_id") or props.get("doc_id"),
                "content_hash": parsed_meta.get("content_hash"),
            })
        except Exception as e:
            logger.warning("Failed to parse/inspect object: %s", e)
            dropped_parse_fail += 1
            continue

    # sort & limit
    parsed_results.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)
    final_results = parsed_results[:k]

    logger.info(
        "Retrieval summary: fetched=%d, postfilter_candidates=%d, returned=%d, dropped_gate2=%d, dropped_gate3=%d, dropped_parse_fail=%d, mode=%s",
        fetched_count,
        len(parsed_results),
        len(final_results),
        dropped_gate2,
        dropped_gate3,
        dropped_parse_fail,
        used_mode,
    )

    if debug:
        logger.info("Sample _additional (first candidate): %s", json.dumps(sample_additional or {}, default=str))

    if show_meta and final_results:
        logger.info("Parsed meta for returned rows:")
        for r in final_results:
            try:
                logger.info("id=%s meta=%s", r.get("id"), json.dumps(r.get("meta") or {}, indent=2)[:2000])
            except Exception:
                logger.info("meta (raw) for id=%s: %s", r.get("id"), str(r.get("meta"))[:2000])

    return final_results


# ---------------------------
# New: Prompt Topology for Llama 3.2 (strict templating)
# ---------------------------

def format_rag_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Build a templated RAG prompt using the required Llama 3.2 tokens.

    Structure must follow:
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    ... (system instructions) <|eot_id|><|start_header_id|>user<|end_header_id|>
    Context: ... Question: ... <|eot_id|><|start_header_id|>assistant<|end_header_id|>

    The function injects a short, numbered context block of chunk ids + text (only chunks
    that passed filters should be passed here).
    """
    # System instruction - be explicit and conservative to reduce hallucination
    system_instr = (
        "You are a helpful, precise assistant. Use ONLY the provided Context sections "
        "to answer the user's question. When you cite facts, include the source id "
        "for traceability. If the context does not support an answer, reply: "
        "'No verifiable information found in the provided context.'"
    )

    # Build Context body (include id, score, and short content up to a sensible length)
    ctx_parts = []
    for i, c in enumerate(chunks, start=1):
        cid = c.get("id") or f"chunk{i}"
        score = c.get("score")
        content = c.get("content") or ""
        # keep context short per chunk to avoid token explosion
        snippet = (content.strip()[:1200] + "...") if len(content) > 1200 else content.strip()
        ctx_parts.append(f"---\nID: {cid}\nSCORE: {score:.4f}\n\n{snippet}\n---")

    context_block = "\n\n".join(ctx_parts) if ctx_parts else "<NO_CONTEXT>"

    # Now assemble with required tokens
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_instr}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        "Context:\n"
        f"{context_block}\n\n"
        f"Question: {query}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return prompt


# ---------------------------
# New: Remote generation flow (Modal lookup at runtime only)
# ---------------------------

def generate_answer_via_modal(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """
    Lookup and call Modal remote function dynamically (soft import).
    If modal is not installed or function lookup fails, raise ImportError/RuntimeError accordingly.
    """
    # Soft import (Dependency isolation)
    try:
        import modal  # type: ignore
    except Exception as e:
        raise ImportError("Modal client not available (import modal failed). Install 'modal' to enable generation.") from e

    # Use lookup pattern (function from name) to avoid importing a module that itself imports modal at top-level.
    try:
        chat_fn = modal.Function.from_name("rag-llama3-3b", "chat_completion_remote")
    except Exception as e:
        raise RuntimeError(f"Failed to lookup remote function: {e}") from e

    # Inform user about potential cold start
    logger.info("Waking up Agent (remote Modal app may be cold-starting)...")

    # Call the remote function using .remote(...) (correct Modal API)
    try:
        # Call signature mirrors modal_llm.chat_completion_remote(prompt, max_tokens, temperature)
        result = chat_fn.remote(prompt, max_tokens=max_tokens, temperature=temperature)
        if result is None:
            return ""
        return str(result).strip()
    except Exception as e:
        raise RuntimeError(f"Remote generation failed: {e}") from e


# ---------------------------
# CLI Entrypoint (extended)
# ---------------------------

if __name__ == "__main__":
    import argparse

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
    p.add_argument("--hybrid-alpha", type=float, default=0.5, help="Alpha for hybrid operator (0..1). Higher => more lexical emphasis")
    p.add_argument("--generate", action="store_true", help="If set, synthesize a natural language answer using the remote Modal LLM (rag-llama3-3b).")
    p.add_argument("--gen-max-tokens", type=int, default=512, help="Max tokens for LLM generation")
    p.add_argument("--gen-temp", type=float, default=0.1, help="Temperature for generation")
    args = p.parse_args()

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

        # Print retrieval results first (always)
        print(f"\nReturned {len(rows)} items\n")
        for i, r in enumerate(rows, start=1):
            print(f"=== {i}. id={r['id']} score={r['score']:.4f} char_length={r.get('char_length')}")
            print("title:", r.get("title"))
            if r.get("content"):
                print("content (first 200 chars):")
                print((r.get("content") or "")[:200].replace("\n", " "))
            else:
                print("content: <none stored in 'text' or extractable from meta>")
                print("doc_id:", r.get("doc_id"))
                print("content_hash:", r.get("content_hash"))
                if args.show_meta:
                    print("meta:", json.dumps(r.get("meta") or {}, indent=2)[:2000])

        # Generation path (optional)
        if args.generate:
            # Attempt to build prompt and call remote function
            if not rows:
                print("\n[generate] No retrieved chunks to synthesize from. Aborting generation.")
            else:
                # Assemble the prompt with only the chunks that passed filters
                prompt = format_rag_prompt(args.query, rows)
                try:
                    # Soft import is inside generate_answer_via_modal
                    answer = generate_answer_via_modal(prompt, max_tokens=args.gen_max_tokens, temperature=args.gen_temp)
                    print("\n--- Generated Answer (Modal LLM) ---\n")
                    print(answer)
                    print("\n--- End Generated Answer ---\n")
                except ImportError as ie:
                    logger.warning("Generation disabled: %s", ie)
                    print("\n[generate] Modal client missing. Install 'modal' if you want to enable remote generation.")
                except Exception as e:
                    logger.exception("Generation attempt failed: %s", e)
                    print(f"\n[generate] Generation failed: {e}")

    except WeaviateConnectionError as e:
        logger.exception("Weaviate unreachable: %s", e)
        raise
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        raise
