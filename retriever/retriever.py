# retriever.py (refactored: hybrid search + fallback + WeaviateConnectionError + local scoring fallback)
"""
Schema-accurate hardened retriever for GameChunk (Weaviate).
- Adds optional hybrid search (BM25 + vector) with automatic fallback to vector-only.
- Raises WeaviateConnectionError when Weaviate is unreachable.
- Telemetry logs whether response came from 'Hybrid' or 'Vector-Fallback'.
- Keeps original 3-gate filtering: scope (DB where), quality (char_length), relevance (score).
- Adds local cosine-similarity fallback when Weaviate returns no certainty/distance for hybrid results.
"""
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


class WeaviateConnectionError(Exception):
    """Raised when we cannot reach the configured Weaviate HTTP endpoint."""
    pass


def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    return SentenceTransformer(model_name)


def _weaviate_schema_properties(weaviate_url: str, class_name: str, timeout: int = 5) -> Set[str]:
    """
    Return set of property names for the given Weaviate class via /v1/schema.
    If schema request fails, return an empty set.
    Raises WeaviateConnectionError on connection problems.
    """
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
        # Circuit breaking behaviour: surface connection errors to callers
        raise WeaviateConnectionError(f"Failed to connect to Weaviate at {weaviate_url}: {e}")
    except Exception as e:
        logger.debug("Failed to fetch schema: %s", e)
    return set()


def _parse_meta_prop(meta_prop: Optional[Any]) -> Dict[str, Any]:
    """
    Meta in your schema is a free-form JSON string. Parse if possible.
    """
    if not meta_prop:
        return {}
    if isinstance(meta_prop, dict):
        return meta_prop
    try:
        return json.loads(meta_prop)
    except Exception:
        return {"raw": str(meta_prop)}


def _score_from_additional(additional: Dict[str, Any]) -> Optional[float]:
    """
    Prefer 'certainty' if present, otherwise convert 'distance' to a 0..1-ish score.
    """
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
    """
    POST GraphQL query to Weaviate and return a response bundle.
    Converts requests ConnectionError into WeaviateConnectionError (circuit-break).
    """
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
    Perform a hardened retrieval with 3 gates and telemetry.

    New Args:
      use_hybrid: attempt Weaviate hybrid search (BM25 + vector) first when True.
      hybrid_alpha: alpha weight for hybrid operator (0..1). Higher -> more lexical emphasis.

    All other args and behaviour retained from original implementation.
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")
    if k <= 0:
        raise ValueError("k must be > 0")
    if fetch_multiplier < 1:
        raise ValueError("fetch_multiplier must be >= 1")
    if not (0.0 <= hybrid_alpha <= 1.0):
        raise ValueError("hybrid_alpha must be between 0.0 and 1.0")

    # load encoder
    model = _load_model(model_name)
    try:
        vec = model.encode([query], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        logger.exception("Failed to encode query: %s", e)
        raise

    # inspect schema to request only valid props (prevents GraphQL unknown-field errors)
    class_props = _weaviate_schema_properties(weaviate_url, class_name)
    if debug:
        logger.info("Schema properties for %s: %s", class_name, sorted(list(class_props)))

    desired_props = ["title", "text", "meta", "char_length", "doc_id", "content_hash"]
    requested_props = [p for p in desired_props if p in class_props]
    if "char_length" in class_props and "char_length" not in requested_props:
        requested_props.append("char_length")

    # always request _additional
    prop_list = ["_additional { distance certainty id }"] + requested_props
    props_gql = "\n          ".join(prop_list)

    # fetch limit
    fetch_limit = max(k * int(fetch_multiplier), k)

    # unified id mapping
    unified_prop_name = None
    for candidate in ("unified_id", "unified_game_id", "unified"):
        if candidate in class_props:
            unified_prop_name = candidate
            break

    where_clause = ""
    if unified_game_id and unified_prop_name:
        safe_value = json.dumps(str(unified_game_id))
        where_clause = f", where: {{ path: [\"{unified_prop_name}\"], operator: Equal, valueString: {safe_value} }}"

    # build vector literal safely
    vec_list_literal = ", ".join([repr(float(v)) for v in vec])

    # Strategy attempt loop: try hybrid (if requested) then fallback to nearVector
    response_bundle = None
    used_mode = "None"
    last_error = None

    # Helper to construct the nearVector GraphQL
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

    # Helper to construct the hybrid GraphQL (safe query JSON-escaped)
    def _hybrid_query(limit: int = fetch_limit) -> str:
        # json.dumps ensures the string is escaped properly for insertion into GraphQL
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

    # Attempt 1: Hybrid (if requested)
    if use_hybrid:
        gql_h = _hybrid_query()
        if debug:
            logger.info("GraphQL query (hybrid):\n%s", gql_h)
        try:
            response_bundle = _post_graphql(weaviate_url, gql_h)
            resp_json = response_bundle["json"]
            # If GraphQL returned 'errors' we treat as failure of hybrid and fall back
            if isinstance(resp_json, dict) and "errors" in resp_json and resp_json.get("errors"):
                logger.warning("Hybrid GraphQL returned errors: %s", json.dumps(resp_json.get("errors"), default=str)[:2000])
                last_error = resp_json.get("errors")
                response_bundle = None
                # fall through to vector fallback
            else:
                used_mode = "Hybrid"
        except WeaviateConnectionError:
            # re-raise connection problem immediately (circuit-break)
            raise
        except Exception as e:
            # any other error -> log and fall back
            logger.warning("Hybrid search attempt failed: %s. Falling back to vector-only.", e)
            last_error = str(e)
            response_bundle = None

    # Attempt 2: nearVector (vector fallback or default)
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
            # cannot reach or parse; raise for caller to handle
            logger.exception("Vector search failed: %s", e)
            raise

    # log which mode succeeded
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

    # extract items (same as original)
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

    # if zero fetched and debug, run probes (reuse original logic)
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

    # Gate2 & Gate3 processing (with robust scoring fallback)
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

            # content determination
            content_value = None
            if isinstance(text_field, str) and text_field.strip():
                content_value = text_field
            else:
                extracted = _extract_content_from_meta(parsed_meta)
                if extracted:
                    content_value = extracted

            # --- robust scoring fallback: prefer Weaviate scores, otherwise compute local cosine ---
            score = _score_from_additional(additional)
            numeric_score = None

            # 1) Use Weaviate-provided score when available
            if score is not None:
                try:
                    numeric_score = float(score)
                except Exception:
                    numeric_score = None

            # 2) Fallback: compute cosine similarity between query vector and chunk text embedding
            if numeric_score is None:
                # choose text to embed: prefer stored 'text' field, else parsed content_value, else meta.text/content/raw
                candidate_text_for_embedding = None
                if isinstance(text_field, str) and text_field.strip():
                    candidate_text_for_embedding = text_field
                elif content_value:
                    candidate_text_for_embedding = content_value
                else:
                    candidate_text_for_embedding = parsed_meta.get("text") or parsed_meta.get("content") or parsed_meta.get("raw")

                if candidate_text_for_embedding and candidate_text_for_embedding.strip():
                    try:
                        # encode the candidate text using the same model that encoded the query
                        emb = model.encode([candidate_text_for_embedding], convert_to_numpy=True)[0]
                        qvec = np.array(vec, dtype=float)
                        emb_norm = np.linalg.norm(emb)
                        qvec_norm = np.linalg.norm(qvec)
                        if emb_norm > 0 and qvec_norm > 0:
                            cos_sim = float(np.dot(qvec, emb) / (qvec_norm * emb_norm))
                            # normalize from [-1,1] -> [0,1] so existing 0..1 thresholds remain valid
                            numeric_score = (cos_sim + 1.0) / 2.0
                        else:
                            numeric_score = 0.0
                    except Exception as e:
                        logger.debug("Fallback embedding similarity failed: %s", e)
                        numeric_score = 0.0
                else:
                    # nothing available to embed — give minimal score
                    numeric_score = 0.0
            # --- end robust scoring fallback ---

            # char_length extraction (schema has char_length int)
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

    # sort and limit to k
    parsed_results.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)
    final_results = parsed_results[:k]

    # telemetry
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

    # optionally show parsed meta for returned rows
    if show_meta and final_results:
        logger.info("Parsed meta for returned rows:")
        for r in final_results:
            try:
                logger.info("id=%s meta=%s", r.get("id"), json.dumps(r.get("meta") or {}, indent=2)[:2000])
            except Exception:
                logger.info("meta (raw) for id=%s: %s", r.get("id"), str(r.get("meta"))[:2000])

    return final_results


# CLI entrypoint (unchanged aside from new args)
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="retriever", description="Schema-accurate Hardened Retriever for GameChunk")
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
        print(f"Returned {len(rows)} items")
        for i, r in enumerate(rows, start=1):
            print(f"\n=== {i}. id={r['id']} score={r['score']} char_length={r.get('char_length')}")
            print("title:", r.get("title"))
            print("site_detail_url:", r.get("site_detail_url"))
            if r.get("content"):
                print("content (first 200 chars):")
                print((r.get("content") or "")[:200].replace("\n", " "))
            else:
                print("content: <none stored in 'text' or extractable from meta>")
                print("doc_id:", r.get("doc_id"))
                print("content_hash:", r.get("content_hash"))
                if args.show_meta:
                    print("meta:", json.dumps(r.get("meta") or {}, indent=2)[:2000])
    except WeaviateConnectionError as e:
        logger.exception("Weaviate unreachable: %s", e)
        raise
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        raise
