# RAG_ent/retriever/simple_retriever.py
"""
Vector-only retriever (SBERT -> Weaviate GraphQL HTTP).

Usage (from project root):
    python -m retriever.simple_retriever --query "far cry 5 gameplay" --k 5 --weaviate http://localhost:8080

Dependencies:
    pip install sentence-transformers requests
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer
import requests

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_CLASS = "GameChunk"


def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model


def _parse_meta_prop(meta_prop: Optional[Any]) -> Dict[str, Any]:
    """
    Try to parse meta property: could be dict or JSON-string or None.
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
    Prefer certainty if present, else compute 1/(1+distance) if distance present.
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


def _weaviate_health_check(weaviate_url: str) -> bool:
    """
    Lightweight health check using /v1/meta. Returns True if reachable.
    """
    try:
        meta_url = weaviate_url.rstrip("/") + "/v1/meta"
        r = requests.get(meta_url, timeout=5)
        return r.ok
    except Exception:
        return False


def retrieve(
    query: str,
    k: int = 5,
    weaviate_url: str = DEFAULT_WEAVIATE_URL,
    class_name: str = DEFAULT_CLASS,
    model_name: str = DEFAULT_MODEL,
    properties: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Vector-only retrieval pipeline using GraphQL HTTP POST to Weaviate.

    Returns list of dicts:
      {
        "id": "<weaviate-uuid>",
        "title": "<title or name>",
        "site_detail_url": "<meta.site_detail_url if present>",
        "content": "<text trimmed to 500 chars>",
        "score": <float or None>,
        "raw_additional": {...},
        "meta": {...}
      }
    """
    if not query or not isinstance(query, str):
        raise ValueError("query must be a non-empty string")

    model = _load_model(model_name)
    try:
        vec = model.encode([query], convert_to_numpy=True)[0].tolist()
    except Exception as e:
        logger.exception("Failed to encode query: %s", e)
        raise

    # health check (optional)
    if not _weaviate_health_check(weaviate_url):
        logger.warning("Weaviate health check failed for %s (continuing, request may still work)", weaviate_url)

    # Build GraphQL query
    requested_props = properties or ["title", "text", "meta"]
    # _additional we request distance and certainty and id for reference
    prop_list = ["_additional { distance certainty id }"] + requested_props
    props_gql = "\n          ".join(prop_list)

    # Build vector literal: ensure floats are JSON-friendly and not in scientific that GraphQL parser dislikes
    vec_list_literal = ", ".join([repr(float(v)) for v in vec])

    graphql_query = f"""
    {{
      Get {{
        {class_name}(nearVector: {{ vector: [{vec_list_literal}] }}, limit: {int(k)}) {{
          {props_gql}
        }}
      }}
    }}
    """

    gw_url = weaviate_url.rstrip("/") + "/v1/graphql"

    try:
        headers = {"Content-Type": "application/json"}
        r = requests.post(gw_url, json={"query": graphql_query}, headers=headers, timeout=30)
        r.raise_for_status()
        resp = r.json()
    except Exception as e:
        logger.exception("Weaviate GraphQL HTTP request failed: %s", e)
        raise

    # Parse response
    items = []
    try:
        get_block = resp.get("data", {}).get("Get", {})
        if isinstance(get_block, dict) and class_name in get_block:
            items = get_block[class_name] or []
        else:
            # try to find any list in Get
            for v in get_block.values():
                if isinstance(v, list):
                    items = v
                    break
    except Exception:
        items = []

    results: List[Dict[str, Any]] = []
    for obj in items:
        try:
            props = obj or {}
            # GraphQL returns properties in the same object; _additional is nested under _additional
            additional = props.get("_additional") or {}
            # properties requested are top-level in the returned object
            title = props.get("title") or props.get("name") or None
            text_prop = props.get("text") or props.get("content") or ""
            meta_prop = props.get("meta") or props.get("metadata") or None
            parsed_meta = _parse_meta_prop(meta_prop)
            site_detail_url = parsed_meta.get("site_detail_url") or parsed_meta.get("site_detail_url".lower()) or None

            score = _score_from_additional(additional)

            # id may be under additional.id or not present
            wid = additional.get("id") or props.get("id") or None

            results.append({
                "id": wid,
                "title": title,
                "site_detail_url": site_detail_url,
                "content": (text_prop or "")[:500],
                "score": score,
                "raw_additional": additional,
                "meta": parsed_meta,
            })
        except Exception as e:
            logger.warning("Failed to parse one object: %s", e)
            continue

    return results


# CLI
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="simple_retriever", description="Vector-only retriever (SBERT -> Weaviate GraphQL)")
    p.add_argument("--query", "-q", required=True, help="Query text")
    p.add_argument("--k", type=int, default=5, help="Top-k to return")
    p.add_argument("--weaviate", default=DEFAULT_WEAVIATE_URL, help="Weaviate URL, e.g. http://localhost:8080")
    p.add_argument("--class-name", default=DEFAULT_CLASS, help="Weaviate class name (default GameChunk)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformers model name")
    args = p.parse_args()

    try:
        rows = retrieve(args.query, k=args.k, weaviate_url=args.weaviate, class_name=args.class_name, model_name=args.model)
        print(f"Retrieved {len(rows)} items")
        for i, r in enumerate(rows, start=1):
            print(f"\n=== {i}. id={r['id']} score={r['score']}")
            print("title:", r.get("title"))
            print("site_detail_url:", r.get("site_detail_url"))
            print("content (first 200 chars):")
            print((r.get("content") or "")[:200].replace("\n", " "))
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        raise
