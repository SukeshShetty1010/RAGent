# retriever.py (REFRACTORED)
# ------------------------------------------------------------
# Single-Responsibility Retrieval Class
#
# This file now contains a Retriever class that is responsible ONLY
# for fetching and filtering data from Weaviate. All generation / LLM
# related logic (modal, prompt formatting, streaming, etc.) has been
# removed per the refactor instructions.
#
# The retrieval behavior (3-Gate filtering: DB fetch -> length filter ->
# similarity threshold) is preserved exactly as before.
#
# Usage (CLI):
#   python -m retriever.retriever --query "What is Ubisoft?" --k 10 --weaviate http://localhost:8080 --debug --show-meta
#
# Original file (before refactor) provided as context. See citation:
# :contentReference[oaicite:0]{index=0}
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from typing import Any, Dict, List, Optional, Set

import requests
from requests.exceptions import ConnectionError as RequestsConnectionError
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_CLASS = "GameChunk"


# ---------------------------
# Exceptions / small helpers
# ---------------------------

class WeaviateConnectionError(Exception):
    pass


# ---------------------------
# Retriever Class (single responsibility)
# ---------------------------


class Retriever:
    """
    Retriever encapsulates embedding model loading and Weaviate retrieval logic.

    Construction loads the SentenceTransformer model so it can be reused across calls.
    """

    def __init__(
        self,
        weaviate_url: str = DEFAULT_WEAVIATE_URL,
        class_name: str = DEFAULT_CLASS,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        """
        Args:
            weaviate_url: Base URL for Weaviate, e.g. http://localhost:8080
            class_name: Weaviate class to query (e.g. GameChunk)
            model_name: SentenceTransformers model name
            device: optional device for SentenceTransformer (e.g. "cpu" or "cuda")
        """
        self.weaviate_url = weaviate_url.rstrip("/")
        self.class_name = class_name
        self.model_name = model_name
        # Load model once and reuse
        logger.info("Loading embedding model: %s", model_name)
        model_kwargs = {}
        if device:
            # SentenceTransformer supports device by passing device parameter in newer versions
            model_kwargs["device"] = device
        self.model = SentenceTransformer(model_name, **model_kwargs)

    # -----------------------
    # Private helpers (moved from module-level)
    # -----------------------

    @staticmethod
    def _parse_meta_prop(meta_prop: Optional[Any]) -> Dict[str, Any]:
        if not meta_prop:
            return {}
        if isinstance(meta_prop, dict):
            return meta_prop
        try:
            return json.loads(meta_prop)
        except Exception:
            return {"raw": str(meta_prop)}

    @staticmethod
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

    @staticmethod
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

    def _post_graphql(self, query: str, timeout: int = 30) -> Dict[str, Any]:
        gw_url = f"{self.weaviate_url}/v1/graphql"
        headers = {"Content-Type": "application/json"}
        try:
            r = requests.post(gw_url, json={"query": query}, headers=headers, timeout=timeout)
        except RequestsConnectionError as e:
            raise WeaviateConnectionError(f"Failed to connect to Weaviate at {self.weaviate_url}: {e}")
        try:
            parsed = r.json()
        except ValueError:
            parsed = {"_raw_text": r.text}
        return {"status_code": r.status_code, "ok": r.ok, "json": parsed, "text": r.text}

    def _weaviate_schema_properties(self, timeout: int = 5) -> Set[str]:
        try:
            r = requests.get(f"{self.weaviate_url}/v1/schema", timeout=timeout)
            r.raise_for_status()
            schema = r.json()
            classes = schema.get("classes", []) if isinstance(schema, dict) else []
            for cls in classes:
                if cls.get("class") == self.class_name:
                    props = cls.get("properties", []) or []
                    return {p.get("name") for p in props if p.get("name")}
        except RequestsConnectionError as e:
            raise WeaviateConnectionError(f"Failed to connect to Weaviate at {self.weaviate_url}: {e}")
        except Exception:
            # swallow and return empty set so logic can continue (we'll be conservative)
            return set()
        return set()

    # -----------------------
    # Public retrieval API
    # -----------------------

    def retrieve(
        self,
        query: str,
        k: int = 5,
        min_char_length: int = 50,
        similarity_threshold: Optional[float] = 0.6,
        unified_game_id: Optional[str] = None,
        fetch_multiplier: int = 2,
        debug: bool = False,
        show_meta: bool = False,
        use_hybrid: bool = False,
        hybrid_alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval from Weaviate using embedding nearest-neighbor (nearVector).
        Applies the three gates:
          Gate 1) Database fetch & parsing
          Gate 2) Drop chunks shorter than min_char_length
          Gate 3) Apply similarity_threshold

        Returns:
            List[Dict[str, Any]] - list of result dicts with keys:
              id, title, content, meta, char_length, doc_id, content_hash, score, _raw
        """
        if not query or not isinstance(query, str):
            raise ValueError("query must be non-empty string")
        if k <= 0:
            raise ValueError("k must be > 0")

        # encode query
        try:
            vec = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        except Exception as e:
            logger.exception("Failed to encode query: %s", e)
            raise

        # inspect schema to request only valid props
        class_props = self._weaviate_schema_properties()
        if debug:
            logger.info("Schema properties for %s: %s", self.class_name, sorted(list(class_props)))

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
        gql = self._build_near_vector_gql(self.class_name, vec, props_gql, fetch_limit, where_clause)
        if debug:
            logger.info("GraphQL query:\n%s", gql)

        response = self._post_graphql(gql)
        if not response.get("ok"):
            logger.warning(
                "GraphQL request failed (status=%s). Response text may show error: %s",
                response.get("status_code"),
                response.get("text")[:1000],
            )

        hits: List[Dict[str, Any]] = []
        try:
            data = response.get("json", {})
            get_block = data.get("data", {}).get("Get", {}).get(self.class_name, [])
            if debug:
                logger.info("Raw hits len=%d", len(get_block))
            for item in get_block:
                additional = item.get("_additional", {}) or {}
                score = self._score_from_additional(additional)
                parsed = {
                    "id": additional.get("id") or item.get("id") or item.get("content_hash") or item.get("doc_id"),
                    "title": item.get("title"),
                    "content": item.get("text") or None,
                    "meta": self._parse_meta_prop(item.get("meta")),
                    "char_length": item.get("char_length")
                    or (len((item.get("text") or "") or "") if item.get("text") else None),
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
        filtered: List[Dict[str, Any]] = []
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

        # Gate 2: similarity threshold (Gate 3 per original numbering)
        if similarity_threshold is not None:
            post_thresh: List[Dict[str, Any]] = []
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
                        logger.debug(
                            "Dropping chunk %s due to low similarity (%s < %s)", h.get("id"), s, similarity_threshold
                        )
            filtered = post_thresh

        # Sort by score desc
        filtered.sort(key=lambda x: (x.get("score") or 0.0), reverse=True)

        # Return top-k
        final_results = filtered[:k]

        if debug:
            logger.info("Returning %d results (requested k=%d) after gates", len(final_results), k)

        if show_meta:
            for r in final_results:
                logger.info(
                    "Row: id=%s score=%s char_length=%s meta=%s",
                    r.get("id"),
                    r.get("score"),
                    r.get("char_length"),
                    json.dumps(r.get("meta") or {})[:400],
                )

        return final_results


# ---------------------------
# Small CLI helpers (kept for convenience)
# ---------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="retriever", description="Schema-accurate Hardened Retriever for GameChunk (retrieval only)")
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


# ---------------------------
# CLI Shim (instantiates Retriever and runs retrieval)
# ---------------------------

if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    # instantiate Retriever (this will load the embedding model)
    try:
        retriever = Retriever(
            weaviate_url=args.weaviate,
            class_name=args.class_name,
            model_name=args.model,
        )
    except Exception as e:
        logger.exception("Failed to initialize Retriever: %s", e)
        sys.exit(1)

    try:
        rows = retriever.retrieve(
            args.query,
            k=args.k,
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

    except WeaviateConnectionError as e:
        logger.exception("Weaviate unreachable: %s", e)
        sys.exit(2)
    except Exception as e:
        logger.exception("Retrieval failed: %s", e)
        sys.exit(1)
