#!/usr/bin/env python3
"""
retriever/simple_retriever.py

Vector-only retriever for Weaviate Python client v4, schema-aware and aligned with
the GameChunk schema provided by the user (see weaviate_gamechunk_schema.json).

Usage:
    python retriever/simple_retriever.py --query "Far Cry 5 gameplay" --k 5 --weaviate http://localhost:8080
    python retriever/simple_retriever.py --dump-schema --weaviate http://localhost:8080

Requirements:
    pip install "weaviate-client>=4" sentence-transformers numpy
"""

from __future__ import annotations
import argparse
import json
import logging
import sys
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import numpy as np
from sentence_transformers import SentenceTransformer

# Weaviate v4 client imports
try:
    import weaviate
    from weaviate.classes.init import AdditionalConfig, Timeout
    from weaviate.classes.query import MetadataQuery
except Exception:
    weaviate = None  # handled at runtime

LOG = logging.getLogger("simple_retriever")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CLASS = "GameChunk"

# Preferred properties based on the uploaded schema (weaviate_gamechunk_schema.json).
# See uploaded schema for exact property names. :contentReference[oaicite:1]{index=1}
PREFERRED_PROPS = [
    "unified_id",
    "doc_id",
    "source",
    "title",
    "chunk_index",
    "text",
    "char_length",
    "content_hash",
    "release_date",
    "release_year",
    "platforms",
    "genres",
    "developers",
    "publishers",
    "meta",
]


# -----------------------------
# SBERT embedding utilities
# -----------------------------
def _load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    LOG.info("Loading sentence-transformers model: %s", model_name)
    return SentenceTransformer(model_name)


def _embed_query(model: SentenceTransformer, query: str) -> List[float]:
    vec = model.encode([query], convert_to_numpy=True)[0]
    vec = vec.astype(float)
    norm = np.linalg.norm(vec)
    if norm == 0 or np.isnan(norm):
        return vec.tolist()
    return (vec / norm).tolist()


# -----------------------------
# Weaviate helper utilities
# -----------------------------
def _parse_weaviate_url(weaviate_url: str) -> Dict[str, Any]:
    """
    Accepts:
      - http://localhost:8080
      - https://host:port
      - host:port
      - localhost:8080
    Returns dict with host (str) and port (int)
    """
    if weaviate_url.startswith("http://") or weaviate_url.startswith("https://"):
        parsed = urlparse(weaviate_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        return {"host": host, "port": int(port)}
    if ":" in weaviate_url:
        host, port = weaviate_url.split(":", 1)
        try:
            return {"host": host, "port": int(port)}
        except Exception:
            return {"host": host, "port": 8080}
    return {"host": weaviate_url, "port": 8080}


def _get_class_properties_from_schema(client, class_name: str) -> List[str]:
    """
    Query the running Weaviate schema and return the property names for the given class.
    This reads client.schema.get() output and extracts properties for the requested class.
    """
    props: List[str] = []

    try:
        schema = client.schema.get()
    except Exception as e:
        LOG.warning("Unable to fetch schema via client.schema.get(): %s", e)
        # Best-effort fallback: try to inspect collection metadata
        try:
            collection = client.collections.get(class_name)
            meta = getattr(collection, "meta", None) or getattr(collection, "schema", None) or {}
            if isinstance(meta, dict):
                class_props = meta.get("properties") or []
                for p in class_props:
                    if isinstance(p, dict) and p.get("name"):
                        props.append(p["name"])
        except Exception:
            pass
        return props

    classes = schema.get("classes") or []
    for c in classes:
        cname = c.get("class") or c.get("name") or ""
        if cname and cname.lower() == class_name.lower():
            for p in c.get("properties", []) or []:
                pn = p.get("name")
                if pn:
                    props.append(pn)
            break
    return props


# -----------------------------
# Response parsing
# -----------------------------
def _parse_v4_objects(objects) -> List[Dict[str, Any]]:
    """
    Convert v4 response.objects list to our standard list of dicts.
    Each entry includes: id (uuid), title, site_detail_url (if present), content (500 chars), score {certainty,distance}
    """
    out = []
    for o in objects:
        uuid = getattr(o, "uuid", None) or getattr(o, "id", None) or None
        props = getattr(o, "properties", None) or {}

        # Prefer the 'text' field (schema has 'text') otherwise use 'meta' or dump props
        content = (
            props.get("text")
            or (lambda: (json.loads(props.get("meta")) if isinstance(props.get("meta"), str) else (props.get("meta") or None)))()
            if props.get("meta")
            else None
        )
        # If content is still not a string (meta might be dict), try other fields
        if not content:
            content = props.get("content") or props.get("excerpt") or props.get("body") or None

        title = props.get("title") or props.get("name") or None

        # We don't have 'site_detail_url' in the schema by default; still attempt to fetch if available
        site_detail_url = props.get("site_detail_url") or props.get("url") or None

        if content is None:
            try:
                content = json.dumps(props, ensure_ascii=False)
            except Exception:
                content = str(props)

        metadata = getattr(o, "metadata", None)
        certainty = None
        distance = None
        if metadata:
            if isinstance(metadata, dict):
                certainty = metadata.get("certainty")
                distance = metadata.get("distance")
            else:
                certainty = getattr(metadata, "certainty", None)
                distance = getattr(metadata, "distance", None)

        out.append({
            "id": uuid,
            "title": title,
            "site_detail_url": site_detail_url,
            "content": content[:500] if isinstance(content, str) else content,
            "score": {"certainty": certainty, "distance": distance},
            "_raw_properties": props,
            "_raw_metadata": metadata,
        })
    return out


# -----------------------------
# Main retrieval function
# -----------------------------
def retrieve(
    query: str,
    k: int = 5,
    weaviate_url: str = "http://localhost:8080",
    class_name: str = DEFAULT_CLASS,
    model_name: str = DEFAULT_MODEL,
    timeout_init: int = 5,
    timeout_query: int = 30,
    dump_schema: bool = False,
) -> List[Dict[str, Any]]:
    """
    Encode query with SBERT, query Weaviate v4 collection.near_vector, return top-K chunks.
    """
    if weaviate is None:
        raise RuntimeError("weaviate client v4 not installed. Install with: pip install 'weaviate-client>=4'")

    # load model and embed
    model = _load_model(model_name)
    q_vec = _embed_query(model, query)

    # parse host/port
    parsed = _parse_weaviate_url(weaviate_url)
    host = parsed["host"]
    port = parsed["port"]

    LOG.info("Connecting to Weaviate at %s:%s (parsed from %s)", host, port, weaviate_url)
    client = weaviate.connect_to_local(
        host=host,
        port=port,
        grpc_port=50051,
        additional_config=AdditionalConfig(timeout=Timeout(init=timeout_init, query=timeout_query)),
    )

    # Determine which properties exist on the class
    class_props = _get_class_properties_from_schema(client, class_name)
    LOG.info("Detected properties for class '%s': %s", class_name, class_props)

    if dump_schema:
        # If user asked to dump schema, print and exit early
        client.close()
        print(json.dumps({"class": class_name, "properties": class_props}, indent=2, ensure_ascii=False))
        return []

    # Intersect preferred props with discovered props, preserving order
    return_props = [p for p in PREFERRED_PROPS if p in class_props]

    # If none of the preferred props exist, request the first available properties
    if not return_props:
        if class_props:
            LOG.warning("No preferred properties present; requesting first %d available properties from schema.", min(10, len(class_props)))
            return_props = class_props[:10]
        else:
            LOG.warning("No properties discovered for class '%s'; requesting no explicit properties.", class_name)
            return_props = []

    LOG.info("Running near_vector query, top_k=%d, class=%s; return_properties=%s", k, class_name, return_props)
    try:
        collection = client.collections.get(class_name)
    except Exception as e:
        client.close()
        raise RuntimeError(f"Failed to access collection '{class_name}': {e}") from e

    try:
        if return_props:
            response = collection.query.near_vector(
                near_vector=q_vec,
                limit=k,
                return_properties=return_props,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )
        else:
            response = collection.query.near_vector(
                near_vector=q_vec,
                limit=k,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )
    except Exception as e:
        client.close()
        raise RuntimeError(f"Weaviate query failed: {e}") from e

    objects = getattr(response, "objects", []) or []
    results = _parse_v4_objects(objects)

    client.close()
    return results


# -------------------------
# CLI
# -------------------------
def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(prog="simple_retriever", description="Vector-only retriever using SBERT + Weaviate v4 (schema-aware)")
    parser.add_argument("--query", "-q", required=False, help="Text query")
    parser.add_argument("--k", type=int, default=5, help="Top-K results")
    parser.add_argument("--weaviate", default="http://localhost:8080", help="Weaviate endpoint (http://host:port or host:port)")
    parser.add_argument("--class-name", default=DEFAULT_CLASS, help="Weaviate collection/class name (default: GameChunk)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model to use for query encoding")
    parser.add_argument("--dump-schema", action="store_true", help="Print detected properties for the class and exit")
    args = parser.parse_args(argv)

    if not args.query and not args.dump_schema:
        parser.error("Either --query or --dump-schema must be provided.")

    try:
        results = retrieve(
            query=args.query or "",
            k=args.k,
            weaviate_url=args.weaviate,
            class_name=args.class_name,
            model_name=args.model,
            dump_schema=args.dump_schema,
        )
    except Exception as e:
        LOG.exception("Retrieval failed: %s", e)
        sys.exit(2)

    if args.dump_schema:
        # already printed in retrieve()
        return

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()