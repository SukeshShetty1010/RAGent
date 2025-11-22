#!/usr/bin/env python3
# ingest/upsert.py
"""
Robust upsert module for canonical + chunk objects into Weaviate.

Exposes:
 - upsert_canonical_and_chunks(merged_canonical, chunks, batch_size=128, dry_run=True, close_client=False)
 - upsert_pairs(pairs, batch_size=128, dry_run=True, workers=1)
 - CLI: python -m ingest.upsert --pairs-file out/upsert_pairs.json [--commit]

Changes:
 - Uses centralized embed_texts from vector.embed (single source of truth for embeddings)
 - Optional client.close() via close_client flag to avoid ResourceWarning in CLI scripts
 - Preserves collection.batch, legacy client.batch, and HTTP batch fallbacks
"""

from __future__ import annotations
import logging
import sys
import time
import uuid
import os
import json
from typing import Any, Dict, List, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("ingest.upsert")
logger.setLevel(logging.INFO)

# environment fallback for HTTP batch fallback
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Import vector/index_manager client & COLLECTION_NAME (shared global used by this module)
try:
    from vector.index_manager import client, COLLECTION_NAME
except Exception as e:
    logger.exception("Failed to import Weaviate client or COLLECTION_NAME from vector.index_manager: %s", e)
    raise

# Use centralized embed helper
try:
    from vector.embed import embed_texts
except Exception as e:
    logger.exception("Failed to import embed_texts from vector.embed: %s", e)
    raise

# -----------------------------
# Utils
# -----------------------------
def _deterministic_uuid_from_id(object_id: str) -> str:
    """Create deterministic UUIDv5 from string id (stable UUID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, object_id))


def _get_schema_property_names() -> List[str]:
    """
    Fetch property names for the configured class from Weaviate schema.
    If schema fetch fails, return an empty list (caller will then send all properties).
    """
    try:
        schema = getattr(client, "schema", None)
        if schema and hasattr(schema, "get"):
            full = schema.get()
            classes = full.get("classes", []) if isinstance(full, dict) else []
            for c in classes:
                if c.get("class") == COLLECTION_NAME or c.get("class", "").lower() == COLLECTION_NAME.lower():
                    props = c.get("properties", [])
                    names = [p.get("name") for p in props if "name" in p]
                    logger.info("Schema properties for %s discovered: %s", COLLECTION_NAME, names)
                    return [n for n in names if n]
        collections = getattr(client, "collections", None)
        if collections and hasattr(collections, "get"):
            try:
                col = collections.get(COLLECTION_NAME)
                cfg = getattr(col, "config", None)
                if cfg:
                    p = getattr(cfg, "properties", None)
                    if p:
                        names = [prop.get("name") for prop in p if isinstance(prop, dict) and "name" in prop]
                        logger.info("Schema properties (via collection.config) for %s: %s", COLLECTION_NAME, names)
                        return [n for n in names if n]
            except Exception:
                pass
    except Exception as e:
        logger.debug("Failed to fetch schema properties: %s", e)
    logger.info("Unable to determine schema properties for %s; will not filter properties.", COLLECTION_NAME)
    return []


def _batch_iter(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# -----------------------------
# Object preparation
# -----------------------------
def _prepare_objects(merged_canonical: Dict[str, Any], chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Produce list of objects: { "properties": {...}, "text": "...", "id_for_uuid": "<string>" }
    id_for_uuid is used only locally to generate deterministic UUIDs.
    """
    objects: List[Dict[str, Any]] = []
    canonical = dict(merged_canonical)
    unified = canonical.get("unified_game_id") or canonical.get("slug") or f"canon-{int(time.time())}"
    canonical["unified_game_id"] = unified
    canonical["slug"] = canonical.get("slug", unified)
    canonical["doc_type"] = canonical.get("doc_type", "canonical")
    canonical_text = canonical.get("description") or canonical.get("text") or canonical.get("title") or ""
    objects.append({"properties": canonical, "text": str(canonical_text), "id_for_uuid": unified})

    for idx, c in enumerate(chunks):
        props: Dict[str, Any] = {}
        if isinstance(c, dict):
            for k, v in c.items():
                if k != "metadata":
                    props[k] = v
            meta = c.get("metadata") or {}
            if isinstance(meta, dict):
                for mk, mv in meta.items():
                    if mk not in props:
                        props[mk] = mv
        else:
            props["raw"] = c
        chunk_unified = props.get("unified_game_id") or unified
        props["unified_game_id"] = chunk_unified
        chunk_uuid = props.get("chunk_uuid") or props.get("id") or f"chunk-{idx}"
        obj_id = f"{chunk_unified}__chunk__{chunk_uuid}"
        props["doc_type"] = props.get("doc_type", "chunk")
        text = props.get("text") or props.get("content") or props.get("body") or ""
        props["content_length"] = props.get("content_length", len(text) if text else 0)
        objects.append({"properties": props, "text": str(text), "id_for_uuid": obj_id})

    return objects


# -----------------------------
# Upsert strategy: collection.batch.fixed_size
# -----------------------------
def _try_collection_batch(collection, objects: List[Dict[str, Any]], batch_size: int) -> bool:
    try:
        ctx = collection.batch.fixed_size(batch_size)
    except Exception as e:
        logger.debug("collection.batch.fixed_size not available: %s", e)
        return False

    try:
        with ctx as batch:
            logger.info("Using collection.batch.fixed_size context manager (batch type=%s)", type(batch))
            added = 0
            for o in objects:
                props = o["properties"]
                vec = o.get("vector")
                obj_uuid = _deterministic_uuid_from_id(o["id_for_uuid"])
                succeeded = False
                try_methods = [
                    lambda: batch.add_object(properties=props, uuid=obj_uuid, vector=vec),
                    lambda: batch.add_object(props, obj_uuid, vec),
                    lambda: batch.add_object(properties=props, vector=vec),
                ]
                for fn in try_methods:
                    try:
                        fn()
                        succeeded = True
                        added += 1
                        break
                    except TypeError as te:
                        logger.debug("add_object TypeError (signature mismatch) for %s: %s", o.get("id_for_uuid"), te)
                        continue
                    except Exception as exc:
                        logger.exception("add_object raised for %s: %s", o.get("id_for_uuid"), exc)
                        break
                if not succeeded:
                    logger.error("Failed to add object via batch.add_object for id_for_uuid=%s", o.get("id_for_uuid"))
        try:
            failed = collection.batch.failed_objects
            if failed:
                logger.error("Batch committed but %d objects failed. Sample: %s", len(failed), json.dumps(failed[:3], default=str))
                return False
        except Exception:
            pass
        logger.info("Batch add completed, added %d objects", added)
        return True
    except Exception as e:
        logger.exception("Error during collection.batch.fixed_size usage: %s", e)
        return False


# -----------------------------
# Upsert strategy: legacy client.batch
# -----------------------------
def _try_client_batch_legacy(objects: List[Dict[str, Any]]) -> bool:
    batch = getattr(client, "batch", None)
    if not batch:
        logger.debug("client.batch not present")
        return False
    try:
        if hasattr(batch, "add_data_object"):
            logger.info("Using client.batch.add_data_object fallback")
            for o in objects:
                try:
                    obj_uuid = _deterministic_uuid_from_id(o["id_for_uuid"])
                    try:
                        batch.add_data_object(o["properties"], COLLECTION_NAME, uuid=obj_uuid, vector=o.get("vector"))
                    except TypeError:
                        batch.add_data_object(o["properties"], COLLECTION_NAME, obj_uuid, o.get("vector"))
                except Exception as exc:
                    logger.exception("client.batch.add_data_object failed for %s: %s", o["id_for_uuid"], exc)
            if hasattr(batch, "create_objects"):
                batch.create_objects()
            elif hasattr(batch, "send"):
                batch.send()
            logger.info("Legacy client.batch fallback completed")
            return True
        if hasattr(batch, "create_objects"):
            logger.info("Using client.batch.create_objects(entries) fallback")
            entries = []
            for o in objects:
                obj_uuid = _deterministic_uuid_from_id(o["id_for_uuid"])
                entries.append({"class": COLLECTION_NAME, "properties": o["properties"], "vector": o.get("vector"), "id": obj_uuid})
            batch.create_objects(entries)
            logger.info("client.batch.create_objects completed")
            return True
    except Exception as e:
        logger.exception("Legacy client.batch approach failed: %s", e)
        return False
    return False


# -----------------------------
# Upsert strategy: HTTP batch POST
# -----------------------------
def _try_http_batch(objects: List[Dict[str, Any]]) -> bool:
    try:
        import requests
    except Exception:
        logger.debug("requests not available for HTTP batch fallback")
        return False

    payload_objects = []
    for o in objects:
        obj_uuid = _deterministic_uuid_from_id(o["id_for_uuid"])
        p = {"class": COLLECTION_NAME, "id": obj_uuid, "properties": o["properties"]}
        if o.get("vector") is not None:
            p["vector"] = o["vector"]
        payload_objects.append(p)

    endpoint = WEAVIATE_URL.rstrip("/") + "/v1/batch/objects"
    try:
        logger.info("HTTP batch fallback: posting %d objects to %s", len(payload_objects), endpoint)
        resp = requests.post(endpoint, json={"objects": payload_objects}, timeout=120)
        logger.info("HTTP batch response status: %s", resp.status_code)
        if resp.status_code in (200, 201):
            try:
                j = resp.json()
                if isinstance(j, dict) and j.get("results") is None and j.get("status") == "error":
                    logger.error("HTTP batch returned error body: %s", j)
                    return False
            except Exception:
                pass
            logger.info("HTTP batch fallback successful.")
            return True
        else:
            body = resp.text
            logger.error("HTTP batch fallback failed: status=%s body=%s", resp.status_code, (body[:200] + "...") if body else "")
            return False
    except Exception as e:
        logger.exception("HTTP batch fallback encountered exception: %s", e)
        return False


# -----------------------------
# Top-level upsert function
# -----------------------------
def upsert_canonical_and_chunks(
    merged_canonical: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    batch_size: int = 128,
    dry_run: bool = True,
    close_client: bool = False,
) -> None:
    """
    Upsert a single merged canonical and its chunk documents.

    If close_client=True the global client will be closed at the end to avoid resource warnings.
    Raises RuntimeError on unrecoverable failure.
    """
    if not merged_canonical:
        raise ValueError("merged_canonical is required")

    # Basic preflight validation
    uid = merged_canonical.get("unified_game_id") or merged_canonical.get("slug") or None
    if uid is None:
        logger.warning("merged_canonical missing unified_game_id; will generate temporary id for upsert (not recommended)")

    objects = _prepare_objects(merged_canonical, chunks)
    logger.info("Prepared %d objects for upsert (1 canonical + %d chunks)", len(objects), max(0, len(objects) - 1))

    # Filter properties to known schema props to avoid server rejecting unknown fields
    allowed_props = set(_get_schema_property_names())
    if allowed_props:
        for o in objects:
            props = o["properties"]
            filtered = {k: v for k, v in props.items() if k in allowed_props}
            removed = set(props.keys()) - set(filtered.keys())
            if removed:
                logger.debug("Removed %d unknown props for id_for_uuid=%s: %s", len(removed), o.get("id_for_uuid"), removed)
            o["properties"] = filtered

    # embed texts — use centralized embed_texts
    texts = [o.get("text", "") or "" for o in objects]
    vectors = embed_texts(texts, batch_size=64)
    for i, o in enumerate(objects):
        o["vector"] = vectors[i] if i < len(vectors) else None

    if dry_run:
        logger.info("========== DRY RUN MODE ==========")
        for o in objects[:3]:
            logger.info("id_for_uuid: %s", o.get("id_for_uuid"))
            logger.info("properties_keys: %s", sorted(list(o["properties"].keys())))
            logger.info("vector_len: %s", len(o.get("vector")) if o.get("vector") else None)
        logger.info("Dry-run done; not writing to Weaviate.")
        if close_client:
            try:
                client.close()
            except Exception:
                pass
        return

    # 1) Try collection.batch.fixed_size (preferred)
    try:
        collection = client.collections.use(COLLECTION_NAME)
    except Exception as e:
        logger.exception("Could not access collection via client.collections.use: %s", e)
        collection = None

    if collection:
        ok = _try_collection_batch(collection, objects, batch_size)
        if ok:
            logger.info("Upsert succeeded via collection.batch.fixed_size")
            if close_client:
                try:
                    client.close()
                except Exception:
                    pass
            return
        else:
            logger.info("collection.batch.fixed_size path failed; trying fallbacks")

    # 2) Try legacy client.batch APIs
    if _try_client_batch_legacy(objects):
        logger.info("Upsert succeeded via legacy client.batch")
        if close_client:
            try:
                client.close()
            except Exception:
                pass
        return

    # 3) Try HTTP batch endpoint /v1/batch/objects
    if _try_http_batch(objects):
        logger.info("Upsert succeeded via HTTP batch /v1/batch/objects")
        if close_client:
            try:
                client.close()
            except Exception:
                pass
        return

    logger.error("All upsert fallbacks failed. Please enable DEBUG logging and inspect client & server logs.")
    if close_client:
        try:
            client.close()
        except Exception:
            pass
    raise RuntimeError("Failed to upsert objects: all fallback methods failed.")


# -----------------------------
# Helper to upsert multiple pairs with concurrency and validation
# -----------------------------
def _validate_pair(pair: Dict[str, Any], allow_empty_chunks: bool = False) -> Optional[str]:
    """
    Validate a single {'canonical':..., 'chunks':[...]} pair.
    Return None if OK, else return an error message.
    """
    if not isinstance(pair, dict):
        return "pair is not a dict"
    canonical = pair.get("canonical")
    if not isinstance(canonical, dict):
        return "missing or invalid 'canonical' object"
    uid = canonical.get("unified_game_id") or canonical.get("slug")
    if not uid:
        return "canonical missing 'unified_game_id' and 'slug'"
    chunks = pair.get("chunks", [])
    if not allow_empty_chunks and (not isinstance(chunks, list) or len(chunks) == 0):
        return f"no chunks provided for {uid}"
    return None


def upsert_pairs(pairs: List[Dict[str, Any]], batch_size: int = 128, dry_run: bool = True, workers: int = 1, allow_empty_chunks: bool = False, close_client: bool = False):
    """
    Upsert multiple pairs concurrently (workers threads).
    Returns list of result dicts: {"ok": True/False, "id": uid, "error": ...}
    """
    results = []
    logger.info("Starting upsert_pairs: count=%d dry_run=%s workers=%d", len(pairs), dry_run, workers)

    def _worker(pair):
        err = _validate_pair(pair, allow_empty_chunks=allow_empty_chunks)
        if err:
            return {"ok": False, "error": err, "id": pair.get("canonical", {}).get("unified_game_id")}
        try:
            upsert_canonical_and_chunks(pair["canonical"], pair.get("chunks", []), batch_size=batch_size, dry_run=dry_run, close_client=False)
            return {"ok": True, "id": pair["canonical"].get("unified_game_id")}
        except Exception as e:
            logger.exception("Upsert failed for %s: %s", pair.get("canonical", {}).get("unified_game_id"), e)
            return {"ok": False, "error": str(e), "id": pair.get("canonical", {}).get("unified_game_id")}

    if workers <= 1:
        for p in pairs:
            results.append(_worker(p))
    else:
        with ThreadPoolExecutor(max_workers=workers) as exe:
            futures = [exe.submit(_worker, p) for p in pairs]
            for fut in as_completed(futures):
                results.append(fut.result())

    logger.info("Completed upsert_pairs: succeeded=%d failed=%d", sum(1 for r in results if r.get("ok")), sum(1 for r in results if not r.get("ok")))

    if close_client:
        try:
            client.close()
        except Exception:
            pass

    return results


# -----------------------------
# CLI: operate on a pairs-file (list of pairs) or single pair JSON
# -----------------------------
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _cli_main():
    import argparse
    p = argparse.ArgumentParser(description="Ingest Upsert CLI - upsert canonical+chunks pairs to Weaviate")
    p.add_argument("--pairs-file", "-p", help="JSON file containing a list of {'canonical','chunks'} pairs (or a single pair object).")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size hint")
    p.add_argument("--workers", type=int, default=1, help="Concurrent worker threads for upserting pairs")
    p.add_argument("--commit", action="store_true", help="If set, actually write to Weaviate; default is dry-run")
    p.add_argument("--allow-empty-chunks", action="store_true", help="Allow upserting canonicals with zero chunks")
    p.add_argument("--close-client", action="store_true", help="Close the Weaviate client after run to avoid ResourceWarning")
    args = p.parse_args()

    if not args.pairs_file:
        print("Provide --pairs-file with a JSON list of pairs or a single pair object.")
        sys.exit(2)

    path = args.pairs_file
    try:
        data = _load_json(path)
    except Exception as e:
        logger.error("Failed to read JSON from %s: %s", path, e)
        sys.exit(2)

    # Normalize to list
    if isinstance(data, dict) and "canonical" in data:
        pairs = [data]
    elif isinstance(data, list):
        pairs = data
    else:
        logger.error("pairs-file must be a JSON list of pairs or a single pair object")
        sys.exit(2)

    dry_run = not args.commit
    results = upsert_pairs(pairs, batch_size=args.batch_size, dry_run=dry_run, workers=args.workers, allow_empty_chunks=args.allow_empty_chunks, close_client=args.close_client)
    succeeded = sum(1 for r in results if r.get("ok"))
    failed = sum(1 for r in results if not r.get("ok"))
    logger.info("CLI upsert summary: total=%d succeeded=%d failed=%d", len(results), succeeded, failed)
    # print small JSON report
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # close client when requested (double-check)
    if args.close_client:
        try:
            client.close()
        except Exception:
            pass

    if failed:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _cli_main()
