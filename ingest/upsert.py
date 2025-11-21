# ingest/upsert.py
"""
Robust upsert module for canonical + chunk objects into Weaviate.

Main behavior:
 - Prepare objects (only properties present in schema are sent)
 - Compute embeddings with sentence-transformers/all-MiniLM-L6-v2
 - Try multiple upsert methods in order:
     1) collection.batch.fixed_size(...) (preferred)
     2) legacy client.batch.* APIs
     3) HTTP batch endpoint POST /v1/batch/objects (robust fallback)
 - Extensive logging and diagnostics on failures

Important: This module uses the shared `client` imported from vector.index_manager.
It does NOT close that client after operations (so other parts of your app can continue
to use the shared connection).
"""

from __future__ import annotations
import logging
import sys
import time
import uuid
import os
from typing import Any, Dict, List, Iterable, Optional
import json

logger = logging.getLogger("ingest.upsert")
logger.setLevel(logging.INFO)

# ensure environment var fallback (used for HTTP batch fallback)
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

try:
    # index_manager provides a connected client and COLLECTION_NAME
    from vector.index_manager import client, COLLECTION_NAME
except Exception as e:
    logger.exception("Failed to import Weaviate client or COLLECTION_NAME from vector.index_manager: %s", e)
    raise

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_MODEL: Optional[Any] = None


# -----------------------------
# Embedding helpers
# -----------------------------
def _load_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("[embed] Loading SentenceTransformer model: %s", EMBED_MODEL_NAME)
            _EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)
            # guard (optional)
            try:
                _EMBED_MODEL.max_seq_length = 512
            except Exception:
                pass
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            raise
    return _EMBED_MODEL


def _batch_iter(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Compute embeddings for texts in batches. Returns list of vectors (list of floats).
    """
    if not texts:
        return []
    model = _load_embed_model()
    embeddings: List[List[float]] = []
    for i, chunk in enumerate(_batch_iter(texts, batch_size)):
        start = i * batch_size
        end = start + len(chunk) - 1
        logger.info("[embed] Embedding batch %d..%d", start, end)
        arr = model.encode(chunk, convert_to_numpy=True, show_progress_bar=False)
        for vec in arr:
            # convert numpy -> list
            embeddings.append(vec.tolist())
    logger.info("[embed] Generated %d embeddings", len(embeddings))
    return embeddings


# -----------------------------
# Utils: deterministic UUID and schema helper
# -----------------------------
def _deterministic_uuid_from_id(object_id: str) -> str:
    """
    Create deterministic UUIDv5 from string id (safe stable UUID).
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, object_id))


def _get_schema_property_names() -> List[str]:
    """
    Fetch property names for the configured class from Weaviate schema.
    If schema fetch fails, return an empty list (caller will then send all properties).
    """
    try:
        # Weaviate python client v4 exposes schema APIs; try to read the class schema
        schema = getattr(client, "schema", None)
        if schema and hasattr(schema, "get"):
            # client.schema.get() returns full schema; find our class
            full = schema.get()
            classes = full.get("classes", []) if isinstance(full, dict) else []
            for c in classes:
                if c.get("class") == COLLECTION_NAME or c.get("class", "").lower() == COLLECTION_NAME.lower():
                    props = c.get("properties", [])
                    names = [p.get("name") for p in props if "name" in p]
                    logger.info("Schema properties for %s discovered: %s", COLLECTION_NAME, names)
                    return [n for n in names if n]
        # Try client.collections.get(...) shape (some wrappers)
        collections = getattr(client, "collections", None)
        if collections and hasattr(collections, "get"):
            try:
                col = collections.get(COLLECTION_NAME)
                # some clients return config with properties list
                cfg = getattr(col, "config", None)
                if cfg:
                    # attempt to find property names
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
# Upsert attempts (multiple strategies)
# -----------------------------
def _try_collection_batch(collection, objects: List[Dict[str, Any]], batch_size: int) -> bool:
    """
    Preferred path: use collection.batch.fixed_size(...) context manager.
    Try to adapt to multiple possible add_object signatures.
    """
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
                if vec is not None and len(vec) not in (0, 384):
                    logger.warning(
                        "Vector length mismatch (%d) for id_for_uuid=%s", len(vec), o.get("id_for_uuid")
                    )
                obj_uuid = _deterministic_uuid_from_id(o["id_for_uuid"])
                # Try a few method signatures that wrappers might expect
                succeeded = False
                try_methods = [
                    # keyword uuid
                    lambda: batch.add_object(properties=props, uuid=obj_uuid, vector=vec),
                    # positional args (properties, uuid, vector)
                    lambda: batch.add_object(props, obj_uuid, vec),
                    # without uuid (let server generate) - some wrappers/versions prefer this
                    lambda: batch.add_object(properties=props, vector=vec),
                ]
                for fn in try_methods:
                    try:
                        fn()
                        succeeded = True
                        added += 1
                        break
                    except TypeError as te:
                        # signature mismatch; try next
                        logger.debug("add_object TypeError (signature mismatch) for %s: %s", o.get("id_for_uuid"), te)
                        continue
                    except Exception as exc:
                        # an operational error (server / connection / payload) - log and continue to next object
                        logger.exception("add_object raised for %s: %s", o.get("id_for_uuid"), exc)
                        break
                if not succeeded:
                    logger.error("Failed to add object via batch.add_object for id_for_uuid=%s", o.get("id_for_uuid"))
            # exit context -> flush
        # inspect batch.failed_objects if available
        try:
            failed = collection.batch.failed_objects
            if failed:
                logger.error("Batch committed but %d objects failed. Sample: %s", len(failed), json.dumps(failed[:3], default=str))
                return False
        except Exception:
            # attribute may not exist on some wrappers
            pass
        logger.info("Batch add completed, added %d objects", added)
        return True
    except Exception as e:
        logger.exception("Error during collection.batch.fixed_size usage: %s", e)
        return False


def _try_client_batch_legacy(objects: List[Dict[str, Any]]) -> bool:
    """
    Legacy python client.batch fallback (add_data_object/create_objects/send)
    """
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
                    # some versions accept 'uuid' kw, some 'id', try both
                    try:
                        batch.add_data_object(o["properties"], COLLECTION_NAME, uuid=obj_uuid, vector=o.get("vector"))
                    except TypeError:
                        batch.add_data_object(o["properties"], COLLECTION_NAME, obj_uuid, o.get("vector"))
                except Exception as exc:
                    logger.exception("client.batch.add_data_object failed for %s: %s", o["id_for_uuid"], exc)
            # flush/send/create depending on wrapper
            if hasattr(batch, "create_objects"):
                batch.create_objects()
            elif hasattr(batch, "send"):
                batch.send()
            logger.info("Legacy client.batch fallback completed")
            return True
        # other possible old API: create_objects(entries)
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


def _try_http_batch(objects: List[Dict[str, Any]]) -> bool:
    """
    Robust fallback: call Weaviate's batch API:
      POST {WEAVIATE_URL}/v1/batch/objects
    Request body: {"objects": [ { "class": "<class>", "id": "<uuid>", "properties": {...}, "vector": [...] }, ... ] }
    This bypasses client wrappers entirely.
    """
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
            # Weaviate returns array of results; check for errors
            try:
                j = resp.json()
                # j may include "results" or similar; best-effort: if string "error" or similar appears, log
                if isinstance(j, dict) and j.get("results") is None and j.get("status") == "error":
                    logger.error("HTTP batch returned error body: %s", j)
                    return False
            except Exception:
                # non-json or no extra info
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
# Top-level: upsert function
# -----------------------------
def upsert_canonical_and_chunks(
    merged_canonical: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    batch_size: int = 128,
    dry_run: bool = True,
) -> None:
    if not merged_canonical:
        raise ValueError("merged_canonical is required")

    objects = _prepare_objects(merged_canonical, chunks)
    logger.info("Prepared %d objects for upsert (1 canonical + %d chunks)", len(objects), max(0, len(objects) - 1))

    # filter properties to known schema props to avoid server rejecting unknown fields
    allowed_props = set(_get_schema_property_names())

    if allowed_props:
        for o in objects:
            props = o["properties"]
            filtered = {k: v for k, v in props.items() if k in allowed_props}
            # warn if we filtered out keys
            removed = set(props.keys()) - set(filtered.keys())
            if removed:
                logger.debug("Removed %d unknown props for id_for_uuid=%s: %s", len(removed), o.get("id_for_uuid"), removed)
            o["properties"] = filtered

    # embed texts
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
            # IMPORTANT: do NOT close the shared client imported from vector.index_manager
            return
        else:
            logger.info("collection.batch.fixed_size path failed; trying fallbacks")

    # 2) Try legacy client.batch APIs
    if _try_client_batch_legacy(objects):
        logger.info("Upsert succeeded via legacy client.batch")
        # keep shared client open
        return

    # 3) Try HTTP batch endpoint /v1/batch/objects (robust fallback)
    if _try_http_batch(objects):
        logger.info("Upsert succeeded via HTTP batch /v1/batch/objects")
        # keep shared client open
        return

    # If reached here, all methods failed
    logger.error("All upsert fallbacks failed. Please enable DEBUG logging and inspect client & server logs.")
    raise RuntimeError("Failed to upsert objects: all fallback methods failed.")


# -----------------------------
# Smoke test when run directly
# -----------------------------
if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Running dry-run smoke test for upsert module")
    test_can = {"unified_game_id": "test-game-2025", "title": "Test Game", "description": "Test Description"}
    test_chunks = [
        {"metadata": {"chunk_uuid": "c1", "unified_game_id": "test-game-2025"}, "text": "Chunk one text"},
        {"metadata": {"chunk_uuid": "c2", "unified_game_id": "test-game-2025"}, "text": "Chunk two text"},
    ]
    upsert_canonical_and_chunks(test_can, test_chunks, batch_size=16, dry_run=True)
