# ingest/upsert.py
"""
Native Weaviate batch upsert for RAG_ent.

- Uses deterministic canonical unified_game_id as the Weaviate object UUID for canonical objects.
- Uses deterministic chunk UUIDs (unified_game_id + "__chunk__" + chunk_uuid) for chunk objects.
- Preserves array fields as native lists (TEXT_ARRAY in Weaviate).
- Computes content_hash for deduplication.
- Embeds canonical.description (if present) as the canonical vector and each chunk text as chunk vectors.
- Uses the client object from vector.index_manager (expected to be a Weaviate v4-compatible client).
- Includes a dry-run mode that only logs objects prepared for upsert.
"""

from __future__ import annotations
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

import weaviate

from vector.index_manager import client, COLLECTION_NAME
from vector.embed import get_embedding_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ---------- helpers ----------
def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_date_for_prop(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    try:
        # best-effort: if it's already ISO, return with Z
        if "T" in value or value.endswith("Z") or "+" in value:
            # naive normalization; more robust parsing can be added if needed
            v = value.replace("+00:00", "Z")
            return v
    except Exception:
        pass
    return value


def _ensure_list(v: Optional[Any]) -> Optional[List[str]]:
    """
    Accepts None, list, or string. Returns None or list[str].
    This keeps arrays as native lists for Weaviate TEXT_ARRAY fields.
    """
    if v is None:
        return None
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    # if it's a JSON string of a list, try parse it
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None]
            except Exception:
                pass
        # fallback: return single-item list
        return [s]
    # fallback
    return [str(v)]


def _prepare_object_properties(metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Map metadata -> Weaviate properties dict.
    Preserve arrays as lists, coerce numerics, and normalize date-like strings minimally.
    """
    props: Dict[str, Any] = {"text": text}

    # simple scalar fields to copy (if present)
    scalar_fields = [
        "source",
        "slug",
        "title",
        "description",
        "site_detail_url",
        "language",
    ]
    for f in scalar_fields:
        v = metadata.get(f)
        if v is not None:
            props[f] = v

    # numeric scalars
    numeric_fields = ["game_id", "release_year", "rating", "rating_count", "score_normalized", "score_count", "articles_count", "reviews_count"]
    for f in numeric_fields:
        v = metadata.get(f)
        if v is not None:
            try:
                props[f] = int(v) if isinstance(v, (int, float, str)) else v
            except Exception:
                try:
                    props[f] = float(v)
                except Exception:
                    # skip if cannot coerce
                    pass

    # date-like fields
    for df in ("release_date", "created_at", "updated_at", "merge_time"):
        v = metadata.get(df)
        if v:
            nv = _normalize_date_for_prop(str(v))
            if nv:
                props[df] = nv

    # arrays: keep native lists
    array_fields = ["genres", "platforms", "developers", "publishers", "tags", "themes", "stores"]
    for af in array_fields:
        raw = metadata.get(af)
        lst = _ensure_list(raw)
        if lst:
            # dedupe preserving order
            seen = set()
            dedup = []
            for item in lst:
                key = str(item).strip()
                if key and key not in seen:
                    seen.add(key)
                    dedup.append(key)
            props[af] = dedup

    # content metadata
    if "content_length" in metadata:
        try:
            props["content_length"] = int(metadata.get("content_length"))
        except Exception:
            props["content_length"] = metadata.get("content_length")

    if "content_hash" in metadata:
        props["content_hash"] = metadata.get("content_hash")
    else:
        # compute fallback hash from text
        props["content_hash"] = _compute_content_hash(text)

    # keep unified_game_id and chunk identifiers
    if metadata.get("unified_game_id"):
        props["unified_game_id"] = metadata.get("unified_game_id")
    if metadata.get("chunk_uuid"):
        props["chunk_uuid"] = metadata.get("chunk_uuid")
    if metadata.get("chunk_type"):
        props["chunk_type"] = metadata.get("chunk_type")

    return props


# ---------- Weaviate helpers ----------
def _get_collection():
    try:
        coll = client.collections.get(COLLECTION_NAME)
        return coll
    except Exception as e:
        logger.exception("Failed to get collection '%s': %s", COLLECTION_NAME, e)
        raise


def _object_exists(collection, obj_id: str) -> bool:
    try:
        # collection.objects.get may raise if not exists
        obj = collection.objects.get(obj_id)
        return obj is not None
    except Exception:
        return False


# ---------- embedding ----------
def _embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    emb_model = get_embedding_model()
    vectors: List[List[float]] = []
    # support several embedding model interfaces (as in your previous code)
    if hasattr(emb_model, "embed_documents"):
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(emb_model.embed_documents(batch))
        return vectors
    if hasattr(emb_model, "client") and hasattr(emb_model.client, "encode"):
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors.extend(emb_model.client.encode(batch, normalize_embeddings=True).tolist())
        return vectors
    # fallback single-call
    for t in texts:
        vectors.append(emb_model.embed([t])[0])
    return vectors


# ---------- top-level upsert ----------
def upsert_canonical_and_chunks(
    canonical: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    batch_size: int = 128,
    dry_run: bool = False,
) -> None:
    """
    Upsert one canonical object and its chunk objects to Weaviate.

    canonical: merged canonical dict (contains unified_game_id, raw_source_blob etc.)
    chunks: list of chunk dicts produced by chunking (each: {"text":..., "metadata":{...}})
    """
    if not canonical:
        logger.warning("No canonical provided to upsert.")
        return

    collection = _get_collection()

    unified_id = canonical.get("unified_game_id")
    if not unified_id:
        # fallback: generate an id but warn (should be deterministic from loader)
        slug = canonical.get("slug") or canonical.get("title") or "unknown"
        year = canonical.get("release_year") or "unknown"
        unified_id = f"{slug}-{year}-{hashlib.sha1((slug + str(year)).encode()).hexdigest()[:8]}"
        logger.warning("Canonical missing unified_game_id — generated fallback id: %s", unified_id)

    # --- Prepare canonical object ---
    # canonical text to embed: prefer description -> title
    canonical_text = canonical.get("description") or canonical.get("title") or ""
    canonical_meta = dict(canonical)  # copy
    # ensure content_hash and content_length for canonical too
    canonical_meta["content_hash"] = canonical_meta.get("content_hash") or _compute_content_hash(canonical_text or "")
    canonical_meta["content_length"] = canonical_meta.get("content_length") or len((canonical_text or "").split())

    canonical_props = _prepare_object_properties(canonical_meta, canonical_text)

    # add raw_source_blob on canonical as a JSON string property (per your decision)
    raw_blob = canonical.get("raw_source_blob")
    if raw_blob is not None:
        # canonical_props stores raw blob as string (Weaviate text property)
        canonical_props["raw_source_blob"] = raw_blob if isinstance(raw_blob, str) else json.dumps(raw_blob, ensure_ascii=False)

    # compute embedding for canonical
    to_embed_texts = []
    if canonical_text:
        to_embed_texts.append(canonical_text)

    # Prepare chunk texts and metadata for embedding
    chunk_texts: List[str] = []
    chunk_props_list: List[Tuple[str, Dict[str, Any], str]] = []  # (text, metadata, object_id)
    for ch in chunks:
        txt = ch.get("text") or ""
        md = dict(ch.get("metadata") or {})
        # ensure unified_game_id present
        md.setdefault("unified_game_id", unified_id)
        # content_hash
        md["content_hash"] = md.get("content_hash") or _compute_content_hash(txt)
        md["content_length"] = md.get("content_length") or len(txt.split())
        # build object id for chunk: unifiedid__chunk__{chunk_uuid or index}
        chunk_uuid = md.get("chunk_uuid") or md.get("chunk_index") or ""
        obj_id = f"{unified_id}__chunk__{chunk_uuid}"
        chunk_props = _prepare_object_properties(md, txt)
        chunk_texts.append(txt)
        chunk_props_list.append((txt, chunk_props, obj_id))

    # embed all texts (canonical + chunks) in batches
    embed_texts_all = []
    if to_embed_texts:
        embed_texts_all.extend(to_embed_texts)
    embed_texts_all.extend(chunk_texts)

    vectors = []
    if embed_texts_all:
        logger.info("Generating embeddings for %d texts (canonical + chunks)...", len(embed_texts_all))
        vectors = _embed_texts(embed_texts_all, batch_size=batch_size)
    else:
        vectors = []

    # map vectors back
    idx = 0
    if to_embed_texts:
        canonical_vector = vectors[0]
        idx = 1
    else:
        canonical_vector = None

    chunk_vectors = vectors[idx : idx + len(chunk_texts)] if len(vectors) >= idx else []

    # --- Build upsert payload using native Weaviate batch API ---
    # We will create objects where canonical has id = unified_game_id
    # and chunks have id = unified_id__chunk__<chunk_uuid>
    operations = []
    # canonical object
    canonical_obj = {
        "class": COLLECTION_NAME,
        "id": unified_id,
        "properties": canonical_props,
    }
    if canonical_vector is not None:
        canonical_obj["vector"] = canonical_vector
    operations.append(canonical_obj)

    # chunk objects
    for i, (txt, props, obj_id) in enumerate(chunk_props_list):
        obj = {"class": COLLECTION_NAME, "id": obj_id, "properties": props}
        if i < len(chunk_vectors):
            obj["vector"] = chunk_vectors[i]
        operations.append(obj)

    logger.info("Prepared %d objects to upsert (1 canonical + %d chunks).", 1, len(chunk_props_list))

    if dry_run:
        logger.info("Dry-run enabled — not sending to Weaviate. Printing first 2 prepared objects:")
        logger.info(json.dumps(operations[:2], ensure_ascii=False, indent=2))
        return

    # Send batch to Weaviate
    try:
        batch = collection.objects.batch()
    except Exception:
        # fallback to client-level batch if collection batch not available
        batch = client.batch

    try:
        # Add objects in controlled batches to avoid huge requests
        B = batch_size if batch_size and batch_size > 0 else 128
        for i in range(0, len(operations), B):
            chunk_ops = operations[i : i + B]
            try:
                # prefer collection batch add API if available
                if hasattr(batch, "add"):
                    for o in chunk_ops:
                        # some clients accept vector in 'vector' key; add expects (obj, vector?) depending on client
                        # safe approach: use collection.objects.create with id+props+vector if available
                        batch.add(o)
                    # flush the batch if method exists
                    if hasattr(batch, "flush"):
                        batch.flush()
                else:
                    # fallback: use client.data_object.create for each object (slower)
                    for o in chunk_ops:
                        obj_id = o.get("id")
                        props = o.get("properties")
                        vec = o.get("vector", None)
                        if vec is not None:
                            client.data_object.create(data_object=props, class_name=COLLECTION_NAME, uuid=obj_id, vector=vec)
                        else:
                            client.data_object.create(data_object=props, class_name=COLLECTION_NAME, uuid=obj_id)
            except Exception as e:
                logger.exception("Failed to add a batch of %d objects: %s", len(chunk_ops), e)
                # continue trying remaining batches
                continue

        logger.info("Batch upsert completed.")
    except Exception as e:
        logger.exception("Batch upsert failed: %s", e)
        raise
    finally:
        # close client connection if client supports it (harmless if not)
        try:
            client.close()
        except Exception:
            pass
