# ingest/upsert.py
import hashlib
import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from langchain_core.documents import Document
from langchain_weaviate import WeaviateVectorStore
import weaviate
import ast

from vector.index_manager import client, COLLECTION_NAME
from vector.embed import get_embedding_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def normalize_text_array(value):
    """Convert messy genre/platform JSON-like strings into clean text arrays."""
    if not value:
        return []
    if isinstance(value, list):
        clean = []
        for v in value:
            # Try to safely parse stringified dicts like "{'name': 'Action'}"
            if isinstance(v, str):
                try:
                    parsed = ast.literal_eval(v)
                    if isinstance(parsed, dict) and "name" in parsed:
                        clean.append(parsed["name"])
                    else:
                        clean.append(str(parsed))
                except Exception:
                    clean.append(v)
            else:
                clean.append(str(v))
        # Remove duplicates and normalize spacing
        return sorted(set([x.strip() for x in clean if x.strip()]))
    # if passed a string that looks like a JSON list, try to parse it
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return normalize_text_array(parsed)
        except Exception:
            # fallthrough: treat as a single-item list
            pass
    return [str(value).strip()]


def _compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_date(value: str) -> Optional[str]:
    """Convert raw date strings into RFC3339 format (YYYY-MM-DDTHH:MM:SSZ)."""
    if not value:
        return None
    try:
        s = str(value).strip()
        # If already ISO-ish (has 'T' or timezone), try to normalize
        if "T" in s or s.endswith("Z") or "+" in s:
            # Try parse with fromisoformat (py3.11+ robust) or fallback to returning z-normalized
            try:
                # Python's fromisoformat accepts many variants; ensure timezone is Z if none present
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            except Exception:
                # sanitise common patterns
                v = s.replace(" ", "T")
                if v.endswith("+00:00"):
                    v = v.replace("+00:00", "Z")
                if v.endswith("ZZ"):
                    v = v.replace("ZZ", "Z")
                return v
        # Handle simple "YYYY-MM-DD HH:MM:SS" or "YYYY-MM-DD"
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s, fmt)
                return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
            except ValueError:
                continue
        return None
    except Exception:
        return None


def _jsonify_list_field(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return json.dumps([str(x) for x in v])
    if isinstance(v, str):
        # if already json-like, try to keep
        try:
            parsed = json.loads(v)
            if isinstance(parsed, list):
                return json.dumps(parsed)
        except Exception:
            # try to split pipes or commas
            if "|" in v:
                return json.dumps([s for s in v.split("|") if s])
            if "," in v:
                return json.dumps([s.strip() for s in v.split(",") if s.strip()])
            return json.dumps([v])
    return json.dumps([str(v)])


def _prepare_weaviate_object(metadata: Dict[str, Any], text: str) -> Dict[str, Any]:
    """
    Map chunk metadata -> Weaviate object properties.
    Accepts partial metadata; missing keys are omitted.
    Ensures array fields are properly formatted for Weaviate (TEXT_ARRAY).
    """
    obj = {"text": text}

    # Basic scalar fields
    scalar_fields = [
        "source",
        "game_id",
        "unified_game_id",
        "slug",
        "title",
        "description",
        "release_date",
        "release_year",
        "created_at",
        "updated_at",
        "content_length",
        "content_hash",
        "site_detail_url",
        "esrb_rating",
        "language",
    ]
    for field in scalar_fields:
        v = metadata.get(field)
        if v is None:
            continue
        # Normalize date fields for RFC3339
        if field in ("release_date", "created_at", "updated_at"):
            v = _normalize_date(str(v))
            if v is None:
                continue
        obj[field] = v

    # Normalize and clean array fields
    array_fields = [
        "genres",
        "platforms",
        "developers",
        "publishers",
        "tags",
        "themes",
        "stores",
    ]
    for field in array_fields:
        raw = metadata.get(field)
        if raw is None:
            continue
        obj[field] = normalize_text_array(raw)

    # Numeric / float fields
    numeric_fields = [
        "rating",
        "rating_count",
        "user_rating",
        "critic_rating",
        "metacritic",
        "playtime",
        "articles_count",
        "reviews_count",
    ]
    for nf in numeric_fields:
        v = metadata.get(nf)
        if v is not None:
            try:
                obj[nf] = float(v)
            except (ValueError, TypeError):
                pass

    return obj


def _embed_texts(emb_model, texts: List[str], batch_size: int = 64) -> List[List[float]]:
    vectors = []
    if hasattr(emb_model, "embed_documents"):
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vecs = emb_model.embed_documents(batch)
            vectors.extend(batch_vecs)
        return vectors
    if hasattr(emb_model, "client") and hasattr(emb_model.client, "encode"):
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vecs = emb_model.client.encode(batch, normalize_embeddings=True)
            vectors.extend(batch_vecs)
        return vectors
    logger.warning("Embedding model missing bulk API; falling back to single calls.")
    for t in texts:
        v = emb_model.embed([t])[0]
        vectors.append(v)
    return vectors


def _find_existing_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """
    Query Weaviate for an existing canonical record by slug. Returns first object dict or None.
    """
    try:
        collection = client.collections.get(COLLECTION_NAME)
        resp = collection.query.fetch_objects(
            filters=weaviate.classes.query.Filter.by_property("slug").equal(slug),
            limit=1,
        )
        # resp may be dict-like
        objects = None
        if isinstance(resp, dict):
            objects = resp.get("objects") or resp.get("data") or []
        else:
            objects = getattr(resp, "objects", None)
        if objects:
            # try to pluck props
            obj = objects[0]
            if isinstance(obj, dict) and "properties" in obj:
                return obj["properties"]
            return obj
    except Exception as e:
        logger.debug("Existing-by-slug lookup failed: %s", e)
    return None


def _merge_metadata(existing, new_meta):
    """
    Merge metadata from an existing Weaviate object and a new document.
    Keeps arrays unique and concatenates text fields if needed.
    Handles None values gracefully.
    """
    # Handle Weaviate Object (v4) case
    if hasattr(existing, "properties"):
        existing = existing.properties

    if not isinstance(existing, dict):
        existing = {}

    merged = dict(existing)

    for key, val in (new_meta or {}).items():
        if val is None:
            continue

        # Merge arrays
        if key in (
            "genres",
            "platforms",
            "developers",
            "publishers",
            "tags",
            "themes",
            "stores",
        ):
            existing_vals = merged.get(key) or []
            if isinstance(existing_vals, str):
                existing_vals = [existing_vals]
            if isinstance(val, str):
                val = [val]
            merged[key] = list({*map(str, existing_vals), *map(str, val)})

        # Merge numeric fields (overwrite if newer)
        elif isinstance(val, (int, float)):
            merged[key] = val

        # Merge text fields (append if not identical)
        elif isinstance(val, str):
            old = merged.get(key)
            if not isinstance(old, str):
                old = ""  # safely handle None or non-string
            if val not in old:
                merged[key] = (old + " " + val).strip()

        else:
            merged[key] = val

    return merged


def upsert_chunks(chunks: List[Document], batch_size: int = 100):
    if not chunks:
        logger.info("No chunks to upsert.")
        return

    texts = [c.page_content for c in chunks]
    hashes = [_compute_content_hash(t) for t in texts]
    for c, h in zip(chunks, hashes):
        c.metadata = {**(c.metadata or {}), "content_hash": h}

    emb_model = get_embedding_model()
    logger.info(
        "Generating embeddings for %d chunks (batch_size=%d)...", len(texts), batch_size
    )
    vectors = _embed_texts(emb_model, texts, batch_size=batch_size)

    to_upsert = []
    try:
        collection = client.collections.get(COLLECTION_NAME)
    except Exception as e:
        logger.exception("Unable to fetch collection '%s': %s", COLLECTION_NAME, e)
        raise

    for idx, (doc, vec, c_hash) in enumerate(zip(chunks, vectors, hashes)):
        meta = doc.metadata or {}
        slug = meta.get("slug") or meta.get("unified_game_id")
        # check duplicate by content_hash first
        try:
            resp = collection.query.fetch_objects(
                filters=weaviate.classes.query.Filter.by_property("content_hash").equal(
                    c_hash
                ),
                limit=1,
            )
            existing_objs = None
            if isinstance(resp, dict):
                existing_objs = resp.get("objects") or []
            else:
                existing_objs = getattr(resp, "objects", None)
            if existing_objs:
                logger.debug("Duplicate detected (hash): %s — skipping", c_hash)
                continue
        except Exception as e:
            logger.warning(
                "Hash existence check failed for hash %s: %s — will attempt to upsert",
                c_hash,
                e,
            )

        # try to find canonical by slug and merge metadata if found
        if slug:
            existing = _find_existing_by_slug(slug)
            if existing:
                merged_props = _merge_metadata(existing, meta)
                # ensure slug/unified_game_id/title set
                merged_props["slug"] = slug
                merged_props["unified_game_id"] = slug
                # perform a partial update to the existing object (Weaviate update flow)
                try:
                    # Weaviate client expects an object id to update; we fetch it via query again
                    # Here we'll use the langchain_weaviate add_documents flow to upsert; keep merged_props for metadata
                    meta = {**meta, **merged_props}
                except Exception as e:
                    logger.debug(
                        "Failed to merge existing metadata for slug %s: %s", slug, e
                    )

        obj_props = _prepare_weaviate_object(meta, doc.page_content)

        # DEBUG: log first two objects so we can confirm types being sent to Weaviate
        if idx < 2:
            logger.info("🧩 DEBUG OBJECT PREVIEW (idx=%d):", idx)
            for k, v in obj_props.items():
                if k in ("genres", "platforms", "developers", "publishers", "tags", "themes"):
                    logger.info("  %s: %s (type=%s)", k, v, type(v))
        to_upsert.append((obj_props, vec))

    if not to_upsert:
        logger.info("No new chunks to upsert (all duplicates).")
        return

    try:
        docs_for_langchain = []
        vectors_to_send = []
        for props, vec in to_upsert:
            d = Document(page_content=props.get("text", ""), metadata=props)
            docs_for_langchain.append(d)
            vectors_to_send.append(vec)

        vectorstore = WeaviateVectorStore(
            client=client,
            index_name=COLLECTION_NAME,
            text_key="text",
            embedding=None,
            attributes=[
                "source",
                "game_id",
                "unified_game_id",
                "slug",
                "title",
                "description",
                "release_date",
                "release_year",
                "created_at",
                "updated_at",
                "genres",
                "platforms",
                "developers",
                "publishers",
                "tags",
                "themes",
                "stores",
                "rating",
                "rating_count",
                "metacritic",
                "esrb_rating",
                "playtime",
                "articles_count",
                "reviews_count",
                "language",
                "content_length",
                "content_hash",
                "site_detail_url",
            ],
        )

        logger.info("Upserting %d new chunks into '%s'...", len(docs_for_langchain), COLLECTION_NAME)
        vectorstore.add_documents(docs_for_langchain, vectors=vectors_to_send, batch_size=batch_size)
        logger.info("Successfully upserted %d chunks.", len(docs_for_langchain))
    except Exception as e:
        logger.exception("Upsert failed: %s", e)
        raise
    finally:
        try:
            client.close()
        except Exception:
            pass
