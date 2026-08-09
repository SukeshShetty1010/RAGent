"""
upsert/upsert_editorial_source.py

Stage 4 container upsert for the additive editorial providers
(Wikipedia, Steam — see ingest/editorial_providers.py). GameSpot keeps
its own dedicated GameSpot_Game path (upsert/upsert_gamespot_chunks.py);
Wikipedia and Steam containers share one metadata-only collection,
EditorialSource, distinguished by their `source` payload field.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)

EDITORIAL_SOURCE_COLLECTION = "EditorialSource"


def upsert_editorial_source_container(
    client: QdrantClient,
    container: Optional[Dict[str, Any]],
    game_name: str,
) -> Optional[str]:
    """
    Idempotently upsert a single normalized editorial container payload
    (from ingest/wikipedia_editorial_normalize.py or
    ingest/steam_editorial_normalize.py) into the EditorialSource
    collection, linked to the canonical Game via `game_uuid`.

    Returns the container UUID, or None if `container` is None (the
    provider had no data — not an error).
    """
    if not container:
        return None

    obj_uuid = container.get("uuid")
    properties = container.get("properties")
    if not obj_uuid or not isinstance(properties, dict):
        raise RuntimeError("Invalid editorial source container structure")

    source = properties.get("source", "unknown")

    existing = client.retrieve(collection_name=EDITORIAL_SOURCE_COLLECTION, ids=[obj_uuid])
    if existing:
        logger.info(
            "EditorialSource (%s) already exists for '%s'. Skipping.", source, game_name
        )
        return obj_uuid

    client.upsert(
        collection_name=EDITORIAL_SOURCE_COLLECTION,
        points=[PointStruct(id=obj_uuid, vector=[0.0], payload=properties)],
    )

    logger.info("EditorialSource (%s) container upserted for '%s'.", source, game_name)
    return obj_uuid
