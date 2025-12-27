# usage.py
"""
Live integration test for upsert_platform_specs.py (Weaviate v4)

This script:
1. Connects to a local Weaviate v4 instance.
2. Ensures the Game collection exists (creates a minimal one if missing).
3. Creates a real canonical Game object (via RAWG-backed identity ingest).
4. Calls upsert_platform_specs with a real game name and UUID.
5. Verifies PlatformSpec objects via GraphQL filtered by game reference.

Requirements:
- Weaviate v4 running locally (http://localhost:8080)
- RAWG_API_KEY set in environment
- weaviate-client >= 4.x
"""

import logging
import sys
from typing import Any, Dict

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.util import generate_uuid5

from upsert.upsert_platform_specs import upsert_platform_specs
from ingest.rawg_identity_ingest import fetch_and_prepare_identity

# ------------------------------------------------------------------
# Logging (project-consistent)
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

WEAVIATE_URL = "http://localhost:8080"
GAME_COLLECTION = "Game"

# Fixed namespace UUID for deterministic Game UUIDs
GAME_NAMESPACE_UUID = "12345678-1234-5678-1234-567812345678"


# ------------------------------------------------------------------
# Schema helpers (Weaviate v4)
# ------------------------------------------------------------------
def ensure_game_collection(client: weaviate.WeaviateClient) -> None:
    """
    Ensure the Game collection exists.
    If missing, create a minimal collection sufficient for references.
    """
    if client.collections.exists(GAME_COLLECTION):
        logger.info("Game collection already exists.")
        return

    logger.warning("Game collection missing. Creating minimal Game collection...")

    client.collections.create(
        name=GAME_COLLECTION,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(
                name="game_id",
                data_type=DataType.INT,
            ),
            Property(
                name="title",
                data_type=DataType.TEXT,
            ),
            Property(
                name="release_year",
                data_type=DataType.INT,
            ),
        ],
    )

    logger.info("Game collection created.")


# ------------------------------------------------------------------
# Canonical Game creation (Weaviate v4)
# ------------------------------------------------------------------
def create_canonical_game(
    client: weaviate.WeaviateClient, game_name: str
) -> str:
    """
    Fetch RAWG-backed canonical identity and upsert a Game object.
    Returns the deterministic Game UUID.
    """
    logger.info("Fetching canonical identity for '%s'...", game_name)
    game_obj = fetch_and_prepare_identity(game_name)

    rawg_game_id = game_obj.get("game_id")
    if rawg_game_id is None:
        raise RuntimeError("Canonical identity missing RAWG game_id")

    game_uuid = generate_uuid5(GAME_NAMESPACE_UUID, str(game_obj["game_id"]))

    logger.info("Upserting Game '%s' with UUID %s", game_name, game_uuid)

    game_collection = client.collections.get(GAME_COLLECTION)
    game_collection.data.insert(
        uuid=game_uuid,
        properties=game_obj,
    )

    return str(game_uuid)


# ------------------------------------------------------------------
# Verification (GraphQL v4-compatible)
# ------------------------------------------------------------------
def verify_platform_specs(
    client: weaviate.WeaviateClient, game_uuid: str
) -> None:
    """
    Query PlatformSpec objects filtered by game reference and print them.
    """
    logger.info("Verifying PlatformSpec objects linked to Game %s", game_uuid)

    query = f"""
    {{
      Get {{
        PlatformSpec(
          where: {{
            path: ["game"],
            operator: Equal,
            valueString: "weaviate://localhost/Game/{game_uuid}"
          }}
        ) {{
          platform_name
          platform_family
          release_date
          requirements_minimum
          requirements_recommended
        }}
      }}
    }}
    """

    result = client.graphql_raw(query)
    specs = (
        result.get("data", {})
        .get("Get", {})
        .get("PlatformSpec", [])
    )

    logger.info("Retrieved %d PlatformSpec objects.", len(specs))
    for spec in specs:
        print(spec)


# ------------------------------------------------------------------
# Main execution
# ------------------------------------------------------------------
def main() -> None:
    game_name = "Far Cry 5"  # real example

    try:
        client = weaviate.connect_to_local()
    except Exception as exc:
        raise SystemExit(f"❌ Failed to connect to Weaviate: {exc}")

    # 1. Ensure prerequisite schema
    ensure_game_collection(client)

    # 2. Create canonical Game anchor
    try:
        game_uuid = create_canonical_game(client, game_name)
    except Exception as exc:
        logger.error("Failed to create canonical Game: %s", exc)
        client.close()
        sys.exit(1)

    # 3. Run PlatformSpec upsert (still uses client object)
    logger.info("Running PlatformSpec upsert for '%s'...", game_name)
    count = upsert_platform_specs(
        client=client,  # v4 client is compatible for batch/REST usage
        game_name=game_name,
        game_uuid=game_uuid,
    )
    logger.info("Upserted %d PlatformSpec objects.", count)

    # 4. Verify linkage
    verify_platform_specs(client, game_uuid)

    client.close()


if __name__ == "__main__":
    main()
