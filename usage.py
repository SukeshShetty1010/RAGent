"""
usage.py

Live integration test for Editorial Chunking (GameSpot → Chunks → JSON).

This script validates:
GameSpot API → Normalization → EditorialChunker → Retrieval-ready chunks → JSON export

Focus:
- Real editorial data (no mocks)
- Token-bounded chunking (500 / overlap 50)
- Strict adherence to chunk_contract.md
- Parent/lineage integrity (No Orphan Chunks)
- JSON export of chunks
"""

import uuid
import json
from collections import Counter
from pprint import pprint

# ---------------------------------------------------------
# Live imports
# ---------------------------------------------------------
from ingest.gamespot_editorial_normalize import fetch_and_prepare_gamespot
from chunking.editorial_chunker import EditorialChunker

def main():
    # -----------------------------------------------------
    # 1. Narrative-heavy target
    # -----------------------------------------------------
    game_name = "Far Cry 5"
    game_uuid = str(uuid.uuid4())

    print(f"\n🎮 Editorial Chunking Target: {game_name}")
    print(f"🧬 Canonical Game UUID: {game_uuid}")

    # -----------------------------------------------------
    # 2. Fetch + Normalize (live GameSpot)
    # -----------------------------------------------------
    gamespot_payload = fetch_and_prepare_gamespot(
        game_name=game_name,
        canonical_game_uuid=game_uuid,
    )

    if gamespot_payload is None:
        print("\n⚠️  WARNING: No GameSpot editorial content found.")
        print("Chunking skipped.")
        return

    gamespot_uuid = gamespot_payload["uuid"]
    editorial_object = gamespot_payload["properties"]

    # -----------------------------------------------------
    # 3. Chunking
    # -----------------------------------------------------
    chunker = EditorialChunker(chunk_size=500, overlap=50)
    chunks = chunker.process_game_editorial(
        editorial_object=editorial_object,
        game_uuid=game_uuid,
        gamespot_uuid=gamespot_uuid,
    )

    if not chunks:
        print("\n⚠️  WARNING: 0 chunks generated (no usable editorial text).")
        return

    # -----------------------------------------------------
    # 4. Contract validation (chunk_contract.md)
    # -----------------------------------------------------
    for chunk in chunks:
        # Identity
        assert "chunk_id" in chunk and isinstance(chunk["chunk_id"], str), "Invalid chunk_id"
        assert chunk.get("game_uuid") == game_uuid, "game_uuid mismatch"
        assert (
            chunk.get("parent_editorial_uuid") == gamespot_uuid
        ), "parent_editorial_uuid mismatch"

        # Content
        assert chunk.get("content"), "Empty chunk content"
        assert chunk.get("source") == "gamespot", "Invalid source"
        assert chunk.get("content_type") in ("review", "article"), "Invalid content_type"

        # Token boundary check
        token_count = len(chunker.tokenizer.encode(chunk["content"]))
        assert token_count <= 520, f"Chunk exceeds token limit: {token_count}"

    # -----------------------------------------------------
    # 5. Reporting
    # -----------------------------------------------------
    type_counts = Counter(c["content_type"] for c in chunks)

    print("\n📊 Chunk Extraction Summary")
    print(
        f"  Extracted {len(chunks)} chunks "
        f"({type_counts.get('review', 0)} Reviews, "
        f"{type_counts.get('article', 0)} Articles)"
    )

    total_tokens = sum(len(chunker.tokenizer.encode(c["content"])) for c in chunks)
    print(f"  Total tokens across all chunks: {total_tokens}")

    # -----------------------------------------------------
    # 6. Boundary inspection
    # -----------------------------------------------------
    print("\n🔍 FIRST CHUNK (start of editorial):")
    pprint(chunks[0])

    print("\n🔍 LAST CHUNK (end of editorial):")
    pprint(chunks[-1])

    # -----------------------------------------------------
    # 7. Save chunks to JSON file
    # -----------------------------------------------------
    output_filename = f"{game_name.replace(' ', '_')}_chunks.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Chunks saved to: {output_filename}")

if __name__ == "__main__":
    main()
