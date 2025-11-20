# rough.py
import json
from ingest.loader import load_and_prepare
from ingest.merge import merge_canonical_objects
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    rawg_canonicals, rawg_docs = load_and_prepare("json/rawg_Far_Cry_3.json")
    igdb_canonicals, igdb_docs = load_and_prepare("json/igdb_Far_Cry_3.json")
    gs_canonicals, gs_docs = load_and_prepare("json/gamespot_Far_Cry_3.json")

    all_canonicals = rawg_canonicals + igdb_canonicals + gs_canonicals

    # merge all canonicals that belong together by grouping on slug or title+year
    # for this sample, we assume they all refer to the same game (as in your tests)
    merged = merge_canonical_objects(all_canonicals)

    # combine docs (they are plain dicts already)
    all_docs = rawg_docs + igdb_docs + gs_docs

    # attach unified_game_id to chunks (so they are ready for upsert)
    unified_id = merged.get("unified_game_id")
    for d in all_docs:
        md = d.get("metadata", {})
        if not md.get("unified_game_id"):
            md["unified_game_id"] = unified_id
        # ensure language normalized
        if md.get("language") in (None, "", "unknown"):
            md["language"] = "en"

    # SAVE outputs
    with open("merged_canonical.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    with open("all_docs.json", "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    logger.info("Wrote merged_canonical.json and all_docs.json")
    print(f"Canonicals merged → unified_game_id={unified_id}")
    print(f"Total chunks: {len(all_docs)}")


if __name__ == "__main__":
    main()
