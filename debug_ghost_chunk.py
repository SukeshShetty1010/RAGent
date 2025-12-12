#!/usr/bin/env python3
"""
debug_ghost_chunk.py

Forensic tool to diagnose “Ghost Data” — chunks that appear to be upserted
in logs but are not retrieved later by the agent.

This script performs 3 checks:

1. Direct ID lookup (ground truth)
2. Real vector search via RetrieverTool
3. Shadow-data inspection (is some old chunk outranking Valhalla?)

Requires:
- Weaviate running at localhost:8080
- Existing project structure so we can import RetrieverTool
"""

import requests
import json
import sys
from pprint import pprint

# --- Project imports ---
from agent.tools.retriever_tool import RetrieverTool  # uses real Retriever
# CAUTION: REQUIRE your project root in PYTHONPATH:
# export PYTHONPATH=.

WEAVIATE_URL = "http://localhost:8080"
CHUNK_CLASS = "GameChunk"
TARGET_ID = "9035187e-9359-56ca-b031-cc8109659cce"
QUERY = "What are the minimum requirements for Assassin's Creed Valhalla?"
K = 20


# -------------------------------------------------------
# CHECK 1 — DIRECT ID LOOKUP (is the chunk actually alive?)
# -------------------------------------------------------
def check_chunk_exists():
    print("\n=== CHECK 1 — Direct ID Lookup ===")
    url = f"{WEAVIATE_URL}/v1/objects/{CHUNK_CLASS}/{TARGET_ID}"
    try:
        r = requests.get(url, timeout=10)
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return False

    if r.status_code == 200:
        print("✅ Chunk EXISTS in DB.")
        obj = r.json()
        title = obj.get("properties", {}).get("title")
        print(f"   Title: {title}")
        return True
    else:
        print(f"❌ Chunk NOT found. Status {r.status_code}")
        print("   Upsert may have failed or was not committed.")
        return False


# -------------------------------------------------------
# CHECK 2 — VECTOR SEARCH (where does the chunk rank?)
# -------------------------------------------------------
def check_vector_search():
    print("\n=== CHECK 2 — Vector Search via RetrieverTool ===")

    retriever = RetrieverTool(
        weaviate_url=WEAVIATE_URL,
        class_name=CHUNK_CLASS,
    )

    # NOTE: similarity_threshold=None → RETURN ALL results; no filtering
    results = retriever.execute({
        "query": QUERY,
        "k": K,
        "similarity_threshold": None,  # critical for diagnostics
        "debug": False,
        "show_meta": True
    })

    if not results:
        print("❌ No retrieval results at all.")
        return None

    found = None
    for rank, chunk in enumerate(results, start=1):
        cid = chunk.get("id") or chunk.get("_additional", {}).get("id")
        score = chunk.get("certainty") or chunk.get("score") or \
                chunk.get("_additional", {}).get("certainty")

        if cid == TARGET_ID:
            found = (rank, score)
            print(f"🎯 FOUND Valhalla chunk at rank {rank} with score={score}")
            if score is not None and score < 0.6:
                print("⚠️ WARNING: Similarity < 0.6 — your agent will REJECT this chunk.")
            break

    if not found:
        print("❌ Target chunk NOT present in top-K search results.")
    return results, found


# -------------------------------------------------------
# CHECK 3 — SHADOW DATA (are old chunks still winning?)
# -------------------------------------------------------
def check_shadow_data(results, valhalla_info):
    print("\n=== CHECK 3 — Shadow Data Check ===")

    if not results:
        print("No results → cannot analyze shadow data.")
        return

    top = results[0]
    top_id = top.get("id")
    top_title = top.get("title") or top.get("meta", {}).get("title")
    top_score = top.get("certainty") or top.get("score") or \
                top.get("_additional", {}).get("certainty")

    print(f"🏆 Top retrieved chunk:")
    print(f"   ID: {top_id}")
    print(f"   Title: {top_title}")
    print(f"   Score: {top_score}")

    if valhalla_info:
        rank, score = valhalla_info
        print(f"\n📌 Valhalla chunk rank = {rank}, score = {score}")
        if top_score and score and top_score > score:
            print("⚠️ SHADOW DATA DETECTED: Another chunk outranks Valhalla.")
        else:
            print("✅ No shadow data issue: Valhalla is not suppressed by older chunks.")
    else:
        print("\n❌ No Valhalla chunk found → likely ingestion or filtering issue.")


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":
    print("\n🔍 Running Ghost Chunk Forensics...\n")

    exists = check_chunk_exists()

    results, valhalla_info = (None, None)
    if exists:
        out = check_vector_search()
        if out is not None:
            results, valhalla_info = out

    check_shadow_data(results, valhalla_info)

    print("\n🔚 Done.\n")
