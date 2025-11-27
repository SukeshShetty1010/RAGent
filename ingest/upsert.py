#!/usr/bin/env python3
"""
ingest/upsert.py  (reworked)

- Provides reusable functions for upserting vector JSONL objects into Weaviate.
- Provides a CLI/main orchestration that can:
    * generate merged->chunks->vectors for a game (via ingest.embeddings.create_embeddings_for_game)
    * OR take an explicit --vectors JSONL file
    * then upsert the vectors into Weaviate using batch API.

Vector JSONL format expected (one JSON object per line):
  {"id": "<chunk-id>", "embedding": [...], "meta": {...}}

This file purposefully separates:
 - upsert_vectors(...) : pure upsert logic (can be imported & reused)
 - CLI/main orchestration that calls the project's ingest pipeline if needed.

python -m ingest.upsert --game "Far Cry 5" --outdir ./out --weaviate http://localhost:8080
python -m ingest.upsert --vectors ./out/far_cry_5_vectors.jsonl --weaviate http://localhost:8080

"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

# --- Constants & defaults ---
DEFAULT_DIM = 384
DEFAULT_BATCH = 64
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5

# --- Utilities (ID normalization + type coercion) ---
def normalize_id_to_uuid(id_str: str) -> str:
    if not id_str:
        return str(uuid.uuid4())
    try:
        u = uuid.UUID(id_str)
        return str(u)
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, id_str))

def coerce_text(x: Any):
    if x is None:
        return None
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, str):
        return x
    return json.dumps(x, ensure_ascii=False)

def coerce_text_array(x: Any):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(e) for e in x if e is not None]
    return [str(x)]

def coerce_int(x: Any):
    try:
        return int(x)
    except Exception:
        return None

def meta_to_props(meta: dict) -> dict:
    """Map chunk/meta fields -> Weaviate properties with simple coercion."""
    if meta is None:
        meta = {}
    props: Dict[str, Any] = {}
    for key in ("unified_id", "doc_id", "source", "title", "content_hash", "release_date"):
        if key in meta and meta.get(key) is not None:
            v = coerce_text(meta.get(key))
            if v is not None:
                props[key] = v
    # numeric fields
    if "chunk_index" in meta:
        ci = coerce_int(meta.get("chunk_index"))
        if ci is not None:
            props["chunk_index"] = ci
    if "char_length" in meta:
        cl = coerce_int(meta.get("char_length"))
        if cl is not None:
            props["char_length"] = cl
    if "release_year" in meta:
        ry = coerce_int(meta.get("release_year"))
        if ry is not None:
            props["release_year"] = ry
    # arrays
    props["platforms"] = coerce_text_array(meta.get("platforms"))
    props["genres"] = coerce_text_array(meta.get("genres"))
    props["developers"] = coerce_text_array(meta.get("developers"))
    props["publishers"] = coerce_text_array(meta.get("publishers"))
    if meta.get("text") is not None:
        t = coerce_text(meta.get("text"))
        if t is not None:
            props["text"] = t
    props["meta"] = json.dumps(meta, ensure_ascii=False)
    return {k: v for k, v in props.items() if v is not None}

# --- Read vectors JSONL (tolerant) ---
def read_vectors_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Vectors file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, start=1):
            ln = line.strip()
            if not ln:
                continue
            # tolerate trailing comma
            if ln.endswith(","):
                ln = ln[:-1].rstrip()
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {ln_no} in {path}: {e}")
            if "id" not in obj or "embedding" not in obj:
                raise RuntimeError(f"Missing required keys (id, embedding) at line {ln_no} in {path}")
            yield obj

# --- chunked iterable ---
def chunked_iterable(it: Iterable, size: int):
    buf = []
    for item in it:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# --- HTTP batch POST with retry/backoff ---
def batch_post(weaviate_url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT):
    url = f"{weaviate_url.rstrip('/')}/v1/batch/objects"
    headers = {"Content-Type": "application/json"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code in (200, 201):
                try:
                    return resp.json()
                except Exception:
                    return resp.text
            else:
                text = resp.text
                raise RuntimeError(f"HTTP {resp.status_code} {text}")
        except Exception as e:
            backoff = BACKOFF_FACTOR ** (attempt - 1)
            print(f"[WARN] Batch post attempt {attempt} failed: {e}. Retrying in {backoff:.1f}s...", file=sys.stderr)
            time.sleep(backoff)
    raise RuntimeError("Failed to POST batch after retries")

def parse_batch_response(data):
    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], dict) and "objects" in data["results"]:
            return data["results"]["objects"]
        if "objects" in data and isinstance(data["objects"], list):
            return data["objects"]
        for v in data.values():
            if isinstance(v, list):
                return v
        return []
    elif isinstance(data, list):
        return data
    else:
        return []

def result_has_errors(item):
    if not item:
        return None
    if isinstance(item, dict):
        if "result" in item and isinstance(item["result"], dict):
            errs = item["result"].get("errors")
            if errs:
                return errs
        if "errors" in item and item["errors"]:
            return item["errors"]
        if "status" in item and isinstance(item["status"], dict) and "error" in item["status"]:
            return [item["status"]["error"]]
    return None

# optional: aggregate check (keeps original behaviour)
def aggregate_count(weaviate_url: str, class_name: str):
    gql = {"query": f"{{ Aggregate {{ {class_name} {{ meta {{ count }} }} }} }}"}
    r = requests.post(f"{weaviate_url.rstrip('/')}/v1/graphql", json=gql, timeout=20)
    if r.status_code == 200:
        try:
            d = r.json()
            return d.get("data", {}).get("Aggregate", {}).get(class_name, [])[0].get("meta", {}).get("count")
        except Exception:
            return None
    return None

# --- Core upsert function (reusable) ---
def upsert_vectors(
    vectors_path: str,
    weaviate_url: str = "http://localhost:8080",
    class_name: str = "GameChunk",
    batch_size: int = DEFAULT_BATCH,
    dim: int = DEFAULT_DIM,
    timeout: int = DEFAULT_TIMEOUT,
    dry_run: bool = False,
) -> Dict[str, int]:
    """
    Read vectors JSONL and upsert into Weaviate using batch API.
    Returns a summary dict: {"processed": n, "success": s, "failed": f}
    """
    vec_p = Path(vectors_path)
    processed = 0
    successes = 0
    failures = 0
    failures_list = []

    print(f"[INFO] Validating vectors file: {vec_p}")
    objs_gen = read_vectors_jsonl(vec_p)

    for batch in chunked_iterable(objs_gen, batch_size):
        batch_payload = []
        batch_map = []
        for item in batch:
            processed += 1
            raw_id = item["id"]
            vid = normalize_id_to_uuid(raw_id)
            embedding = item["embedding"]
            if not isinstance(embedding, list):
                raise RuntimeError(f"Embedding for id={raw_id} is not a list")
            if len(embedding) != dim:
                raise RuntimeError(f"Embedding dim mismatch for id={raw_id}: expected {dim}, got {len(embedding)}")
            meta = item.get("meta", {}) or {}
            props = meta_to_props(meta)
            obj = {
                "class": class_name,
                "id": vid,
                "properties": props,
                "vector": embedding
            }
            batch_payload.append(obj)
            batch_map.append({"raw_id": raw_id, "uuid": vid})
            # print sample
            if processed == 1:
                print("[DEBUG] Sample object to upsert (truncated):")
                print(json.dumps(obj, indent=2, ensure_ascii=False)[:1000])

        if dry_run:
            print(f"[DRY-RUN] would upsert batch of {len(batch_payload)} objects")
            successes += len(batch_payload)
            continue

        try:
            resp = batch_post(weaviate_url, {"objects": batch_payload}, timeout=timeout)
        except Exception as e:
            print(f"[ERROR] Batch post failed: {e}", file=sys.stderr)
            failures += len(batch_map)
            for m in batch_map:
                failures_list.append({"raw_id": m["raw_id"], "uuid": m["uuid"], "error": str(e)})
            continue

        per_results = parse_batch_response(resp)
        if not per_results:
            print("[WARN] no per-object results in response; treating batch as success (but check server).")
            successes += len(batch_map)
            continue

        for m, res in zip(batch_map, per_results):
            errs = result_has_errors(res)
            if errs:
                failures += 1
                failures_list.append({"raw_id": m["raw_id"], "uuid": m["uuid"], "errors": errs, "response_fragment": res})
            else:
                successes += 1

        if len(per_results) < len(batch_map):
            for m in batch_map[len(per_results):]:
                failures += 1
                failures_list.append({"raw_id": m["raw_id"], "uuid": m["uuid"], "errors": ["no result returned for object"]})

        print(f"[INFO] Processed batch of {len(batch_payload)} → successes so far: {successes} failures so far: {failures}")

    # write summaries
    with open("upsert_success.jsonl", "w", encoding="utf-8") as s:
        # intentionally minimal success records
        for i in range(successes):
            # we don't keep per-success mapping here by default to avoid huge files; kept for parity with original you can extend
            s.write(json.dumps({"index": i}, ensure_ascii=False) + "\n")
    with open("upsert_failed.jsonl", "w", encoding="utf-8") as f:
        for obj in failures_list:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n==== SUMMARY ====")
    print("Total processed:", processed)
    print("Success count:", successes)
    print("Failed count:", failures)
    if failures_list:
        print("Sample failures (first 10):")
        for it in failures_list[:10]:
            print(json.dumps(it, ensure_ascii=False)[:500])

    try:
        cnt = aggregate_count(weaviate_url, class_name)
        if cnt is not None:
            print(f"[INFO] Weaviate aggregate count for class {class_name}: {cnt}")
    except Exception:
        pass

    return {"processed": processed, "success": successes, "failed": failures}

# --- Orchestration / CLI: generate vectors (if needed) then upsert ---
def main():
    p = argparse.ArgumentParser(prog="ingest.upsert", description="Create embeddings (optional) and upsert into Weaviate")
    p.add_argument("--game", "-g", help="Game name to fetch, merge, chunk, embed and upsert (if --vectors omitted)")
    p.add_argument("--merged", "-m", help="Path to pre-merged JSON (skip fetch/merge)")
    p.add_argument("--vectors", "-v", help="Path to vectors JSONL (if omitted and --game provided, vectors will be produced)")
    p.add_argument("--outdir", "-o", default=".", help="Output dir for pipeline-generated files")
    p.add_argument("--weaviate", default="http://localhost:8080", help="Weaviate base URL")
    p.add_argument("--class", dest="class_name", default="GameChunk", help="Weaviate class name")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Expected embedding dimension")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--dry-run", action="store_true", help="Validate and print sample objects without sending")
    p.add_argument("--no-generate", dest="no_generate", action="store_true", help="Do NOT generate embeddings even if --game provided (use only --vectors)")
    args = p.parse_args()

    vectors_path = args.vectors

    # If vectors path not provided but game or merged is provided, call the embeddings orchestration
    if not vectors_path:
        if not args.no_generate:
            if args.merged:
                # use provided merged json to create chunks & embeddings
                merged_path = args.merged
                try:
                    # Use ingest.embeddings.create_embeddings_for_game if available
                    from ingest.embeddings import create_embeddings_for_game
                except Exception as e:
                    print("ERROR: ingest.embeddings not importable. Run this from project root so 'ingest' package is available.", file=sys.stderr)
                    raise

                print(f"[INFO] Creating embeddings from merged file: {merged_path}")
                res = create_embeddings_for_game(game_name=None, merged_path=merged_path, outdir=args.outdir)
                vectors_path = res.get("vectors")
            elif args.game:
                # full pipeline: fetch+merge+chunk+embed
                try:
                    from ingest.embeddings import create_embeddings_for_game
                except Exception as e:
                    print("ERROR: ingest.embeddings not importable. Run this from project root so 'ingest' package is available.", file=sys.stderr)
                    raise
                print(f"[INFO] Running full pipeline for game: {args.game}")
                res = create_embeddings_for_game(game_name=args.game, merged_path=None, outdir=args.outdir)
                vectors_path = res.get("vectors")
            else:
                print("ERROR: either --vectors must be provided, or --game/--merged must be given to generate vectors.", file=sys.stderr)
                sys.exit(2)
        else:
            print("ERROR: --no-generate set and no --vectors provided. Nothing to upsert.", file=sys.stderr)
            sys.exit(2)

    if not vectors_path:
        print("ERROR: failed to determine vectors file path.", file=sys.stderr)
        sys.exit(3)

    # At this point vectors_path should be set
    print(f"[INFO] Upserting vectors from: {vectors_path} -> Weaviate: {args.weaviate} (class: {args.class_name})")
    summary = upsert_vectors(
        vectors_path=vectors_path,
        weaviate_url=args.weaviate,
        class_name=args.class_name,
        batch_size=args.batch_size,
        dim=args.dim,
        timeout=args.timeout,
        dry_run=args.dry_run,
    )
    # exit code mapping: 0 success, 3 failures
    if summary["failed"] > 0:
        sys.exit(3)
    sys.exit(0)

if __name__ == "__main__":
    main()
