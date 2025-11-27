#!/usr/bin/env python3
"""
ingest/upsert.py  (patched)

Batch-upserts precomputed embeddings (jsonl) into Weaviate via REST API with:
 - type coercion to match schema
 - per-object result parsing and failure logging
 - deterministic UUIDv5 id mapping (so re-runs are idempotent)

Input lines: {"id": "<chunk-id>", "embedding": [...], "meta": {...}}
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Any
import requests
import sys
import uuid

DEFAULT_DIM = 384
DEFAULT_BATCH = 64
DEFAULT_TIMEOUT = 30  # seconds for HTTP requests
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5

def normalize_id_to_uuid(id_str: str) -> str:
    if not id_str:
        return str(uuid.uuid4())
    try:
        u = uuid.UUID(id_str)
        return str(u)
    except Exception:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, id_str))

# --- Type coercion helpers ---
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

# Map meta keys -> schema property names (with coercion)
def meta_to_props(meta: dict) -> dict:
    if meta is None:
        meta = {}
    props = {}
    # text fields (must be strings in schema)
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
    # array fields
    props["platforms"] = coerce_text_array(meta.get("platforms"))
    props["genres"] = coerce_text_array(meta.get("genres"))
    props["developers"] = coerce_text_array(meta.get("developers"))
    props["publishers"] = coerce_text_array(meta.get("publishers"))
    # optional chunk text
    if meta.get("text") is not None:
        t = coerce_text(meta.get("text"))
        if t is not None:
            props["text"] = t
    # always keep a meta snapshot for traceability
    props["meta"] = json.dumps(meta, ensure_ascii=False)
    # drop empty-array fields? Weaviate accepts empty arrays for text[] so keep them.
    # Remove None values
    return {k: v for k, v in props.items() if v is not None}

# read vectors jsonl
def read_vectors(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for ln_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON at line {ln_no} in {path}: {e}")
            if "id" not in obj or "embedding" not in obj:
                raise RuntimeError(f"Missing required keys (id, embedding) at line {ln_no}")
            yield obj

def chunked_iterable(it, size):
    buf = []
    for item in it:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# Batch send with retries (HTTP)
def batch_post(weaviate_url: str, payload: dict, timeout: int = DEFAULT_TIMEOUT):
    url = f"{weaviate_url.rstrip('/')}/v1/batch/objects"
    headers = {"Content-Type": "application/json"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if resp.status_code in (200, 201):
                # attempt to parse JSON
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

# Parse Weaviate batch response into list of per-object results (handles multiple shapes)
def parse_batch_response(data):
    if isinstance(data, dict):
        # common v4 style: {"results": {"objects": [ ... ]}}
        if "results" in data and isinstance(data["results"], dict) and "objects" in data["results"]:
            return data["results"]["objects"]
        # sometimes top-level objects key
        if "objects" in data and isinstance(data["objects"], list):
            return data["objects"]
        # fallback: find first list value
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
        # result.errors
        if "result" in item and isinstance(item["result"], dict):
            errs = item["result"].get("errors")
            if errs:
                return errs
        # direct errors
        if "errors" in item and item["errors"]:
            return item["errors"]
        # status.error
        if "status" in item and isinstance(item["status"], dict) and "error" in item["status"]:
            return [item["status"]["error"]]
    return None

# optional: GraphQL aggregate check
def aggregate_count(weaviate_url: str, class_name: str):
    gql = {"query": f"{{ Aggregate {{ {class_name} {{ meta {{ count }} }} }} }}"}
    r = requests.post(f"{weaviate_url.rstrip('/')}/v1/graphql", json=gql, timeout=20)
    if r.status_code == 200:
        try:
            d = r.json()
            return d.get("data",{}).get("Aggregate",{}).get(class_name,[{}])[0].get("meta",{}).get("count")
        except Exception:
            return None
    return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vectors", required=True, help="Path to vectors jsonl")
    p.add_argument("--weaviate", default="http://localhost:8080", help="Weaviate base URL")
    p.add_argument("--class", dest="class_name", default="GameChunk", help="Weaviate class name")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--dim", type=int, default=DEFAULT_DIM, help="Expected embedding dimension")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--dry-run", action="store_true", help="Validate and print sample objects without sending")
    args = p.parse_args()

    vec_path = Path(args.vectors)
    if not vec_path.exists():
        print(f"[ERROR] Vectors file not found: {vec_path}", file=sys.stderr)
        sys.exit(2)

    print(f"[INFO] Validating vectors file: {vec_path}")
    objs_gen = read_vectors(vec_path)
    total = 0
    sample_printed = False

    successes = []
    failures = []

    for batch in chunked_iterable(objs_gen, args.batch_size):
        batch_payload = []
        batch_mapping = []  # keep raw_id to map results
        for item in batch:
            total += 1
            raw_vid = item["id"]
            vid = normalize_id_to_uuid(raw_vid)
            embedding = item["embedding"]
            if not isinstance(embedding, list):
                raise RuntimeError(f"Embedding for id={raw_vid} is not a list")
            if len(embedding) != args.dim:
                raise RuntimeError(f"Embedding dim mismatch for id={raw_vid}: expected {args.dim}, got {len(embedding)}")
            meta = item.get("meta", {}) or {}
            props = meta_to_props(meta)
            obj = {
                "class": args.class_name,
                "id": vid,
                "properties": props,
                "vector": embedding
            }
            batch_payload.append(obj)
            batch_mapping.append({"raw_id": raw_vid, "uuid": vid})
            if not sample_printed:
                print("[DEBUG] Sample object to upsert:")
                print(json.dumps(obj, indent=2, ensure_ascii=False)[:1000])
                sample_printed = True

        if args.dry_run:
            print(f"[DRY-RUN] would upsert batch of {len(batch_payload)} objects")
            continue

        # send batch and capture response
        try:
            resp = batch_post(args.weaviate, {"objects": batch_payload}, timeout=args.timeout)
        except Exception as e:
            print(f"[ERROR] Batch post failed: {e}", file=sys.stderr)
            # mark all in this batch as failures
            for m in batch_mapping:
                failures.append({"raw_id": m["raw_id"], "uuid": m["uuid"], "error": str(e)})
            continue

        per_results = parse_batch_response(resp)
        if not per_results:
            # no per-object results: we treat as success but log a warning
            print("[WARN] no per-object results in response; treating batch as success (but check server).")
            for m in batch_mapping:
                successes.append({"raw_id": m["raw_id"], "uuid": m["uuid"]})
            continue

        # iterate and map results (zip - handle shorter lists)
        for m, res in zip(batch_mapping, per_results):
            errs = result_has_errors(res)
            if errs:
                failures.append({"raw_id": m["raw_id"], "uuid": m["uuid"], "errors": errs, "response_fragment": res})
            else:
                successes.append({"raw_id": m["raw_id"], "uuid": m["uuid"]})

        # if response shorter than sent batch, treat remaining as failures
        if len(per_results) < len(batch_mapping):
            for m in batch_mapping[len(per_results):]:
                failures.append({"raw_id": m["raw_id"], "uuid": m["uuid"], "errors": ["no result returned for object"]})

        print(f"[INFO] Processed batch of {len(batch_payload)} → successes so far: {len(successes)} failures so far: {len(failures)}")

    # write outputs
    with open("upsert_success.jsonl", "w", encoding="utf-8") as s:
        for obj in successes:
            s.write(json.dumps(obj, ensure_ascii=False) + "\n")
    with open("upsert_failed.jsonl", "w", encoding="utf-8") as f:
        for obj in failures:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print("\n==== SUMMARY ====")
    print("Total processed:", total)
    print("Success count:", len(successes))
    print("Failed count:", len(failures))
    if failures:
        print("Sample failures (first 10):")
        for it in failures[:10]:
            print(json.dumps(it, ensure_ascii=False)[:500])
    # final aggregate check
    try:
        cnt = aggregate_count(args.weaviate, args.class_name)
        if cnt is not None:
            print(f"[INFO] Weaviate aggregate count for class {args.class_name}: {cnt}")
    except Exception:
        pass

    if failures:
        print("Wrote upsert_success.jsonl and upsert_failed.jsonl")
        sys.exit(3)
    else:
        print("All items upserted successfully.")
        sys.exit(0)

if __name__ == "__main__":
    main()
