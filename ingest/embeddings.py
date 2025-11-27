# ingest/embeddings.py

'''
python -m ingest.embeddings --game "Far Cry 5" --outdir ./out
python -m ingest.embeddings --merged ./out/far_cry_5_merged.json --outdir ./out
'''


from __future__ import annotations
import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional, Set

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# Defaults
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH = 32

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ingest.embeddings")


# --------------------------
# I/O helpers (JSONL / Array)
# --------------------------
def _iter_chunks_from_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON line %d in %s: %s", i + 1, path, e)


def _iter_chunks_from_json_array(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON file does not contain an array of chunks")
        for obj in data:
            yield obj


def _iter_chunks_from_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a file with one JSON object per line (JSONL)."""
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            # tolerate trailing commas accidentally present (e.g., last line: "},")
            if line.endswith(","):
                line = line[:-1].rstrip()
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # skip bad lines but log if you want; for now re-raise with context
                raise json.JSONDecodeError(f"Invalid JSON on line {lineno} of {path}: {e.msg}", e.doc, e.pos)

def load_chunks(path: str) -> List[Dict[str, Any]]:
    """
    Load chunks from either:
      - a JSON array file: [ {...}, {...}, ... ]
      - a JSONL file: one JSON object per line
    Returns list of chunk dicts.
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    # Try the fast path: parse whole file as JSON (array expected)
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return data
            # If the file is a single object (dict), but you expect list of chunks,
            # wrap single object in list for compatibility.
            if isinstance(data, dict):
                return [data]
            # if some other JSON type (string/number) fallback to jsonl path
    except json.JSONDecodeError:
        # fallback to JSONL parsing below
        pass

    # Fallback: parse as JSONL (one JSON object per line)
    chunks = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            if raw.endswith(","):
                raw = raw[:-1].rstrip()
            try:
                obj = json.loads(raw)
                chunks.append(obj)
            except json.JSONDecodeError as e:
                # If you prefer to be strict, re-raise with more info:
                raise json.JSONDecodeError(f"Failed to parse JSONL at {path} line {lineno}: {e.msg}", e.doc, e.pos)
    return chunks

# --------------------------
# Checkpoint helpers
# --------------------------
def _load_checkpoint(checkpoint_path: str) -> Set[str]:
    if not checkpoint_path:
        return set()
    p = pathlib.Path(checkpoint_path)
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return set(data)
            if isinstance(data, dict) and "processed_ids" in data:
                return set(data.get("processed_ids") or [])
    except Exception:
        logger.warning("Failed to read checkpoint file %s — starting fresh", checkpoint_path)
    return set()


def _save_checkpoint(checkpoint_path: str, processed_ids: Set[str]) -> None:
    if not checkpoint_path:
        return
    tmp = str(pathlib.Path(checkpoint_path).with_suffix(".tmp"))
    payload = {"processed_ids": sorted(list(processed_ids))}
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, checkpoint_path)


# --------------------------
# Normalize
# --------------------------
def _normalize_vectors(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# --------------------------
# Core embed function (re-usable)
# --------------------------
def embed_and_save(
    chunks_path: str,
    out_path: str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    resume: bool = True,
    normalize: bool = True,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Create embeddings from a chunks file (JSONL or array) and write JSONL vectors.
    Output lines: {"id":..., "embedding":[...], "meta":{...}}
    """
    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning("No chunks found in %s", chunks_path)
        return

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)
    out_p = pathlib.Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if resume and not checkpoint_path:
        checkpoint_path = str(out_p.with_suffix(out_p.suffix + ".checkpoint.json"))

    processed_ids = _load_checkpoint(checkpoint_path) if resume else set()
    logger.info("Already processed IDs (from checkpoint): %d", len(processed_ids))

    model = SentenceTransformer(model_name)
    logger.info("Loaded model: %s", model_name)

    mode = "a" if resume and out_p.exists() else "w"
    written = 0

    with open(out_path, mode, encoding="utf-8") as out_fh:
        if mode == "w":
            out_fh.truncate(0)

        items = []
        for c in chunks:
            cid = c.get("id") or c.get("chunk_id") or None
            text = c.get("text") or c.get("content") or ""
            meta = c.get("meta") or c.get("metadata") or {}
            if cid is None:
                cid = f"chunk_{hash(text) & 0xFFFFFFFF:08x}"
            if cid in processed_ids:
                continue
            items.append({"id": str(cid), "text": text, "meta": meta})

        total = len(items)
        logger.info("Embedding %d new chunks (batch size=%d).", total, batch_size)

        for i in tqdm(range(0, total, batch_size), desc="Embedding", unit="batch"):
            batch = items[i : i + batch_size]
            texts = [b["text"] for b in batch]
            ids = [b["id"] for b in batch]
            metas = [b["meta"] for b in batch]

            try:
                vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            except Exception as e:
                logger.exception("Model failed to encode batch starting at %d: %s", i, e)
                for bid, meta in zip(ids, metas):
                    out_obj = {"id": bid, "embedding": None, "meta": meta}
                    out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    processed_ids.add(bid)
                _save_checkpoint(checkpoint_path, processed_ids)
                continue

            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            if normalize:
                try:
                    vectors = _normalize_vectors(vectors)
                except Exception:
                    logger.exception("Normalization failed; continuing without normalization")

            vectors = vectors.astype(float).tolist()

            for bid, vec, meta in zip(ids, vectors, metas):
                out_obj = {"id": bid, "embedding": vec, "meta": meta}
                out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                processed_ids.add(bid)
                written += 1

            out_fh.flush()
            _save_checkpoint(checkpoint_path, processed_ids)

    logger.info("Finished embeddings. Wrote %d vectors to %s", written, out_path)


# --------------------------
# Orchestration: fetch -> merge -> chunk -> embed
# --------------------------
def create_embeddings_for_game(
    game_name: Optional[str] = None,
    merged_path: Optional[str] = None,
    outdir: str = ".",
    chunks_filename: Optional[str] = None,
    vectors_filename: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = DEFAULT_BATCH,
    chunk_tokens: int = 800,
    overlap_tokens: int = 200,
    resume: bool = True,
) -> Dict[str, str]:
    """
    Orchestrate the pipeline:
      - if merged_path is provided, use it; otherwise fetch+merge live using ingest.merge.merge_and_save
      - chunk using ingest.chunking.build_chunks_from_merged and save JSONL
      - embed using embed_and_save

    Returns dict with paths: {"merged":..., "chunks":..., "vectors":...}
    """
    outdir = str(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Try to import project helpers (loader/merge/chunking). If unavailable,
    # we raise informative error.
    try:
        from ingest import merge as ingest_merge  # merge.merge_and_save, merge_three_sources
        from ingest import chunking as ingest_chunking  # build_chunks_from_merged, save helpers
    except Exception as e:
        raise RuntimeError("Failed to import ingest.merge or ingest.chunking. Run from project root so 'ingest' is importable.") from e

    # 1) obtain merged object
    if merged_path:
        merged_path = str(merged_path)
        if not pathlib.Path(merged_path).exists():
            raise FileNotFoundError(f"Provided merged file not found: {merged_path}")
        with open(merged_path, "r", encoding="utf-8") as fh:
            merged_obj = json.load(fh)
        saved_merged = merged_path
    else:
        if not game_name:
            raise ValueError("Either game_name or merged_path must be provided.")
        # merge_and_save will fetch and produce a merged json in outdir
        saved_merged = ingest_merge.merge_and_save(game_name, outdir=outdir, validate=True)
        with open(saved_merged, "r", encoding="utf-8") as fh:
            merged_obj = json.load(fh)

    # 2) chunk
    base = (merged_obj.get("title") or game_name or "merged").strip().lower().replace(" ", "_")
    chunks_filename = chunks_filename or os.path.join(outdir, f"{base}_chunks.jsonl")
    # build chunks (token-splitting uses tiktoken internally in chunking.py)
    chunks = ingest_chunking.build_chunks_from_merged(
        merged_obj,
        chunk_tokens=chunk_tokens,
        overlap_tokens=overlap_tokens,
        model_encoding="cl100k_base",
        namespace=None,
    )
    # Save chunks as JSONL
    ingest_chunking.save_chunks_jsonl(chunks, chunks_filename)

    # 3) embeddings
    vectors_filename = vectors_filename or os.path.join(outdir, f"{base}_vectors.jsonl")
    embed_and_save(
        chunks_path=chunks_filename,
        out_path=vectors_filename,
        model_name=model_name,
        batch_size=batch_size,
        resume=resume,
        normalize=True,
        checkpoint_path=None,
    )

    return {"merged": saved_merged, "chunks": chunks_filename, "vectors": vectors_filename}


# --------------------------
# CLI
# --------------------------
def _cli():
    p = argparse.ArgumentParser(prog="ingest.embeddings", description="Create embeddings (optionally fetch+merge+chunk first)")
    p.add_argument("--game", "-g", help="Game name to fetch & process (used when --merged is omitted)")
    p.add_argument("--merged", "-m", help="Path to a pre-existing merged JSON (skip fetch/merge)")
    p.add_argument("--outdir", "-o", default=".", help="Output directory")
    p.add_argument("--chunks", help="Explicit chunks JSONL path to save/use (default: <title>_chunks.jsonl in outdir)")
    p.add_argument("--vectors", help="Explicit vectors JSONL path to save (default: <title>_vectors.jsonl in outdir)")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"SentenceTransformers model name (default: {DEFAULT_MODEL})")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--chunk-tokens", type=int, default=800, help="Chunk size in tokens (if tiktoken available)")
    p.add_argument("--overlap-tokens", type=int, default=200, help="Overlap size in tokens")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume from checkpoint; overwrite output")
    args = p.parse_args()

    try:
        res = create_embeddings_for_game(
            game_name=args.game,
            merged_path=args.merged,
            outdir=args.outdir,
            chunks_filename=args.chunks,
            vectors_filename=args.vectors,
            model_name=args.model,
            batch_size=args.batch,
            chunk_tokens=args.chunk_tokens,
            overlap_tokens=args.overlap_tokens,
            resume=args.resume,
        )
        print("[done] Produced files:")
        for k, v in res.items():
            print(f" - {k}: {v}")
    except Exception as e:
        logger.exception("Embedding pipeline failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    _cli()
