"""
ingest/embeddings.py

Create embeddings for chunk files produced by ingest.chunking.

Decisions (per user):
 - Model: sentence-transformers all-MiniLM-L6-v2 (HuggingFace)
 - Input: JSONL of chunks (each line: {"id","text","meta"}) or a JSON array file.
 - Output: JSONL with lines: {"id":..., "embedding":[...], "meta":{...}}
 - Batch size: 32
 - Normalize embeddings to unit length
 - Checkpointing: save a .checkpoint file listing processed ids (resumable)
 - CLI + importable function API

Usage (CLI):
 python -m ingest.embeddings --chunks ./chunks/far_cry_5_chunks.jsonl --out ./vectors/far_cry_5_vectors.jsonl

Functions exported:
 - embed_and_save(chunks_path, out_path, model_name, batch_size, resume, normalize, checkpoint_path)

"""
from __future__ import annotations
import argparse
import json
import logging
import math
import os
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Optional, Set

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# --------------------------
# Defaults (per your choices)
# --------------------------
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_BATCH = 32

# --------------------------
# Logging
# --------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("ingest.embeddings")


# --------------------------
# I/O helpers
# --------------------------

def _iter_chunks_from_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield chunk dicts from a JSONL file."""
    with open(path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skipping invalid JSON line %d in %s: %s", i + 1, path, e)
                continue
            yield obj


def _iter_chunks_from_json_array(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("JSON file does not contain an array of chunks")
        for obj in data:
            yield obj


def load_chunks(path: str) -> List[Dict[str, Any]]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Chunks file not found: {path}")

    # Try JSONL first (fast, streaming). Heuristics: if file has multiple lines and many lines start with '{', treat as JSONL.
    try:
        # small sniff: count lines and braces
        with p.open("r", encoding="utf-8") as fh:
            first_k = [next(fh) for _ in range(10)]
    except StopIteration:
        first_k = []

    is_jsonl = False
    if len(first_k) >= 2:
        # if multiple lines and each looks like a JSON object, assume JSONL
        if all(line.strip().startswith("{") for line in first_k if line.strip()):
            is_jsonl = True

    if is_jsonl:
        return list(_iter_chunks_from_jsonl(path))
    # otherwise assume an array JSON
    return list(_iter_chunks_from_json_array(path))


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
# Embedding + normalize
# --------------------------

def _normalize_vectors(arr: np.ndarray) -> np.ndarray:
    """L2-normalize a 2D numpy array along axis=1."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# --------------------------
# Main embedding function
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
    """Embed chunks from `chunks_path` and write JSONL embeddings to `out_path`.

    Each output line is a JSON object: {"id": <chunk id>, "embedding": [...], "meta": {...}}
    """
    chunks_path = str(chunks_path)
    out_path = str(out_path)

    chunks = load_chunks(chunks_path)
    if not chunks:
        logger.warning("No chunks found in %s", chunks_path)
        return

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # Ensure output dir exists
    out_p = pathlib.Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    # checkpoint
    if resume and not checkpoint_path:
        checkpoint_path = str(out_p.with_suffix(out_p.suffix + ".checkpoint.json"))

    processed_ids = _load_checkpoint(checkpoint_path) if resume else set()
    logger.info("Already processed IDs (from checkpoint): %d", len(processed_ids))

    model = SentenceTransformer(model_name)
    logger.info("Loaded model: %s", model_name)

    # Open output file in append mode if resuming and file exists
    mode = "a" if resume and out_p.exists() else "w"
    written = 0

    with open(out_path, mode, encoding="utf-8") as out_fh:
        # If not appending (new file) and mode == 'w', ensure truncation
        if mode == "w":
            out_fh.truncate(0)

        # Build list of (id, text, meta)
        items: List[Dict[str, Any]] = []
        for c in chunks:
            cid = c.get("id") or c.get("chunk_id") or None
            text = c.get("text") or c.get("content") or ""
            meta = c.get("meta") or c.get("metadata") or {}
            if cid is None:
                # try to construct an id — not ideal but better than dropping
                cid = f"chunk_{hash(text) & 0xFFFFFFFF:08x}"
            if cid in processed_ids:
                continue
            items.append({"id": str(cid), "text": text, "meta": meta})

        # Progress bar over batches
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
                # write empty placeholders and continue
                for bid, meta in zip(ids, metas):
                    out_obj = {"id": bid, "embedding": None, "meta": meta}
                    out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    processed_ids.add(bid)
                _save_checkpoint(checkpoint_path, processed_ids)
                continue

            # ensure 2D
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)

            if normalize:
                try:
                    vectors = _normalize_vectors(vectors)
                except Exception:
                    logger.exception("Normalization failed; continuing without normalization")

            # Convert to python lists
            vectors = vectors.astype(float).tolist()

            # Write each vector line
            for bid, vec, meta in zip(ids, vectors, metas):
                out_obj = {"id": bid, "embedding": vec, "meta": meta}
                out_fh.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                processed_ids.add(bid)
                written += 1

            # flush + checkpoint after each batch
            out_fh.flush()
            _save_checkpoint(checkpoint_path, processed_ids)

    logger.info("Finished embeddings. Wrote %d vectors to %s", written, out_path)


# --------------------------
# CLI
# --------------------------

def _cli():
    p = argparse.ArgumentParser(prog="ingest.embeddings", description="Create embeddings for chunk JSONL using sentence-transformers")
    p.add_argument("--chunks", "-c", required=True, help="Path to chunks JSONL or chunks JSON array")
    p.add_argument("--out", "-o", required=True, help="Output JSONL path for vectors")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"SentenceTransformers model name (default: {DEFAULT_MODEL})")
    p.add_argument("--batch", type=int, default=DEFAULT_BATCH, help=f"Batch size (default: {DEFAULT_BATCH})")
    p.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume from checkpoint; overwrite output")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="Do not L2-normalize embeddings")
    p.add_argument("--checkpoint", default=None, help="Explicit checkpoint path (optional). By default uses <out>.checkpoint.json")
    args = p.parse_args()

    try:
        embed_and_save(
            chunks_path=args.chunks,
            out_path=args.out,
            model_name=args.model,
            batch_size=args.batch,
            resume=args.resume,
            normalize=args.normalize,
            checkpoint_path=args.checkpoint,
        )
    except Exception as e:
        logger.exception("Embedding run failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    _cli()