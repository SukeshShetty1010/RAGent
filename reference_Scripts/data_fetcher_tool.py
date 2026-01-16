"""
agent/tools/data_fetcher_tool.py

Tool wrapper around the project's ingest pipeline that programmatically
orchestrates: fetch/merge -> chunk -> embed -> upsert.

This updated variant returns a canonical/resolved game name when available
(the RAWG-corrected / merged title) and, when possible, will relocate
intermediate artifacts into a canonical outdir so downstream consumers
(agents, logs, filenames) can rely on a stable, normalized title.

See: original file uploaded by user for context. :contentReference[oaicite:0]{index=0}
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import traceback
from typing import Any, Dict, List, Optional

from agent.base import Tool

# Programmatic imports from the ingest pipeline (no subprocess)
from ingest.merge import merge_and_save
from ingest import chunking as ingest_chunking
from ingest.embeddings import embed_and_save
from ingest.upsert import upsert_vectors

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class DataFetcherTool(Tool):
    """
    Tool that runs the project's ingestion pipeline programmatically:
      1) merge_and_save (fetch + merge)
      2) build_chunks_from_merged + save_chunks_jsonl
      3) embed_and_save
      4) upsert_vectors

    The tool stores intermediate artifacts under a base_outdir (default "temp_data/").

    Returns a dict with keys including (when available):
      - success: bool
      - game: original game_name argument
      - canonical_name: name discovered in merged metadata (preferred for downstream use)
      - chunks_upserted / processed / failed
      - merged_path / chunks_path / vectors_path
      - summary, notes
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        base_outdir: str = "temp_data",
        class_name: str = "GameChunk",
    ) -> None:
        """
        Initialize the DataFetcherTool.

        Args:
            weaviate_url: URL to Weaviate instance for upsert.
            base_outdir: Directory to store intermediate files (merged, chunks, vectors).
            class_name: Weaviate class name to upsert into.
        """
        super().__init__(
            name="data_fetcher",
            description="Fetch, process, embed and upsert game data into Weaviate.",
        )
        self.weaviate_url = weaviate_url
        self.base_outdir = str(base_outdir)
        self.class_name = class_name
        os.makedirs(self.base_outdir, exist_ok=True)

    def _safe_outdir_for_game(self, game_name: str) -> str:
        # create a safe subdirectory for outputs per game
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in (game_name or "")).strip().replace(" ", "_")
        outdir = os.path.join(self.base_outdir, safe or "game")
        os.makedirs(outdir, exist_ok=True)
        return outdir

    def execute(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the full ingest pipeline for a given game.

        Expected args:
            - game_name: str (required)
            - outdir: Optional[str] (overrides base_outdir if provided)
            - chunk_tokens / overlap_tokens / model_name / batch_size / resume: optional args forwarded to embed_and_save
            - min_char_length: Optional[int] -- if provided, chunks shorter than this will be removed BEFORE embedding/upsert

        Returns:
            A dict containing success status, game name, canonical_name (if found),
            chunks_upserted, and a summary or error.
        """
        try:
            if not isinstance(args, dict):
                raise ValueError("args must be a dict")

            game_name = args.get("game_name") or args.get("game")
            if not game_name or not isinstance(game_name, str):
                raise ValueError("Missing required 'game_name' (string) in args")

            # pick out optional parameters
            explicit_outdir = args.get("outdir")
            # If caller provided explicit outdir, honor it; otherwise create safe outdir now
            initial_outdir = str(explicit_outdir) if explicit_outdir else self._safe_outdir_for_game(game_name)
            outdir = initial_outdir
            os.makedirs(outdir, exist_ok=True)

            # optional embed/upsert params
            chunk_tokens = int(args.get("chunk_tokens", 800))
            overlap_tokens = int(args.get("overlap_tokens", 200))
            model_name = str(args.get("model_name", "all-MiniLM-L6-v2"))
            embed_batch_size = int(args.get("batch_size", 32))
            resume = bool(args.get("resume", True))
            vectors_filename = args.get("vectors_filename")  # optional override

            # optional: drop very short chunks before embedding/upsert
            min_char_length = args.get("min_char_length", None)
            if min_char_length is not None:
                min_char_length = int(min_char_length)

            logger.info("Starting ingestion pipeline for game=%s outdir=%s", game_name, outdir)

            # -------------------------
            # Step 1: Fetch & Merge
            # -------------------------
            logger.info("Step 1: fetch & merge (merge_and_save)")
            # merge_and_save returns path to saved merged file (and merged file contains canonical metadata)
            merged_path = merge_and_save(game_name, outdir=outdir, validate=True)
            if not merged_path or not pathlib.Path(merged_path).exists():
                raise RuntimeError(f"merge_and_save did not produce merged file (game={game_name})")

            logger.info("Merged file saved at: %s", merged_path)

            # Read merged object to detect canonical/resolved name (RAWG-corrected)
            import json

            with open(merged_path, "r", encoding="utf-8") as fh:
                merged_obj = json.load(fh)

            # Determine canonical name from merged object if possible
            canonical_name = None
            # common fields that might contain canonical/resolved titles
            for key in ("canonical_name", "resolved_name", "title", "rawg_name", "name"):
                val = merged_obj.get(key)
                if val and isinstance(val, str) and val.strip():
                    canonical_name = val.strip()
                    break

            notes: List[str] = []
            if canonical_name and canonical_name != game_name:
                notes.append(f"RAWG-corrected name: '{canonical_name}' (original request: '{game_name}')")
                logger.info("Detected canonical name from merged metadata: %s", canonical_name)
                # If caller did not provide an explicit outdir, relocate artifacts into canonical-named outdir
                if not explicit_outdir:
                    canonical_outdir = self._safe_outdir_for_game(canonical_name)
                    if canonical_outdir != outdir:
                        logger.info("Relocating artifacts into canonical outdir: %s", canonical_outdir)
                        os.makedirs(canonical_outdir, exist_ok=True)
                        # Move merged file into canonical_outdir
                        try:
                            new_merged_path = os.path.join(canonical_outdir, os.path.basename(merged_path))
                            shutil.move(merged_path, new_merged_path)
                            merged_path = new_merged_path
                            outdir = canonical_outdir
                            notes.append(f"Relocated merged file into canonical outdir: {canonical_outdir}")
                        except Exception as ex_move:
                            # If move fails, log and continue with original outdir (non-fatal)
                            logger.exception("Failed to relocate merged file to canonical outdir: %s", ex_move)
                            notes.append(f"Failed to relocate merged file to canonical outdir: {ex_move}")

            else:
                if canonical_name:
                    logger.info("Canonical name equals requested name or not useful: %s", canonical_name)
                else:
                    logger.info("No canonical name discovered in merged metadata; continuing with original game_name.")
                    notes.append("No canonical name discovered in merged metadata")

            # -------------------------
            # Step 2: Chunking
            # -------------------------
            logger.info("Step 2: chunking (build_chunks_from_merged)")
            chunks = ingest_chunking.build_chunks_from_merged(
                merged_obj,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                model_encoding="cl100k_base",
                namespace=None,
            )

            if not isinstance(chunks, list):
                raise RuntimeError("chunking.build_chunks_from_merged returned unexpected type")

            original_chunk_count = len(chunks)

            # Optional filtering of short chunks before saving / embedding
            filtered_out_count = 0
            if min_char_length is not None:
                kept_chunks = []
                for c in chunks:
                    clen = c.get("char_length") or len((c.get("text") or ""))
                    if clen >= min_char_length:
                        kept_chunks.append(c)
                    else:
                        filtered_out_count += 1
                chunks = kept_chunks
                logger.info("Filtered %d chunks shorter than min_char_length=%d; remaining=%d", filtered_out_count, min_char_length, len(chunks))

            # Name chunk file using canonical title if available, else merged title or original game_name
            chunk_title = (merged_obj.get("title") or canonical_name or game_name).strip()
            safe_chunk_title = chunk_title.lower().replace(" ", "_")
            chunks_path = os.path.join(outdir, f"{safe_chunk_title}_chunks.jsonl")
            ingest_chunking.save_chunks_jsonl(chunks, chunks_path)
            logger.info("Chunks saved to %s (count=%d, original=%d)", chunks_path, len(chunks), original_chunk_count)

            # -------------------------
            # Step 3: Embed
            # -------------------------
            logger.info("Step 3: embedding (embed_and_save)")
            vectors_path = vectors_filename or os.path.join(outdir, f"{safe_chunk_title}_vectors.jsonl")
            embed_and_save(
                chunks_path=chunks_path,
                out_path=vectors_path,
                model_name=model_name,
                batch_size=embed_batch_size,
                resume=resume,
                normalize=True,
                checkpoint_path=None,
            )
            logger.info("Vectors written to %s", vectors_path)

            # -------------------------
            # Step 4: Upsert
            # -------------------------
            logger.info("Step 4: upsert (upsert_vectors)")
            upsert_result = upsert_vectors(
                vectors_path=vectors_path,
                weaviate_url=self.weaviate_url,
                class_name=self.class_name,
                batch_size=int(args.get("upsert_batch_size", 64)),
                dim=int(args.get("dim", 384)),
                timeout=int(args.get("timeout", 30)),
                dry_run=bool(args.get("dry_run", False)),
            )

            if not isinstance(upsert_result, dict):
                raise RuntimeError("upsert_vectors returned unexpected result")

            processed = int(upsert_result.get("processed", 0))
            success_count = int(upsert_result.get("success", 0))
            failed_count = int(upsert_result.get("failed", 0))

            summary = (
                f"Ingestion completed for requested='{game_name}' canonical='{canonical_name or game_name}'. "
                f"Merged: {merged_path}. Chunks: {chunks_path} ({len(chunks)} chunks, original={original_chunk_count}). "
                f"Vectors: {vectors_path}. Upsert processed={processed} success={success_count} failed={failed_count}."
            )

            logger.info(summary)

            # Build result payload including canonical name if present
            result_payload: Dict[str, Any] = {
                "success": True,
                "game": game_name,
                "canonical_name": canonical_name,
                "chunks_upserted": success_count,
                "processed": processed,
                "failed": failed_count,
                "summary": summary,
                "merged_path": merged_path,
                "chunks_path": chunks_path,
                "vectors_path": vectors_path,
                "original_chunk_count": original_chunk_count,
                "filtered_out_count": filtered_out_count,
                "notes": notes,
                "merged_obj": merged_obj,  # include merged metadata for downstream inspection if needed
            }

            # Optionally provide upsert result meta
            result_payload["upsert_result"] = upsert_result

            return result_payload

        except Exception as exc:
            logger.exception("DataFetcherTool failed: %s", exc)
            tb = traceback.format_exc()
            return {"success": False, "error": str(exc), "traceback": tb}
