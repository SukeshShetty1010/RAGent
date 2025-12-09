# agent/tools/data_fetcher_tool.py
"""
agent/tools/data_fetcher_tool.py

Tool wrapper around the project's ingest pipeline that programmatically
orchestrates: fetch/merge -> chunk -> embed -> upsert.

Usage example:
    from agent.tools.data_fetcher_tool import DataFetcherTool
    t = DataFetcherTool(weaviate_url="http://localhost:8080")
    res = t.execute({"game_name": "Far Cry 5"})
"""

from __future__ import annotations

import logging
import os
import pathlib
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
        safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in game_name).strip().replace(" ", "_")
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

        Returns:
            A dict containing success status, game name, chunks_upserted, and a summary or error.
        """
        try:
            if not isinstance(args, dict):
                raise ValueError("args must be a dict")

            game_name = args.get("game_name") or args.get("game")
            if not game_name or not isinstance(game_name, str):
                raise ValueError("Missing required 'game_name' (string) in args")

            # pick out optional parameters
            explicit_outdir = args.get("outdir")
            outdir = str(explicit_outdir) if explicit_outdir else self._safe_outdir_for_game(game_name)
            os.makedirs(outdir, exist_ok=True)

            # optional embed/upsert params
            chunk_tokens = int(args.get("chunk_tokens", 800))
            overlap_tokens = int(args.get("overlap_tokens", 200))
            model_name = str(args.get("model_name", "all-MiniLM-L6-v2"))
            embed_batch_size = int(args.get("batch_size", 32))
            resume = bool(args.get("resume", True))
            vectors_filename = args.get("vectors_filename")  # optional override

            logger.info("Starting ingestion pipeline for game=%s outdir=%s", game_name, outdir)

            # -------------------------
            # Step 1: Fetch & Merge
            # -------------------------
            logger.info("Step 1: fetch & merge (merge_and_save)")
            # merge_and_save returns path to saved merged file
            merged_path = merge_and_save(game_name, outdir=outdir, validate=True)
            if not merged_path or not pathlib.Path(merged_path).exists():
                raise RuntimeError(f"merge_and_save did not produce merged file (game={game_name})")

            logger.info("Merged file saved at: %s", merged_path)

            # -------------------------
            # Step 2: Chunking
            # -------------------------
            logger.info("Step 2: chunking (build_chunks_from_merged)")
            # load merged object (build_chunks_from_merged expects merged obj; but also accepts the merged dict)
            # The ingest.chunking.build_chunks_from_merged function signature expects a merged_obj (dict)
            import json

            with open(merged_path, "r", encoding="utf-8") as fh:
                merged_obj = json.load(fh)

            chunks = ingest_chunking.build_chunks_from_merged(
                merged_obj,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                model_encoding="cl100k_base",
                namespace=None,
            )

            if not isinstance(chunks, list):
                raise RuntimeError("chunking.build_chunks_from_merged returned unexpected type")

            chunks_path = os.path.join(outdir, f"{(merged_obj.get('title') or game_name).strip().lower().replace(' ', '_')}_chunks.jsonl")
            ingest_chunking.save_chunks_jsonl(chunks, chunks_path)
            logger.info("Chunks saved to %s (count=%d)", chunks_path, len(chunks))

            # -------------------------
            # Step 3: Embed
            # -------------------------
            logger.info("Step 3: embedding (embed_and_save)")
            vectors_path = vectors_filename or os.path.join(outdir, f"{(merged_obj.get('title') or game_name).strip().lower().replace(' ', '_')}_vectors.jsonl")
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
                f"Ingestion completed for '{game_name}'. "
                f"Merged: {merged_path}. Chunks: {chunks_path} ({len(chunks)} chunks). "
                f"Vectors: {vectors_path}. Upsert processed={processed} success={success_count} failed={failed_count}."
            )

            logger.info(summary)

            return {
                "success": True,
                "game": game_name,
                "chunks_upserted": success_count,
                "processed": processed,
                "failed": failed_count,
                "summary": summary,
                "merged_path": merged_path,
                "chunks_path": chunks_path,
                "vectors_path": vectors_path,
            }

        except Exception as exc:
            logger.exception("DataFetcherTool failed: %s", exc)
            tb = traceback.format_exc()
            return {"success": False, "error": str(exc), "traceback": tb}
