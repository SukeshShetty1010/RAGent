"""
agent/tools/retriever_tool.py

Wrapper that adapts the Retriever class (retriever.retriever.Retriever) to the
agent Tool interface.

This updated wrapper respects an explicit 'similarity_threshold' key in the
args dict (including None) so callers (the agent) can request unthresholded
results by passing {"similarity_threshold": None}.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.base import Tool
from retriever.retriever import Retriever


class RetrieverTool(Tool):
    """
    Tool wrapper around Retriever.

    Description:
        Useful for retrieving specific information about games, companies,
        or entities from the vector database.
    """

    def __init__(
        self,
        weaviate_url: str = "http://localhost:8080",
        model_name: str = "all-MiniLM-L6-v2",
        class_name: str = "GameChunk",
        device: Optional[str] = None,
    ) -> None:
        """
        Instantiate the Retriever and set up tool metadata.

        Parameters
        ----------
        weaviate_url : str
            URL to the Weaviate instance (e.g. "http://localhost:8080").
        model_name : str
            SentenceTransformers model name to load (the Retriever will load it).
        class_name : str
            Weaviate class name to query.
        device : Optional[str]
            Optional device string passed to SentenceTransformer (e.g., "cpu" or "cuda").
        """
        super().__init__(
            name="retriever",
            description=(
                "Useful for retrieving specific information about games, companies, "
                "or entities from the vector database."
            ),
        )
        # instantiate and reuse the Retriever instance
        self.retriever = Retriever(
            weaviate_url=weaviate_url,
            class_name=class_name,
            model_name=model_name,
            device=device,
        )

    def execute(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute retrieval.

        Expected args keys (all optional except 'query'):
            - query: str (required)
            - k: int (optional, default 5)
            - min_char_length: int (optional)
            - similarity_threshold: float | None (optional). IMPORTANT:
                If the 'similarity_threshold' key is present and its value is None,
                the retriever will NOT apply the similarity gate and will return all
                hits that pass the minimum-char-length filter. If the key is omitted,
                the wrapper uses the default 0.6 (backwards-compatible).
            - unified_game_id: str (optional)
            - fetch_multiplier: int (optional)
            - debug: bool (optional)
            - show_meta: bool (optional)
            - use_hybrid: bool (optional)
            - hybrid_alpha: float (optional)

        Returns
        -------
        List[Dict[str, Any]]
            List of retrieved chunks (each chunk is a dict).
        """
        if not isinstance(args, dict):
            raise ValueError("args must be a dict")

        query = args.get("query")
        if not query or not isinstance(query, str):
            raise ValueError("Missing required 'query' in args (must be non-empty string)")

        # Extract optional parameters with sensible defaults (mirror Retriever defaults)
        k = int(args.get("k", 5))
        min_char_length = int(args.get("min_char_length", 50))

        # IMPORTANT: respect explicit presence of similarity_threshold in args.
        # If the key is present and set to None, we pass None -> no thresholding.
        if "similarity_threshold" in args:
            similarity_threshold = args.get("similarity_threshold")
            # keep None as-is; otherwise coerce to float
            if similarity_threshold is not None:
                similarity_threshold = float(similarity_threshold)
        else:
            # backwards-compatible default if not provided
            similarity_threshold = 0.6

        unified_game_id = args.get("unified_game_id")
        fetch_multiplier = int(args.get("fetch_multiplier", 2))
        debug = bool(args.get("debug", False))
        show_meta = bool(args.get("show_meta", False))
        use_hybrid = bool(args.get("use_hybrid", False))
        hybrid_alpha = float(args.get("hybrid_alpha", 0.5))

        # Delegate strictly to the Retriever instance — no duplication of logic.
        results = self.retriever.retrieve(
            query=query,
            k=k,
            min_char_length=min_char_length,
            similarity_threshold=similarity_threshold,
            unified_game_id=unified_game_id,
            fetch_multiplier=fetch_multiplier,
            debug=debug,
            show_meta=show_meta,
            use_hybrid=use_hybrid,
            hybrid_alpha=hybrid_alpha,
        )

        return results
