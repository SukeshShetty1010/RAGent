from __future__ import annotations

import hashlib
import html
import re
from typing import Dict, List
from uuid import UUID, uuid5

# Namespace UUID for deterministic chunk IDs
CHUNK_NAMESPACE = UUID("12345678-1234-5678-1234-567812345678")


class LocalTokenizer:
    """
    Robust local tokenizer.

    - Splits on ANY whitespace (spaces, tabs, newlines, unicode)
    """

    @staticmethod
    def encode(text: str) -> List[str]:
        if not text:
            return []
        return re.split(r"\s+", text.strip())


class EditorialChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        assert overlap < chunk_size, "overlap must be smaller than chunk_size"
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = LocalTokenizer()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def process_game_editorial(
        self,
        editorial_object: Dict,
        game_uuid: str,
        parent_uuid: str,
        source: str = "gamespot",
    ) -> List[Dict]:
        chunks: List[Dict] = []

        reviews = editorial_object.get("reviews", {}).get("items", []) or []
        for review in reviews:
            chunks.extend(
                self._chunk_text(
                    body=review.get("body"),
                    title=review.get("title"),
                    game_uuid=game_uuid,
                    parent_uuid=parent_uuid,
                    content_type="review",
                    source=source,
                )
            )

        articles = editorial_object.get("articles", []) or []
        for article in articles:
            chunks.extend(
                self._chunk_text(
                    body=article.get("body"),
                    title=article.get("title"),
                    game_uuid=game_uuid,
                    parent_uuid=parent_uuid,
                    content_type="article",
                    source=source,
                )
            )

        return chunks

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _chunk_text(
        self,
        body: str,
        title: str,
        game_uuid: str,
        parent_uuid: str,
        content_type: str,
        source: str = "gamespot",
    ) -> List[Dict]:
        if not body or not isinstance(body, str):
            return []

        body = self._normalize_text(body)
        tokens = self.tokenizer.encode(body)

        if not tokens:
            return []

        chunks: List[Dict] = []
        start = 0
        index = 0
        total_tokens = len(tokens)

        while start < total_tokens:
            end = min(start + self.chunk_size, total_tokens)
            window_tokens = tokens[start:end]

            # SAFETY: pathological single-token guard
            if len(window_tokens) == 1 and len(window_tokens[0]) > 2000:
                window_tokens = [window_tokens[0][:2000]]

            content = " ".join(window_tokens)

            if title:
                content = f"{title}\n\n{content}"

            chunk_id = self._deterministic_uuid(content)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "content": content,
                    "game_uuid": game_uuid,
                    "parent_editorial_uuid": parent_uuid,
                    "source": source,
                    "content_type": content_type,
                    "chunk_index": index,
                    "source_title": title,
                }
            )

            index += 1
            start += self.chunk_size - self.overlap

        return chunks

    # --------------------------------------------------
    # NORMALIZATION (FIXED)
    # --------------------------------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Aggressive HTML-safe normalization.

        Steps:
        1. HTML entity unescape
        2. Strip ALL HTML tags
        3. Collapse whitespace
        """

        # 1. Decode HTML entities (&nbsp;, &gt;, etc.)
        text = html.unescape(text)

        # 2. Strip all HTML tags safely
        text = re.sub(r"<[^>]+>", " ", text)

        # 3. Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{2,}", "\n", text)

        return text.strip()

    @staticmethod
    def _deterministic_uuid(text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        return str(uuid5(CHUNK_NAMESPACE, digest))
