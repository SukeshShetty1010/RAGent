from __future__ import annotations

import hashlib
import html
import re
from typing import Dict, List
from uuid import UUID, uuid5

# Namespace UUID for deterministic chunk IDs
CHUNK_NAMESPACE = UUID("12345678-1234-5678-1234-567812345678")


class WordSplitter:
    """
    Robust whitespace word splitter.

    - Splits on ANY whitespace (spaces, tabs, newlines, unicode)
    - Returns words, not model tokens — there is no tokenizer here
    """

    @staticmethod
    def split(text: str) -> List[str]:
        if not text:
            return []
        return re.split(r"\s+", text.strip())


class EditorialChunker:
    def __init__(self, chunk_words: int = 500, overlap_words: int = 50):
        # The 500 default is not the production value — the only caller
        # (embed/prepare_editorial_payloads.py) passes chunk_words=300.
        assert overlap_words < chunk_words, "overlap_words must be smaller than chunk_words"
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.splitter = WordSplitter()

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
        words = self.splitter.split(body)

        if not words:
            return []

        chunks: List[Dict] = []
        start = 0
        index = 0
        total_words = len(words)

        while start < total_words:
            end = min(start + self.chunk_words, total_words)
            window_words = words[start:end]

            # SAFETY: pathological single-word guard
            if len(window_words) == 1 and len(window_words[0]) > 2000:
                window_words = [window_words[0][:2000]]

            content = " ".join(window_words)

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
            start += self.chunk_words - self.overlap_words

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
