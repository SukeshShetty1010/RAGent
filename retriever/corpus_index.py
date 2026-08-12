# ============================================================
# retriever/corpus_index.py
# Corpus entity grounding — "does the corpus actually anchor
# the game this query is asking about?"
# ============================================================
"""
retriever/quality_gate.py needs a second signal beyond relevance: a
query can retrieve chunks that score well against the *wrong* game
(e.g. "Grand Theft Auto VI" pulling GTA V plot chunks) because the
embedding space clusters same-franchise content tightly. Comparing the
capitalized entity span in the query against the corpus's actual
`Game` anchors (vector/create_schema.py's `Game` collection,
payload contract from upsert/upsert_canonical_game.py:93-100) catches
that case independent of relevance score.

Token-tuple equality, not substring matching: "grand theft auto v" is
a raw substring of "grand theft auto vi", so naive `in` checks would
silently pass the exact query this module exists to catch.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from qdrant_client import QdrantClient

logger = logging.getLogger("RAG_CORPUS_INDEX")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

_STOPWORDS: Set[str] = {
    "the", "a", "an", "what", "which", "who", "whose", "where", "when",
    "why", "how", "is", "are", "was", "were", "can", "could", "should",
    "would", "will", "does", "do", "did", "i", "in", "on", "for", "of",
    "to", "and", "or", "about", "me", "you", "it", "me", "playing",
    "play",
}

_ROMAN_NUMERALS: Set[str] = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi",
    "xii", "xiii",
}

# Lowercase words that can bridge a span mid-title ("Star Rail - Then
# Wake to Weep", "Legend of Zelda") without breaking it. Deliberately
# excludes "and"/"or"/"vs"/"versus" — those separate two *different*
# entities in comparison queries ("Doom Eternal and Crusader Kings
# III") and must keep splitting the span, not bridge it.
_CONNECTORS: Set[str] = {"of", "to", "the", "in", "on", "for", "at"}


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text or "")


def _normalize(text: str) -> Tuple[str, ...]:
    return tuple(t.lower() for t in _tokenize(text))


class CorpusEntityIndex:
    """
    Fail-soft, lazily-loaded set of game titles the corpus anchors.

    Any load failure leaves `known_titles` empty and `assess_grounding`
    always returns None ("unknown") — a broken index must never cause a
    refusal, it must simply disable the entity signal.
    """

    def __init__(self) -> None:
        self.known_titles: Set[Tuple[str, ...]] = set()
        self._load_failed = False

        try:
            self._load()
        except Exception as exc:
            logger.warning(f"CorpusEntityIndex load failed (fail-soft): {exc}")
            self._load_failed = True
            self.known_titles = set()

    # --------------------------------------------------------
    # Test / offline construction
    # --------------------------------------------------------

    @classmethod
    def from_titles(cls, titles: Iterable[str]) -> "CorpusEntityIndex":
        """
        Build an index directly from known titles, skipping the Qdrant
        scroll. Used by hermetic unit tests (see tests/test_corpus_index.py,
        tests/test_quality_gate.py) so the gate's entity signal can be
        exercised with no network access.
        """
        instance = cls.__new__(cls)
        instance._load_failed = False
        instance.known_titles = {_normalize(t) for t in titles if t}
        return instance

    # --------------------------------------------------------
    # Loading
    # --------------------------------------------------------

    def _load(self) -> None:
        from dotenv import load_dotenv
        load_dotenv()

        url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        api_key = os.environ.get("QDRANT_API_KEY", "")
        client = QdrantClient(url=url, api_key=api_key or None)

        try:
            titles: Set[Tuple[str, ...]] = set()
            offset = None

            while True:
                points, offset = client.scroll(
                    collection_name="Game",
                    with_payload=["title"],
                    limit=200,
                    offset=offset,
                )

                for point in points:
                    title = (point.payload or {}).get("title")
                    if title:
                        titles.add(_normalize(title))

                if offset is None:
                    break

            self.known_titles = titles
            logger.info(f"CorpusEntityIndex loaded {len(titles)} game titles")
        finally:
            client.close()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def candidate_spans(self, query: str) -> List[Tuple[str, ...]]:
        """
        Maximal runs of capitalized tokens in `query`, normalized to
        lowercase token tuples. The sentence-initial token is never
        considered (golden-set queries are always interrogative:
        "What...", "Which...", "Can I...") so it never seeds a span.
        Digits and Roman numerals continue an in-progress span so
        "Far Cry 5" and "Grand Theft Auto VI" stay whole.
        """
        tokens = _tokenize(query)
        spans: List[Tuple[str, ...]] = []
        current: List[str] = []

        def flush() -> None:
            if current:
                normalized = tuple(t.lower() for t in current)
                if not all(t in _STOPWORDS for t in normalized):
                    spans.append(normalized)
            current.clear()

        for idx, tok in enumerate(tokens):
            if idx == 0:
                continue

            lower = tok.lower()
            is_capitalized = tok[:1].isupper() and lower not in _STOPWORDS
            is_numeric_continuation = bool(current) and (
                tok.isdigit() or (tok.isupper() and lower in _ROMAN_NUMERALS)
            )
            # A lowercase connector only continues the span if a real
            # entity word follows it — otherwise it's just a stray
            # preposition after the span already ended.
            is_connector_bridge = (
                bool(current)
                and lower in _CONNECTORS
                and idx + 1 < len(tokens)
                and tokens[idx + 1][:1].isupper()
                and tokens[idx + 1].lower() not in _STOPWORDS
            )

            if is_capitalized or is_numeric_continuation or is_connector_bridge:
                current.append(tok)
            else:
                flush()

        flush()
        return spans

    def assess_grounding(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
    ) -> Optional[bool]:
        """
        None: the query names no candidate entity, or the index failed
              to load — the relevance floor is the only signal.
        True: some candidate span matches a known corpus title, or
              appears in a retrieved chunk's source_title (covers
              Game.title drifting from EditorialChunk.source_title).
        False: the query names an entity the corpus does not anchor.
        """
        if self._load_failed or not self.known_titles:
            return None

        spans = self.candidate_spans(query)
        if not spans:
            return None

        # Web results routinely echo the query in their page title
        # (search engines rank on term overlap), which would make this
        # fallback trivially confirm grounding for exactly the
        # web-rescued off-corpus queries the gate needs to catch after
        # a re-merge (orchestrator.py's post-web-merge re-gate). Only
        # local corpus chunks count for the title-drift guard.
        source_titles = " ".join(
            (c.get("source_title") or "").lower()
            for c in evidence
            if c.get("source_type") != "web"
        )

        for span in spans:
            if span in self.known_titles:
                return True
            span_text = " ".join(span)
            if span_text and span_text in source_titles:
                return True

        return False


# ============================================================
# Lazy singleton accessor (same pattern as
# rag_retriever._get_dense_encoder / _get_reranker)
# ============================================================

_entity_index: Optional[CorpusEntityIndex] = None


def _get_entity_index() -> CorpusEntityIndex:
    global _entity_index
    if _entity_index is None:
        _entity_index = CorpusEntityIndex()
    return _entity_index
