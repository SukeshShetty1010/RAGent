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
import threading
import time
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


def _strip_leading_stopwords(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    """Drop leading _STOPWORDS tokens ("The", "It", ...). candidate_spans()
    never seeds a span on a stopword, so a real title that opens with one
    ("The Legend of Zelda...", "It Takes Two") produces a query span
    missing that word — the title-prefix check in assess_grounding must
    skip the same leading run or it can never anchor at position 0."""
    i = 0
    while i < len(tokens) and tokens[i] in _STOPWORDS:
        i += 1
    return tokens[i:]


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

    def candidate_spans(
        self, query: str, *, include_sentence_initial: bool = False
    ) -> List[Tuple[str, ...]]:
        """
        Maximal runs of capitalized tokens in `query`, normalized to
        lowercase token tuples. Digits and Roman numerals continue an
        in-progress span so "Far Cry 5" and "Grand Theft Auto VI" stay
        whole.

        By default the sentence-initial token never seeds a span. Many
        golden-set queries are interrogative ("What...", "Which...",
        "Can I...") where token 0 is a stopword anyway, but plenty of
        real queries are not ("Far Cry 5 combat", "Compare X and Y",
        "Tell me about...") — for those, skipping index 0 unconditionally
        either misses the entity entirely or, if index 0 is treated like
        any other token, turns non-entity leading words ("Tell", "What's")
        into spurious spans. `assess_grounding` resolves this by using
        two calls: this default (conservative) as the "does the query
        name any entity at all" verdict, and `include_sentence_initial=True`
        (greedy) as the "what could match a known title" search — so a
        stray leading word can only ever help a match, never manufacture
        a false refusal on its own.

        Pass include_sentence_initial=True to also let index 0 seed a
        span.
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
            if idx == 0 and not include_sentence_initial:
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
        True: some candidate span matches a known corpus title, or is a
              token-prefix of a retrieved chunk's source_title once that
              title's own leading stopwords are stripped (covers
              Game.title drifting from EditorialChunk.source_title, e.g.
              "Far Cry 5" vs "Far Cry 5 Review — GameSpot", and titles
              that open with a word candidate_spans() never seeds a span
              on, e.g. "It Takes Two" or "The Legend of Zelda...").
        False: the query names an entity the corpus does not anchor.

        The source_title fallback is a PREFIX test, not substring
        containment: span tokens must match the title's tokens starting
        at position 0 (after stripping the title's leading stopwords —
        see _strip_leading_stopwords). A raw substring test used to
        accept g047's search-adjacent evidence ("Beyond Good and Evil 2")
        as grounded via an unrelated corpus title ("Resident Evil 2")
        because its fragmented span ('evil', '2') is a substring of that
        title anywhere; prefix comparison rejects that (('resident',
        'evil') != ('evil', '2')) while still accepting the drift case
        above, since a real (post-strip) title always leads with the
        game name.
        """
        if self._load_failed or not self.known_titles:
            return None

        # Conservative verdict set: does the query name any entity at
        # all? (Sentence-initial token excluded — see candidate_spans.)
        verdict_spans = self.candidate_spans(query)
        if not verdict_spans:
            return None

        # Web results routinely echo the query in their page title
        # (search engines rank on term overlap), which would make this
        # fallback trivially confirm grounding for exactly the
        # web-rescued off-corpus queries the gate needs to catch after
        # a re-merge (orchestrator.py's post-web-merge re-gate). Only
        # local corpus chunks count for the title-drift guard.
        #
        # Tokenized per-chunk, not concatenated into one string: a
        # prefix test needs each title's own token boundary, and joining
        # titles together would let a span prefix-match across a
        # boundary that was never a real title.
        source_title_tokens: List[Tuple[str, ...]] = [
            _strip_leading_stopwords(_normalize(c.get("source_title") or ""))
            for c in evidence
            if c.get("source_type") != "web"
        ]

        # Greedy match set: includes the sentence-initial token, so a
        # bare title query ("Far Cry 5 combat") grounds even though its
        # entity opens the sentence. Checked ALONGSIDE verdict_spans,
        # not instead of it: including index 0 can merge a leading
        # non-entity word into what would otherwise be a clean span
        # ("Compare Far Cry 5 and..." -> "compare far cry 5" instead of
        # "far cry 5"), which breaks the match that verdict_spans (idx0
        # excluded) already gets right. Only used to look for a match,
        # never to decide there is none — verdict_spans already
        # established the query names *something*, so checking a wider
        # span set here can only turn a False into a True, never
        # manufacture a new False.
        match_spans = self.candidate_spans(query, include_sentence_initial=True)
        for span in (*verdict_spans, *match_spans):
            if span in self.known_titles:
                return True
            span_len = len(span)
            if any(title[:span_len] == span for title in source_title_tokens):
                return True

        return False


# ============================================================
# Lazy singleton accessor (same pattern as
# rag_retriever._get_dense_encoder / _get_reranker), with a TTL so
# titles ingested after process start eventually become groundable
# without a restart.
# ============================================================

_ENTITY_INDEX_TTL_ENV = "CORPUS_INDEX_TTL_SECONDS"
_DEFAULT_TTL_SECONDS = 900.0

_entity_index: Optional[CorpusEntityIndex] = None
_entity_index_loaded_at: float = 0.0
_entity_index_lock = threading.Lock()


def _ttl_seconds() -> float:
    """Read per call, not cached — same rationale as
    quality_gate._resolve_floors(): a test's monkeypatch.setenv must
    take effect without a module reload. <= 0 disables refresh, which
    an evaluation run wants (a mid-run rebuild would change the
    grounding verdict between two queries of the same run — see
    evaluation/calibrate_relevance.py, which holds one reference for
    the whole run)."""
    try:
        return float(os.environ.get(_ENTITY_INDEX_TTL_ENV, _DEFAULT_TTL_SECONDS))
    except ValueError:
        return _DEFAULT_TTL_SECONDS


def _get_entity_index() -> CorpusEntityIndex:
    global _entity_index, _entity_index_loaded_at

    current = _entity_index
    if current is None:
        with _entity_index_lock:
            if _entity_index is None:
                _entity_index = CorpusEntityIndex()
                _entity_index_loaded_at = time.monotonic()
            return _entity_index

    ttl = _ttl_seconds()
    if ttl <= 0 or (time.monotonic() - _entity_index_loaded_at) < ttl:
        return current

    # Stale. Exactly one thread rebuilds; the rest keep serving the
    # current index rather than queueing behind a Qdrant scroll.
    if not _entity_index_lock.acquire(blocking=False):
        return current
    try:
        # Bump BEFORE loading: if Qdrant is down, CorpusEntityIndex
        # swallows the error into an empty known_titles rather than
        # raising, and without this every request during the outage
        # would retry the failing scroll instead of waiting a full TTL.
        _entity_index_loaded_at = time.monotonic()
        refreshed = CorpusEntityIndex()
        if refreshed.known_titles:
            _entity_index = refreshed
        else:
            logger.warning(
                "CorpusEntityIndex refresh returned no titles — keeping previous index"
            )
    finally:
        _entity_index_lock.release()

    return _entity_index


def invalidate_entity_index() -> None:
    """Force a rebuild on the next access. A test seam and future-
    proofing hook, not a real ingest-invalidation path: the ingest CLIs
    (scripts/bulk_ingest.py, upsert/*) run in a separate process from
    the API and cannot reach this module's state — the TTL above is
    what actually covers them."""
    global _entity_index, _entity_index_loaded_at
    with _entity_index_lock:
        _entity_index = None
        _entity_index_loaded_at = 0.0
