"""
tests/test_corpus_index_ttl.py

Hermetic tests for retriever/corpus_index.py's module-level entity
index singleton: TTL-based refresh, single-flight rebuild, fail-soft
keep-previous-on-empty-refresh, and the invalidate_entity_index() seam
(AUDIT_TASKS §9). CorpusEntityIndex._load is monkeypatched so no Qdrant
or network access is needed.
"""

import time

import pytest

from retriever import corpus_index
from retriever.corpus_index import CorpusEntityIndex, invalidate_entity_index

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def reset_singleton():
    """Module-level state must not leak between tests."""
    invalidate_entity_index()
    yield
    invalidate_entity_index()


def _patch_load(monkeypatch, title_batches):
    """Make CorpusEntityIndex._load pop one title list per call and
    populate known_titles from it, mimicking a real Qdrant scroll
    result without any network access."""
    batches = list(title_batches)

    def fake_load(self):
        titles = batches.pop(0) if batches else []
        self.known_titles = {corpus_index._normalize(t) for t in titles}

    monkeypatch.setattr(CorpusEntityIndex, "_load", fake_load)


def test_index_is_cached_within_ttl(monkeypatch):
    _patch_load(monkeypatch, [["Far Cry 5"], ["Doom Eternal"]])
    monkeypatch.setenv("CORPUS_INDEX_TTL_SECONDS", "900")

    first = corpus_index._get_entity_index()
    second = corpus_index._get_entity_index()

    assert first is second
    assert first.known_titles == {("far", "cry", "5")}


def test_expired_ttl_triggers_rebuild(monkeypatch):
    _patch_load(monkeypatch, [["Far Cry 5"], ["Doom Eternal"]])
    monkeypatch.setenv("CORPUS_INDEX_TTL_SECONDS", "0.05")

    first = corpus_index._get_entity_index()
    assert first.known_titles == {("far", "cry", "5")}

    time.sleep(0.1)
    second = corpus_index._get_entity_index()

    assert second is not first
    assert second.known_titles == {("doom", "eternal")}


def test_failed_refresh_keeps_previous_index(monkeypatch):
    """A refresh that yields no titles (the fail-soft outcome of a
    Qdrant outage — CorpusEntityIndex swallows load errors into an
    empty set) must not replace a good index with an empty one, since
    an empty index makes every query un-groundable via assess_grounding's
    `not self.known_titles` short-circuit."""
    _patch_load(monkeypatch, [["Far Cry 5"], []])
    monkeypatch.setenv("CORPUS_INDEX_TTL_SECONDS", "0.05")

    first = corpus_index._get_entity_index()
    assert first.known_titles == {("far", "cry", "5")}

    time.sleep(0.1)
    second = corpus_index._get_entity_index()

    assert second is first
    assert second.known_titles == {("far", "cry", "5")}


def test_ttl_disabled_never_refreshes(monkeypatch):
    _patch_load(monkeypatch, [["Far Cry 5"], ["Doom Eternal"]])
    monkeypatch.setenv("CORPUS_INDEX_TTL_SECONDS", "0")

    first = corpus_index._get_entity_index()
    time.sleep(0.05)
    second = corpus_index._get_entity_index()

    assert second is first
    assert second.known_titles == {("far", "cry", "5")}


def test_invalidate_forces_rebuild_even_with_ttl_disabled(monkeypatch):
    _patch_load(monkeypatch, [["Far Cry 5"], ["Doom Eternal"]])
    monkeypatch.setenv("CORPUS_INDEX_TTL_SECONDS", "0")

    first = corpus_index._get_entity_index()
    assert first.known_titles == {("far", "cry", "5")}

    invalidate_entity_index()
    second = corpus_index._get_entity_index()

    assert second is not first
    assert second.known_titles == {("doom", "eternal")}
