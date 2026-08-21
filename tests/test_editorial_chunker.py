"""
tests/test_editorial_chunker.py

Unit tests for chunking.editorial_chunker. Fully local, no network calls.

These pin down what "chunk_words"/"overlap_words" actually mean: whitespace
word counts, not model tokens or characters. That distinction drifted
undetected for a while because no test existed for this module at all.
"""

import pytest

from chunking.editorial_chunker import EditorialChunker, WordSplitter

pytestmark = pytest.mark.unit


def _editorial(body: str, title: str = "Test Title") -> dict:
    return {
        "reviews": {"items": []},
        "articles": [{"title": title, "body": body}],
    }


def test_word_splitter_counts_whitespace_words():
    assert WordSplitter.split("one two three") == ["one", "two", "three"]
    assert WordSplitter.split("") == []
    assert WordSplitter.split("  spaced   out  ") == ["spaced", "out"]


def test_word_splitter_treats_long_word_as_one_unit():
    long_word = "a" * 5000
    result = WordSplitter.split(f"{long_word} short")
    assert len(result) == 2
    assert result[0] == long_word


def test_chunk_count_and_stride_at_small_window():
    words = [f"w{i}" for i in range(25)]
    body = " ".join(words)

    chunker = EditorialChunker(chunk_words=10, overlap_words=2)
    chunks = chunker._chunk_text(
        body=body,
        title=None,
        game_uuid="game-uuid",
        parent_uuid="parent-uuid",
        content_type="article",
    )

    # stride = chunk_words - overlap_words = 8; starts at 0, 8, 16, 24
    assert len(chunks) == 4
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]


def test_chunk_words_is_a_word_count_not_char_or_token_count():
    # 20 short words, each far under a "token" worth of characters
    words = ["ab"] * 20
    body = " ".join(words)

    chunker = EditorialChunker(chunk_words=5, overlap_words=0)
    chunks = chunker._chunk_text(
        body=body,
        title=None,
        game_uuid="game-uuid",
        parent_uuid="parent-uuid",
        content_type="article",
    )

    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk["content"].split()) == 5


def test_overlap_words_must_be_smaller_than_chunk_words():
    with pytest.raises(AssertionError):
        EditorialChunker(chunk_words=10, overlap_words=10)

    with pytest.raises(AssertionError):
        EditorialChunker(chunk_words=10, overlap_words=15)


def test_production_configuration_constructs_and_chunks():
    chunker = EditorialChunker(chunk_words=300, overlap_words=50)
    assert chunker.chunk_words == 300
    assert chunker.overlap_words == 50

    words = [f"word{i}" for i in range(700)]
    body = " ".join(words)

    chunks = chunker.process_game_editorial(
        editorial_object=_editorial(body),
        game_uuid="game-uuid",
        parent_uuid="parent-uuid",
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["game_uuid"] == "game-uuid"
        assert chunk["parent_editorial_uuid"] == "parent-uuid"
        assert chunk["chunk_id"]
