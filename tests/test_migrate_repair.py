"""
tests/test_migrate_repair.py

Unit tests for scripts/migrate_embeddings_to_gemini.py's --repair mode:
the gap-split detector and the direction-verification step that decides
which cluster is already migrated. Fully hermetic — no Qdrant, no Gemini,
embed_fn is injected as a fake.
"""

import random

import pytest

from scripts.migrate_embeddings_to_gemini import (
    find_gap_split,
    identify_migrated_cluster,
)

pytestmark = pytest.mark.unit


def _random_unit_vector(rng: random.Random, dim: int, bias: list[float], spread: float) -> list[float]:
    """A unit vector clustered around `bias` (itself a unit vector)."""
    v = [b + rng.uniform(-spread, spread) for b in bias]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def _unit_vector(dim: int, hot_index: int) -> list[float]:
    v = [0.0] * dim
    v[hot_index] = 1.0
    return v


def test_gap_split_separates_two_orthogonal_populations():
    rng = random.Random(0)
    dim = 32
    bias_a = _unit_vector(dim, 0)
    bias_b = _unit_vector(dim, 1)

    cluster_a_vecs = [_random_unit_vector(rng, dim, bias_a, 0.02) for _ in range(20)]
    cluster_b_vecs = [_random_unit_vector(rng, dim, bias_b, 0.02) for _ in range(6)]

    vectors = cluster_a_vecs + cluster_b_vecs
    expected_a_idx = set(range(0, 20))
    expected_b_idx = set(range(20, 26))

    split = find_gap_split(vectors)
    assert split is not None

    found_a, found_b = split
    found_a, found_b = set(found_a), set(found_b)

    # The detector doesn't know which side is "a" vs "b" -- either
    # assignment that reproduces the true clusters is correct.
    assert {frozenset(found_a), frozenset(found_b)} == {
        frozenset(expected_a_idx),
        frozenset(expected_b_idx),
    }


def test_gap_split_returns_none_for_uniform_population():
    rng = random.Random(1)
    dim = 32
    bias = _unit_vector(dim, 0)

    # One tight population -- no wide gap should exist anywhere.
    vectors = [_random_unit_vector(rng, dim, bias, 0.05) for _ in range(50)]

    assert find_gap_split(vectors) is None


def test_gap_split_returns_none_below_min_gap_threshold():
    rng = random.Random(2)
    dim = 32
    bias_a = _unit_vector(dim, 0)
    bias_b = [(1.0 / dim ** 0.5)] * dim  # close-ish direction, not orthogonal

    # Two loosely separated blobs with only a small gap between them --
    # below the default min_gap, this must not be treated as a real split.
    cluster_a_vecs = [_random_unit_vector(rng, dim, bias_a, 0.3) for _ in range(10)]
    cluster_b_vecs = [_random_unit_vector(rng, dim, bias_b, 0.3) for _ in range(10)]
    vectors = cluster_a_vecs + cluster_b_vecs

    assert find_gap_split(vectors, min_gap=0.9) is None


def test_identify_migrated_cluster_handles_unmigrated_majority():
    """
    The common real-world case: unmigrated (old-space) points outnumber
    migrated ones. A majority-vote heuristic gets this backwards --
    direction verification via re-embedding must not.
    """
    migrated_idx = [0, 1]  # minority
    unmigrated_idx = [2, 3, 4, 5, 6]  # majority

    contents = {i: f"chunk-{i}" for i in migrated_idx + unmigrated_idx}
    # Stored vectors: migrated points store the "true" embedding already;
    # unmigrated points store something unrelated (simulating stale E5).
    stored_vectors = {}
    for i in migrated_idx:
        stored_vectors[i] = [1.0, 0.0, 0.0]
    for i in unmigrated_idx:
        stored_vectors[i] = [0.0, 1.0, 0.0]

    contents_list = [contents[i] for i in range(7)]
    vectors_list = [stored_vectors[i] for i in range(7)]

    def fake_embed_fn(texts: list[str]) -> list[list[float]]:
        # "Fresh" embeddings always match the migrated-cluster vector,
        # regardless of which chunk text was asked for.
        return [[1.0, 0.0, 0.0] for _ in texts]

    migrated, unmigrated = identify_migrated_cluster(
        migrated_idx,
        unmigrated_idx,
        contents_list,
        vectors_list,
        embed_fn=fake_embed_fn,
        sample_size=2,
    )

    assert set(migrated) == set(migrated_idx)
    assert set(unmigrated) == set(unmigrated_idx)


def test_identify_migrated_cluster_handles_migrated_majority():
    """Same check with the migrated cluster as the numeric majority, to
    make sure the function isn't secretly relying on list order/size."""
    migrated_idx = [0, 1, 2, 3, 4]
    unmigrated_idx = [5, 6]

    stored_vectors = {}
    for i in migrated_idx:
        stored_vectors[i] = [1.0, 0.0, 0.0]
    for i in unmigrated_idx:
        stored_vectors[i] = [0.0, 1.0, 0.0]

    contents_list = [f"chunk-{i}" for i in range(7)]
    vectors_list = [stored_vectors[i] for i in range(7)]

    def fake_embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    # Pass the unmigrated (smaller) cluster first as "cluster_a" to make
    # sure the function doesn't assume its first argument is migrated.
    migrated, unmigrated = identify_migrated_cluster(
        unmigrated_idx,
        migrated_idx,
        contents_list,
        vectors_list,
        embed_fn=fake_embed_fn,
        sample_size=2,
    )

    assert set(migrated) == set(migrated_idx)
    assert set(unmigrated) == set(unmigrated_idx)
