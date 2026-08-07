"""
tests/test_ingest_success_gate.py

scripts/bulk_ingest.py's success gate: a game whose identity resolves
but whose Stage 5 upserts 0 editorial chunks must not be counted a
SUCCESS, and a Stage 2/3 failure must not abort the game. Every stage
function is monkeypatched at scripts.bulk_ingest's module namespace
(where they're imported directly), so this is hermetic — no network,
no credentials, no Qdrant.
"""

import pytest

import scripts.bulk_ingest as bulk_ingest


class _DummyClient:
    def close(self):
        pass


def _patch_all_stages(
    monkeypatch,
    *,
    game_uuid="uuid-1",
    platform_specs=lambda **kw: 1,
    igdb=lambda **kw: 1,
    gamespot=lambda **kw: "gamespot-uuid",
    chunks=lambda name, uuid: [{"uuid": "c1"}],
    upsert_chunks=lambda payloads: len(payloads),
):
    monkeypatch.setattr(bulk_ingest, "upsert_game_anchor", lambda client, name: game_uuid)
    monkeypatch.setattr(bulk_ingest, "upsert_platform_specs", lambda **kw: platform_specs(**kw))
    monkeypatch.setattr(bulk_ingest, "upsert_igdb_context", lambda **kw: igdb(**kw))
    monkeypatch.setattr(bulk_ingest, "upsert_gamespot_container", lambda **kw: gamespot(**kw))
    monkeypatch.setattr(bulk_ingest, "generate_chunk_payloads", chunks)
    monkeypatch.setattr(bulk_ingest, "upsert_chunk_batch", upsert_chunks)


@pytest.mark.unit
def test_ingest_single_game_returns_chunk_count_on_full_success(monkeypatch):
    _patch_all_stages(
        monkeypatch,
        chunks=lambda name, uuid: [{"uuid": "a"}, {"uuid": "b"}],
        upsert_chunks=lambda payloads: len(payloads),
    )

    count = bulk_ingest._ingest_single_game(client=None, game_name="Hades")

    assert count == 2


@pytest.mark.unit
def test_ingest_single_game_zero_chunks_is_not_success(monkeypatch):
    _patch_all_stages(monkeypatch, chunks=lambda name, uuid: [])

    count = bulk_ingest._ingest_single_game(client=None, game_name="Some Obscure Game")

    assert count == 0


@pytest.mark.unit
def test_stage2_failure_does_not_abort_the_game(monkeypatch):
    def broken_platform_specs(**kw):
        raise RuntimeError("simulated RAWG outage")

    _patch_all_stages(
        monkeypatch,
        platform_specs=broken_platform_specs,
        chunks=lambda name, uuid: [{"uuid": "a"}],
        upsert_chunks=lambda payloads: len(payloads),
    )

    count = bulk_ingest._ingest_single_game(client=None, game_name="Hades")

    assert count == 1


@pytest.mark.unit
def test_stage3_failure_does_not_abort_the_game(monkeypatch):
    def broken_igdb(**kw):
        raise RuntimeError("simulated IGDB outage")

    _patch_all_stages(
        monkeypatch,
        igdb=broken_igdb,
        chunks=lambda name, uuid: [{"uuid": "a"}],
        upsert_chunks=lambda payloads: len(payloads),
    )

    count = bulk_ingest._ingest_single_game(client=None, game_name="Hades")

    assert count == 1


@pytest.mark.unit
def test_stage4_failure_does_not_abort_the_game(monkeypatch):
    def broken_gamespot(**kw):
        raise RuntimeError("simulated GameSpot outage")

    _patch_all_stages(
        monkeypatch,
        gamespot=broken_gamespot,
        chunks=lambda name, uuid: [{"uuid": "a"}],
        upsert_chunks=lambda payloads: len(payloads),
    )

    count = bulk_ingest._ingest_single_game(client=None, game_name="Hades")

    assert count == 1


@pytest.mark.unit
def test_stage1_failure_propagates(monkeypatch):
    def broken_anchor(client, name):
        raise RuntimeError("simulated total identity failure")

    monkeypatch.setattr(bulk_ingest, "upsert_game_anchor", broken_anchor)

    with pytest.raises(RuntimeError):
        bulk_ingest._ingest_single_game(client=None, game_name="Hades")


@pytest.mark.unit
def test_main_loop_logs_partial_not_success_for_zero_chunk_game(monkeypatch, tmp_path):
    _patch_all_stages(monkeypatch, chunks=lambda name, uuid: [])

    monkeypatch.setattr(bulk_ingest, "LOG_DIR", tmp_path)
    monkeypatch.setattr(bulk_ingest, "SUCCESS_LOG", tmp_path / "success_games.log")
    monkeypatch.setattr(bulk_ingest, "PARTIAL_LOG", tmp_path / "partial_games.log")
    monkeypatch.setattr(bulk_ingest, "FAILED_LOG", tmp_path / "failed_games.log")
    monkeypatch.setattr(bulk_ingest, "_get_qdrant_client", lambda: _DummyClient())
    monkeypatch.setattr(bulk_ingest, "TOP_100_GAMES", ["Some Obscure Game"])
    monkeypatch.setattr(
        "sys.argv",
        ["bulk_ingest.py", "--start", "0", "--end", "1", "--delay", "0"],
    )

    bulk_ingest.main()

    assert (tmp_path / "partial_games.log").exists()
    assert not (tmp_path / "success_games.log").exists()


@pytest.mark.unit
def test_main_loop_logs_success_for_nonzero_chunk_game(monkeypatch, tmp_path):
    _patch_all_stages(
        monkeypatch,
        chunks=lambda name, uuid: [{"uuid": "a"}],
        upsert_chunks=lambda payloads: len(payloads),
    )

    monkeypatch.setattr(bulk_ingest, "LOG_DIR", tmp_path)
    monkeypatch.setattr(bulk_ingest, "SUCCESS_LOG", tmp_path / "success_games.log")
    monkeypatch.setattr(bulk_ingest, "PARTIAL_LOG", tmp_path / "partial_games.log")
    monkeypatch.setattr(bulk_ingest, "FAILED_LOG", tmp_path / "failed_games.log")
    monkeypatch.setattr(bulk_ingest, "_get_qdrant_client", lambda: _DummyClient())
    monkeypatch.setattr(bulk_ingest, "TOP_100_GAMES", ["Hades"])
    monkeypatch.setattr(
        "sys.argv",
        ["bulk_ingest.py", "--start", "0", "--end", "1", "--delay", "0"],
    )

    bulk_ingest.main()

    assert (tmp_path / "success_games.log").exists()
    assert not (tmp_path / "partial_games.log").exists()
