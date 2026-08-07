"""
tests/test_identity_resolver.py

Provider-independent identity resolution (ingest/identity_resolver.py).
Provider adapters are monkeypatched at their import source, so these
tests are hermetic — no network, no credentials.
"""

import pytest

import data.igdb_data as igdb_data_mod
import ingest.rawg_identity_ingest as rawg_identity_ingest_mod
from ingest.identity_resolver import (
    CanonicalIdentity,
    IdentityResolutionError,
    make_game_uuid,
    make_unified_game_id,
    resolve_identity,
    slug,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "title,expected",
    [
        ("Elden Ring", "elden-ring"),
        ("The Witcher 3: Wild Hunt", "the-witcher-3-wild-hunt"),
        ("Resident Evil 4 (2023)", "resident-evil-4"),
        ("Dead Space (2023)", "dead-space"),
        ("Tom Clancy's Rainbow Six Siege", "tom-clancy-s-rainbow-six-siege"),
    ],
)
def test_slug_normalization(title, expected):
    assert slug(title) == expected


@pytest.mark.unit
def test_unified_game_id_appends_year_once_for_parenthetical_titles():
    # "Resident Evil 4 (2023)" + release_year=2023 must not double-append
    unified_id = make_unified_game_id("Resident Evil 4 (2023)", 2023)
    assert unified_id == "resident-evil-4-2023"
    assert unified_id.count("2023") == 1


@pytest.mark.unit
def test_unified_game_id_missing_year():
    assert make_unified_game_id("Undertale", None) == "undertale"


@pytest.mark.unit
def test_game_uuid_is_deterministic():
    uid = make_unified_game_id("Hades", 2020)
    assert make_game_uuid(uid) == make_game_uuid(uid)


@pytest.mark.unit
def test_resolve_identity_prefers_igdb(monkeypatch):
    rawg_calls = []

    def fake_igdb(query, strip_visual=True, limit=1):
        return {
            "query_name": query,
            "resolved_name": "Hades",
            "clean": [{"id": 42, "name": "Hades", "first_release_date": 1597881600}],
        }

    def track_rawg(game_name):
        # Fail-soft adapters swallow exceptions, so an AssertionError here
        # would silently be treated as "RAWG couldn't resolve" rather than
        # failing the test — track calls instead.
        rawg_calls.append(game_name)
        return {"game_id": 1, "title": "Hades", "release_year": 2020}

    monkeypatch.setattr(igdb_data_mod, "fetch_igdb_game_data", fake_igdb)
    monkeypatch.setattr(rawg_identity_ingest_mod, "fetch_and_prepare_identity", track_rawg)

    identity = resolve_identity("Hades")

    assert isinstance(identity, CanonicalIdentity)
    assert identity.identity_source == "igdb"
    assert identity.title == "Hades"
    assert identity.release_year == 2020
    assert identity.source_ids == {"igdb": 42}
    assert identity.unified_game_id == "hades-2020"
    assert rawg_calls == [], "RAWG adapter should not run when IGDB already resolved"


@pytest.mark.unit
def test_resolve_identity_falls_back_to_rawg_when_igdb_unavailable(monkeypatch):
    def broken_igdb(*args, **kwargs):
        raise RuntimeError("simulated IGDB outage")

    def fake_rawg(game_name):
        return {"game_id": 123, "title": "Hades", "release_year": 2020}

    monkeypatch.setattr(igdb_data_mod, "fetch_igdb_game_data", broken_igdb)
    monkeypatch.setattr(rawg_identity_ingest_mod, "fetch_and_prepare_identity", fake_rawg)

    identity = resolve_identity("Hades")

    assert identity.identity_source == "rawg"
    assert identity.source_ids == {"rawg": 123}
    assert identity.unified_game_id == "hades-2020"


@pytest.mark.unit
def test_resolve_identity_is_provider_independent(monkeypatch):
    """Same title+year via IGDB vs RAWG must produce an identical
    game_uuid — this is what makes the upsert idempotent across a
    provider outage (e.g. RAWG rejoining after being down)."""

    def fake_igdb(query, strip_visual=True, limit=1):
        return {"clean": [{"id": 1, "name": "Hades", "first_release_date": 1597881600}]}

    monkeypatch.setattr(igdb_data_mod, "fetch_igdb_game_data", fake_igdb)
    via_igdb = resolve_identity("Hades")

    def broken_igdb(*args, **kwargs):
        raise RuntimeError("simulated outage")

    def fake_rawg(game_name):
        return {"game_id": 999, "title": "Hades", "release_year": 2020}

    monkeypatch.setattr(igdb_data_mod, "fetch_igdb_game_data", broken_igdb)
    monkeypatch.setattr(rawg_identity_ingest_mod, "fetch_and_prepare_identity", fake_rawg)
    via_rawg = resolve_identity("Hades")

    assert via_igdb.game_uuid == via_rawg.game_uuid
    assert via_igdb.unified_game_id == via_rawg.unified_game_id


@pytest.mark.unit
def test_resolve_identity_raises_when_all_providers_fail(monkeypatch):
    def broken_igdb(*args, **kwargs):
        raise RuntimeError("simulated IGDB outage")

    def broken_rawg(game_name):
        raise RuntimeError("simulated RAWG outage")

    monkeypatch.setattr(igdb_data_mod, "fetch_igdb_game_data", broken_igdb)
    monkeypatch.setattr(rawg_identity_ingest_mod, "fetch_and_prepare_identity", broken_rawg)

    with pytest.raises(IdentityResolutionError):
        resolve_identity("Hades")


@pytest.mark.unit
def test_resolve_identity_rejects_empty_name():
    with pytest.raises(IdentityResolutionError):
        resolve_identity("")


# Frozen fixture list — a future refactor of slug()/make_unified_game_id()
# must not silently re-key an already-ingested corpus. If one of these
# assertions needs to change, that change is a breaking migration to call
# out explicitly, not a routine edit.
_FROZEN_UNIFIED_IDS = [
    ("Grand Theft Auto V", 2013, "grand-theft-auto-v-2013"),
    (
        "The Legend of Zelda: Breath of the Wild",
        2017,
        "the-legend-of-zelda-breath-of-the-wild-2017",
    ),
    ("Resident Evil 2 (2019)", 2019, "resident-evil-2-2019"),
    ("Dead Space (2023)", 2023, "dead-space-2023"),
    ("EA Sports FC 24", 2023, "ea-sports-fc-24-2023"),
]


@pytest.mark.unit
@pytest.mark.parametrize("title,year,expected", _FROZEN_UNIFIED_IDS)
def test_unified_game_id_stability(title, year, expected):
    assert make_unified_game_id(title, year) == expected
