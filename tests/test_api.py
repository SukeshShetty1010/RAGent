import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app
from engine.execution_engine_streaming import StreamingResult

client = TestClient(app)

@pytest.mark.unit
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ============================================================
# AUDIT_TASKS T14: ChatRequest.history is optional and flows through
# to StreamingRageEngine.run_streaming unchanged.
# ============================================================

class _FakeEngine:
    def __init__(self):
        self.received_history = None

    def run_streaming(self, query, on_token_callback=None, on_stage_callback=None,
                       cancel_event=None, options=None, history=None):
        self.received_history = history
        return StreamingResult(
            final_answer="ok",
            agent_decisions={},
            evidence=[],
            kpis={"cancelled": False},
            raw_metrics={},
            stages=[],
        )


@pytest.mark.unit
def test_chat_history_field_is_optional(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(api_main, "get_engine", lambda: fake)

    response = client.post("/api/chat", json={"query": "hi"})

    assert response.status_code == 200
    assert fake.received_history == []


@pytest.mark.unit
def test_chat_history_field_passed_through(monkeypatch):
    fake = _FakeEngine()
    monkeypatch.setattr(api_main, "get_engine", lambda: fake)

    response = client.post("/api/chat", json={
        "query": "what about its story?",
        "history": [
            {"role": "user", "content": "Tell me about Far Cry 5"},
            {"role": "assistant", "content": "Far Cry 5 is an open-world FPS."},
        ],
    })

    assert response.status_code == 200
    assert fake.received_history == [
        {"role": "user", "content": "Tell me about Far Cry 5"},
        {"role": "assistant", "content": "Far Cry 5 is an open-world FPS."},
    ]


@pytest.mark.unit
def test_chat_history_rejects_oversized_turn(monkeypatch):
    monkeypatch.setattr(api_main, "get_engine", lambda: _FakeEngine())

    response = client.post("/api/chat", json={
        "query": "hi",
        "history": [{"role": "user", "content": "x" * 4001}],
    })

    assert response.status_code == 422
