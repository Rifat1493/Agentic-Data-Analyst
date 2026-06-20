"""Offline tests for the async FastAPI surface.

We override the `get_workflow` dependency with a fake graph whose `ainvoke` is a
coroutine returning a canned state, so the HTTP/serialization layer and the
interrupt-detection logic are tested without any LLM/network.

Run:  pytest tests/test_api.py
"""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app, get_workflow


class FakeWorkflow:
    """Minimal stand-in: records calls and returns a preset result from ainvoke."""
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def ainvoke(self, payload, config):
        self.calls.append((payload, config))
        return self.result


@pytest.fixture
def client_with(monkeypatch):
    """Return a factory: given a fake result, yield (TestClient, FakeWorkflow)."""
    created = {}

    def _make(result):
        fake = FakeWorkflow(result)
        app.dependency_overrides[get_workflow] = lambda: fake
        created["fake"] = fake
        return TestClient(app), fake

    yield _make
    app.dependency_overrides.clear()


def test_health():
    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_analyze_completed(client_with):
    client, fake = client_with({
        "messages": [SimpleNamespace(content="Done. Report ready.")],
        "collected_data": "{...}",
        "chart_path": "/out/chart.png",
        "report_path": "/out/report.md",
        "denied": [],
    })
    resp = client.post("/analyze", json={"prompt": "Analyze AAPL and report."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["report_path"] == "/out/report.md"
    assert body["thread_id"]  # auto-generated
    # The prompt was forwarded and require_approval defaulted to False.
    _payload, config = fake.calls[0]
    assert config["configurable"]["require_approval"] is False


def test_analyze_pending_approval(client_with):
    pending = SimpleNamespace(value={"type": "report_approval", "question": "Approve?"})
    client, _fake = client_with({
        "messages": [SimpleNamespace(content="Waiting for approval.")],
        "__interrupt__": [pending],
    })
    resp = client.post("/analyze", json={
        "prompt": "Analyze AAPL and report.",
        "require_approval": True,
    })
    body = resp.json()
    assert body["status"] == "pending_approval"
    assert body["interrupt"]["type"] == "report_approval"


def test_resume_uses_thread_id(client_with):
    client, fake = client_with({
        "messages": [SimpleNamespace(content="Report written after approval.")],
        "report_path": "/out/report.md",
    })
    resp = client.post("/analyze/thread-123/resume",
                       json={"approved": True, "reason": "looks good"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["thread_id"] == "thread-123"
    # The resume Command carried the human verdict into the graph.
    payload, config = fake.calls[0]
    assert config["configurable"]["thread_id"] == "thread-123"
    assert getattr(payload, "resume", None) == {"approved": True, "reason": "looks good"}
