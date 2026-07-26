"""Offline tests for the human-in-the-loop (HITL) approval gate.

No LLM/network: we monkeypatch `interrupt` (to simulate the human's resumed
verdict) and the report sub-agent (so the approve path doesn't hit DashScope),
then call `report_generator_node` directly. This exercises all three branches:
approval disabled, approved, and rejected.

Run:  pytest tests/test_hitl.py
"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import src.workflows.analyst_workflow as wf
from src.middleware.authentication import anonymous_principal


def _base_state():
    # report_generator is open to anonymous principals, so this passes the guard.
    return {
        "messages": [HumanMessage("Write a short report on the data.")],
        "principal": anonymous_principal().to_dict(),
        "task": "write report",
        "collected_data": "{}",
    }


class _FakeAgent:
    """Stands in for the real report sub-agent; records that it was invoked."""
    def __init__(self):
        self.invoked = False

    def invoke(self, _payload):
        self.invoked = True
        # report_path is parsed from the LAST ToolMessage, mirroring the real agent.
        return {"messages": [
            ToolMessage("Report saved to: /tmp/report.md", tool_call_id="t1"),
            AIMessage("Saved the report to /tmp/report.md"),
        ]}


def test_is_approved_interprets_verdicts():
    assert wf._is_approved(True) is True
    assert wf._is_approved("yes") is True
    assert wf._is_approved("Approve") is True
    assert wf._is_approved(False) is False
    assert wf._is_approved("no") is False
    assert wf._is_approved({"approved": True}) is True
    assert wf._is_approved({"approved": False, "reason": "looks wrong"}) is False


def test_no_approval_required_runs_agent(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(wf, "_report_generator", fake)
    # interrupt must NOT be reached when approval isn't requested.
    monkeypatch.setattr(wf, "interrupt", lambda payload: (_ for _ in ()).throw(
        AssertionError("interrupt should not be called")))

    out = wf.report_generator_node(_base_state(), {"configurable": {}})

    assert fake.invoked is True
    assert "report_generator" in out["messages"][0].content


def test_approval_granted_runs_agent(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(wf, "_report_generator", fake)
    monkeypatch.setattr(wf, "interrupt", lambda payload: {"approved": True})

    out = wf.report_generator_node(
        _base_state(), {"configurable": {"require_approval": True}})

    assert fake.invoked is True
    assert out["report_path"] == "/tmp/report.md"


def test_approval_rejected_skips_agent(monkeypatch):
    fake = _FakeAgent()
    monkeypatch.setattr(wf, "_report_generator", fake)
    monkeypatch.setattr(
        wf, "interrupt", lambda payload: {"approved": False, "reason": "stale data"})

    out = wf.report_generator_node(
        _base_state(), {"configurable": {"require_approval": True}})

    assert fake.invoked is False
    msg = out["messages"][0].content.lower()
    assert "rejected" in msg and "stale data" in msg
