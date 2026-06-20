"""Async FastAPI surface for the supervisor-orchestrated analyst workflow.

Run locally:
    uv run uvicorn app:app --reload
Then open the interactive Swagger UI at  http://localhost:8000/docs
(ReDoc is also served at /redoc, and the raw OpenAPI schema at /openapi.json).

Why this layer is ASYNC
-----------------------
Every request spends almost all of its wall-clock time *waiting on the network*:
LLM calls to DashScope, market-data downloads from Yahoo Finance, and (optionally)
Postgres/Redis round-trips. That is I/O-bound work, not CPU-bound. With async
handlers a single worker process can `await` one request's network call and use
that idle time to make progress on other requests, instead of one thread sitting
blocked per in-flight request. LangGraph graphs and LangChain agents expose a
native async path (`ainvoke`), so we drive the whole pipeline with `await` and
let FastAPI/uvicorn (an ASGI server) interleave many concurrent analyses on one
event loop. See the README for the fuller explanation.

Human-in-the-loop
-----------------
`POST /analyze` with `require_approval=true` runs until the report step, then
pauses (HTTP 200 with status="pending_approval") and returns the pending
decision plus a `thread_id`. The reviewer calls `POST /analyze/{thread_id}/resume`
with their verdict, which feeds `Command(resume=...)` back into the graph so it
continues from exactly where it paused.
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import Depends, FastAPI
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.workflows.analyst_workflow import build_analyst_workflow


# --------------------------------------------------------------------------- #
# Workflow singleton (built once, reused across requests)                      #
# --------------------------------------------------------------------------- #
# A checkpointer is REQUIRED so multi-turn threads and human-in-the-loop
# interrupts can be resumed by thread_id. Swap the in-memory backends for the
# Postgres saver/store (see src/workflows/*_memory.py) in production.
_workflow = build_analyst_workflow(
    checkpointer=InMemorySaver(),
    store=InMemoryStore(),
)


def get_workflow():
    """FastAPI dependency returning the compiled workflow.

    Exposed as a dependency (rather than referenced directly) so tests can swap
    in a fake graph via `app.dependency_overrides[get_workflow]`.
    """
    return _workflow


# --------------------------------------------------------------------------- #
# Request / response schemas (these power the Swagger UI)                      #
# --------------------------------------------------------------------------- #
class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., description="The analyst's natural-language request.",
                        examples=["Analyze AAPL over the last month and write a short report."])
    thread_id: Optional[str] = Field(
        None, description="Conversation id. Omit to start a new thread (one is generated).")
    user_id: Optional[str] = Field(
        None, description="Stable user id used for long-term preference memory.")
    jwt: Optional[str] = Field(
        None, description="Auth0 access token. Omit for an anonymous principal.")
    require_approval: bool = Field(
        False, description="If true, pause for human sign-off before the report is written.")


class ResumeRequest(BaseModel):
    approved: bool = Field(..., description="The human reviewer's verdict.")
    reason: Optional[str] = Field(None, description="Optional note explaining the decision.")


class AnalyzeResponse(BaseModel):
    thread_id: str
    status: str = Field(..., description='"completed" or "pending_approval".')
    message: str = Field("", description="The workflow's final assistant message.")
    collected_data: Optional[str] = None
    chart_path: Optional[str] = None
    report_path: Optional[str] = None
    denied: list[str] = Field(default_factory=list,
                              description="Agents the principal was blocked from.")
    interrupt: Optional[dict] = Field(
        None, description="When status is pending_approval, the decision the human must make.")


# --------------------------------------------------------------------------- #
# App                                                                          #
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Agentic Data Analyst",
    description="A supervisor-orchestrated multi-agent stock-analysis API "
                "(data collection → charting → reporting) with auth, rate "
                "limiting, and human-in-the-loop approval.",
    version="0.1.0",
)


def _extract_interrupt(result: dict) -> Optional[dict]:
    """Return the interrupt payload if the graph paused, else None."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    # Interrupt objects expose the payload on `.value`; fall back to the raw item.
    return getattr(first, "value", first)


def _final_message(result: dict) -> str:
    msgs = result.get("messages") or []
    if not msgs:
        return ""
    content = getattr(msgs[-1], "content", "")
    return content if isinstance(content, str) else str(content)


def _to_response(thread_id: str, result: dict) -> AnalyzeResponse:
    pending = _extract_interrupt(result)
    return AnalyzeResponse(
        thread_id=thread_id,
        status="pending_approval" if pending else "completed",
        message=_final_message(result),
        collected_data=result.get("collected_data"),
        chart_path=result.get("chart_path"),
        report_path=result.get("report_path"),
        denied=list(result.get("denied", []) or []),
        interrupt=pending if isinstance(pending, dict) else None,
    )


@app.get("/health", tags=["ops"])
async def health() -> dict:
    """Liveness/readiness probe (used by Docker/K8s and CI)."""
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
async def analyze(req: AnalyzeRequest, workflow=Depends(get_workflow)) -> AnalyzeResponse:
    """Run the analyst workflow for a request.

    Returns status="completed" with the produced artifacts, or
    status="pending_approval" (with the pending `interrupt`) when human sign-off
    is required before the report step.
    """
    thread_id = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {
        "thread_id": thread_id,
        "user_id": req.user_id,
        "jwt": req.jwt,
        "require_approval": req.require_approval,
    }}
    result = await workflow.ainvoke(
        {"messages": [HumanMessage(req.prompt)]}, config
    )
    return _to_response(thread_id, result)


@app.post("/analyze/{thread_id}/resume", response_model=AnalyzeResponse, tags=["analysis"])
async def resume(thread_id: str, req: ResumeRequest,
                 workflow=Depends(get_workflow)) -> AnalyzeResponse:
    """Resume a paused (human-in-the-loop) run with the reviewer's verdict."""
    config = {"configurable": {"thread_id": thread_id}}
    result = await workflow.ainvoke(
        Command(resume={"approved": req.approved, "reason": req.reason}), config
    )
    return _to_response(thread_id, result)
