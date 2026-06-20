"""Standalone FastAPI micro-app for the report_generator agent.

Run independently:
    uvicorn src.agents.report_generator.app:app --port 8003

Or via Docker:
    docker build -f Dockerfile.report-generator -t report-generator .
    docker run -p 8003:8000 --env-file .env.dev report-generator

Swagger UI: http://localhost:8003/docs
"""
from typing import Optional

from fastapi import FastAPI
from langchain_core.messages import HumanMessage, ToolMessage

from .agent import build_report_generator_agent
from .schemas import ReportGeneratorRequest, ReportGeneratorResponse

app = FastAPI(
    title="Report Generator Agent",
    description="Writes structured Markdown financial analysis reports.",
    version="0.1.0",
)

_agent = build_report_generator_agent()


def _parse_report_path(tool_output: Optional[str]) -> Optional[str]:
    if tool_output and "Report saved to:" in tool_output:
        return tool_output.split("Report saved to:", 1)[1].strip()
    return None


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "agent": "report_generator"}


@app.post("/invoke", response_model=ReportGeneratorResponse, tags=["agent"])
async def invoke(req: ReportGeneratorRequest) -> ReportGeneratorResponse:
    """Invoke the report_generator agent with a task and optional artifacts."""
    parts = [req.task]
    if req.collected_data:
        parts.append(f"\nAvailable OHLCV data (JSON):\n{req.collected_data}")
    if req.chart_path:
        parts.append(f"\nAvailable chart image path: {req.chart_path}")

    result = await _agent.ainvoke({"messages": [HumanMessage("\n".join(parts))]})

    tool_out = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage):
            tool_out = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    last = result["messages"][-1]
    summary = last.content if isinstance(getattr(last, "content", None), str) else ""

    return ReportGeneratorResponse(
        report_path=_parse_report_path(tool_out),
        summary=summary,
    )
