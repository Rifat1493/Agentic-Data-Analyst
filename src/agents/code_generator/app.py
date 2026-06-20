"""Standalone FastAPI micro-app for the code_generator agent.

Run independently:
    uvicorn src.agents.code_generator.app:app --port 8002

Or via Docker:
    docker build -f Dockerfile.code-generator -t code-generator .
    docker run -p 8002:8000 --env-file .env.dev code-generator

Swagger UI: http://localhost:8002/docs
"""
from typing import Optional

from fastapi import FastAPI
from langchain_core.messages import HumanMessage, ToolMessage

from .agent import build_code_generator_agent
from .schemas import CodeGeneratorRequest, CodeGeneratorResponse

app = FastAPI(
    title="Code Generator Agent",
    description="Writes and safely executes Plotly visualization code to produce chart PNGs.",
    version="0.1.0",
)

_agent = build_code_generator_agent()


def _parse_chart_path(tool_output: Optional[str]) -> Optional[str]:
    if tool_output and "Chart saved to:" in tool_output:
        return tool_output.split("Chart saved to:", 1)[1].strip()
    return None


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "agent": "code_generator"}


@app.post("/invoke", response_model=CodeGeneratorResponse, tags=["agent"])
async def invoke(req: CodeGeneratorRequest) -> CodeGeneratorResponse:
    """Invoke the code_generator agent with a task and optional OHLCV data."""
    prompt = req.task
    if req.collected_data:
        prompt += f"\n\nAvailable OHLCV data (JSON):\n{req.collected_data}"

    result = await _agent.ainvoke({"messages": [HumanMessage(prompt)]})

    tool_out = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage):
            tool_out = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    last = result["messages"][-1]
    summary = last.content if isinstance(getattr(last, "content", None), str) else ""

    return CodeGeneratorResponse(
        chart_path=_parse_chart_path(tool_out),
        summary=summary,
    )
