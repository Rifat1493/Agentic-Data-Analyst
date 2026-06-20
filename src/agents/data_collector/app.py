"""Standalone FastAPI micro-app for the data_collector agent.

Run independently:
    uvicorn src.agents.data_collector.app:app --port 8001

Or via Docker:
    docker build -f Dockerfile.data-collector -t data-collector .
    docker run -p 8001:8000 --env-file .env.dev data-collector

Swagger UI: http://localhost:8001/docs
"""
from langchain_core.messages import HumanMessage, ToolMessage
from fastapi import FastAPI

from .agent import build_data_collector_agent
from .schemas import DataCollectorRequest, DataCollectorResponse

app = FastAPI(
    title="Data Collector Agent",
    description="Downloads historical OHLCV market data from Yahoo Finance.",
    version="0.1.0",
)

_agent = build_data_collector_agent()


@app.get("/health", tags=["ops"])
async def health() -> dict:
    return {"status": "ok", "agent": "data_collector"}


@app.post("/invoke", response_model=DataCollectorResponse, tags=["agent"])
async def invoke(req: DataCollectorRequest) -> DataCollectorResponse:
    """Invoke the data_collector agent with a natural language task."""
    result = await _agent.ainvoke({"messages": [HumanMessage(req.task)]})

    data = None
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage):
            data = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    last = result["messages"][-1]
    summary = last.content if isinstance(getattr(last, "content", None), str) else ""

    return DataCollectorResponse(data=data, summary=summary)
