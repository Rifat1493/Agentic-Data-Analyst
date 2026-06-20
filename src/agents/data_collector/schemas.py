from typing import Optional
from pydantic import BaseModel, Field


class DataCollectorRequest(BaseModel):
    task: str = Field(
        ...,
        description="Natural language instruction for the agent.",
        examples=["Download AAPL data for the last month, daily interval."],
    )


class DataCollectorResponse(BaseModel):
    data: Optional[str] = Field(None, description="OHLCV payload as a JSON string.")
    summary: str = Field("", description="Agent's natural language summary of what was collected.")
    error: Optional[str] = Field(None, description="Error message if the request failed.")
