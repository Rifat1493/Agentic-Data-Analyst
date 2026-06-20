from typing import Optional
from pydantic import BaseModel, Field


class CodeGeneratorRequest(BaseModel):
    task: str = Field(
        ...,
        description="Natural language instruction for the agent.",
        examples=["Generate a candlestick chart from the provided OHLCV data."],
    )
    collected_data: Optional[str] = Field(
        None, description="OHLCV payload as a JSON string (from data_collector)."
    )


class CodeGeneratorResponse(BaseModel):
    chart_path: Optional[str] = Field(None, description="Absolute path to the saved PNG chart.")
    summary: str = Field("", description="Agent's natural language summary.")
    error: Optional[str] = Field(None, description="Error message if the request failed.")
