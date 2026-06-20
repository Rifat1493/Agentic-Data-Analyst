from typing import Optional
from pydantic import BaseModel, Field


class ReportGeneratorRequest(BaseModel):
    task: str = Field(
        ...,
        description="Natural language instruction for the agent.",
        examples=["Write a financial analysis report for the provided MSFT data."],
    )
    collected_data: Optional[str] = Field(
        None, description="OHLCV payload as a JSON string (from data_collector)."
    )
    chart_path: Optional[str] = Field(
        None, description="Absolute path to a chart PNG (from code_generator)."
    )


class ReportGeneratorResponse(BaseModel):
    report_path: Optional[str] = Field(None, description="Absolute path to the saved Markdown report.")
    summary: str = Field("", description="Agent's natural language confirmation message.")
    error: Optional[str] = Field(None, description="Error message if the request failed.")
