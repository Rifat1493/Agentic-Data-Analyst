from langchain.agents import create_agent

from src.agents.models import get_foundation_model
from .tools import download_financial_data


SYSTEM_PROMPT = """You are a financial data collector agent. Your job is to download \
historical market data from Yahoo Finance using the download_financial_data tool.

When the user provides a ticker symbol (and optionally a timeframe / granularity), \
call the tool with those values. If no timeframe or interval is specified, use the \
defaults: period='1wk' and interval='1d'.

After the tool returns data, summarise the result in a short message: ticker, date \
range covered, number of rows, and columns available. Do not print the raw JSON."""


def build_data_collector_agent():
    """Return a compiled LangChain agent that downloads financial data."""
    llm = get_foundation_model()
    return create_agent(
        model=llm,
        tools=[download_financial_data],
        system_prompt=SYSTEM_PROMPT,
    )
