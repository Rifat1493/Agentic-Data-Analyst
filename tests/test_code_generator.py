"""Standalone test for the code_generator agent.

Uses mock OHLCV data so the data_collector agent is not required.
Run:  python tests/test_code_generator.py
Requires DASHSCOPE_API_KEY in the environment.
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.code_generator import build_code_generator_agent  # noqa: E402

# --- Mock data (replaces data_collector output) --------------------------------
MOCK_DATA = json.dumps({
    "ticker": "MSFT",
    "period": "1wk",
    "interval": "1d",
    "rows": 5,
    "columns": ["Open", "High", "Low", "Close", "Volume"],
    "data": {
        "2026-06-09 00:00:00": {"Open": 448.50, "High": 452.30, "Low": 446.10, "Close": 450.20, "Volume": 18500000},
        "2026-06-10 00:00:00": {"Open": 450.20, "High": 455.80, "Low": 449.00, "Close": 454.60, "Volume": 21000000},
        "2026-06-11 00:00:00": {"Open": 454.60, "High": 458.00, "Low": 452.40, "Close": 456.90, "Volume": 19800000},
        "2026-06-12 00:00:00": {"Open": 456.90, "High": 460.50, "Low": 455.10, "Close": 459.30, "Volume": 22300000},
        "2026-06-13 00:00:00": {"Open": 459.30, "High": 463.10, "Low": 457.80, "Close": 461.70, "Volume": 20100000},
    },
})

# --- Natural language query ----------------------------------------------------
USER_QUERY = (
    "Here is the financial data:\n"
    f"{MOCK_DATA}\n\n"
    "Plot a piechart of the closing price distribution"
)

# --- Run -----------------------------------------------------------------------
if __name__ == "__main__":
    agent = build_code_generator_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": USER_QUERY}]})
    last = result["messages"][-1]
    response = (
        last.content
        if hasattr(last, "content") and isinstance(last.content, str)
        else "".join(b.text for b in last.content_blocks if hasattr(b, "text"))
    )
    print(response)
