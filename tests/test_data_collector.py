"""Test invocations for the data_collector agent."""
import sys
import os

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.data_collector import build_data_collector_agent  # noqa: E402

# --------------------------------------------------------------------------- #
# Requires DASHSCOPE_API_KEY in the environment.                               #
# Optionally set DASHSCOPE_BASE_URL to override the default DashScope endpoint.#
# --------------------------------------------------------------------------- #

agent = build_data_collector_agent()


def call_agent(prompt: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    last = result["messages"][-1]
    # LangChain 1.x AIMessage: prefer .content, fall back to content_blocks.
    if hasattr(last, "content") and isinstance(last.content, str):
        return last.content
    if hasattr(last, "content_blocks"):
        return "".join(
            b.text for b in last.content_blocks if hasattr(b, "text")
        )
    return str(last)


if __name__ == "__main__":
    # --- Test 1: defaults (1 week, daily) ---
    print("=== Test 1: ABX.TO – default period and interval ===")
    print(call_agent("Download financial data for Microsoft"))

    # --- Test 2: custom period and interval ---
    # print("\n=== Test 2: MSFT – last 1 month, weekly interval ===")
    # print(call_agent("Get MSFT stock data for the last 1 month at weekly granularity."))

    # # --- Test 3: intraday data ---
    # print("\n=== Test 3: GOOGL – last 5 days, 1-hour interval ===")
    # print(call_agent("Fetch GOOGL data for the past 5 days with 1-hour candles."))
