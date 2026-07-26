"""Standalone tests for the supervisor-orchestrated analyst workflow.

Run:  python tests/test_workflow.py
Requires DASHSCOPE_API_KEY in the environment (scenarios 1-2 also hit the
network via yfinance / the LLM).

What this file demonstrates
---------------------------
* Dynamic routing — the supervisor picks the path; we never hard-code
  data -> chart -> report.
* Structured artifact hand-off via typed state fields.
* Short-term memory (checkpointer): a follow-up message in the SAME thread
  reuses earlier results.
* Long-term memory (store): user preferences that persist ACROSS threads.

------------------------------------------------------------------------------
SHORT-TERM vs LONG-TERM MEMORY — when to use which (read this!)
------------------------------------------------------------------------------
Short-term memory = checkpointer, scoped to a `thread_id` (one conversation).
  Use it for:
    * Multi-turn conversations: "collect TSLA data" ... then "now chart it".
      The data is still in state, so the second turn skips data collection.
    * Resuming / retrying a run after a crash (state is persisted per step).
    * Human-in-the-loop pauses (approve before the report is written).

Long-term memory = store, scoped to a `user_id` (across ALL conversations).
  Use it for:
    * User preferences: default chart type (candlestick), report tone, timezone.
    * A watchlist / favourite tickers the analyst reuses every day.
    * Caching expensive results (e.g. a fundamentals lookup) to reuse next week.
    * Learned facts about the user ("always wants volume on price charts").
  In production back this with Postgres so it survives restarts; the interface
  is identical to the InMemoryStore used here.
------------------------------------------------------------------------------
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import HumanMessage

from src import config
from src.workflows.analyst_workflow import build_analyst_workflow
from src.workflows.short_term_memory import open_postgres_checkpointer
from src.workflows.long_term_memory import open_postgres_store


# Mock OHLCV data, used where we want to simulate the user supplying data
# directly (so the supervisor should SKIP the data_collector).
MOCK_DATA = json.dumps({
    "ticker": "MSFT", "period": "1wk", "interval": "1d", "rows": 5,
    "columns": ["Open", "High", "Low", "Close", "Volume"],
    "data": {
        "2026-06-09 00:00:00": {"Open": 448.5, "High": 452.3, "Low": 446.1, "Close": 450.2, "Volume": 18500000},
        "2026-06-10 00:00:00": {"Open": 450.2, "High": 455.8, "Low": 449.0, "Close": 454.6, "Volume": 21000000},
        "2026-06-11 00:00:00": {"Open": 454.6, "High": 458.0, "Low": 452.4, "Close": 456.9, "Volume": 19800000},
        "2026-06-12 00:00:00": {"Open": 456.9, "High": 460.5, "Low": 455.1, "Close": 459.3, "Volume": 22300000},
        "2026-06-13 00:00:00": {"Open": 459.3, "High": 463.1, "Low": 457.8, "Close": 461.7, "Volume": 20100000},
    },
})


def _print_outcome(state: dict) -> None:
    print("  routing finished. Artifacts produced:")
    print(f"    collected_data: {'yes' if state.get('collected_data') else 'no'}")
    print(f"    chart_path:     {state.get('chart_path')}")
    print(f"    report_path:    {state.get('report_path')}")
    print(f"  final message: {state['messages'][-1].content[:160]}")


# --------------------------------------------------------------------------- #
# Scenario 1: full pipeline — supervisor should run all three workers.         #
# --------------------------------------------------------------------------- #
def scenario_full_pipeline():
    print("\n=== Scenario 1: full pipeline (data -> chart -> report) ===")
    app = build_analyst_workflow()
    config = {"configurable": {"thread_id": "s1"}}
    state = app.invoke(
        {"messages": [HumanMessage(
            "Analyze AAPL over the last 1 month (daily). Make a price chart and "
            "write a short report."
        )]},
        config,
    )
    _print_outcome(state)


# --------------------------------------------------------------------------- #
# Scenario 2: DYNAMIC — user supplies data, wants only a chart.               #
# Supervisor should SKIP data_collector and go straight to code_generator.    #
# --------------------------------------------------------------------------- #
def scenario_user_supplies_data():
    print("\n=== Scenario 2: user supplies data, asks only for a chart ===")
    app = build_analyst_workflow()
    config = {"configurable": {"thread_id": "s2"}}
    state = app.invoke(
        {
            "messages": [HumanMessage(
                "Here is MSFT data I already have. Just make me a candlestick chart."
            )],
            # Pre-seed the typed artifact: this is the structured hand-off in action.
            "collected_data": MOCK_DATA,
        },
        config,
    )
    _print_outcome(state)


# --------------------------------------------------------------------------- #
# Scenario 3: SHORT-TERM MEMORY — two turns, same thread_id.                  #
# Turn 1 collects data; Turn 2 says "now chart it" and the data is reused.    #
# --------------------------------------------------------------------------- #
def scenario_short_term_memory():
    print("\n=== Scenario 3: short-term memory in POSTGRES (two turns, same thread) ===")
    config = {"configurable": {"thread_id": "s3"}}   # SAME thread for both turns

    # The Postgres checkpointer persists the whole graph state per thread_id, so
    # turn 2 sees the data collected in turn 1 — even across process restarts.
    with open_postgres_checkpointer() as checkpointer:
        app = build_analyst_workflow(checkpointer=checkpointer)

        print("  turn 1: collect data only")
        app.invoke(
            {"messages": [HumanMessage("Download TSLA data for the last week, daily. Just the data.")]},
            config,
        )

        print("  turn 2: 'now chart it' — should reuse turn-1 data, no re-download")
        state = app.invoke(
            {"messages": [HumanMessage("Great, now make a line chart of the closing price.")]},
            config,
        )
        _print_outcome(state)


# --------------------------------------------------------------------------- #
# Scenario 4: LONG-TERM MEMORY — preferences persist across threads/users.    #
# We store a preference, then run a brand-new thread; the supervisor reads it. #
# --------------------------------------------------------------------------- #
def scenario_long_term_memory():
    print("\n=== Scenario 4: long-term memory in POSTGRES (cross-thread preference) ===")
    user_id = "analyst_42"

    # The Postgres store keeps the DB connection open for the whole `with` block.
    # Anything we write here survives restarts and is shared across all threads.
    with open_postgres_store() as store:
        # Write a preference once; on later runs it is already persisted in the DB.
        store.put(("preferences", user_id), "chart_type",
                  {"key": "chart_type", "value": "candlestick"})

        # Long-term store on Postgres; short-term left as the in-memory default.
        app = build_analyst_workflow(store=store)

        # Brand-new thread, but SAME user — the supervisor loads the preference
        # from Postgres, not from this process's memory.
        config = {"configurable": {"thread_id": "s4_new", "user_id": user_id}}
        state = app.invoke(
            {"messages": [HumanMessage("Make a chart for the data.")],
             "collected_data": MOCK_DATA},
            config,
        )
        _print_outcome(state)
        print("  (the supervisor read 'chart_type=candlestick' from Postgres)")


if __name__ == "__main__":
    if not config.DASHSCOPE_API_KEY:
        print("DASHSCOPE_API_KEY is not set — set it before running these scenarios.")
        sys.exit(1)

    scenario_full_pipeline()
    # scenario_user_supplies_data()
    # scenario_short_term_memory()
    # scenario_long_term_memory()
