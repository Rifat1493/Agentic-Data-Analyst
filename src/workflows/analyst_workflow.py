"""Supervisor-orchestrated multi-agent workflow for stock data analysis.

Architecture (LangGraph "supervisor" pattern)
---------------------------------------------
A central *supervisor* LLM looks at the conversation and the artifacts produced
so far, then decides which worker agent should run next — or that the work is
done. Workers always return to the supervisor, which re-evaluates and routes
again. This loop handles DYNAMIC requests: the analyst may ask only for a chart,
only for a report from data they pasted in, or the full data -> chart -> report
pipeline. The supervisor figures out the path; we do not hard-code the order.

    START -> supervisor -> {data_collector | code_generator | report_generator} -> supervisor -> ... -> END

How agents share context
------------------------
Two complementary mechanisms, both living in the graph *state*:

1. `messages` — the running conversation. Every agent can see it. This is the
   shared "working memory" of the turn.
2. Typed artifact fields (`collected_data`, `chart_path`, `report_path`) — a
   RELIABLE, structured hand-off. Downstream agents read the data JSON or chart
   path straight from these fields instead of re-parsing chat history. This is
   the production-grade part: structured state beats string-scraping.

Memory
------
* Short-term memory  = a *checkpointer* (here `InMemorySaver`). It persists the
  whole state per `thread_id`, so a follow-up message in the SAME conversation
  reuses previously collected data / charts. Swap for a Postgres/SQLite saver in
  production — same interface.
* Long-term memory   = a *store* (here `InMemoryStore`). It survives ACROSS
  conversations (threads) and users — e.g. user chart preferences or a watchlist.
  Swap for a Postgres-backed store later.
"""
import operator
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from src.agents.models import get_foundation_model
from src.agents.data_collector import build_data_collector_agent
from src.agents.code_generator import build_code_generator_agent
from src.agents.report_generator import build_report_generator_agent
from src.middleware.authentication import authenticate, Principal, AuthenticationError
from src.middleware.authorization import check_access, permitted_agents
from src.middleware.rate_limiter import get_default_limiter
from src.middleware.otel_setup import configure_tracing


# --------------------------------------------------------------------------- #
# 1. Shared graph state                                                        #
# --------------------------------------------------------------------------- #
class AnalystState(TypedDict):
    """The single source of truth passed between every node in the graph."""

    # Conversation history. `add_messages` is a *reducer*: node updates are
    # appended to the list rather than overwriting it.
    messages: Annotated[list, add_messages]

    # Structured artifacts — the reliable hand-off between agents.
    collected_data: Optional[str]   # OHLCV JSON from data_collector (or user-supplied)
    chart_path: Optional[str]       # PNG path from code_generator (or user-supplied)
    report_path: Optional[str]      # Markdown path from report_generator

    # Auth — set once by the authenticate node, read by guards + supervisor.
    principal: dict                 # serialized Principal (user_id, roles, authenticated)
    auth_error: Optional[str]       # set if a supplied token failed verification
    denied: Annotated[list, operator.add]  # agents the user was blocked from (accumulates)

    # Rate limiting — set by the rate_limit node at workflow entry.
    rate_limited: Optional[bool]    # True if the run was rejected for exceeding the limit

    # Supervisor's routing decision for the current step.
    next: str                       # which worker to run, or "FINISH"
    task: str                       # focused instruction the supervisor hands the worker


# --------------------------------------------------------------------------- #
# 2. Build the worker agents once (reused across invocations)                  #
# --------------------------------------------------------------------------- #
_data_collector = build_data_collector_agent()
_code_generator = build_code_generator_agent()
_report_generator = build_report_generator_agent()


# --------------------------------------------------------------------------- #
# 3. Helpers                                                                    #
# --------------------------------------------------------------------------- #
def _final_text(result: dict) -> str:
    """Extract the final assistant text from a sub-agent's result."""
    last = result["messages"][-1]
    if isinstance(getattr(last, "content", None), str) and last.content:
        return last.content
    blocks = getattr(last, "content_blocks", None)
    if blocks:
        return "".join(b.text for b in blocks if hasattr(b, "text"))
    return str(last)


def _last_tool_output(result: dict) -> Optional[str]:
    """Return the content of the LAST tool call a sub-agent made.

    This is how we capture an agent's structured artifact: the data JSON, the
    'Chart saved to: ...' line, or the 'Report saved to: ...' line — none of
    which are reliably in the agent's final natural-language message.
    """
    for msg in reversed(result["messages"]):
        if isinstance(msg, ToolMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return None


def _path_after(marker: str, text: Optional[str]) -> Optional[str]:
    """Parse a path out of e.g. 'Chart saved to: C:/.../chart.png'."""
    if text and marker in text:
        return text.split(marker, 1)[1].strip()
    return None


def _build_worker_input(state: AnalystState) -> str:
    """Construct a FOCUSED prompt for a worker.

    We deliberately do NOT dump the entire conversation into every worker.
    Instead each worker gets: the supervisor's task + the latest user request +
    whatever typed artifacts already exist. This keeps worker context small and
    relevant while still covering the case where the user pasted data/a chart in.
    """
    last_user = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_user = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    parts = [f"Task: {state.get('task', '')}", f"\nUser request:\n{last_user}"]
    if state.get("collected_data"):
        parts.append(f"\nAvailable OHLCV data (JSON):\n{state['collected_data']}")
    if state.get("chart_path"):
        parts.append(f"\nAvailable chart image path: {state['chart_path']}")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# 4a. Authentication node — runs first, establishes the principal              #
# --------------------------------------------------------------------------- #
def authenticate_node(state: AnalystState, config) -> dict:
    """Verify the caller's Auth0 token (if any) and store the Principal in state.

    The raw token is passed in via config["configurable"]["jwt"]. No token means
    an anonymous principal (still allowed to use the open agents).
    """
    token = (config or {}).get("configurable", {}).get("jwt")
    try:
        principal = authenticate(token)
        return {"principal": principal.to_dict(), "auth_error": None}
    except AuthenticationError as exc:
        # A bad token is treated as anonymous, but we record why so the supervisor
        # / caller can explain it. (We do NOT raise — report_generator is open.)
        from src.middleware.authentication import anonymous_principal
        return {"principal": anonymous_principal().to_dict(), "auth_error": str(exc)}


# --------------------------------------------------------------------------- #
# 4a-bis. Rate-limit node — caps how often a user can start a run              #
# --------------------------------------------------------------------------- #
def rate_limit_node(state: AnalystState) -> dict:
    """Per-user token-bucket check at the workflow entry.

    Keyed by the principal's user_id (anonymous users share one 'anonymous'
    bucket — in a real deployment you'd key those by client IP instead).
    """
    principal = Principal.from_dict(state.get("principal"))
    key = principal.user_id or "anonymous"
    result = get_default_limiter().allow(key)
    if result.allowed:
        return {"rate_limited": False}
    return {
        "rate_limited": True,
        "messages": [AIMessage(
            content=f"[rate_limit] Rate limit exceeded for '{key}'. "
                    f"Try again in {result.retry_after:.1f}s."
        )],
    }


def after_rate_limit(state: AnalystState) -> str:
    """Conditional edge: stop early if rate limited, else proceed to supervisor."""
    return "blocked" if state.get("rate_limited") else "ok"


# --------------------------------------------------------------------------- #
# 4b. Supervisor node — the router                                             #
# --------------------------------------------------------------------------- #
class _Router(TypedDict):
    """Structured decision the supervisor LLM must return."""
    next: Literal["data_collector", "code_generator", "report_generator", "FINISH"]
    task: str


# Built once: the supervisor LLM constrained to emit a _Router decision.
_supervisor_llm = get_foundation_model().with_structured_output(_Router)


_WORKERS_DOC = """\
- data_collector: downloads historical OHLCV market data from Yahoo Finance for a ticker.
- code_generator: writes Python (plotly) code and runs it to produce a chart image (PNG).
- report_generator: writes a Markdown financial analysis report (can embed a chart).
"""

_SUPERVISOR_SYSTEM = """You are the supervisor of a stock-analysis team. Based on the \
conversation and the artifacts already produced, decide which ONE worker should act \
next, or FINISH when the user's request is fully satisfied.

Workers:
{workers}

Routing rules:
- Only collect data if it is needed and not already available.
- The user's request is DYNAMIC. They might supply data or a chart themselves and ask \
only for the next step. Do not redo work that is already done.
- Route to exactly one worker at a time. After it runs you will be asked again.
- When the request is satisfied (the asked-for artifact exists), return next=FINISH.

ACCESS CONTROL (must obey):
- This user may ONLY use these agents: {permitted}.
- Never route to an agent outside that list. If the request needs a forbidden \
agent, do NOT route to it — instead return next=FINISH (the system will explain \
the access restriction to the user).
- Agents already denied this run: {denied}.

Current artifact status:
- collected_data available: {has_data}
- chart available: {has_chart}
- report available: {has_report}
{preferences}
In `task`, give the chosen worker a short, specific instruction."""


def _load_preferences(config) -> str:
    """Read long-term user preferences from the store, if one is configured.

    Returns a short text block to inject into the supervisor prompt. Safe to call
    even when no store / no user_id is set — returns an empty string.
    """
    try:
        from langgraph.config import get_store
        store = get_store()
    except Exception:
        return ""

    user_id = (config or {}).get("configurable", {}).get("user_id")
    if not user_id:
        return ""

    items = store.search(("preferences", user_id))
    if not items:
        return ""
    prefs = "; ".join(f"{i.value.get('key')}={i.value.get('value')}" for i in items)
    return f"\nKnown user preferences (long-term memory): {prefs}\n"


def supervisor_node(state: AnalystState, config) -> dict:
    principal = Principal.from_dict(state.get("principal"))
    allowed = permitted_agents(principal)
    system = _SUPERVISOR_SYSTEM.format(
        workers=_WORKERS_DOC,
        permitted=", ".join(allowed) or "none",
        denied=", ".join(state.get("denied", [])) or "none",
        has_data=bool(state.get("collected_data")),
        has_chart=bool(state.get("chart_path")),
        has_report=bool(state.get("report_path")),
        preferences=_load_preferences(config),
    )
    decision: _Router = _supervisor_llm.invoke(
        [{"role": "system", "content": system}, *state["messages"]]
    )
    # Hard backstop: even if the LLM ignores the rule, never route to a forbidden
    # agent. Redirect a disallowed choice to FINISH.
    choice = decision["next"]
    if choice != "FINISH" and choice not in allowed:
        return {"next": "FINISH", "task": decision["task"]}
    return {"next": choice, "task": decision["task"]}


def route_decision(state: AnalystState) -> str:
    """Conditional-edge function: send the graph to the supervisor's choice."""
    return state["next"]


# --------------------------------------------------------------------------- #
# 5. Worker wrapper nodes                                                       #
# --------------------------------------------------------------------------- #
def _deny(agent_name: str, reason: str) -> dict:
    """Standard state update when a guard blocks an agent."""
    return {
        "denied": [agent_name],
        "messages": [AIMessage(content=f"[{agent_name}] ACCESS DENIED: {reason}")],
    }


def data_collector_node(state: AnalystState) -> dict:
    # SECURITY BOUNDARY: deterministic guard, runs before any work / LLM call.
    principal = Principal.from_dict(state.get("principal"))
    allowed, reason = check_access(principal, "data_collector")
    if not allowed:
        return _deny("data_collector", reason)

    result = _data_collector.invoke(
        {"messages": [{"role": "user", "content": _build_worker_input(state)}]}
    )
    raw_json = _last_tool_output(result)            # the JSON the tool returned
    summary = _final_text(result)
    return {
        "collected_data": raw_json,
        "messages": [AIMessage(content=f"[data_collector] {summary}")],
    }


def code_generator_node(state: AnalystState) -> dict:
    # SECURITY BOUNDARY: deterministic guard, runs before any work / LLM call.
    principal = Principal.from_dict(state.get("principal"))
    allowed, reason = check_access(principal, "code_generator")
    if not allowed:
        return _deny("code_generator", reason)

    result = _code_generator.invoke(
        {"messages": [{"role": "user", "content": _build_worker_input(state)}]}
    )
    tool_out = _last_tool_output(result)
    chart_path = _path_after("Chart saved to:", tool_out)
    return {
        "chart_path": chart_path,
        "messages": [AIMessage(content=f"[code_generator] {_final_text(result)}")],
    }


def report_generator_node(state: AnalystState) -> dict:
    # report_generator is open to everyone, but we still call the guard so the
    # policy stays in ONE place (AGENT_POLICY) rather than hard-coded here.
    principal = Principal.from_dict(state.get("principal"))
    allowed, reason = check_access(principal, "report_generator")
    if not allowed:
        return _deny("report_generator", reason)

    result = _report_generator.invoke(
        {"messages": [{"role": "user", "content": _build_worker_input(state)}]}
    )
    tool_out = _last_tool_output(result)
    report_path = _path_after("Report saved to:", tool_out)
    return {
        "report_path": report_path,
        "messages": [AIMessage(content=f"[report_generator] {_final_text(result)}")],
    }


# --------------------------------------------------------------------------- #
# 6. Assemble the graph                                                         #
# --------------------------------------------------------------------------- #
def build_analyst_workflow(
    checkpointer: Optional[object] = None,
    store: Optional[object] = None,
):
    """Compile and return the supervisor-orchestrated analyst workflow.

    Args:
        checkpointer: short-term memory (thread-scoped). Defaults to InMemorySaver.
        store:        long-term memory (cross-thread). Defaults to InMemoryStore.

    Pass your own (e.g. a Postgres saver/store) in production.
    """
    # Set up OTEL tracing if enabled in .env (no-op otherwise). Idempotent.
    configure_tracing()

    graph = StateGraph(AnalystState)

    graph.add_node("authenticate", authenticate_node)
    graph.add_node("rate_limit", rate_limit_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("data_collector", data_collector_node)
    graph.add_node("code_generator", code_generator_node)
    graph.add_node("report_generator", report_generator_node)

    # Auth runs first (establishes the principal), then the rate-limit gate caps
    # how often that principal can start a run. Only then do we reach the
    # supervisor, whose routing the principal also governs.
    graph.add_edge(START, "authenticate")
    graph.add_edge("authenticate", "rate_limit")
    graph.add_conditional_edges(
        "rate_limit",
        after_rate_limit,
        {"ok": "supervisor", "blocked": END},
    )
    graph.add_conditional_edges(
        "supervisor",
        route_decision,
        {
            "data_collector": "data_collector",
            "code_generator": "code_generator",
            "report_generator": "report_generator",
            "FINISH": END,
        },
    )
    # Every worker reports back to the supervisor, which decides the next step.
    graph.add_edge("data_collector", "supervisor")
    graph.add_edge("code_generator", "supervisor")
    graph.add_edge("report_generator", "supervisor")

    return graph.compile(
        checkpointer=checkpointer or InMemorySaver(),
        store=store or InMemoryStore(),
    )
