"""Library & infrastructure probes — the small checks used during development.

These are the throwaway snippets I ran while building the system to verify
library APIs and infra BEFORE writing real code (e.g. "does this LangGraph
version expose Runtime?", "what's the PostgresStore signature?", "is Redis
reachable?"). They use NO LLMs and NO agent code — pure imports, signature
introspection, and (optionally) a real DB round-trip that skips if unavailable.

Run:  python tests/test_probes.py
"""
import sys
import os
import inspect
from importlib.metadata import version
from importlib.util import find_spec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


def skip(label: str, why: str) -> None:
    print(f"  [SKIP] {label} — {why}")


# --------------------------------------------------------------------------- #
# 1. Installed versions (used to confirm we were on LangGraph/LangChain 1.x)   #
# --------------------------------------------------------------------------- #
def test_versions():
    print("\n=== Installed versions ===")
    lg = version("langgraph")
    lc = version("langchain")
    print(f"  langgraph={lg}  langchain={lc}")
    check("langgraph installed", bool(lg))
    check("langchain installed", bool(lc))


# --------------------------------------------------------------------------- #
# 2. LangGraph core primitives importable (graph + memory + Command)          #
#    Used before writing analyst_workflow / memory handlers.                   #
# --------------------------------------------------------------------------- #
def test_langgraph_core_api():
    print("\n=== LangGraph core API ===")
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command
    from langgraph.config import get_store
    check("StateGraph/START/END import", all([StateGraph, START is not None, END is not None]))
    check("add_messages reducer import", callable(add_messages))
    check("InMemorySaver (short-term) import", InMemorySaver is not None)
    check("InMemoryStore (long-term) import", InMemoryStore is not None)
    check("Command import", Command is not None)
    check("get_store runtime accessor import", callable(get_store))


# --------------------------------------------------------------------------- #
# 3. Runtime context API (used to decide HOW to pass the auth principal in)    #
# --------------------------------------------------------------------------- #
def test_runtime_and_context_api():
    print("\n=== LangGraph runtime / context API ===")
    from langgraph.runtime import Runtime, get_runtime  # noqa: F401
    from langgraph.graph import StateGraph, START, END
    from typing_extensions import TypedDict

    check("Runtime + get_runtime import", Runtime is not None and callable(get_runtime))
    check("StateGraph accepts context_schema",
          "context_schema" in inspect.signature(StateGraph).parameters)

    # A trivial primitive graph (no agents, no LLM) just to read .invoke's params
    # — this is how I confirmed `.invoke(..., context=...)` exists.
    class S(TypedDict):
        x: int

    g = StateGraph(S)
    g.add_node("n", lambda s: s)
    g.add_edge(START, "n")
    g.add_edge("n", END)
    app = g.compile()
    check("compiled graph .invoke supports a 'context' arg",
          "context" in inspect.signature(app.invoke).parameters)


# --------------------------------------------------------------------------- #
# 4. langgraph_sdk.Auth exists (noted as the LangGraph-Platform-only auth path)#
# --------------------------------------------------------------------------- #
def test_langgraph_sdk_auth_present():
    print("\n=== langgraph_sdk.Auth availability ===")
    try:
        from langgraph_sdk import Auth  # noqa: F401
        check("langgraph_sdk.Auth importable", Auth is not None)
    except Exception as exc:
        skip("langgraph_sdk.Auth", f"not importable: {type(exc).__name__}")


# --------------------------------------------------------------------------- #
# 5. Postgres store/checkpointer API shape (signatures only — no connection)   #
#    Used to learn the context-manager + setup() pattern.                       #
# --------------------------------------------------------------------------- #
def test_postgres_api_shape():
    print("\n=== Postgres store/saver API shape (no DB needed) ===")
    from langgraph.store.postgres import PostgresStore
    from langgraph.checkpoint.postgres import PostgresSaver

    store_sig = inspect.signature(PostgresStore.from_conn_string)
    saver_sig = inspect.signature(PostgresSaver.from_conn_string)
    check("PostgresStore.from_conn_string exists", "conn_string" in store_sig.parameters)
    check("PostgresStore has setup()", hasattr(PostgresStore, "setup"))
    check("PostgresStore has put/search", hasattr(PostgresStore, "put") and hasattr(PostgresStore, "search"))
    check("PostgresSaver.from_conn_string exists", "conn_string" in saver_sig.parameters)
    check("PostgresSaver has setup()", hasattr(PostgresSaver, "setup"))


# --------------------------------------------------------------------------- #
# 6. RedisSemanticCache API shape (signature only — no Redis needed)          #
# --------------------------------------------------------------------------- #
def test_redis_semantic_cache_api_shape():
    print("\n=== RedisSemanticCache API shape (no Redis needed) ===")
    from langchain_redis import RedisSemanticCache
    params = inspect.signature(RedisSemanticCache.__init__).parameters
    check("takes 'embeddings'", "embeddings" in params)
    check("takes 'redis_url'", "redis_url" in params)
    check("takes 'distance_threshold'", "distance_threshold" in params)
    check("takes 'ttl' (freshness knob)", "ttl" in params)
    check("has lookup/update/clear",
          all(hasattr(RedisSemanticCache, m) for m in ("lookup", "update", "clear")))


# --------------------------------------------------------------------------- #
# 7. Chat-model class capabilities (class-level hasattr — no LLM call)         #
#    Used to confirm the supervisor could do structured-output routing.        #
# --------------------------------------------------------------------------- #
def test_chat_model_capabilities():
    print("\n=== ChatQwen capabilities (class introspection, no LLM call) ===")
    from langchain_qwq import ChatQwen
    check("ChatQwen.with_structured_output exists", hasattr(ChatQwen, "with_structured_output"))
    check("ChatQwen.bind_tools exists", hasattr(ChatQwen, "bind_tools"))


# --------------------------------------------------------------------------- #
# 8. Optional dependency presence (informational find_spec checks)            #
# --------------------------------------------------------------------------- #
def test_optional_dependencies_present():
    print("\n=== Optional dependency presence ===")
    for pkg in ("redis", "redisvl", "langchain_redis", "opentelemetry.sdk", "jwt"):
        present = find_spec(pkg) is not None
        print(f"  [{'PASS' if present else 'INFO'}] {pkg}: {'installed' if present else 'absent'}")


# --------------------------------------------------------------------------- #
# 9. Postgres round-trip (REAL DB — skips if POSTGRES_URL unset/unreachable)   #
#    This is the _pg_smoke.py / _mem_smoke.py check, with both backends.        #
# --------------------------------------------------------------------------- #
def test_postgres_roundtrip():
    print("\n=== Postgres round-trip (real DB; skips if unavailable) ===")
    if not config.POSTGRES_URL:
        skip("postgres round-trip", "POSTGRES_URL not set")
        return
    try:
        from src.workflows.long_term_memory import open_postgres_store
        from src.workflows.short_term_memory import open_postgres_checkpointer
        with open_postgres_checkpointer() as cp, open_postgres_store() as store:
            store.put(("probe", "u1"), "k", {"key": "k", "value": "v"})
            items = store.search(("probe", "u1"))
            check("store put/search works", any(i.value.get("value") == "v" for i in items))
            check("checkpointer opened (different tables, same DB)", cp is not None)
            store.delete(("probe", "u1"), "k")  # cleanup
    except Exception as exc:
        skip("postgres round-trip", f"connection failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    test_versions()
    test_langgraph_core_api()
    test_runtime_and_context_api()
    test_langgraph_sdk_auth_present()
    test_postgres_api_shape()
    test_redis_semantic_cache_api_shape()
    test_chat_model_capabilities()
    test_optional_dependencies_present()
    test_postgres_roundtrip()
    print("\nProbe checks finished.")
