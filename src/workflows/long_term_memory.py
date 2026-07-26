"""Long-term memory — cross-thread user memories stored in the `long_term` table.

LangGraph's native PostgresStore powers the in-workflow store API (table: store).
Additionally user memories are persisted to the `long_term` table so they are
directly queryable in Supabase.
"""
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

import psycopg
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from src import config


# ─── LangGraph store (powers in-workflow memory access) ───────────────────── #

def in_memory_store() -> InMemoryStore:
    """Volatile store — lost on restart.  Use for tests / local dev."""
    return InMemoryStore()


@contextmanager
def open_postgres_store(create_tables: bool = True):
    """Postgres store via POSTGRES_URL.  Build the workflow inside the `with` block."""
    if not config.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set (.env.dev).")
    with PostgresStore.from_conn_string(config.POSTGRES_URL) as store:
        if create_tables:
            store.setup()
        yield store


# ─── long_term table (Supabase-queryable) ────────────────────────────────── #

_CREATE_LONG_TERM = """
CREATE TABLE IF NOT EXISTS long_term (
    user_id     TEXT        NOT NULL,
    key         TEXT        NOT NULL,
    value       JSONB       NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, key)
);
"""


def _connect() -> psycopg.Connection:
    if not config.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set (.env.dev).")
    # urlparse uses the LAST '@' as the userinfo/host boundary, so passwords
    # that contain '@' (e.g. "Cindrell@15") are parsed correctly.
    p = urlparse(config.POSTGRES_URL)
    return psycopg.connect(
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=unquote(p.password or ""),
        sslmode="require",
    )


def setup_long_term_table() -> None:
    """Create the long_term table if it does not already exist (idempotent)."""
    with _connect() as conn:
        conn.execute(_CREATE_LONG_TERM)
        conn.commit()


_UPSERT = """
INSERT INTO long_term (user_id, key, value, updated_at)
VALUES (%s, %s, %s, %s)
ON CONFLICT (user_id, key) DO UPDATE
    SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
"""


def save_long_term(user_id: str, key: str, value: dict) -> None:
    """Write (or update) a single memory entry for a user."""
    with _connect() as conn:
        conn.execute(_UPSERT, (user_id, key, json.dumps(value), datetime.now(timezone.utc)))
        conn.commit()


def _load_key(conn: psycopg.Connection, user_id: str, key: str) -> dict | None:
    """Read one key's value within an existing connection (no commit)."""
    row = conn.execute(
        "SELECT value FROM long_term WHERE user_id = %s AND key = %s",
        (user_id, key),
    ).fetchone()
    return row[0] if row else None


def persist_session_signals(user_id: str, result: dict, prompt: str) -> None:
    """Extract meaningful signals from a completed workflow run and accumulate them.

    Stores three concepts per user — all keyed by concept name, not by session:

      watched_tickers   list of tickers the user has asked about (most recent first,
                        capped at 20). Moves a re-queried ticker to the front.

      preferred_period  frequency map  {"1wk": 3, "1mo": 1} so the UI / supervisor
                        can surface the user's usual lookback window.

      preferred_interval same idea for data granularity {"1d": 5, "1h": 1}.

    Nothing is stored if no data was collected (e.g. the run failed or the user
    only asked a question that didn't trigger the data_collector agent).
    """
    collected_data_str = result.get("collected_data")
    if not collected_data_str:
        return

    try:
        data = json.loads(collected_data_str)
    except (ValueError, TypeError):
        return

    ticker = data.get("ticker")
    period = data.get("period")
    interval = data.get("interval")

    if not ticker:
        return

    now = datetime.now(timezone.utc)

    with _connect() as conn:
        # 1. watched_tickers — deduplicated, most-recent-first, max 20
        tickers_val = _load_key(conn, user_id, "watched_tickers") or {"tickers": []}
        tickers: list = tickers_val.get("tickers", [])
        if ticker in tickers:
            tickers.remove(ticker)
        tickers.insert(0, ticker)
        conn.execute(_UPSERT, (user_id, "watched_tickers", json.dumps({"tickers": tickers[:20]}), now))

        # 2. preferred_period — frequency map
        if period:
            period_map = _load_key(conn, user_id, "preferred_period") or {}
            period_map[period] = period_map.get(period, 0) + 1
            conn.execute(_UPSERT, (user_id, "preferred_period", json.dumps(period_map), now))

        # 3. preferred_interval — frequency map
        if interval:
            interval_map = _load_key(conn, user_id, "preferred_interval") or {}
            interval_map[interval] = interval_map.get(interval, 0) + 1
            conn.execute(_UPSERT, (user_id, "preferred_interval", json.dumps(interval_map), now))

        conn.commit()


def load_long_term(user_id: str) -> list[dict]:
    """Return all memory entries for a user, newest first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT key, value, updated_at FROM long_term "
            "WHERE user_id = %s ORDER BY updated_at DESC",
            (user_id,),
        ).fetchall()
    return [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in rows]


def delete_long_term(user_id: str, key: str | None = None) -> None:
    """Remove one memory entry (specific key) or all entries for a user.

    Pass key=None to wipe all memories for the user (used by tests for cleanup).
    """
    with _connect() as conn:
        if key is None:
            conn.execute("DELETE FROM long_term WHERE user_id = %s", (user_id,))
        else:
            conn.execute(
                "DELETE FROM long_term WHERE user_id = %s AND key = %s", (user_id, key)
            )
        conn.commit()
