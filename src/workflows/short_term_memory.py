"""Short-term memory — per-thread conversation checkpoints.

LangGraph's native PostgresSaver powers multi-turn / HITL inside the workflow
(tables: checkpoints, checkpoint_blobs, checkpoint_writes).  Additionally each
completed workflow run is persisted to the `short_term` table so the history is
directly queryable in Supabase.
"""
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from urllib.parse import unquote, urlparse

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from src import config


# ─── LangGraph checkpointer (powers multi-turn / HITL) ───────────────────── #

def in_memory_checkpointer() -> InMemorySaver:
    """Volatile checkpointer — lost on restart.  Use for tests / local dev."""
    return InMemorySaver()


@contextmanager
def open_postgres_checkpointer(create_tables: bool = True):
    """Postgres checkpointer via POSTGRES_URL.  Build the workflow inside the `with` block."""
    if not config.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set (.env.dev).")
    with PostgresSaver.from_conn_string(config.POSTGRES_URL) as checkpointer:
        if create_tables:
            checkpointer.setup()
        yield checkpointer


# ─── short_term table (Supabase-queryable) ───────────────────────────────── #

_CREATE_SHORT_TERM = """
CREATE TABLE IF NOT EXISTS short_term (
    thread_id   TEXT        NOT NULL,
    step        INT         NOT NULL DEFAULT 0,
    messages    JSONB       NOT NULL DEFAULT '[]',
    state       JSONB       NOT NULL DEFAULT '{}',
    saved_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, step)
);
"""


def _connect() -> psycopg.Connection:
    if not config.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set (.env.dev).")
    # urlparse uses the LAST '@' as the userinfo/host boundary, so passwords
    # that contain '@' (e.g. "Cindrell@15") are parsed correctly.  psycopg's
    # own URL parser splits on the FIRST '@' and would mangle such passwords.
    p = urlparse(config.POSTGRES_URL)
    return psycopg.connect(
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=unquote(p.password or ""),
        sslmode="require",
    )


def setup_short_term_table() -> None:
    """Create the short_term table if it does not already exist (idempotent)."""
    with _connect() as conn:
        conn.execute(_CREATE_SHORT_TERM)
        conn.commit()


def save_short_term(
    thread_id: str,
    messages: list,
    state: dict | None = None,
    step: int = 0,
) -> None:
    """Write (or overwrite) a conversation snapshot to the short_term table."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO short_term (thread_id, step, messages, state, saved_at)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (thread_id, step) DO UPDATE
                SET messages = EXCLUDED.messages,
                    state    = EXCLUDED.state,
                    saved_at = EXCLUDED.saved_at
            """,
            (
                thread_id,
                step,
                json.dumps(messages),
                json.dumps(state or {}),
                datetime.now(timezone.utc),
            ),
        )
        conn.commit()


def load_short_term(thread_id: str) -> list[dict]:
    """Return all snapshots for a thread ordered by step."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT step, messages, state, saved_at "
            "FROM short_term WHERE thread_id = %s ORDER BY step",
            (thread_id,),
        ).fetchall()
    return [{"step": r[0], "messages": r[1], "state": r[2], "saved_at": r[3]} for r in rows]


def delete_short_term(thread_id: str) -> None:
    """Remove all snapshots for a thread (used by tests for cleanup)."""
    with _connect() as conn:
        conn.execute("DELETE FROM short_term WHERE thread_id = %s", (thread_id,))
        conn.commit()
