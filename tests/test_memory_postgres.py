"""Integration test — verifies that short_term and long_term tables exist in
Supabase and that data written through the memory helpers actually appears there.

Run with:
    pytest tests/test_memory_postgres.py -v -s
"""
import os
import uuid
from urllib.parse import unquote, urlparse

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.dev"))

from src.workflows.long_term_memory import (
    delete_long_term,
    load_long_term,
    save_long_term,
    setup_long_term_table,
)
from src.workflows.short_term_memory import (
    delete_short_term,
    load_short_term,
    save_short_term,
    setup_short_term_table,
)


# ── helpers ──────────────────────────────────────────────────────────────── #

def _connect() -> psycopg.Connection:
    """Connect using urlparse so passwords containing '@' are handled correctly."""
    url = os.getenv("POSTGRES_URL")
    if not url:
        pytest.skip("POSTGRES_URL not set in .env.dev")
    p = urlparse(url)
    return psycopg.connect(
        host=p.hostname,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=unquote(p.password or ""),
        sslmode="require",
    )


# ── table setup ──────────────────────────────────────────────────────────── #

def test_setup_short_term_table():
    """setup_short_term_table() creates the table (idempotent, no error if run twice)."""
    setup_short_term_table()
    with _connect() as conn:
        result = conn.execute(
            "SELECT to_regclass('public.short_term')"
        ).fetchone()
    assert result[0] is not None, "short_term table was not created"
    print("\n  short_term table: OK")


def test_setup_long_term_table():
    """setup_long_term_table() creates the table (idempotent)."""
    setup_long_term_table()
    with _connect() as conn:
        result = conn.execute(
            "SELECT to_regclass('public.long_term')"
        ).fetchone()
    assert result[0] is not None, "long_term table was not created"
    print("\n  long_term table: OK")


# ── short_term round-trip ─────────────────────────────────────────────────── #

def test_short_term_save_and_load():
    """save_short_term → load_short_term round-trip, then verify via raw psycopg."""
    tid = f"test-thread-{uuid.uuid4()}"
    messages = [
        {"role": "human", "content": "Analyze AAPL"},
        {"role": "ai",    "content": "Here is the AAPL analysis…"},
    ]

    save_short_term(tid, messages, state={"step": "done"}, step=0)

    rows = load_short_term(tid)
    assert len(rows) == 1
    assert rows[0]["messages"] == messages
    assert rows[0]["state"] == {"step": "done"}

    # Verify via raw SQL so we know it's actually in Supabase.
    with _connect() as conn:
        raw = conn.execute(
            "SELECT thread_id, messages, saved_at FROM short_term WHERE thread_id = %s",
            (tid,),
        ).fetchall()
    assert len(raw) == 1
    assert raw[0][0] == tid
    print(f"\n  short_term row:")
    print(f"    thread_id : {raw[0][0]}")
    print(f"    messages  : {raw[0][1]}")
    print(f"    saved_at  : {raw[0][2]}")

    # Cleanup.
    delete_short_term(tid)
    assert load_short_term(tid) == []


def test_short_term_upsert():
    """Writing to the same (thread_id, step) overwrites the row."""
    tid = f"test-thread-{uuid.uuid4()}"
    save_short_term(tid, [{"role": "human", "content": "v1"}], step=0)
    save_short_term(tid, [{"role": "human", "content": "v2"}], step=0)

    rows = load_short_term(tid)
    assert len(rows) == 1
    assert rows[0]["messages"][0]["content"] == "v2"

    delete_short_term(tid)


# ── long_term round-trip ──────────────────────────────────────────────────── #

def test_long_term_save_and_load():
    """save_long_term → load_long_term round-trip, then verify via raw psycopg."""
    uid = f"test-user-{uuid.uuid4()}"

    save_long_term(uid, "watchlist", {"tickers": ["AAPL", "MSFT", "NVDA"]})
    save_long_term(uid, "risk_profile", {"level": "moderate", "horizon_years": 5})

    rows = load_long_term(uid)
    assert len(rows) == 2
    keys = {r["key"] for r in rows}
    assert keys == {"watchlist", "risk_profile"}

    # Verify via raw SQL.
    with _connect() as conn:
        raw = conn.execute(
            "SELECT user_id, key, value, updated_at FROM long_term WHERE user_id = %s ORDER BY key",
            (uid,),
        ).fetchall()
    assert len(raw) == 2
    print(f"\n  long_term rows for user {uid}:")
    for r in raw:
        print(f"    key={r[1]}  value={r[2]}  updated_at={r[3]}")

    # Cleanup.
    delete_long_term(uid)
    assert load_long_term(uid) == []


def test_long_term_upsert():
    """Saving the same key overwrites the value."""
    uid = f"test-user-{uuid.uuid4()}"
    save_long_term(uid, "pref", {"color": "blue"})
    save_long_term(uid, "pref", {"color": "red"})

    rows = load_long_term(uid)
    assert len(rows) == 1
    assert rows[0]["value"]["color"] == "red"

    delete_long_term(uid, "pref")


def test_long_term_delete_single_key():
    """delete_long_term(uid, key) removes only that key."""
    uid = f"test-user-{uuid.uuid4()}"
    save_long_term(uid, "a", {"x": 1})
    save_long_term(uid, "b", {"x": 2})

    delete_long_term(uid, "a")
    rows = load_long_term(uid)
    assert len(rows) == 1
    assert rows[0]["key"] == "b"

    delete_long_term(uid)
