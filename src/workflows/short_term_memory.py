"""Short-term memory handlers (the LangGraph *checkpointer*).

Short-term memory is scoped to a `thread_id` (one conversation). It persists the
WHOLE graph state after every step, which gives you:
  * multi-turn conversations (a follow-up turn reuses earlier results),
  * crash recovery / resume,
  * human-in-the-loop pauses.

Every backend below exposes the SAME interface, so the workflow code never
changes when you swap one for another.

Tables: the Postgres checkpointer uses its own `checkpoints`, `checkpoint_blobs`
and `checkpoint_writes` tables — separate from the long-term store's `store`
table (see long_term_memory.py), so both can share one database safely.
"""
from contextlib import contextmanager

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

from src import config


def in_memory_checkpointer() -> InMemorySaver:
    """Volatile short-term memory — lost on restart. Good for tests/local dev."""
    return InMemorySaver()


@contextmanager
def open_postgres_checkpointer(create_tables: bool = True):
    """Postgres-backed short-term memory, using POSTGRES_URL from .env.dev.

    The DB connection lives for the `with` block, so build and use the workflow
    inside it:

        with open_postgres_checkpointer() as cp:
            app = build_analyst_workflow(checkpointer=cp)
            app.invoke(..., {"configurable": {"thread_id": "t1"}})

    Args:
        create_tables: run `cp.setup()` (idempotent) to create the checkpoint
            tables if missing. Set False to skip the check once they exist.
    """
    if not config.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set (.env.dev).")

    with PostgresSaver.from_conn_string(config.POSTGRES_URL) as checkpointer:
        if create_tables:
            checkpointer.setup()
        yield checkpointer


# --------------------------------------------------------------------------- #
# Alternative backend: Redis (uncomment to use)                                #
# --------------------------------------------------------------------------- #
# Install:  uv add langgraph-checkpoint-redis
# Set REDIS_URL in .env.dev, e.g. redis://localhost:6379
#
# from langgraph.checkpoint.redis import RedisSaver
#
# @contextmanager
# def open_redis_checkpointer():
#     """Redis-backed short-term memory, using REDIS_URL from .env.dev."""
#     load_dotenv(".env.dev")
#     url = os.getenv("REDIS_URL")
#     if not url:
#         raise RuntimeError("REDIS_URL is not set (.env.dev).")
#     with RedisSaver.from_conn_string(url) as checkpointer:
#         checkpointer.setup()
#         yield checkpointer


# --------------------------------------------------------------------------- #
# Alternative backend: SQLite (uncomment to use)                               #
# --------------------------------------------------------------------------- #
# Install:  uv add langgraph-checkpoint-sqlite
#
# from langgraph.checkpoint.sqlite import SqliteSaver
#
# @contextmanager
# def open_sqlite_checkpointer(db_path: str = "checkpoints.sqlite"):
#     """File-based short-term memory — simplest persistent option, no server."""
#     with SqliteSaver.from_conn_string(db_path) as checkpointer:
#         yield checkpointer
