"""Long-term memory handlers (the LangGraph *store*).

Long-term memory is scoped to a `user_id` and lives ACROSS conversations and
process restarts. Use it for user preferences, watchlists, cached results, and
learned facts. Every backend below exposes the SAME interface (`put`, `get`,
`search`), so the workflow code never changes when you swap one for another.

Tables: the Postgres store uses its own `store` table — separate from the
checkpointer's `checkpoints*` tables (see short_term_memory.py), so both can
share one database without colliding.
"""
from contextlib import contextmanager

from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from src import config


def in_memory_store() -> InMemoryStore:
    """Volatile long-term store — lost on restart. Good for tests/local dev."""
    return InMemoryStore()


@contextmanager
def open_postgres_store(create_tables: bool = True):
    """Postgres-backed long-term store, using POSTGRES_URL from .env.dev.

    The DB connection lives for the `with` block, so build and use the workflow
    inside it:

        with open_postgres_store() as store:
            app = build_analyst_workflow(store=store)
            app.invoke(..., {"configurable": {"thread_id": "t1", "user_id": "u1"}})

    Args:
        create_tables: run `store.setup()` (idempotent) to create the `store`
            table if missing. Set False to skip the check once it exists.
    """
    if not config.POSTGRES_URL:
        raise RuntimeError("POSTGRES_URL is not set (.env.dev).")

    with PostgresStore.from_conn_string(config.POSTGRES_URL) as store:
        if create_tables:
            store.setup()
        yield store


# --------------------------------------------------------------------------- #
# Alternative backend: Redis (uncomment to use)                                #
# --------------------------------------------------------------------------- #
# Install:  uv add langgraph-checkpoint-redis
# Set REDIS_URL in .env.dev, e.g. redis://localhost:6379
#
# from langgraph.store.redis import RedisStore
#
# @contextmanager
# def open_redis_store(create_indexes: bool = True):
#     """Redis-backed long-term store, using REDIS_URL from .env.dev."""
#     load_dotenv(".env.dev")
#     url = os.getenv("REDIS_URL")
#     if not url:
#         raise RuntimeError("REDIS_URL is not set (.env.dev).")
#     with RedisStore.from_conn_string(url) as store:
#         if create_indexes:
#             store.setup()
#         yield store
