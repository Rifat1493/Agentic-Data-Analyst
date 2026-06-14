"""Caching middleware — a semantic LLM cache (Redis-backed) to cut LLM calls.

What this is (and is NOT)
-------------------------
This is an APPLICATION RESPONSE CACHE, the only kind that reduces the *number*
of LLM calls: we store the model's output keyed by the meaning of its input, and
on a semantically-similar input we return the stored output WITHOUT calling the
model. It is unrelated to the transformer's internal KV cache (intra-request) and
to provider prompt caching (which only lowers per-call cost, not call count).

How "semantic" works
--------------------
Each prompt is embedded into a vector. On a new prompt we vector-search Redis for
a previously cached prompt within `distance_threshold` (cosine distance: smaller =
stricter). A hit returns the cached response. This catches paraphrases that an
exact-match cache would miss ("AAPL last week" ~ "Apple, past 7 days").

Requirements
------------
* Redis Stack (RediSearch/vector module) — plain Redis will NOT work.
  Quick start:  docker run -d -p 6379:6379 redis/redis-stack:latest
* An embeddings model (we use DashScope embeddings via the OpenAI-compatible API).

Environment (.env.dev)
----------------------
    REDIS_URL                 e.g. redis://localhost:6379
    EMBEDDING_MODEL           default 'text-embedding-v3'
    DASHSCOPE_API_KEY         (already used by the chat model)
    DASHSCOPE_BASE_URL        default DashScope compatible-mode endpoint
    SEMANTIC_CACHE_DISTANCE   cosine-distance threshold, default 0.1 (strict)
    SEMANTIC_CACHE_TTL        seconds; default unset (no expiry)

IMPORTANT — scoping / correctness
---------------------------------
`enable_global_semantic_cache()` installs the cache for EVERY chat-model call,
including the supervisor's routing and the agents' tool-calling. A semantic
near-miss could therefore return a stale ROUTING decision or stale generated
code. Mitigations: keep `distance_threshold` strict (small), set a TTL, or attach
the cache to only a specific model instance via `ChatModel(cache=<cache>)` instead
of globally. For time-sensitive financial answers, always set a short TTL.
"""
from langchain_core.globals import set_llm_cache
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisSemanticCache

from src import config


def build_embeddings() -> OpenAIEmbeddings:
    """DashScope embeddings via the OpenAI-compatible endpoint.

    `check_embedding_ctx_length=False` disables OpenAI-specific tiktoken
    chunking, which is required when talking to a non-OpenAI endpoint.
    """
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        base_url=config.DASHSCOPE_BASE_URL,
        api_key=config.DASHSCOPE_API_KEY,
        check_embedding_ctx_length=False,
    )


def build_semantic_cache(embeddings=None) -> RedisSemanticCache:
    """Construct a Redis-backed semantic cache. Requires REDIS_URL."""
    if not config.REDIS_URL:
        raise RuntimeError("REDIS_URL is not set (.env.dev). Semantic cache needs Redis Stack.")

    return RedisSemanticCache(
        embeddings=embeddings or build_embeddings(),
        redis_url=config.REDIS_URL,
        distance_threshold=config.SEMANTIC_CACHE_DISTANCE,
        ttl=config.SEMANTIC_CACHE_TTL,
    )


def enable_global_semantic_cache(embeddings=None) -> RedisSemanticCache:
    """Install the semantic cache for ALL chat-model calls in the process.

    Call this ONCE at startup, before building the workflow:

        from src.middleware.cache import enable_global_semantic_cache
        enable_global_semantic_cache()
        app = build_analyst_workflow()

    Returns the cache so you can `.clear()` it in tests/admin.
    """
    cache = build_semantic_cache(embeddings)
    set_llm_cache(cache)
    return cache
