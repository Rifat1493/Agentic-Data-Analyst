"""Caching middleware — Redis-backed semantic LLM cache.

Responses that contain tool_calls are never cached. Storing them would
embed stale tool_call_ids that break LangGraph's routing on the next hit.
Only pure text-generation responses (final summaries, routing decisions)
are cached — which is where the real LLM savings are anyway.

Environment (.env.dev)
----------------------
    REDIS_URL                 e.g. redis://default:pass@host:port
    SEMANTIC_CACHE_DISTANCE   cosine-distance threshold, default 0.1 (strict)
    SEMANTIC_CACHE_TTL        seconds; default unset (no expiry)
"""
import logging
from typing import Sequence

from langchain_core.globals import set_llm_cache
from langchain_core.outputs import ChatGeneration, Generation
from langchain_core.messages import AIMessage
from langchain_redis import RedisSemanticCache
from langchain_huggingface import HuggingFaceEmbeddings

from src import config

log = logging.getLogger(__name__)


def _has_tool_calls(return_val: Sequence[Generation]) -> bool:
    """Return True if any generation contains tool_calls.

    Checks both the normalized AIMessage.tool_calls attribute and
    additional_kwargs['tool_calls'] because ChatQwen may populate either.
    """
    for gen in return_val:
        if isinstance(gen, ChatGeneration):
            msg = gen.message
            if isinstance(msg, AIMessage):
                if msg.tool_calls:
                    return True
                if msg.additional_kwargs.get("tool_calls"):
                    return True
    return False


class SafeSemanticCache(RedisSemanticCache):
    """RedisSemanticCache that never stores or returns responses with tool_calls.

    A cached AIMessage with tool_calls carries stale tool_call_ids from a
    prior run. When LangGraph sees those IDs it thinks the tools already ran,
    hits the fallback branch, and raises KeyError: 'model'.

    Both write (update) and read (lookup) are guarded so old Redis entries
    from before this fix are also ignored.
    """

    def lookup(self, prompt: str, llm_string: str):
        result = super().lookup(prompt, llm_string)
        if result and _has_tool_calls(result):
            log.debug("Cache: ignoring hit — stored response contains tool_calls.")
            return None
        return result

    async def alookup(self, prompt: str, llm_string: str):
        result = await super().alookup(prompt, llm_string)
        if result and _has_tool_calls(result):
            log.debug("Cache: ignoring hit — stored response contains tool_calls.")
            return None
        return result

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if _has_tool_calls(return_val):
            log.debug("Cache: skipping write — response contains tool_calls.")
            return
        super().update(prompt, llm_string, return_val)

    async def aupdate(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if _has_tool_calls(return_val):
            log.debug("Cache: skipping write — response contains tool_calls.")
            return
        await super().aupdate(prompt, llm_string, return_val)


def build_semantic_cache() -> SafeSemanticCache:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    if not config.REDIS_URL:
        raise RuntimeError("REDIS_URL is not set.")
    return SafeSemanticCache(
        embeddings=embeddings,
        redis_url=config.REDIS_URL,
        distance_threshold=config.SEMANTIC_CACHE_DISTANCE,
        ttl=config.SEMANTIC_CACHE_TTL,
    )


def enable_global_cache() -> str:
    """Install the semantic Redis cache globally for all LLM calls.

    Raises if Redis is unavailable or embeddings fail.
    Returns "semantic" on success.
    """
    if not config.REDIS_URL:
        raise RuntimeError("REDIS_URL is not set.")
    cache = build_semantic_cache()
    set_llm_cache(cache)
    log.info("Semantic LLM cache enabled (Redis + HuggingFace embeddings).")
    return "semantic"


# Alias only — do NOT call here; the actual activation happens in app.py lifespan.
enable_global_semantic_cache = enable_global_cache
