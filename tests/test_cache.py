"""Standalone test for the semantic LLM cache.

Strategy: we use FAKE embeddings and a FAKE chat model so the test needs no
DashScope/network — but a REAL Redis (Stack) so it genuinely exercises the
vector-search cache. The fake embeddings map any 'apple' text to one vector and
everything else to another, so two different 'apple' phrasings are semantically
identical and must produce a cache HIT (only ONE underlying model call).

Prerequisites:
    docker run -d -p 6379:6379 redis/redis-stack:latest
    set REDIS_URL=redis://localhost:6379   (in env or .env.dev)

Run:  python tests/test_cache.py
If REDIS_URL is unset or Redis is unreachable, the test SKIPS (not fails).
"""
import sys
import os
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import config

from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel


class KeywordEmbeddings(Embeddings):
    """Deterministic 2-D embeddings: 'apple' -> [1,0], else -> [0,1]."""
    def _vec(self, text: str):
        return [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]

    def embed_query(self, text: str):
        return self._vec(text)

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


def _redis_reachable(url: str) -> bool:
    try:
        import redis
        redis.Redis.from_url(url).ping()
        return True
    except Exception as exc:
        print(f"  (Redis not reachable: {exc})")
        return False


def test_semantic_cache():
    print("\n=== Semantic LLM cache (paraphrase -> cache hit) ===")
    url = config.REDIS_URL
    if not url:
        print("  SKIP: REDIS_URL not set. Start Redis Stack and set REDIS_URL to run this test.")
        return
    if not _redis_reachable(url):
        print("  SKIP: Redis not reachable.")
        return

    from langchain_redis import RedisSemanticCache

    cache = RedisSemanticCache(
        embeddings=KeywordEmbeddings(),
        redis_url=url,
        distance_threshold=0.1,
        name=f"test_llmcache_{uuid.uuid4().hex[:8]}",  # isolate from other runs
    )
    cache.clear()
    set_llm_cache(cache)

    try:
        # Distinct responses so we can tell a cache HIT (reused) from a MISS (new).
        llm = FakeListChatModel(responses=["FIRST", "SECOND", "THIRD"])

        r1 = llm.invoke("What is Apple stock price today?")      # miss -> FIRST
        r2 = llm.invoke("Tell me the price of apple shares")     # apple == apple -> HIT
        r3 = llm.invoke("How is the banana market doing?")       # different vector -> miss

        check("1st call is a miss -> 'FIRST'", r1.content == "FIRST")
        check("2nd call (paraphrase) is a HIT -> reuses 'FIRST'", r2.content == "FIRST")
        check("3rd call (different topic) is a miss -> next response 'SECOND'",
              r3.content == "SECOND")
        print("  => the paraphrase did NOT consume a new model response: LLM call saved.")
    finally:
        cache.clear()
        set_llm_cache(None)


if __name__ == "__main__":
    test_semantic_cache()
    print("\nCache test finished.")
