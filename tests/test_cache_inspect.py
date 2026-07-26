"""Write a cache entry to Redis and leave it so you can inspect it.

Run:  python tests/test_cache_inspect.py

Open Redis Insight / redis-cli and look for keys matching  llmcache:inspect:*
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env.dev"))

from src import config
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_redis import RedisSemanticCache


class KeywordEmbeddings(Embeddings):
    def _vec(self, text):
        return [1.0, 0.0] if "apple" in text.lower() else [0.0, 1.0]
    def embed_query(self, text): return self._vec(text)
    def embed_documents(self, texts): return [self._vec(t) for t in texts]


def main():
    url = config.REDIS_URL
    if not url:
        print("REDIS_URL not set in .env.dev"); return

    cache = RedisSemanticCache(
        embeddings=KeywordEmbeddings(),
        redis_url=url,
        distance_threshold=0.1,
        name="llmcache:inspect",   # fixed name so you can find it in Redis
    )
    cache.clear()  # start fresh for this demo

    llm = FakeListChatModel(responses=["AAPL-RESPONSE", "BANANA-RESPONSE"])
    set_llm_cache(cache)

    r1 = llm.invoke("What is Apple stock price today?")
    r2 = llm.invoke("Tell me the price of apple shares")   # HIT — same vector
    r3 = llm.invoke("How is the banana market doing?")     # MISS

    print(f"r1 (miss)  : {r1.content}")
    print(f"r2 (HIT)   : {r2.content}   <- came from Redis, no LLM call")
    print(f"r3 (miss)  : {r3.content}")

    # ── inspect what is in Redis ─────────────────────────────────────── #
    import redis as _redis
    r = _redis.Redis.from_url(url, decode_responses=True)

    keys = r.keys("llmcache:inspect*")
    print(f"\nKeys in Redis ({len(keys)} total):")
    for k in sorted(keys):
        t = r.type(k)
        print(f"  {k}  [{t}]")

    # NOTE: we do NOT call cache.clear() here, so the data stays in Redis.
    # Run `python tests/test_cache_inspect.py` again to see hits in your dashboard.
    set_llm_cache(None)
    print("\nData left in Redis — check your Redis Insight / dashboard now.")


if __name__ == "__main__":
    main()
