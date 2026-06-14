"""Rate-limiting middleware — cap how often a user can start a workflow run.

Why this lives at the workflow ENTRY
------------------------------------
One user request fans out through the supervisor into many LLM + external-API
calls (data_collector -> yfinance, code_generator -> LLM, etc.). Limiting at the
entry caps that whole amplified chain per user, which protects:
  * LLM spend / token quotas (DashScope 429s would otherwise hit everyone),
  * the external Yahoo Finance API (avoid IP bans),
  * fairness between users, and contains abuse from a leaked token.

Algorithm: token bucket
-----------------------
Each key (user) owns a bucket with `capacity` tokens that refills at
`rate_per_sec` tokens/second. Every request spends one token. Empty bucket ->
denied, with a `retry_after` hint. This naturally allows short bursts (up to
capacity) while bounding the long-run average rate.

Scope note: this in-memory limiter is per-process. With multiple worker
processes you'd back it with Redis so the limit is shared (see the commented
alternative at the bottom). The interface stays the same.
"""
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from src import config


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after: float   # seconds until the next token (0.0 when allowed)
    remaining: float     # tokens left in the bucket after this check


class TokenBucketRateLimiter:
    """Thread-safe in-memory token-bucket limiter, keyed by an arbitrary string."""

    def __init__(
        self,
        rate_per_sec: float,
        capacity: float,
        *,
        time_func: Callable[[], float] = time.monotonic,
    ):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self._time = time_func
        self._lock = threading.Lock()
        # key -> (tokens, last_refill_timestamp)
        self._buckets: Dict[str, Tuple[float, float]] = {}

    @classmethod
    def from_window(cls, max_requests: int, window_seconds: float, **kw):
        """Build from the intuitive 'N requests per window' form.

        Capacity = N (burst), refill rate = N / window (sustained).
        """
        return cls(rate_per_sec=max_requests / window_seconds,
                   capacity=max_requests, **kw)

    def allow(self, key: str, cost: float = 1.0) -> RateLimitResult:
        """Try to spend `cost` tokens for `key`. Returns a RateLimitResult."""
        now = self._time()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            # Refill based on elapsed time, capped at capacity.
            tokens = min(self.capacity, tokens + (now - last) * self.rate)

            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return RateLimitResult(True, 0.0, tokens)

            # Not enough tokens: deny and estimate when one will be available.
            deficit = cost - tokens
            retry_after = deficit / self.rate if self.rate > 0 else float("inf")
            self._buckets[key] = (tokens, now)
            return RateLimitResult(False, retry_after, tokens)

    def reset(self, key: str = None) -> None:
        """Clear one key's bucket, or all buckets when key is None (tests/admin)."""
        with self._lock:
            if key is None:
                self._buckets.clear()
            else:
                self._buckets.pop(key, None)


# --------------------------------------------------------------------------- #
# Process-wide default limiter, configured from the environment.              #
# --------------------------------------------------------------------------- #
#   RATE_LIMIT_MAX_REQUESTS  (default 5)
#   RATE_LIMIT_WINDOW_SECONDS (default 60)
_default_limiter: TokenBucketRateLimiter = None


def get_default_limiter() -> TokenBucketRateLimiter:
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = TokenBucketRateLimiter.from_window(
            config.RATE_LIMIT_MAX_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS
        )
    return _default_limiter


# --------------------------------------------------------------------------- #
# Alternative: distributed limiter for multi-process deployments (uncomment)   #
# --------------------------------------------------------------------------- #
# Install:  uv add redis
# Use a Redis INCR + EXPIRE or a Lua token-bucket script so all worker
# processes share one limit. Same `allow(key)` interface as above.
#
# import redis
# class RedisRateLimiter:
#     def __init__(self, url, max_requests, window_seconds):
#         self.r = redis.Redis.from_url(url)
#         self.max = max_requests
#         self.window = window_seconds
#     def allow(self, key, cost=1.0):
#         bucket = f"ratelimit:{key}"
#         count = self.r.incr(bucket)
#         if count == 1:
#             self.r.expire(bucket, self.window)
#         allowed = count <= self.max
#         ttl = self.r.ttl(bucket)
#         return RateLimitResult(allowed, 0.0 if allowed else ttl, max(0, self.max - count))
