"""Standalone tests for the rate-limiting middleware.

Runs FULLY OFFLINE (no LLM/network). Refill timing is made deterministic by
injecting a fake clock, so the tests are fast and not flaky.

Run:  python tests/test_rate_limiter.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.middleware.rate_limiter import TokenBucketRateLimiter


class FakeClock:
    """A controllable monotonic clock for deterministic refill tests."""
    def __init__(self):
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def check(label: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    assert condition, label


# --------------------------------------------------------------------------- #
# 1. Burst up to capacity, then deny                                          #
# --------------------------------------------------------------------------- #
def test_burst_then_deny():
    print("\n=== Burst up to capacity, then deny ===")
    clock = FakeClock()
    # 3 requests / 60s -> capacity 3, refill 0.05 tok/s.
    rl = TokenBucketRateLimiter.from_window(3, 60, time_func=clock)

    check("req 1 allowed", rl.allow("alice").allowed)
    check("req 2 allowed", rl.allow("alice").allowed)
    check("req 3 allowed", rl.allow("alice").allowed)

    r = rl.allow("alice")
    check("req 4 denied (bucket empty)", not r.allowed)
    check("retry_after is positive", r.retry_after > 0)


# --------------------------------------------------------------------------- #
# 2. Refill over time restores capacity                                       #
# --------------------------------------------------------------------------- #
def test_refill_over_time():
    print("\n=== Refill over time ===")
    clock = FakeClock()
    rl = TokenBucketRateLimiter.from_window(3, 60, time_func=clock)  # 0.05 tok/s

    for _ in range(3):
        rl.allow("bob")
    check("bob is now limited", not rl.allow("bob").allowed)

    # Advance 20s -> +1 token (0.05 * 20 = 1.0).
    clock.advance(20)
    check("after 20s, one request allowed again", rl.allow("bob").allowed)
    check("but the next is denied again", not rl.allow("bob").allowed)


# --------------------------------------------------------------------------- #
# 3. Per-user isolation                                                       #
# --------------------------------------------------------------------------- #
def test_per_user_isolation():
    print("\n=== Per-user isolation ===")
    clock = FakeClock()
    rl = TokenBucketRateLimiter.from_window(2, 60, time_func=clock)

    rl.allow("user_a")
    rl.allow("user_a")
    check("user_a is limited", not rl.allow("user_a").allowed)
    # user_b has a fresh, independent bucket.
    check("user_b unaffected by user_a", rl.allow("user_b").allowed)
    check("user_b second request allowed", rl.allow("user_b").allowed)
    check("user_b now limited too", not rl.allow("user_b").allowed)


# --------------------------------------------------------------------------- #
# 4. The rate_limit graph node short-circuits when over the limit             #
# --------------------------------------------------------------------------- #
def test_rate_limit_node():
    print("\n=== rate_limit_node integration (no LLM) ===")
    # Configure a tiny limit BEFORE the default limiter is first built.
    from src import config
    import src.middleware.rate_limiter as rl_mod
    config.RATE_LIMIT_MAX_REQUESTS = 2
    config.RATE_LIMIT_WINDOW_SECONDS = 60
    rl_mod._default_limiter = None   # force a rebuild with the new config

    from src.middleware.authentication import Principal
    from src.workflows.analyst_workflow import rate_limit_node, after_rate_limit

    principal = Principal(user_id="u-senior", roles=["senior"], authenticated=True)
    state = {"messages": [], "principal": principal.to_dict()}

    out1 = rate_limit_node(state)
    check("1st run not rate limited", out1["rate_limited"] is False)
    check("1st run routes to 'ok'", after_rate_limit(out1) == "ok")

    out2 = rate_limit_node(state)
    check("2nd run not rate limited", out2["rate_limited"] is False)

    out3 = rate_limit_node(state)
    check("3rd run IS rate limited", out3["rate_limited"] is True)
    check("3rd run routes to 'blocked'", after_rate_limit(out3) == "blocked")
    check("a helpful message is attached", "Rate limit exceeded" in out3["messages"][0].content)


if __name__ == "__main__":
    test_burst_then_deny()
    test_refill_over_time()
    test_per_user_isolation()
    test_rate_limit_node()
    print("\nAll rate-limiter tests passed.")
