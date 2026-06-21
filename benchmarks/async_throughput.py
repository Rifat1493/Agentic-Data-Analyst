"""Async vs blocking request-handling throughput benchmark.

Isolates the ONE architectural variable behind this service's design: async
(event-loop) request handlers vs blocking (threadpool) handlers, under concurrent
I/O-bound load. Downstream I/O (LLM calls, market-data downloads, DB round-trips)
is MOCKED as a fixed wait, so the number reflects the concurrency model itself —
not network variance.

Two otherwise-identical endpoints:
    GET /async -> `await asyncio.sleep(L)`  (non-blocking; runs on the event loop)
    GET /sync  -> `time.sleep(L)`           (blocking; Starlette dispatches it to a
                                             bounded threadpool)

Requests are issued concurrently in-process via httpx's ASGITransport (no socket
or server overhead), so the two paths are compared apples-to-apples. This mirrors
the real app, whose handlers are `async def` and `await workflow.ainvoke(...)`.

Run:  uv run python benchmarks/async_throughput.py
"""
import asyncio
import statistics
import time

import anyio
import httpx
from fastapi import FastAPI

LATENCY_S = 0.05            # simulated downstream I/O per request (50 ms)
CONCURRENCY = [50, 100, 200, 400]
TRIALS = 3

app = FastAPI()


@app.get("/async")
async def async_endpoint():
    await asyncio.sleep(LATENCY_S)      # event loop stays free for other requests
    return {"ok": True}


@app.get("/sync")
def sync_endpoint():
    time.sleep(LATENCY_S)               # blocks a worker thread for the whole wait
    return {"ok": True}


async def _run(path: str, n: int) -> float:
    """Fire n concurrent requests at `path`; return wall-clock seconds."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:
        start = time.perf_counter()
        await asyncio.gather(*(client.get(path) for _ in range(n)))
        return time.perf_counter() - start


async def _median(path: str, n: int) -> float:
    return statistics.median([await _run(path, n) for _ in range(TRIALS)])


async def main():
    # Warm up: the first calls pay one-time threadpool / event-loop spin-up costs.
    await _run("/async", 20)
    await _run("/sync", 20)

    try:
        pool = int(anyio.to_thread.current_default_thread_limiter().total_tokens)
    except Exception:
        pool = "unknown"

    print(f"simulated I/O latency: {LATENCY_S * 1000:.0f} ms/request | trials: {TRIALS} "
          f"(median) | sync threadpool size: {pool}\n")
    header = f"{'concurrency':>11} | {'async req/s':>11} | {'sync req/s':>10} | {'speedup':>7}"
    print(header)
    print("-" * len(header))

    speedups = []
    for n in CONCURRENCY:
        a = await _median("/async", n)
        s = await _median("/sync", n)
        a_rps, s_rps = n / a, n / s
        speedup = a_rps / s_rps
        speedups.append(speedup)
        print(f"{n:>11} | {a_rps:>11.0f} | {s_rps:>10.0f} | {speedup:>6.1f}x")

    print(f"\nHeadline: at concurrency={CONCURRENCY[-1]}, async sustained "
          f"~{speedups[-1]:.1f}x the throughput of blocking handlers "
          f"(range {min(speedups):.1f}x to {max(speedups):.1f}x across the sweep).")


if __name__ == "__main__":
    asyncio.run(main())
