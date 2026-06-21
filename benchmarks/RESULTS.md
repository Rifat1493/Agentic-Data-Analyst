# Benchmark results

## Async vs blocking request handling (`async_throughput.py`)

**What it measures.** The throughput impact of this service's core design choice:
`async def` handlers that `await` I/O (the app's real pattern) vs equivalent
blocking handlers. Downstream I/O (LLM calls, market data, DB) is mocked as a
fixed 50 ms wait so the result reflects the concurrency model, not network noise.
Requests are issued concurrently in-process via httpx ASGITransport.

**Environment.** Single worker process, Python 3.12, Starlette default sync
threadpool (40 threads), median of 3 trials per point.

| Concurrency | Async (req/s) | Blocking (req/s) | Speedup |
|------------:|--------------:|-----------------:|--------:|
| 50          | 785           | 431              | 1.8x    |
| 100         | 1624          | 558              | 2.9x    |
| 200         | 2297          | 639              | 3.6x    |
| 400         | 3036          | 608              | 5.0x    |

**Takeaway.** Async throughput scales with concurrency, while the blocking path
plateaus near ~600 req/s — it is capped by the 40-thread pool (~40 ÷ 50 ms ≈ 800
req/s ceiling, less overhead). At 400 concurrent I/O-bound requests, the async
handlers sustain **~5x** the throughput of blocking ones on the same single worker.

**Honest scope.** This isolates the handler concurrency model with mocked I/O; it
is not an end-to-end load test against live LLM/market-data calls (which would add
real latency and provider rate limits). The multiplier is conditional on the
stated concurrency, latency, and threadpool size — reproduce with:

```bash
uv run python benchmarks/async_throughput.py
```
