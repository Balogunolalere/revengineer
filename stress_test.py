#!/usr/bin/env python3
"""
Stress test for the DeepSeek proxy API.

Tests concurrent requests, measures response times, tracks errors,
and determines rate limits. Run this AFTER starting deepseek_api.py.

=== PREREQUISITES ===

    # Terminal 1: start the API server first
    python3 deepseek_api.py

    # Terminal 2: run stress tests against it

=== EXAMPLES ===

    # Send 10 requests one at a time (default):
    python3 stress_test.py

    # Send 20 requests with 3 running simultaneously:
    python3 stress_test.py --concurrent 3 --total 20

    # Fire 10 requests all at once (burst mode):
    python3 stress_test.py --burst 10

    # Run continuously for 5 minutes:
    python3 stress_test.py --duration 300

    # Target a different server:
    python3 stress_test.py --url http://localhost:8080/v1/chat/completions

=== OUTPUT ===

    Prints per-request status (latency, response length), then a summary
    with min/mean/max/p95 latency, success/failure counts, and rate limits.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class RequestResult:
    index: int
    status: int
    latency: float
    content_length: int
    error: str = ""
    timestamp: float = 0.0


@dataclass
class StressTestStats:
    results: list[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.results if r.status == 200)

    @property
    def failed(self) -> int:
        return self.total - self.successful

    @property
    def latencies(self) -> list[float]:
        return [r.latency for r in self.results if r.status == 200]

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def print_report(self):
        print("\n" + "=" * 60)
        print("  STRESS TEST RESULTS")
        print("=" * 60)
        print(f"  Duration:        {self.duration:.1f}s")
        print(f"  Total requests:  {self.total}")
        print(f"  Successful:      {self.successful} ({self.successful*100//max(self.total,1)}%)")
        print(f"  Failed:          {self.failed}")
        print(f"  Req/min:         {self.successful / max(self.duration/60, 0.01):.1f}")

        if self.latencies:
            print(f"\n  Latency (success only):")
            print(f"    Min:     {min(self.latencies):.1f}s")
            print(f"    Max:     {max(self.latencies):.1f}s")
            print(f"    Mean:    {statistics.mean(self.latencies):.1f}s")
            print(f"    Median:  {statistics.median(self.latencies):.1f}s")
            if len(self.latencies) > 1:
                print(f"    Stdev:   {statistics.stdev(self.latencies):.1f}s")
            p95 = sorted(self.latencies)[int(len(self.latencies) * 0.95)] if len(self.latencies) > 5 else max(self.latencies)
            print(f"    P95:     {p95:.1f}s")

        # Show errors
        errors = [r for r in self.results if r.error]
        if errors:
            print(f"\n  Errors:")
            error_counts: dict[str, int] = {}
            for r in errors:
                key = r.error[:80]
                error_counts[key] = error_counts.get(key, 0) + 1
            for err, count in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"    [{count}x] {err}")

        # Timeline: show when failures started
        if self.results:
            print(f"\n  Timeline:")
            bucket_size = max(self.duration / 10, 1)
            buckets: dict[int, tuple[int, int]] = {}
            for r in self.results:
                b = int((r.timestamp - self.start_time) / bucket_size)
                ok, fail = buckets.get(b, (0, 0))
                if r.status == 200:
                    buckets[b] = (ok + 1, fail)
                else:
                    buckets[b] = (ok, fail + 1)
            for b in sorted(buckets.keys()):
                ok, fail = buckets[b]
                t_start = b * bucket_size
                t_end = t_start + bucket_size
                bar_ok = "█" * ok
                bar_fail = "░" * fail
                print(f"    {t_start:6.0f}-{t_end:4.0f}s: {bar_ok}{bar_fail} ({ok}ok/{fail}err)")

        print("=" * 60)


PROMPTS = [
    "Say hello in one sentence.",
    "What is 2+2? Answer with just the number.",
    "Name one color.",
    "What day comes after Monday? One word.",
    "Is water wet? Yes or no.",
    "Name a fruit.",
    "What's the capital of France? One word.",
    "Count to 3.",
    "What's 10 * 5?",
    "Name an animal.",
]


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    index: int,
    stream: bool = False,
) -> RequestResult:
    """Send a single chat completion request."""
    prompt = PROMPTS[index % len(PROMPTS)]
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
    }

    start = time.time()
    try:
        if stream:
            async with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as resp:
                content = []
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            if "content" in delta:
                                content.append(delta["content"])
                        except json.JSONDecodeError:
                            pass
                latency = time.time() - start
                text = "".join(content)
                status = resp.status_code
                result = RequestResult(index, status, latency, len(text), timestamp=start)
                print(f"  [{index:3d}] {status} | {latency:5.1f}s | {len(text):4d} chars | {text[:50]}...")
                return result
        else:
            resp = await client.post(f"{base_url}/v1/chat/completions", json=payload)
            latency = time.time() - start
            status = resp.status_code

            if status == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                result = RequestResult(index, status, latency, len(content), timestamp=start)
                print(f"  [{index:3d}] {status} | {latency:5.1f}s | {len(content):4d} chars | {content[:50]}...")
            else:
                error = resp.text[:200]
                result = RequestResult(index, status, latency, 0, error=error, timestamp=start)
                print(f"  [{index:3d}] {status} | {latency:5.1f}s | ERROR: {error[:60]}")

            return result

    except Exception as e:
        latency = time.time() - start
        error_msg = str(e)[:200]
        print(f"  [{index:3d}] ERR | {latency:5.1f}s | {error_msg[:60]}")
        return RequestResult(index, 0, latency, 0, error=error_msg, timestamp=start)


async def run_sequential(base_url: str, total: int, stream: bool) -> StressTestStats:
    """Send requests one by one."""
    stats = StressTestStats(start_time=time.time())

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        for i in range(total):
            result = await send_request(client, base_url, i, stream)
            stats.results.append(result)

    stats.end_time = time.time()
    return stats


async def run_concurrent(
    base_url: str, total: int, concurrency: int, stream: bool
) -> StressTestStats:
    """Send requests with limited concurrency."""
    stats = StressTestStats(start_time=time.time())
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client, i):
        async with semaphore:
            return await send_request(client, base_url, i, stream)

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        tasks = [bounded_request(client, i) for i in range(total)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, RequestResult):
                stats.results.append(r)
            else:
                stats.results.append(
                    RequestResult(0, 0, 0, 0, error=str(r), timestamp=time.time())
                )

    stats.end_time = time.time()
    return stats


async def run_duration(
    base_url: str, duration: int, concurrency: int, stream: bool
) -> StressTestStats:
    """Send requests continuously for a given duration."""
    stats = StressTestStats(start_time=time.time())
    end_at = time.time() + duration
    semaphore = asyncio.Semaphore(concurrency)
    index = 0
    lock = asyncio.Lock()

    async def worker(client: httpx.AsyncClient):
        nonlocal index
        while time.time() < end_at:
            async with lock:
                i = index
                index += 1
            async with semaphore:
                result = await send_request(client, base_url, i, stream)
                stats.results.append(result)

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0)) as client:
        workers = [worker(client) for _ in range(concurrency)]
        await asyncio.gather(*workers)

    stats.end_time = time.time()
    return stats


async def main():
    parser = argparse.ArgumentParser(description="Stress test DeepSeek proxy API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--total", type=int, default=10, help="Total requests to send")
    parser.add_argument("--concurrent", type=int, default=1, help="Concurrent requests")
    parser.add_argument("--burst", type=int, default=0,
                        help="Burst N requests simultaneously (overrides --total and --concurrent)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run for N seconds (overrides --total)")
    parser.add_argument("--stream", action="store_true", help="Use streaming mode")
    args = parser.parse_args()

    # Check server is up
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{args.url}/health")
            if r.status_code != 200:
                print(f"Server not healthy: {r.status_code}")
                return
    except Exception as e:
        print(f"Cannot reach server at {args.url}: {e}")
        print("Start the server first: python3 deepseek_api.py --token YOUR_TOKEN")
        return

    print(f"\n  Target: {args.url}")

    if args.burst > 0:
        print(f"  Mode:   BURST ({args.burst} simultaneous requests)")
        print(f"  Stream: {'ON' if args.stream else 'OFF'}")
        print("-" * 60)
        stats = await run_concurrent(args.url, args.burst, args.burst, args.stream)
    elif args.duration > 0:
        print(f"  Mode:   DURATION ({args.duration}s, {args.concurrent} concurrent)")
        print(f"  Stream: {'ON' if args.stream else 'OFF'}")
        print("-" * 60)
        stats = await run_duration(args.url, args.duration, args.concurrent, args.stream)
    elif args.concurrent > 1:
        print(f"  Mode:   CONCURRENT ({args.concurrent} workers, {args.total} total)")
        print(f"  Stream: {'ON' if args.stream else 'OFF'}")
        print("-" * 60)
        stats = await run_concurrent(args.url, args.total, args.concurrent, args.stream)
    else:
        print(f"  Mode:   SEQUENTIAL ({args.total} requests)")
        print(f"  Stream: {'ON' if args.stream else 'OFF'}")
        print("-" * 60)
        stats = await run_sequential(args.url, args.total, args.stream)

    stats.print_report()

    # Fetch server stats
    try:
        async with httpx.AsyncClient() as c:
            r = await c.get(f"{args.url}/stats")
            if r.status_code == 200:
                print("\n  Server stats:", json.dumps(r.json(), indent=2))
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
