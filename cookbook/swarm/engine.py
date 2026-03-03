"""
Swarm execution engine.

Handles:
- Concurrent agent execution with DAG dependency resolution
- Rate limiting (token bucket)
- Retries with exponential backoff + jitter
- Per-agent timeouts
- Live event callbacks for streaming
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Callable, Awaitable

from .models import (
    AgentSpec, AgentResult, AgentStatus, SwarmPlan,
)
from .config import SwarmConfig

# Type for the function that runs a single agent
AgentRunner = Callable[
    [AgentSpec, dict[str, AgentResult], SwarmConfig],
    Awaitable[str],
]

# Event callbacks
OnAgentStart = Callable[[AgentSpec], Awaitable[None] | None]
OnAgentDone = Callable[[AgentResult], Awaitable[None] | None]
OnAgentRetry = Callable[[AgentSpec, int, Exception], Awaitable[None] | None]


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rpm: int = 60, burst: int = 10):
        self._rate = rpm / 60.0  # tokens per second
        self._burst = burst
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
            if self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


class SwarmEngine:
    """
    Executes a SwarmPlan by running agents concurrently,
    respecting dependency order, rate limits, and retries.
    """

    def __init__(
        self,
        config: SwarmConfig,
        runner: AgentRunner,
        on_start: OnAgentStart | None = None,
        on_done: OnAgentDone | None = None,
        on_retry: OnAgentRetry | None = None,
    ):
        self.config = config
        self.runner = runner
        self.on_start = on_start
        self.on_done = on_done
        self.on_retry = on_retry
        self._rate_limiter = RateLimiter(
            rpm=config.rate_limit_rpm,
            burst=config.rate_limit_burst,
        )
        self._semaphore = asyncio.Semaphore(config.max_parallel)
        self._results: dict[str, AgentResult] = {}
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._completion_event = asyncio.Event()

    async def execute(self, plan: SwarmPlan) -> list[AgentResult]:
        """Execute all agents in the plan, respecting dependencies."""
        errors = plan.validate()
        if errors:
            raise ValueError(f"Invalid plan: {'; '.join(errors)}")

        self._results = {}
        self._completed = set()
        self._failed = set()
        pending = {a.agent_id for a in plan.agents}
        running: dict[str, asyncio.Task] = {}

        while pending or running:
            # Find agents ready to run
            ready = [
                a for a in plan.agents
                if a.agent_id in pending
                and all(d in self._completed for d in a.depends_on)
                and not any(d in self._failed for d in a.depends_on)
            ]

            # Check for agents blocked by failed dependencies
            blocked = [
                a for a in plan.agents
                if a.agent_id in pending
                and any(d in self._failed for d in a.depends_on)
            ]
            for a in blocked:
                pending.discard(a.agent_id)
                result = AgentResult(
                    agent_id=a.agent_id,
                    role=a.role,
                    task=a.task,
                    status=AgentStatus.FAILED,
                    error="Dependency failed",
                )
                self._results[a.agent_id] = result
                self._failed.add(a.agent_id)
                if self.on_done:
                    _maybe_await(self.on_done(result))

            # Sort by priority (higher first)
            ready.sort(key=lambda a: -a.priority)

            # Launch ready agents
            for agent in ready:
                pending.discard(agent.agent_id)
                task = asyncio.create_task(
                    self._run_agent(agent),
                    name=f"agent-{agent.agent_id}",
                )
                running[agent.agent_id] = task

            if not running:
                if pending:
                    # Deadlock — remaining agents have unsatisfiable deps
                    for aid in list(pending):
                        a = plan.get_agent(aid)
                        if a:
                            self._results[aid] = AgentResult(
                                agent_id=aid, role=a.role, task=a.task,
                                status=AgentStatus.FAILED,
                                error="Deadlocked — dependencies cannot be satisfied",
                            )
                            self._failed.add(aid)
                    pending.clear()
                break

            # Wait for at least one to finish
            done, _ = await asyncio.wait(
                running.values(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for finished_task in done:
                # Find which agent_id this was
                for aid, t in list(running.items()):
                    if t is finished_task:
                        del running[aid]
                        break

        return list(self._results.values())

    async def _run_agent(self, agent: AgentSpec):
        """Run a single agent with retries, rate limiting, and timeout."""
        async with self._semaphore:
            last_error = None
            max_tries = agent.max_retries + 1
            delay = self.config.retry_base_delay

            for attempt in range(1, max_tries + 1):
                try:
                    # Fire start event
                    if self.on_start:
                        _maybe_await(self.on_start(agent))

                    # Rate limit
                    await self._rate_limiter.acquire()

                    started = time.time()

                    # Run with timeout
                    timeout = agent.timeout or self.config.agent_timeout
                    content = await asyncio.wait_for(
                        self.runner(agent, self._results, self.config),
                        timeout=timeout,
                    )

                    result = AgentResult(
                        agent_id=agent.agent_id,
                        role=agent.role,
                        task=agent.task,
                        content=content,
                        status=AgentStatus.COMPLETED,
                        started_at=started,
                        finished_at=time.time(),
                        attempt=attempt,
                    )
                    self._results[agent.agent_id] = result
                    self._completed.add(agent.agent_id)

                    if self.on_done:
                        _maybe_await(self.on_done(result))
                    return

                except asyncio.TimeoutError:
                    last_error = TimeoutError(
                        f"Agent '{agent.role}' timed out after {agent.timeout}s"
                    )
                except Exception as e:
                    last_error = e

                # Retry logic
                if attempt < max_tries:
                    if self.on_retry:
                        _maybe_await(self.on_retry(agent, attempt, last_error))

                    jitter = random.uniform(0.5, 1.5) if self.config.retry_jitter else 1.0
                    wait = min(delay * jitter, self.config.retry_max_delay)
                    await asyncio.sleep(wait)
                    delay *= 2  # exponential backoff

            # All retries exhausted
            result = AgentResult(
                agent_id=agent.agent_id,
                role=agent.role,
                task=agent.task,
                status=AgentStatus.FAILED,
                error=str(last_error) if last_error else "Unknown error",
                started_at=time.time(),
                finished_at=time.time(),
                attempt=max_tries,
            )
            self._results[agent.agent_id] = result
            self._failed.add(agent.agent_id)

            if self.on_done:
                _maybe_await(self.on_done(result))


def _maybe_await(result):
    """Fire-and-forget for sync or async callbacks."""
    if asyncio.iscoroutine(result):
        asyncio.ensure_future(result)
