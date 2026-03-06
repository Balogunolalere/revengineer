"""
Swarm execution engine.

Handles:
- Concurrent agent execution with DAG dependency resolution
- Adaptive rate limiting (token bucket with dynamic adjustment)
- Circuit breaker for cascading failure protection
- Retries with exponential backoff + jitter
- Per-agent timeouts
- Live event callbacks for streaming
"""

from __future__ import annotations

import asyncio
import enum
import logging
import random
import time
from typing import Any, Callable, Awaitable

from .models import (
    AgentSpec, AgentResult, AgentStatus, SwarmPlan, BudgetTracker,
)
from .config import SwarmConfig

log = logging.getLogger(__name__)

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


class AdaptiveRateLimiter:
    """Token bucket rate limiter with dynamic adjustment based on backend signals.

    On errors/429s: reduces effective RPM by backoff_factor.
    On consecutive successes: gradually recovers toward max RPM.
    """

    def __init__(
        self,
        max_rpm: int = 60,
        min_rpm: int = 5,
        burst: int = 10,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
        recovery_streak: int = 5,
    ):
        self._max_rpm = max_rpm
        self._min_rpm = max(1, min_rpm)
        self._current_rpm = float(max_rpm)
        self._burst = burst
        self._backoff_factor = backoff_factor
        self._recovery_factor = recovery_factor
        self._recovery_streak = recovery_streak

        self._rate = self._current_rpm / 60.0
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

        # Tracking
        self._consecutive_successes = 0
        self._total_successes = 0
        self._total_errors = 0
        self._total_backoffs = 0

    @property
    def current_rpm(self) -> float:
        return self._current_rpm

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "current_rpm": round(self._current_rpm, 1),
            "max_rpm": self._max_rpm,
            "min_rpm": self._min_rpm,
            "successes": self._total_successes,
            "errors": self._total_errors,
            "backoffs": self._total_backoffs,
            "consecutive_successes": self._consecutive_successes,
        }

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

    def record_success(self):
        """Signal a successful request — may increase rate."""
        self._consecutive_successes += 1
        self._total_successes += 1
        if (
            self._consecutive_successes >= self._recovery_streak
            and self._current_rpm < self._max_rpm
        ):
            old = self._current_rpm
            self._current_rpm = min(
                self._max_rpm,
                self._current_rpm * self._recovery_factor,
            )
            self._rate = self._current_rpm / 60.0
            self._consecutive_successes = 0
            if self._current_rpm != old:
                log.debug(f"Rate limiter recovered: {old:.1f} → {self._current_rpm:.1f} RPM")

    def record_error(self, is_rate_limit: bool = False):
        """Signal a failure — reduces rate."""
        self._consecutive_successes = 0
        self._total_errors += 1
        old = self._current_rpm
        factor = self._backoff_factor if is_rate_limit else max(self._backoff_factor, 0.7)
        self._current_rpm = max(self._min_rpm, self._current_rpm * factor)
        self._rate = self._current_rpm / 60.0
        self._total_backoffs += 1
        if self._current_rpm != old:
            log.info(
                f"Rate limiter backed off: {old:.1f} → {self._current_rpm:.1f} RPM"
                f" ({'429' if is_rate_limit else 'error'})"
            )


class CircuitState(enum.Enum):
    CLOSED = "closed"        # normal operation
    OPEN = "open"            # failing, reject requests
    HALF_OPEN = "half_open"  # allow one probe


class CircuitBreaker:
    """Protects against cascading failures by tripping open after consecutive errors.

    States:
      CLOSED   → normal. On N consecutive failures → OPEN
      OPEN     → reject immediately with CircuitOpenError. After cooldown → HALF_OPEN
      HALF_OPEN → allow one request through. Success → CLOSED, Failure → OPEN
    """

    def __init__(self, threshold: int = 5, cooldown: float = 30.0):
        self._threshold = threshold
        self._cooldown = cooldown
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()
        # Stats
        self._total_trips = 0
        self._total_rejections = 0

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "total_trips": self._total_trips,
            "total_rejections": self._total_rejections,
        }

    async def check(self):
        """Check if a request is allowed. Raises CircuitOpenError if not."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._last_failure_time
                if elapsed >= self._cooldown:
                    self._state = CircuitState.HALF_OPEN
                    log.info("Circuit breaker → HALF_OPEN (probing)")
                    return
                self._total_rejections += 1
                raise CircuitOpenError(
                    f"Circuit breaker is OPEN ({self._consecutive_failures} consecutive failures, "
                    f"{self._cooldown - elapsed:.0f}s until probe)"
                )
            # HALF_OPEN — allow through

    def record_success(self):
        """Record a successful request."""
        if self._state != CircuitState.CLOSED:
            log.info(f"Circuit breaker → CLOSED (recovered from {self._state.value})")
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0

    def record_failure(self):
        """Record a failed request."""
        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._total_trips += 1
            log.warning("Circuit breaker → OPEN (probe failed)")
        elif self._consecutive_failures >= self._threshold:
            self._state = CircuitState.OPEN
            self._total_trips += 1
            log.warning(
                f"Circuit breaker → OPEN ({self._consecutive_failures} consecutive failures)"
            )


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open and rejecting requests."""
    pass


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
        budget: BudgetTracker | None = None,
    ):
        self.config = config
        self.runner = runner
        self.on_start = on_start
        self.on_done = on_done
        self.on_retry = on_retry
        self.budget = budget

        # Choose rate limiter based on config
        if config.adaptive_rate_limit:
            self._rate_limiter = AdaptiveRateLimiter(
                max_rpm=config.rate_limit_rpm,
                min_rpm=config.rate_limit_min_rpm,
                burst=config.rate_limit_burst,
                backoff_factor=config.rate_limit_backoff_factor,
                recovery_factor=config.rate_limit_recovery_factor,
                recovery_streak=config.rate_limit_recovery_streak,
            )
        else:
            self._rate_limiter = RateLimiter(
                rpm=config.rate_limit_rpm,
                burst=config.rate_limit_burst,
            )

        # Circuit breaker
        if config.circuit_breaker_enabled:
            self._circuit_breaker = CircuitBreaker(
                threshold=config.circuit_breaker_threshold,
                cooldown=config.circuit_breaker_cooldown,
            )
        else:
            self._circuit_breaker = None

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

        if self.config.swarm_timeout > 0:
            return await asyncio.wait_for(
                self._execute_inner(plan),
                timeout=self.config.swarm_timeout,
            )
        return await self._execute_inner(plan)

    async def _execute_inner(self, plan: SwarmPlan) -> list[AgentResult]:
        """Inner execution loop — handles DAG scheduling and agent killing."""
        self._results = {}
        self._completed = set()
        self._failed = set()
        pending = {a.agent_id for a in plan.agents}
        running: dict[str, asyncio.Task] = {}
        # Track when each running agent started (for agent killing)
        _start_times: dict[str, float] = {}

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
                # Check budget before launching
                if self.budget:
                    budget_err = self.budget.check_budget()
                    if budget_err:
                        pending.discard(agent.agent_id)
                        result = AgentResult(
                            agent_id=agent.agent_id,
                            role=agent.role,
                            task=agent.task,
                            status=AgentStatus.FAILED,
                            error=f"Budget exceeded: {budget_err}",
                        )
                        self._results[agent.agent_id] = result
                        self._failed.add(agent.agent_id)
                        if self.on_done:
                            _maybe_await(self.on_done(result))
                        continue

                pending.discard(agent.agent_id)
                task = asyncio.create_task(
                    self._run_agent(agent),
                    name=f"agent-{agent.agent_id}",
                )
                running[agent.agent_id] = task
                _start_times[agent.agent_id] = time.time()

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

            # Wait for at least one to finish (with periodic timeout for agent killing)
            wait_timeout = None
            total_agents = len(plan.agents)
            min_completed_for_kill = max(3, (total_agents + 1) // 2)
            if (
                self.config.enable_agent_killing
                and len(self._completed) >= min_completed_for_kill
            ):
                wait_timeout = 1.0  # check for slow agents every second

            done, _ = await asyncio.wait(
                running.values(),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=wait_timeout,
            )
            for finished_task in done:
                # Find which agent_id this was
                for aid, t in list(running.items()):
                    if t is finished_task:
                        del running[aid]
                        _start_times.pop(aid, None)
                        break

            # ── Agent killing — cancel slow stragglers ──
            # Require a meaningful majority of agents to be done before
            # using their durations as a baseline. This prevents 2 fast
            # agents from killing all the rest.
            total_agents = len(plan.agents)
            min_completed_for_kill = max(3, (total_agents + 1) // 2)
            if (
                self.config.enable_agent_killing
                and running
                and len(self._completed) >= min_completed_for_kill
            ):
                self._kill_slow_agents(running, _start_times, plan)

        return list(self._results.values())

    def _kill_slow_agents(
        self,
        running: dict[str, asyncio.Task],
        start_times: dict[str, float],
        plan: SwarmPlan,
    ) -> None:
        """Cancel agents that are taking far longer than their completed peers.

        Skip tool-using agents when computing the median — they are expected
        to take multiple LLM round-trips and shouldn't be penalized.
        """
        # Compute median duration of completed NON-tool agents
        durations = []
        for r in self._results.values():
            if r.status != AgentStatus.COMPLETED or r.duration <= 0:
                continue
            agent = plan.get_agent(r.agent_id)
            has_tools = agent and any(
                t.lower() not in ("search", "reasoning")
                for t in agent.tools
            )
            if not has_tools:
                durations.append(r.duration)
        if not durations:
            return

        durations.sort()
        mid = len(durations) // 2
        median = (
            durations[mid]
            if len(durations) % 2 == 1
            else (durations[mid - 1] + durations[mid]) / 2
        )

        threshold = max(
            median * self.config.agent_kill_threshold,
            self.config.agent_kill_min_time,
        )
        now = time.time()

        for aid in list(running):
            elapsed = now - start_times.get(aid, now)
            if elapsed > threshold:
                agent = plan.get_agent(aid)
                role = agent.role if agent else aid
                log.warning(
                    f"Killing slow agent '{role}': {elapsed:.1f}s > "
                    f"{threshold:.1f}s (median {median:.1f}s × {self.config.agent_kill_threshold})"
                )
                running[aid].cancel()
                result = AgentResult(
                    agent_id=aid,
                    role=role,
                    task=agent.task if agent else "",
                    status=AgentStatus.CANCELLED,
                    error=f"Killed: exceeded {self.config.agent_kill_threshold}× median duration ({elapsed:.1f}s > {threshold:.1f}s)",
                    started_at=start_times.get(aid, 0),
                    finished_at=now,
                )
                self._results[aid] = result
                self._failed.add(aid)
                start_times.pop(aid, None)
                del running[aid]
                if self.on_done:
                    _maybe_await(self.on_done(result))

    async def _run_agent(self, agent: AgentSpec):
        """Run a single agent with retries, rate limiting, circuit breaker, and timeout."""
        async with self._semaphore:
            last_error = None
            max_tries = agent.max_retries + 1
            delay = self.config.retry_base_delay

            for attempt in range(1, max_tries + 1):
                try:
                    # Circuit breaker check
                    if self._circuit_breaker:
                        try:
                            await self._circuit_breaker.check()
                        except CircuitOpenError as e:
                            # On first attempt, propagate immediately (will be caught as last_error)
                            # On retries, we'll wait for the cooldown
                            if attempt == max_tries:
                                raise
                            last_error = e
                            # Wait for the cooldown period before retrying
                            await asyncio.sleep(self.config.circuit_breaker_cooldown)
                            continue

                    # Fire start event
                    if self.on_start:
                        _maybe_await(self.on_start(agent))

                    # Rate limit
                    await self._rate_limiter.acquire()

                    started = time.time()

                    # Run with timeout
                    # Tool-using agents get a longer timeout
                    has_real_tools = any(
                        t.lower() not in ("search", "reasoning")
                        for t in agent.tools
                    )
                    if agent.timeout is not None:
                        timeout = agent.timeout
                    elif has_real_tools:
                        timeout = self.config.tool_agent_timeout
                    else:
                        timeout = self.config.agent_timeout
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

                    # Record to budget tracker
                    if self.budget:
                        await self.budget.record_agent()
                        if result.tokens_used > 0:
                            await self.budget.record_tokens(result.tokens_used)

                    # Signal success to adaptive systems
                    if isinstance(self._rate_limiter, AdaptiveRateLimiter):
                        self._rate_limiter.record_success()
                    if self._circuit_breaker:
                        self._circuit_breaker.record_success()

                    if self.on_done:
                        _maybe_await(self.on_done(result))
                    return

                except CircuitOpenError:
                    last_error = CircuitOpenError("Circuit breaker open — backend overloaded")
                except asyncio.TimeoutError:
                    last_error = TimeoutError(
                        f"Agent '{agent.role}' timed out after {timeout}s"
                    )
                except Exception as e:
                    last_error = e
                    # Signal failure to adaptive systems
                    is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                    if isinstance(self._rate_limiter, AdaptiveRateLimiter):
                        self._rate_limiter.record_error(is_rate_limit=is_rate_limit)
                    if self._circuit_breaker:
                        self._circuit_breaker.record_failure()

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
