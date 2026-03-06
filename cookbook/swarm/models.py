"""
Data models for the swarm system.

AgentSpec   — defines a sub-agent's role, task, and dependencies
AgentResult — output from a completed agent
SwarmPlan   — the orchestrator's decomposition of a task
SwarmResult — final aggregated output
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    PENDING = "pending"
    WAITING = "waiting"      # blocked on dependencies
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class SwarmMode(str, Enum):
    AUTO = "auto"            # orchestrator decides decomposition
    MANUAL = "manual"        # user provides explicit agent specs
    ITERATIVE = "iterative"  # orchestrator can re-plan mid-execution


@dataclass
class AgentSpec:
    """Specification for a single sub-agent."""
    role: str                              # e.g. "Security Researcher"
    task: str                              # specific instruction
    agent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    depends_on: list[str] = field(default_factory=list)  # agent_ids this depends on
    model: str = ""                        # override model for this agent
    system_prompt: str = ""                # override system prompt
    tools: list[str] = field(default_factory=list)  # tool names this agent can use
    priority: int = 0                      # higher = runs first when possible
    timeout: float | None = None           # per-agent timeout (None = use config default)
    max_retries: int = 2


@dataclass
class AgentResult:
    """Result from a completed sub-agent."""
    agent_id: str
    role: str
    task: str
    content: str = ""
    status: AgentStatus = AgentStatus.COMPLETED
    error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    tokens_used: int = 0
    attempt: int = 1
    tool_calls_made: int = 0
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return 0.0


@dataclass
class SwarmPlan:
    """The orchestrator's plan — a set of agents with dependency graph."""
    goal: str
    agents: list[AgentSpec] = field(default_factory=list)
    strategy: str = ""     # orchestrator's reasoning about decomposition
    created_at: float = field(default_factory=time.time)

    @property
    def agent_ids(self) -> set[str]:
        return {a.agent_id for a in self.agents}

    def get_agent(self, agent_id: str) -> AgentSpec | None:
        for a in self.agents:
            if a.agent_id == agent_id:
                return a
        return None

    def get_ready_agents(self, completed: set[str]) -> list[AgentSpec]:
        """Return agents whose dependencies are all satisfied."""
        ready = []
        for a in self.agents:
            if a.agent_id in completed:
                continue
            if all(dep in completed for dep in a.depends_on):
                ready.append(a)
        return ready

    def validate(self) -> list[str]:
        """Check for issues in the plan."""
        errors = []
        ids = self.agent_ids
        for a in self.agents:
            for dep in a.depends_on:
                if dep not in ids:
                    errors.append(f"Agent '{a.role}' depends on unknown agent_id '{dep}'")
        # Check for cycles (DFS)
        visited: set[str] = set()
        stack: set[str] = set()
        adj = {a.agent_id: a.depends_on for a in self.agents}

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            stack.add(node)
            for dep in adj.get(node, []):
                if dep not in visited:
                    if _has_cycle(dep):
                        return True
                elif dep in stack:
                    return True
            stack.discard(node)
            return False

        for a in self.agents:
            if a.agent_id not in visited:
                if _has_cycle(a.agent_id):
                    errors.append("Circular dependency detected in agent graph")
                    break
        return errors


class BudgetTracker:
    """Thread-safe token and cost accounting for a swarm.

    Shared across the swarm (including sub-swarms) to enforce
    global limits on tokens, LLM calls, and agent count.
    """

    def __init__(
        self,
        token_budget: int = 0,
        max_llm_calls: int = 0,
        max_agents: int = 0,
    ):
        self._token_budget = token_budget    # 0 = unlimited
        self._max_llm_calls = max_llm_calls  # 0 = unlimited
        self._max_agents = max_agents        # 0 = unlimited

        self._tokens_used = 0
        self._llm_calls = 0
        self._agents_spawned = 0
        self._lock = asyncio.Lock()

    @property
    def tokens_used(self) -> int:
        return self._tokens_used

    @property
    def llm_calls(self) -> int:
        return self._llm_calls

    @property
    def agents_spawned(self) -> int:
        return self._agents_spawned

    @property
    def tokens_remaining(self) -> int | None:
        """Tokens remaining, or None if unlimited."""
        if self._token_budget <= 0:
            return None
        return max(0, self._token_budget - self._tokens_used)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "tokens_used": self._tokens_used,
            "token_budget": self._token_budget or "unlimited",
            "llm_calls": self._llm_calls,
            "max_llm_calls": self._max_llm_calls or "unlimited",
            "agents_spawned": self._agents_spawned,
            "max_agents": self._max_agents or "unlimited",
        }

    async def record_tokens(self, tokens: int) -> None:
        """Record token usage."""
        async with self._lock:
            self._tokens_used += tokens

    async def record_llm_call(self) -> None:
        """Record an LLM call."""
        async with self._lock:
            self._llm_calls += 1

    async def record_agent(self) -> None:
        """Record an agent spawn."""
        async with self._lock:
            self._agents_spawned += 1

    def check_budget(self) -> str | None:
        """Check if budget is exceeded. Returns error message or None."""
        if self._token_budget > 0 and self._tokens_used >= self._token_budget:
            return f"Token budget exhausted ({self._tokens_used}/{self._token_budget})"
        if self._max_llm_calls > 0 and self._llm_calls >= self._max_llm_calls:
            return f"LLM call limit reached ({self._llm_calls}/{self._max_llm_calls})"
        if self._max_agents > 0 and self._agents_spawned >= self._max_agents:
            return f"Agent limit reached ({self._agents_spawned}/{self._max_agents})"
        return None


class ContextWindow:
    """Token-aware context manager for inter-agent communication.

    Assembles context for each agent from completed peer results,
    ensuring the total fits within a configurable token budget.
    Dependencies get full (or summarized) content; peers get short previews.
    Summaries are cached so the LLM summarizer is called at most once per agent.
    """

    CHARS_PER_TOKEN = 4  # rough heuristic for English text

    def __init__(
        self,
        max_tokens: int = 0,
        summary_max_tokens: int = 300,
    ):
        self.max_tokens = max_tokens          # 0 = unlimited (original behavior)
        self.summary_max_tokens = summary_max_tokens
        self._summaries: dict[str, str] = {}  # agent_id → cached summary
        self._lock = asyncio.Lock()

    # ── helpers ──

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate (~4 chars/token)."""
        return max(1, len(text) // ContextWindow.CHARS_PER_TOKEN)

    async def cache_summary(self, agent_id: str, summary: str) -> None:
        async with self._lock:
            self._summaries[agent_id] = summary

    def get_summary(self, agent_id: str) -> str | None:
        return self._summaries.get(agent_id)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "max_tokens": self.max_tokens or "unlimited",
            "summaries_cached": len(self._summaries),
            "summary_max_tokens": self.summary_max_tokens,
        }

    # ── context assembly ──

    def build_context(
        self,
        agent: AgentSpec,
        results: dict[str, AgentResult],
    ) -> str:
        """Assemble context for *agent* from completed *results*.

        If max_tokens is 0 (unlimited), uses the original full-dump
        behavior.  Otherwise prunes to fit within the token budget:
          1. Dependencies — full content (or summary/truncation if too large)
          2. Non-dependency peers — summary or short preview, space permitting
        """
        if not results:
            return ""

        if self.max_tokens <= 0:
            return self._build_unlimited(agent, results)

        return self._build_pruned(agent, results, self.max_tokens)

    # ── private ──

    def _build_unlimited(self, agent: AgentSpec, results: dict[str, AgentResult]) -> str:
        """Original behavior: full dependency content + 300-char peer previews."""
        dep_parts: list[str] = []
        peer_parts: list[str] = []

        for dep_id in agent.depends_on:
            dep = results.get(dep_id)
            if dep and dep.status == AgentStatus.COMPLETED:
                dep_parts.append(f"### Output from {dep.role}:\n{dep.content}")

        for aid, r in results.items():
            if aid in agent.depends_on or r.status != AgentStatus.COMPLETED:
                continue
            preview = r.content[:300] + "..." if len(r.content) > 300 else r.content
            peer_parts.append(f"- **{r.role}** (completed): {preview}")

        section = ""
        if dep_parts:
            section += "## Required context from other agents:\n\n"
            section += "\n\n".join(dep_parts)
        if peer_parts:
            section += "\n\n## Other agents' outputs (for reference):\n\n"
            section += "\n".join(peer_parts)
        return section

    def _build_pruned(
        self,
        agent: AgentSpec,
        results: dict[str, AgentResult],
        budget: int,
    ) -> str:
        """Token-aware context assembly."""
        parts: list[str] = []
        tokens_used = 0

        # 1) Dependencies — full or summarized
        for dep_id in agent.depends_on:
            dep = results.get(dep_id)
            if not dep or dep.status != AgentStatus.COMPLETED:
                continue

            content_tokens = self.estimate_tokens(dep.content)
            if tokens_used + content_tokens <= budget:
                parts.append(f"### Output from {dep.role} (dependency):\n{dep.content}")
                tokens_used += content_tokens
            else:
                # Try cached summary
                summary = self._summaries.get(dep_id)
                if summary:
                    stokens = self.estimate_tokens(summary)
                    parts.append(
                        f"### Output from {dep.role} (dependency, summarized):\n{summary}"
                    )
                    tokens_used += stokens
                else:
                    # Hard truncate to remaining budget
                    remaining_chars = max(200, (budget - tokens_used) * self.CHARS_PER_TOKEN)
                    truncated = dep.content[:remaining_chars]
                    parts.append(
                        f"### Output from {dep.role} (dependency, truncated):\n{truncated}..."
                    )
                    tokens_used += self.estimate_tokens(truncated)

        # 2) Non-dependency peers — summary/preview, space permitting
        for aid, r in results.items():
            if aid in agent.depends_on or r.status != AgentStatus.COMPLETED:
                continue
            if tokens_used >= budget:
                break

            summary = self._summaries.get(aid)
            if summary:
                text = f"- **{r.role}**: {summary}"
            else:
                preview_len = min(300, max(100, (budget - tokens_used) * self.CHARS_PER_TOKEN))
                preview = r.content[:preview_len]
                if len(r.content) > preview_len:
                    preview += "..."
                text = f"- **{r.role}**: {preview}"

            txt_tokens = self.estimate_tokens(text)
            if tokens_used + txt_tokens <= budget:
                parts.append(text)
                tokens_used += txt_tokens

        if not parts:
            return ""

        dep_parts = [p for p in parts if p.startswith("###")]
        peer_parts = [p for p in parts if p.startswith("- ")]

        section = ""
        if dep_parts:
            section += "## Required context from other agents:\n\n"
            section += "\n\n".join(dep_parts)
        if peer_parts:
            if section:
                section += "\n\n"
            section += "## Other agents' outputs (for reference):\n\n"
            section += "\n".join(peer_parts)
        return section


@dataclass
class SwarmResult:
    """Final output from a swarm execution."""
    goal: str
    plan: SwarmPlan
    agent_results: list[AgentResult] = field(default_factory=list)
    synthesis: str = ""           # final aggregated output
    started_at: float = 0.0
    finished_at: float = 0.0
    total_tokens: int = 0
    mode: SwarmMode = SwarmMode.AUTO

    @property
    def duration(self) -> float:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        return 0.0

    @property
    def successful(self) -> list[AgentResult]:
        return [r for r in self.agent_results if r.status == AgentStatus.COMPLETED]

    @property
    def failed(self) -> list[AgentResult]:
        return [r for r in self.agent_results if r.status == AgentStatus.FAILED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "mode": self.mode.value,
            "duration_seconds": round(self.duration, 2),
            "agents_total": len(self.agent_results),
            "agents_succeeded": len(self.successful),
            "agents_failed": len(self.failed),
            "total_tokens": self.total_tokens,
            "strategy": self.plan.strategy,
            "agents": [
                {
                    "role": r.role,
                    "task": r.task,
                    "status": r.status.value,
                    "duration": round(r.duration, 2),
                    "attempt": r.attempt,
                    "content": r.content,
                    "error": r.error or None,
                }
                for r in self.agent_results
            ],
            "synthesis": self.synthesis,
        }
