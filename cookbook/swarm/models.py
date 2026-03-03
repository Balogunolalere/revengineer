"""
Data models for the swarm system.

AgentSpec   — defines a sub-agent's role, task, and dependencies
AgentResult — output from a completed agent
SwarmPlan   — the orchestrator's decomposition of a task
SwarmResult — final aggregated output
"""

from __future__ import annotations

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
    timeout: float = 120.0                 # per-agent timeout in seconds
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
                    "content_length": len(r.content),
                    "error": r.error or None,
                }
                for r in self.agent_results
            ],
            "synthesis": self.synthesis,
        }
