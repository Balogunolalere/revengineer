"""
Swarm Orchestrator — the brain.

Handles:
- Task decomposition into parallel sub-agents via LLM
- Dynamic agent running via OpenAI-compatible API
- Result synthesis into final output
- Optional iterative re-planning
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from .models import (
    AgentSpec, AgentResult, AgentStatus,
    SwarmPlan, SwarmResult, SwarmMode,
)
from .config import SwarmConfig
from .engine import SwarmEngine


DECOMPOSE_SYSTEM = """You are a task orchestrator. Your job is to decompose a complex task into independent sub-tasks that can be executed in parallel by specialized AI agents.

Rules:
1. Create {min_agents}-{max_agents} agents, each with a clear specialist role and specific task
2. Tasks should be as INDEPENDENT as possible to maximize parallelism
3. If a task genuinely depends on another's output, specify the dependency
4. Each agent should produce a self-contained result
5. Assign roles that sound like job titles: "Security Researcher", "Code Architect", etc.

Output ONLY valid JSON (no markdown fencing):
{{
  "strategy": "Brief explanation of your decomposition approach",
  "agents": [
    {{
      "role": "Specialist Title",
      "task": "Specific detailed instructions for this agent",
      "depends_on": [],
      "priority": 0,
      "tools": []
    }}
  ]
}}

Available tools agents can request: ["search", "code", "analyze"]
Set depends_on to the INDEX (0-based) of agents this one must wait for.
Set priority higher (1-5) for more critical agents.
"""

REPLAN_SYSTEM = """You are a task orchestrator reviewing the first round of agent outputs. Your job is to identify GAPS, unanswered questions, or areas that need deeper investigation.

Rules:
1. Only create new agents if there are genuine gaps or important follow-up questions
2. Create 0-{max_agents} additional agents
3. New agents can reference existing outputs (provided as context)
4. If the existing outputs are comprehensive enough, return an empty agents list

Existing agent outputs:
{existing_outputs}

Output ONLY valid JSON (no markdown fencing):
{{
  "strategy": "What gaps you identified and how new agents address them",
  "agents": [
    {{
      "role": "Specialist Title",
      "task": "Specific detailed instructions referencing prior findings",
      "depends_on": [],
      "priority": 0,
      "tools": []
    }}
  ]
}}
"""

AGENT_SYSTEM = """You are {role}.

Your specific task: {task}

{context_section}

Instructions:
- Be thorough and detailed in your analysis
- Use concrete examples, data, and evidence
- Structure your response with clear headers and sections
- If you reference sources, cite them
- Focus ONLY on your assigned task — do not overlap with other agents
"""

SYNTHESIS_SYSTEM = """You are a research synthesizer. You've received outputs from multiple specialized agents who worked on different aspects of a complex task.

Your job:
1. Combine all agent outputs into a single coherent, well-structured report
2. Resolve any contradictions between agents
3. Remove redundancy while preserving all unique insights
4. Add an executive summary at the top
5. Organize with clear headers and logical flow
6. Cite which agent/specialist contributed each insight where relevant

Output a polished, comprehensive final report in markdown format.
"""


class Orchestrator:
    """Decomposes tasks, runs agents, synthesizes results."""

    def __init__(self, config: SwarmConfig | None = None):
        self.config = config or SwarmConfig.from_env()
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.config.agent_timeout + 10),
        )
        return self

    async def __aexit__(self, *exc):
        if self._client:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("Use 'async with Orchestrator(...) as o:' or call __aenter__")
        return self._client

    # ----- LLM calls -----

    async def _chat(
        self,
        messages: list[dict[str, str]],
        model: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        is_orchestrator: bool = False,
        auto_continue: bool = False,
        max_continuations: int = 3,
    ) -> str:
        """Send a chat completion request with retry on server errors.

        If auto_continue=True, detects truncated responses (finish_reason='length')
        and automatically sends follow-up requests to get the complete output.
        """
        if is_orchestrator and self.config.orchestrator_model:
            effective_model = self.config.orchestrator_model
        else:
            effective_model = model or self.config.default_model

        payload = {
            "model": effective_model,
            "messages": list(messages),  # copy so we can extend for continuations
            "temperature": temperature if temperature is not None else self.config.temperature,
            "stream": False,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        full_content = ""

        for continuation in range(max_continuations + 1):
            last_error = None
            for attempt in range(3):
                try:
                    resp = await self.client.post("/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        import asyncio
                        await asyncio.sleep(2 ** attempt)
            else:
                raise last_error  # type: ignore[misc]

            chunk = data["choices"][0]["message"]["content"]
            finish_reason = data["choices"][0].get("finish_reason", "stop")
            full_content += chunk

            # If not truncated or auto_continue disabled, we're done
            if not auto_continue or finish_reason != "length":
                break

            # Truncated — send a continuation request
            # Append the assistant's partial response and ask to continue
            payload["messages"] = list(messages) + [
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": "Continue from exactly where you left off. Do not repeat any content already written."},
            ]

        return full_content

    # ----- Decomposition -----

    async def decompose(self, goal: str, mode: SwarmMode = SwarmMode.AUTO) -> SwarmPlan:
        """Break a goal into a SwarmPlan with agent specs."""
        if mode == SwarmMode.MANUAL:
            raise ValueError("Manual mode requires passing a pre-built SwarmPlan")

        system = DECOMPOSE_SYSTEM.format(
            min_agents=self.config.min_agents,
            max_agents=self.config.max_agents,
        )

        raw = await self._chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Task to decompose:\n\n{goal}"},
            ],
            temperature=0.4,  # more deterministic for planning
            is_orchestrator=True,
        )

        # Parse JSON from response
        plan_data = _extract_json(raw)
        if not plan_data or "agents" not in plan_data:
            raise RuntimeError(f"Orchestrator failed to produce valid plan. Raw:\n{raw[:500]}")

        agents: list[AgentSpec] = []
        raw_agents = plan_data["agents"]

        # Enforce min_agents — if the LLM returned too few, pad with a generic agent
        if len(raw_agents) < self.config.min_agents:
            for i in range(self.config.min_agents - len(raw_agents)):
                raw_agents.append({
                    "role": f"Additional Analyst {i + 1}",
                    "task": (
                        f"Provide an independent perspective on the following task "
                        f"that the other agents may have missed: {goal}"
                    ),
                    "depends_on": [],
                    "priority": 0,
                    "tools": [],
                })

        # Cap at max_agents
        raw_agents = raw_agents[:self.config.max_agents]

        # First pass: create specs
        for i, ag in enumerate(raw_agents):
            spec = AgentSpec(
                role=ag.get("role", f"Agent-{i}"),
                task=ag.get("task", ""),
                priority=ag.get("priority", 0),
                tools=ag.get("tools", []),
                timeout=self.config.agent_timeout,
                max_retries=self.config.max_retries,
                model=ag.get("model", ""),
            )
            agents.append(spec)

        # Second pass: resolve depends_on (indexes → agent_ids)
        for i, ag in enumerate(raw_agents):
            deps = ag.get("depends_on", [])
            for dep_idx in deps:
                if isinstance(dep_idx, int) and 0 <= dep_idx < len(agents):
                    agents[i].depends_on.append(agents[dep_idx].agent_id)

        plan = SwarmPlan(
            goal=goal,
            agents=agents,
            strategy=plan_data.get("strategy", ""),
        )

        errors = plan.validate()
        if errors:
            raise RuntimeError(f"Invalid plan: {'; '.join(errors)}")

        return plan

    # ----- Agent execution -----

    async def run_agent(
        self,
        agent: AgentSpec,
        completed_results: dict[str, AgentResult],
        config: SwarmConfig,
    ) -> str:
        """Run a single agent — this is the AgentRunner callback for the engine."""
        # Build context from dependencies
        context_parts = []
        for dep_id in agent.depends_on:
            dep_result = completed_results.get(dep_id)
            if dep_result and dep_result.status == AgentStatus.COMPLETED:
                context_parts.append(
                    f"### Output from {dep_result.role}:\n{dep_result.content}"
                )

        # Also provide summaries of _all_ completed agents (shared context)
        other_parts = []
        for aid, result in completed_results.items():
            if aid not in agent.depends_on and result.status == AgentStatus.COMPLETED:
                # Short summary for non-dependency peers
                preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
                other_parts.append(f"- **{result.role}** (completed): {preview}")

        context_section = ""
        if context_parts:
            context_section += "## Required context from other agents:\n\n"
            context_section += "\n\n".join(context_parts)
        if other_parts:
            context_section += "\n\n## Other agents' outputs (for reference):\n\n"
            context_section += "\n".join(other_parts)

        system = AGENT_SYSTEM.format(
            role=agent.role,
            task=agent.task,
            context_section=context_section,
        )

        # Pick model — search agents use search model
        model = agent.model or config.default_model
        if "search" in [t.lower() for t in agent.tools]:
            model = config.search_model

        return await self._chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": agent.task},
            ],
            model=model,
            max_tokens=config.max_tokens,
        )

    # ----- Synthesis -----

    async def synthesize(
        self, goal: str, results: list[AgentResult]
    ) -> str:
        """Combine all agent outputs into a final report."""
        successful = [r for r in results if r.status == AgentStatus.COMPLETED]
        if not successful:
            return "⚠️ All agents failed. No results to synthesize."

        agent_outputs = []
        for r in successful:
            agent_outputs.append(
                f"## {r.role}\n"
                f"**Task:** {r.task}\n"
                f"**Duration:** {r.duration:.1f}s\n\n"
                f"{r.content}"
            )

        combined = "\n\n---\n\n".join(agent_outputs)

        return await self._chat(
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM},
                {"role": "user", "content": (
                    f"# Original Goal\n{goal}\n\n"
                    f"# Agent Outputs ({len(successful)} agents)\n\n{combined}"
                )},
            ],
            max_tokens=self.config.max_tokens * 2,  # synthesis can be longer
            temperature=0.3,  # coherent synthesis
            is_orchestrator=True,
            auto_continue=True,  # auto-continue if synthesis gets truncated
            max_continuations=3,
        )

    # ----- Full swarm run -----

    async def run(
        self,
        goal: str,
        mode: SwarmMode = SwarmMode.AUTO,
        plan: SwarmPlan | None = None,
        on_start=None,
        on_done=None,
        on_retry=None,
        on_synthesis_start=None,
    ) -> SwarmResult:
        """Full swarm execution: decompose → execute → [re-plan →] synthesize."""
        swarm_start = time.time()

        # Step 1: Decompose
        if plan is None:
            plan = await self.decompose(goal, mode)

        # Step 2: Execute via engine
        engine = SwarmEngine(
            config=self.config,
            runner=self.run_agent,
            on_start=on_start,
            on_done=on_done,
            on_retry=on_retry,
        )

        agent_results = await engine.execute(plan)

        # Step 2.5: ITERATIVE re-planning — review results and spawn more agents
        if mode == SwarmMode.ITERATIVE and self.config.allow_replan:
            for _ in range(self.config.replan_max):
                new_agents = await self._replan(goal, agent_results)
                if not new_agents:
                    break  # orchestrator decided no gaps remain

                # Create a supplementary plan with just the new agents
                supplement = SwarmPlan(
                    goal=goal,
                    agents=new_agents,
                    strategy="Iterative gap-filling round",
                )
                errors = supplement.validate()
                if errors:
                    break

                # Execute the new agents (they get all prior results as context)
                new_engine = SwarmEngine(
                    config=self.config,
                    runner=self.run_agent,
                    on_start=on_start,
                    on_done=on_done,
                    on_retry=on_retry,
                )
                # Inject prior results so new agents can see them
                new_engine._results = {r.agent_id: r for r in agent_results
                                        if r.status == AgentStatus.COMPLETED}

                new_results = await new_engine.execute(supplement)
                agent_results.extend(new_results)
                plan.agents.extend(new_agents)

        # Step 3: Synthesize
        if on_synthesis_start:
            from .engine import _maybe_await
            _maybe_await(on_synthesis_start())
        synthesis = await self.synthesize(goal, agent_results)

        return SwarmResult(
            goal=goal,
            plan=plan,
            agent_results=agent_results,
            synthesis=synthesis,
            started_at=swarm_start,
            finished_at=time.time(),
            total_tokens=sum(r.tokens_used for r in agent_results),
            mode=mode,
        )

    async def _replan(
        self, goal: str, results: list[AgentResult]
    ) -> list[AgentSpec]:
        """Review existing results and create additional agents for gaps."""
        successful = [r for r in results if r.status == AgentStatus.COMPLETED]
        if not successful:
            return []

        existing_outputs = "\n\n---\n\n".join(
            f"**{r.role}:** {r.content[:500]}{'...' if len(r.content) > 500 else ''}"
            for r in successful
        )

        remaining_budget = self.config.max_agents - len(results)
        if remaining_budget <= 0:
            return []

        system = REPLAN_SYSTEM.format(
            max_agents=min(remaining_budget, 5),
            existing_outputs=existing_outputs,
        )

        raw = await self._chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    f"Original goal: {goal}\n\n"
                    f"Review the {len(successful)} agent outputs above. "
                    f"Are there gaps that need additional agents?"
                )},
            ],
            temperature=0.4,
            is_orchestrator=True,
        )

        plan_data = _extract_json(raw)
        if not plan_data or not plan_data.get("agents"):
            return []

        new_agents = []
        for ag in plan_data["agents"]:
            spec = AgentSpec(
                role=ag.get("role", "Supplementary Agent"),
                task=ag.get("task", ""),
                priority=ag.get("priority", 0),
                tools=ag.get("tools", []),
                timeout=self.config.agent_timeout,
                max_retries=self.config.max_retries,
            )
            new_agents.append(spec)

        return new_agents


# ----- Helpers -----

def _extract_json(text: str) -> dict | None:
    """Extract JSON from LLM output, handling markdown fences."""
    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` blocks
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first { ... } block
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = -1

    return None
