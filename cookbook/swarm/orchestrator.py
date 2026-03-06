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
import logging
import re
import time
from typing import Any

import httpx

from .models import (
    AgentSpec, AgentResult, AgentStatus,
    SwarmPlan, SwarmResult, SwarmMode, BudgetTracker, ContextWindow,
)
from .config import SwarmConfig
from .engine import SwarmEngine
from .tool_registry import ToolRegistry, extract_tool_calls

log = logging.getLogger(__name__)


DECOMPOSE_SYSTEM = """You are a task orchestrator. Your job is to decompose a complex task into independent sub-tasks that can be executed in parallel by specialized AI agents.

Rules:
1. Create {min_agents}-{max_agents} agents, each with a clear specialist role and specific task
2. Tasks should be as INDEPENDENT as possible to maximize parallelism
3. If a task genuinely depends on another's output, specify the dependency
4. Each agent should produce a self-contained result
5. Assign roles that sound like job titles: "Security Researcher", "Code Architect", etc.
6. Use priority 3-5 for agents that other agents depend on (they run first)
7. Avoid redundant agents — never assign the same subtask to two agents
8. If tools are relevant, assign the most specific tool to each agent

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

Available tools agents can request: {available_tools}
IMPORTANT: Agents must use the EXACT tool names listed above. If tools like ig_timeline_feed, ig_like_post etc. are available, assign them to the appropriate agents.
Set depends_on to the INDEX (0-based) of agents this one must wait for.
Set priority 0 (low) to 5 (critical).

--- EXAMPLE 1: Security audit ---
Task: "Perform a security audit of the login system"
Good decomposition:
{{
  "strategy": "Parallel independent audits, then a dependent synthesis agent that needs all results",
  "agents": [
    {{"role": "Authentication Analyst", "task": "Analyze the authentication flow for weaknesses: password hashing, session management, token expiry, brute-force protections. List each finding with severity.", "depends_on": [], "priority": 4, "tools": ["analyze"]}},
    {{"role": "Input Validation Specialist", "task": "Review all login endpoints for injection attacks: SQL injection, XSS, LDAP injection, command injection. Test each input field.", "depends_on": [], "priority": 3, "tools": ["code"]}},
    {{"role": "Cryptography Reviewer", "task": "Audit cryptographic choices: TLS configuration, password hashing algorithm, key derivation, token signing. Compare against current OWASP recommendations.", "depends_on": [], "priority": 3, "tools": ["analyze"]}},
    {{"role": "Security Report Compiler", "task": "Consolidate all findings from the other agents into a prioritized vulnerability report with remediation steps.", "depends_on": [0, 1, 2], "priority": 1, "tools": []}}
  ]
}}

--- EXAMPLE 2: Research task ---
Task: "What are the latest breakthroughs in fusion energy?"
Good decomposition:
{{
  "strategy": "Independent researchers cover distinct subtopics in parallel — no dependencies needed",
  "agents": [
    {{"role": "Magnetic Confinement Researcher", "task": "Research recent breakthroughs in tokamak and stellarator designs. Cover ITER progress, SPARC/ARC, and any record-breaking plasma parameters.", "depends_on": [], "priority": 2, "tools": ["search"]}},
    {{"role": "Inertial Confinement Researcher", "task": "Research laser-driven and Z-pinch fusion progress. Cover NIF ignition results and any follow-up experiments.", "depends_on": [], "priority": 2, "tools": ["search"]}},
    {{"role": "Private Sector Analyst", "task": "Survey private fusion companies: Commonwealth Fusion, TAE Technologies, Helion, First Light. Cover funding, timelines, and technical approaches.", "depends_on": [], "priority": 2, "tools": ["search"]}},
    {{"role": "Materials Science Specialist", "task": "Research advances in plasma-facing materials, superconducting magnets (HTS), and tritium breeding blankets relevant to fusion reactors.", "depends_on": [], "priority": 1, "tools": ["search"]}}
  ]
}}

--- EXAMPLE 3: Code review ---
Task: "Review the AuthManager class for bugs and design issues"
Good decomposition:
{{
  "strategy": "Parallel analysis tracks covering different quality dimensions, with a dependent summary agent",
  "agents": [
    {{"role": "Bug Hunter", "task": "Find concrete bugs: null pointer risks, race conditions, resource leaks, off-by-one errors, exception handling gaps. For each bug, cite the exact code pattern.", "depends_on": [], "priority": 4, "tools": ["code"]}},
    {{"role": "Design Critic", "task": "Evaluate class design: single responsibility, coupling, cohesion, API ergonomics. Suggest specific refactors with before/after code.", "depends_on": [], "priority": 3, "tools": ["code"]}},
    {{"role": "Performance Analyst", "task": "Identify performance issues: unnecessary allocations, O(n²) loops, blocking calls, missing caching opportunities. Provide complexity analysis.", "depends_on": [], "priority": 2, "tools": ["analyze"]}},
    {{"role": "Review Synthesizer", "task": "Merge all findings into a unified code review document. Deduplicate overlapping findings, prioritize by impact, and write a final recommendation.", "depends_on": [0, 1, 2], "priority": 1, "tools": []}}
  ]
}}
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

{tool_section}

Instructions:
- You MUST use the tools provided to accomplish your task. Do NOT fabricate, hallucinate, or invent data.
- Every piece of data in your response must come from an actual tool call result.
- If a tool returns an error, report the error — do NOT make up what the result "would" be.
- NEVER invent IDs, usernames, URLs, numbers, or any other data. Only use what tools return.
- Structure your final response with clear headers and sections.
- Focus ONLY on your assigned task — do not overlap with other agents.
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

CRITIQUE_SYSTEM = """You are a plan quality reviewer. You will receive a task decomposition plan (a set of agents with roles, tasks, dependencies, and tools). Your job is to critique it and suggest improvements.

Check for:
1. REDUNDANCY — Are two agents doing essentially the same work? If so, suggest merging.
2. COVERAGE GAPS — Is an important aspect of the goal not assigned to any agent? If so, suggest adding one.
3. BAD DEPENDENCIES — Are dependencies correct? An agent should only depend on another if it truly needs that output. Unnecessary dependencies reduce parallelism.
4. TASK CLARITY — Are tasks specific enough? Vague tasks like "research the topic" produce poor results.
5. TOOL MISMATCH — Are the right tools assigned? An agent doing web research should have "search", not "code".

Output ONLY valid JSON:
{{
  "verdict": "accept" or "revise",
  "issues": [
    {{"type": "redundancy|gap|dependency|clarity|tool_mismatch", "description": "what's wrong", "fix": "how to fix it"}}
  ],
  "revised_agents": [
    {{
      "role": "Specialist Title",
      "task": "Specific detailed instructions",
      "depends_on": [],
      "priority": 0,
      "tools": []
    }}
  ]
}}

If the plan is good, set verdict to "accept" and leave issues and revised_agents as empty lists.
If the plan needs revision, set verdict to "revise" and provide the complete corrected agent list in revised_agents.
"""


class Orchestrator:
    """Decomposes tasks, runs agents, synthesizes results."""

    def __init__(
        self,
        config: SwarmConfig | None = None,
        tool_registry: ToolRegistry | None = None,
        budget: BudgetTracker | None = None,
        depth: int = 0,
    ):
        self.config = config or SwarmConfig.from_env()
        self.tool_registry = tool_registry
        self.budget = budget
        self.depth = depth  # current sub-swarm nesting depth (0 = root)
        self._client: httpx.AsyncClient | None = None

        # Context window — shared across all agent runs in this orchestrator
        if self.config.enable_context_pruning and self.config.context_window_tokens > 0:
            self.context_window: ContextWindow | None = ContextWindow(
                max_tokens=self.config.context_window_tokens,
                summary_max_tokens=self.config.context_summary_tokens,
            )
        else:
            self.context_window = None

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
        auto_continue: bool = True,
        max_continuations: int = 5,
    ) -> str:
        """Send a chat completion request with retry on server errors.

        If auto_continue=True (default), detects truncated responses
        (finish_reason='length') and automatically sends follow-up requests
        to get the complete output.  This fixes the DeepSeek "Continue"
        issue where responses are cut off mid-generation.
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
        max_retries = self.config.max_retries + 3  # more resilient for LLM calls
        empty_retries = 5  # dedicated budget for empty-response retries
        continuation = 0

        while continuation <= max_continuations:
            # ── HTTP-level retry loop ──
            last_error = None
            data = None
            for attempt in range(max_retries):
                try:
                    resp = await self.client.post("/chat/completions", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        import asyncio
                        delay = min(2 ** attempt + (attempt * 0.5), 30)
                        log.warning(f"_chat attempt {attempt+1}/{max_retries} failed: {e}, retrying in {delay:.1f}s")
                        await asyncio.sleep(delay)
            else:
                raise last_error  # type: ignore[misc]

            chunk = data["choices"][0]["message"]["content"] or ""
            finish_reason = data["choices"][0].get("finish_reason", "stop")

            # ── Empty-response retry (separate from continuation budget) ──
            if not chunk.strip() and not full_content.strip():
                if empty_retries > 0:
                    empty_retries -= 1
                    import asyncio as _aio
                    wait = 3.0 + (5 - empty_retries) * 2.0  # 3s, 5s, 7s, 9s, 11s
                    log.warning(
                        f"LLM returned empty response, retrying in {wait:.0f}s "
                        f"({empty_retries} retries left)..."
                    )
                    await _aio.sleep(wait)
                    continue  # retry without incrementing continuation
                else:
                    log.error("LLM returned empty response after all retries")

            # ── Dedup: strip repeated overlap from continuation chunks ──
            if full_content and chunk:
                chunk = _strip_overlap(full_content, chunk)

            full_content += chunk
            continuation += 1

            # If not truncated or auto_continue disabled, we're done
            if not auto_continue or finish_reason != "length":
                break

            # Truncated — send a continuation request
            log.info(
                f"Response truncated (finish_reason='length'), auto-continuing "
                f"({continuation}/{max_continuations})..."
            )
            payload["messages"] = list(messages) + [
                {"role": "assistant", "content": full_content},
                {"role": "user", "content": (
                    "Your previous response was cut off. Continue EXACTLY from "
                    "where you stopped — do NOT repeat anything already written. "
                    "Pick up mid-sentence if necessary."
                )},
            ]

        return full_content

    # ----- Decomposition -----

    async def decompose(self, goal: str, mode: SwarmMode = SwarmMode.AUTO) -> SwarmPlan:
        """Break a goal into a SwarmPlan with agent specs."""
        if mode == SwarmMode.MANUAL:
            raise ValueError("Manual mode requires passing a pre-built SwarmPlan")

        # Build list of actual available tool names for the decomposer
        available_tool_names = '["search", "code", "analyze"]'
        if self.tool_registry:
            real_tools = self.tool_registry.available()
            if real_tools:
                tool_names = [t.name for t in real_tools]
                tool_desc = [f'{t.name} — {t.description}' for t in real_tools]
                available_tool_names = json.dumps(tool_names)
                available_tool_names += '\nTool descriptions:\n' + '\n'.join(f'  - {d}' for d in tool_desc)

        system = DECOMPOSE_SYSTEM.format(
            min_agents=self.config.min_agents,
            max_agents=self.config.max_agents,
            available_tools=available_tool_names,
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

        # Optional plan critique — LLM reviews the plan before execution
        if self.config.enable_plan_critique:
            plan = await self._critique_plan(goal, plan)

        return plan

    async def _critique_plan(self, goal: str, plan: SwarmPlan) -> SwarmPlan:
        """Send the plan to an LLM critic for quality review.

        If the critic says "revise" and provides corrected agents, rebuild the plan.
        Otherwise return the original plan unchanged.
        """
        # Serialize plan for the critic
        agents_json = []
        for i, a in enumerate(plan.agents):
            # Convert agent_id deps back to indexes for the critic
            dep_indexes = []
            for dep_id in a.depends_on:
                for j, other in enumerate(plan.agents):
                    if other.agent_id == dep_id:
                        dep_indexes.append(j)
                        break
            agents_json.append({
                "role": a.role,
                "task": a.task,
                "depends_on": dep_indexes,
                "priority": a.priority,
                "tools": a.tools,
            })

        plan_summary = json.dumps({
            "strategy": plan.strategy,
            "agents": agents_json,
        }, indent=2)

        raw = await self._chat(
            messages=[
                {"role": "system", "content": CRITIQUE_SYSTEM},
                {"role": "user", "content": (
                    f"Goal: {goal}\n\n"
                    f"Proposed plan:\n{plan_summary}"
                )},
            ],
            temperature=0.3,
            is_orchestrator=True,
        )

        critique = _extract_json(raw)
        if not critique:
            log.warning("Plan critic returned unparseable response — keeping original plan")
            return plan

        verdict = critique.get("verdict", "accept")
        issues = critique.get("issues", [])

        if issues:
            for issue in issues:
                log.info(f"Plan critique [{issue.get('type', '?')}]: {issue.get('description', '')}")

        if verdict != "revise" or not critique.get("revised_agents"):
            return plan

        # Rebuild plan from critic's revised agents
        revised_raw = critique["revised_agents"]
        revised_raw = revised_raw[:self.config.max_agents]

        agents: list[AgentSpec] = []
        for i, ag in enumerate(revised_raw):
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

        for i, ag in enumerate(revised_raw):
            deps = ag.get("depends_on", [])
            for dep_idx in deps:
                if isinstance(dep_idx, int) and 0 <= dep_idx < len(agents):
                    agents[i].depends_on.append(agents[dep_idx].agent_id)

        revised_plan = SwarmPlan(
            goal=goal,
            agents=agents,
            strategy=critique.get("strategy", plan.strategy),
        )

        errors = revised_plan.validate()
        if errors:
            log.warning(f"Critic's revised plan is invalid ({errors}) — keeping original")
            return plan

        log.info(f"Plan critic revised plan: {len(plan.agents)} → {len(revised_plan.agents)} agents")
        return revised_plan

    # ----- Agent execution -----

    async def run_agent(
        self,
        agent: AgentSpec,
        completed_results: dict[str, AgentResult],
        config: SwarmConfig,
    ) -> str:
        """Run a single agent — this is the AgentRunner callback for the engine.

        If the agent has tools and a tool_registry is available, enters an
        iterative tool-calling loop: LLM → extract tool calls → execute →
        feed results → repeat until the LLM stops calling tools or the
        iteration cap is hit.
        """
        # Build context from dependencies
        if self.context_window:
            context_section = self.context_window.build_context(agent, completed_results)
        else:
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

        # Build tool section if agent has tools and registry is available
        has_tools = bool(agent.tools and self.tool_registry)
        # Filter out the "search" meta-tool — it controls model routing, not actual tool calls
        real_tools = [t for t in agent.tools if t.lower() != "search"] if has_tools else []
        has_tools = bool(real_tools and self.tool_registry)

        # If the registry has custom tools but the decomposer didn't assign any,
        # give this agent access to ALL registered tools so it can actually do its job
        if self.tool_registry and not has_tools:
            all_tools = self.tool_registry.available()
            if all_tools:
                real_tools = [t.name for t in all_tools]
                has_tools = True

        tool_section = ""
        if has_tools:
            tool_section = self.tool_registry.build_tool_prompt(real_tools)

        system = AGENT_SYSTEM.format(
            role=agent.role,
            task=agent.task,
            context_section=context_section,
            tool_section=tool_section,
        )

        # Pick model — search agents use search model, reasoning uses reasoning model
        model = agent.model or config.default_model
        if "search" in [t.lower() for t in agent.tools]:
            model = config.search_model
        elif "reasoning" in [t.lower() for t in agent.tools]:
            model = config.reasoning_model

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": agent.task},
        ]

        if not has_tools:
            # Simple path — single LLM call, no tool loop
            content = await self._chat(
                messages=messages,
                model=model,
                max_tokens=config.max_tokens,
            )
            if config.enable_reflection:
                content = await self._reflect(messages, content, model, config)
            return content

        # ── Tool-calling loop ──────────────────────────────────────
        tool_calls_made = 0
        max_iterations = config.max_tool_calls_per_agent
        _forced_retry = False  # track if we already forced a tool-call retry

        for iteration in range(max_iterations):
            llm_response = await self._chat(
                messages=messages,
                model=model,
                max_tokens=config.max_tokens,
            )
            messages.append({"role": "assistant", "content": llm_response})

            # Check for tool calls in the response
            tool_calls = extract_tool_calls(llm_response)
            if not tool_calls:
                # ── Tool-call enforcement ──
                # If this is the FIRST iteration and the agent hasn't called
                # any tools yet, it's likely hallucinating.  Force a retry.
                if iteration == 0 and tool_calls_made == 0 and not _forced_retry:
                    _forced_retry = True
                    log.warning(
                        f"Agent '{agent.role}' did not call any tools on first response. "
                        f"Forcing retry with explicit tool-call instruction."
                    )
                    messages.append({
                        "role": "user",
                        "content": (
                            "You did NOT call any tools. Your task REQUIRES using tools. "
                            "Do NOT describe what you would do — actually call the tool NOW.\n\n"
                            "Output ONLY a JSON tool call like this:\n"
                            '```json\n[{"tool": "tool_name", "args": {"param": "value"}}]\n```\n\n'
                            "Available tools: " + ", ".join(real_tools)
                        ),
                    })
                    continue  # retry this iteration

                # No tool calls — LLM is done, this is the final answer
                if config.enable_reflection:
                    llm_response = await self._reflect(messages, llm_response, model, config)
                return llm_response

            # Execute each tool call
            results_parts = []
            for call in tool_calls:
                tool_name = call.get("tool", "")
                args = call.get("args", {})
                tool_calls_made += 1

                log.info(f"Agent '{agent.role}' calling tool: {tool_name}({list(args.keys())})")
                result = await self.tool_registry.execute(
                    tool_name, args, timeout=config.tool_timeout,
                )
                results_parts.append(f"### {tool_name}\n{result}")

            # Feed results back to LLM
            results_text = "\n\n".join(results_parts)
            messages.append({
                "role": "user",
                "content": (
                    f"Tool results (iteration {iteration + 1}/{max_iterations}):\n\n"
                    f"{results_text}\n\n"
                    "Analyze these results. Call more tools if needed, "
                    "or provide your final answer (without JSON tool calls)."
                ),
            })

        # Max iterations reached — ask for final summary
        messages.append({
            "role": "user",
            "content": (
                f"Tool call limit ({max_iterations}) reached. "
                "Provide your final answer based on all results so far."
            ),
        })
        final = await self._chat(
            messages=messages,
            model=model,
            max_tokens=config.max_tokens,
        )
        return final

    async def _reflect(
        self,
        messages: list[dict[str, str]],
        content: str,
        model: str,
        config: SwarmConfig,
    ) -> str:
        """Self-reflection: agent reviews and revises its own output."""
        reflect_messages = list(messages) + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Review your output above for completeness, accuracy, and missed angles. "
                    "If improvements are needed, provide the revised version. "
                    "If your output is already satisfactory, reproduce it as-is."
                ),
            },
        ]
        return await self._chat(
            messages=reflect_messages,
            model=model,
            max_tokens=config.max_tokens,
        )

    # ----- Sub-swarm -----

    def _can_spawn_sub_swarm(self) -> bool:
        """Check if sub-swarm spawning is allowed at the current depth."""
        if not self.config.enable_sub_swarms:
            return False
        if self.depth >= self.config.sub_swarm_max_depth:
            return False
        if self.budget:
            err = self.budget.check_budget()
            if err:
                return False
        return True

    # ----- Context summarization -----

    async def _summarize_result(self, result: AgentResult) -> None:
        """Summarize a completed agent's output and cache it in the ContextWindow.

        Only called when context pruning is enabled and the output is large
        enough to benefit from summarization.
        """
        if not self.context_window:
            return
        # Skip if already cached or content is short enough
        if self.context_window.get_summary(result.agent_id):
            return
        target_chars = self.context_window.summary_max_tokens * ContextWindow.CHARS_PER_TOKEN
        if len(result.content) <= target_chars:
            # Content is already short — use it as its own summary
            await self.context_window.cache_summary(result.agent_id, result.content)
            return

        try:
            summary = await self._chat(
                messages=[
                    {"role": "system", "content": (
                        "Summarize the following agent output concisely, "
                        "preserving all key findings, data points, and conclusions. "
                        f"Target length: ~{self.context_window.summary_max_tokens} tokens."
                    )},
                    {"role": "user", "content": (
                        f"Agent role: {result.role}\n"
                        f"Agent task: {result.task}\n\n"
                        f"Output to summarize:\n{result.content}"
                    )},
                ],
                temperature=0.2,
                is_orchestrator=True,
            )
            await self.context_window.cache_summary(result.agent_id, summary)
            log.debug(
                f"Summarized '{result.role}' output: "
                f"{len(result.content)} → {len(summary)} chars"
            )
        except Exception as e:
            log.warning(f"Failed to summarize '{result.role}' output: {e}")

    async def _run_sub_swarm(self, goal: str) -> str:
        """Spawn a child swarm to handle a complex subtask.

        Creates a new Orchestrator at depth+1, sharing the same budget
        tracker and API client, with reduced agent limits.
        """
        if not self._can_spawn_sub_swarm():
            return (
                f"Sub-swarm not available (depth={self.depth}, "
                f"max_depth={self.config.sub_swarm_max_depth}, "
                f"enabled={self.config.enable_sub_swarms})"
            )

        # Create child config with reduced limits
        child_config = SwarmConfig(
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            default_model=self.config.default_model,
            search_model=self.config.search_model,
            reasoning_model=self.config.reasoning_model,
            orchestrator_model=self.config.orchestrator_model,
            max_parallel=min(self.config.max_parallel, 5),
            rate_limit_rpm=self.config.rate_limit_rpm,
            rate_limit_burst=self.config.rate_limit_burst,
            max_retries=self.config.max_retries,
            agent_timeout=self.config.agent_timeout,
            swarm_timeout=self.config.agent_timeout * 2,  # sub-swarm timeout
            max_agents=self.config.sub_swarm_max_agents,
            min_agents=2,
            allow_replan=False,  # no re-planning in sub-swarms
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            adaptive_rate_limit=self.config.adaptive_rate_limit,
            circuit_breaker_enabled=self.config.circuit_breaker_enabled,
            enable_plan_critique=False,  # no critique overhead in sub-swarms
            enable_agent_killing=self.config.enable_agent_killing,
            agent_kill_threshold=self.config.agent_kill_threshold,
            enable_sub_swarms=self.config.enable_sub_swarms,
            sub_swarm_max_depth=self.config.sub_swarm_max_depth,
            sub_swarm_max_agents=self.config.sub_swarm_max_agents,
            token_budget=0,  # budget enforcement via shared BudgetTracker
            max_llm_calls=0,
            # Disable file output for sub-swarms
            save_json=False,
            save_markdown=False,
            stream_to_terminal=False,
        )

        child_orc = Orchestrator(
            config=child_config,
            tool_registry=self.tool_registry,
            budget=self.budget,
            depth=self.depth + 1,
        )

        try:
            async with child_orc:
                result = await child_orc.run(goal=goal, mode=SwarmMode.AUTO)

            log.info(
                f"Sub-swarm (depth {self.depth + 1}) completed: "
                f"{len(result.successful)}/{len(result.agent_results)} agents, "
                f"{result.duration:.1f}s"
            )
            return result.synthesis or "Sub-swarm produced no synthesis."
        except Exception as e:
            log.warning(f"Sub-swarm failed: {e}")
            return f"Sub-swarm failed: {e}"

    def register_sub_swarm_tool(self, registry: ToolRegistry) -> None:
        """Register the spawn_sub_swarm tool in the given registry."""
        from .tool_registry import ToolDef

        async def _sub_swarm_handler(goal: str = "", **kwargs) -> str:
            return await self._run_sub_swarm(goal)

        tool = ToolDef(
            name="spawn_sub_swarm",
            description=(
                "Spawn a sub-swarm of AI agents to collaboratively tackle a complex subtask. "
                "Use this when a task is too complex for a single agent and would benefit from "
                "multiple specialists working in parallel. Provide a clear goal description."
            ),
            parameters={"goal": "The complex subtask for the sub-swarm to solve"},
            fn=_sub_swarm_handler,
        )
        registry.register(tool)

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

        # Step 1.5: Register sub-swarm tool if enabled and depth allows
        if self._can_spawn_sub_swarm():
            if self.tool_registry is None:
                self.tool_registry = ToolRegistry()
            if "spawn_sub_swarm" not in self.tool_registry._tools:
                self.register_sub_swarm_tool(self.tool_registry)

        # Wrap on_done to trigger context summarization after each agent completes
        _user_on_done = on_done
        if self.context_window:
            async def _on_done_with_summary(result: AgentResult):
                if result.status == AgentStatus.COMPLETED:
                    await self._summarize_result(result)
                if _user_on_done:
                    from .engine import _maybe_await
                    _maybe_await(_user_on_done(result))
            effective_on_done = _on_done_with_summary
        else:
            effective_on_done = on_done

        # Step 2: Execute via engine
        engine = SwarmEngine(
            config=self.config,
            runner=self.run_agent,
            on_start=on_start,
            on_done=effective_on_done,
            on_retry=on_retry,
            budget=self.budget,
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
                    on_done=effective_on_done,
                    on_retry=on_retry,
                    budget=self.budget,
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


def _strip_overlap(existing: str, continuation: str) -> str:
    """Remove overlapping prefix from a continuation chunk.

    When DeepSeek 'continues' it sometimes repeats the last few sentences.
    We find the longest suffix of *existing* that matches a prefix of
    *continuation* and strip it so the join is seamless.
    """
    if not existing or not continuation:
        return continuation
    # Check up to 500 chars of overlap (more than enough for repeated sentences)
    max_check = min(len(existing), len(continuation), 500)
    best = 0
    for length in range(1, max_check + 1):
        if existing[-length:] == continuation[:length]:
            best = length
    if best > 0:
        log.debug(f"Stripped {best} chars of repeated overlap from continuation")
        return continuation[best:]
    return continuation


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
