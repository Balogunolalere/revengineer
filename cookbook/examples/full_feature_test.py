#!/usr/bin/env python3
"""
Full-feature swarm stress test — exercises every capability added
across Phases 0-5 in a single multi-stage run.

Features tested:
  - AUTO decomposition with few-shot prompts (Phase 5)
  - Plan critique / LLM-reviewed plan (Phase 5)
  - Adaptive rate limiting + circuit breaker (Phase 3)
  - Agent killing of slow stragglers (Phase 5)
  - Budget tracking with token + call limits (Phase 2)
  - Sub-swarm spawning from within agents (Phase 2)
  - Token-aware context pruning + auto-summarization (Phase 4)
  - ITERATIVE re-planning (gap-filling rounds)
  - Manual agent graph with complex dependency chains
  - search / reasoning model routing
  - Tool calling via ToolRegistry (Phase 1)
  - Live renderer callbacks

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Run this script:                uv run python cookbook/examples/full_feature_test.py

Estimated: ~15 agents, 2-5 minutes depending on model speed.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

from cookbook.swarm import (
    Swarm,
    Orchestrator,
    SwarmConfig,
    SwarmEngine,
    SwarmMode,
    SwarmPlan,
    SwarmResult,
    AgentSpec,
    AgentResult,
    AgentStatus,
    BudgetTracker,
    ContextWindow,
    ToolRegistry,
    ToolDef,
)

# ─── Utility ────────────────────────────────────────────────────

DIVIDER = "═" * 70

def banner(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def print_result(result: SwarmResult, label: str = "") -> None:
    tag = f" ({label})" if label else ""
    print(f"\n  Duration:    {result.duration:.1f}s")
    print(f"  Agents:      {len(result.successful)} ok / {len(result.failed)} failed / {len(result.agent_results)} total")
    print(f"  Tokens:      {result.total_tokens}")
    print(f"  Strategy:    {result.plan.strategy[:120]}")
    for r in result.agent_results:
        icon = "✓" if r.status == AgentStatus.COMPLETED else "✗" if r.status == AgentStatus.FAILED else "⊘"
        dur = f"{r.duration:.1f}s" if r.duration else "n/a"
        preview = r.content[:80].replace("\n", " ") if r.content else r.error[:80]
        print(f"    {icon} [{dur:>6}] {r.role}: {preview}...")
    print(f"\n  Synthesis preview{tag}:")
    print(f"    {result.synthesis[:300]}...\n" if len(result.synthesis) > 300 else f"    {result.synthesis}\n")


# ─── Custom tools ───────────────────────────────────────────────

def build_custom_tools() -> ToolRegistry:
    """Create a ToolRegistry with a few lightweight custom tools."""
    registry = ToolRegistry()

    # 1) A "knowledge base" lookup tool
    async def lookup_kb(query: str = "", **kw) -> str:
        """Simulates a knowledge-base search."""
        await asyncio.sleep(0.1)  # simulate I/O
        entries = {
            "quantum": "Quantum computing uses qubits that can exist in superposition. "
                       "Key players: IBM (127-qubit Eagle), Google (Sycamore), IonQ (trapped ions).",
            "fusion": "Fusion energy: NIF achieved ignition Dec 2022. ITER construction ongoing. "
                      "Private: Commonwealth Fusion (SPARC), Helion (FRC approach), TAE Technologies.",
            "llm": "Large language models: GPT-4 (OpenAI), Claude (Anthropic), Gemini (Google), "
                   "DeepSeek-V3, Llama 3 (Meta). Trends: MoE, long context, tool use, agents.",
            "cyber": "Cybersecurity 2025: rise of AI-powered phishing, supply chain attacks, "
                     "zero-trust adoption, CISA KEV catalog expansion, ransomware-as-a-service.",
        }
        for key, value in entries.items():
            if key in query.lower():
                return value
        return f"No knowledge base entry found for: {query}"

    registry.register(ToolDef(
        name="lookup_kb",
        description="Search an internal knowledge base for factual information on a topic.",
        parameters={"query": "The topic or question to look up"},
        fn=lookup_kb,
    ))

    # 2) A "calculator" tool
    async def calculate(expression: str = "", **kw) -> str:
        """Evaluate a safe math expression."""
        # Only allow digits, operators, parens, dots
        import re
        clean = re.sub(r"[^0-9+\-*/().%e ]", "", expression)
        if not clean:
            return "Invalid expression"
        try:
            result = eval(clean, {"__builtins__": {}}, {})  # noqa: S307 — restricted eval
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    registry.register(ToolDef(
        name="calculate",
        description="Evaluate a mathematical expression and return the result.",
        parameters={"expression": "A mathematical expression like '2 * 3 + 1'"},
        fn=calculate,
    ))

    # 3) A "summarize_text" tool
    async def summarize_text(text: str = "", max_length: str = "200", **kw) -> str:
        """Truncate text to a max length with an ellipsis."""
        length = int(max_length)
        if len(text) <= length:
            return text
        return text[:length].rstrip() + "..."

    registry.register(ToolDef(
        name="summarize_text",
        description="Truncate a long text to a specified maximum character length.",
        parameters={
            "text": "The text to summarize",
            "max_length": "Maximum character count (default 200)",
        },
        fn=summarize_text,
    ))

    return registry


# ═════════════════════════════════════════════════════════════════
# Stage 1: AUTO mode with all Phase 3-5 features
# ═════════════════════════════════════════════════════════════════

async def stage_auto_with_features() -> SwarmResult:
    """AUTO decomposition with plan critique, adaptive rate limiting,
    circuit breaker, agent killing, and context pruning."""

    banner("Stage 1: AUTO + Plan Critique + Context Pruning + Agent Killing")

    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_agents=8,
        max_parallel=4,

        # Phase 3: Adaptive concurrency
        adaptive_rate_limit=True,
        rate_limit_rpm=30,
        rate_limit_min_rpm=5,
        circuit_breaker_enabled=True,
        circuit_breaker_threshold=5,

        # Phase 5: Smarter decomposition
        enable_plan_critique=True,           # LLM reviews the plan
        enable_agent_killing=True,           # kill slow agents
        agent_kill_threshold=5.0,            # kill if >5× median

        # Phase 4: Context pruning
        enable_context_pruning=True,
        context_window_tokens=2000,          # ~8K chars budget per agent
        context_summary_tokens=200,

        # Output
        save_json=False,
        save_markdown=False,
        stream_to_terminal=True,
        verbose=True,
    )

    result = await Swarm(
        "Analyze the current state and future trajectory of artificial general "
        "intelligence (AGI). Cover: (1) leading architectures and scaling laws, "
        "(2) safety and alignment research, (3) economic implications and job "
        "displacement forecasts, (4) regulatory landscape across US/EU/China, "
        "(5) the role of open-source models vs closed labs.",
        config=config,
        verbose=True,
    )

    print_result(result, "AUTO + critique + pruning")
    return result


# ═════════════════════════════════════════════════════════════════
# Stage 2: MANUAL mode with dependency chains + tool calling
# ═════════════════════════════════════════════════════════════════

async def stage_manual_with_tools() -> SwarmResult:
    """Manual agent graph with custom tools and complex dependencies."""

    banner("Stage 2: MANUAL + Tools + Dependency Chain")

    registry = build_custom_tools()

    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=5,
        max_tool_calls_per_agent=5,
        tool_timeout=30.0,
        tool_agent_timeout=300.0,
        enable_agent_killing=True,
        agent_kill_threshold=5.0,
        save_json=False,
        save_markdown=False,
        stream_to_terminal=True,
    )

    # Layer 0: Independent researchers (run in parallel)
    quantum_expert = AgentSpec(
        role="Quantum Computing Researcher",
        task=(
            "Use the lookup_kb tool to find information about quantum computing, "
            "then expand on the current state of quantum hardware, error correction, "
            "and quantum advantage claims. Cite specific qubit counts and fidelity metrics."
        ),
        tools=["lookup_kb", "search"],
        priority=5,
    )

    fusion_expert = AgentSpec(
        role="Fusion Energy Researcher",
        task=(
            "Use the lookup_kb tool to find information about fusion energy, "
            "then provide a detailed analysis of the path to commercial fusion power. "
            "Cover tokamaks, inertial confinement, and private fusion ventures. "
            "Use the calculate tool to estimate energy breakeven ratios."
        ),
        tools=["lookup_kb", "calculate"],
        priority=5,
    )

    ai_expert = AgentSpec(
        role="AI/ML Researcher",
        task=(
            "Use the lookup_kb tool to research large language models, "
            "then analyze the trajectory of AI capabilities. Cover: "
            "scaling laws, MoE architectures, multimodal models, "
            "and the open-source vs proprietary dynamics."
        ),
        tools=["lookup_kb", "search"],
        priority=5,
    )

    cyber_expert = AgentSpec(
        role="Cybersecurity Analyst",
        task=(
            "Use the lookup_kb tool to research cybersecurity trends, "
            "then provide a threat landscape analysis for 2025-2026. "
            "Focus on AI-powered attacks, supply chain risks, and zero-trust."
        ),
        tools=["lookup_kb"],
        priority=4,
    )

    # Layer 1: Cross-domain analyst (depends on ALL layer 0 agents)
    cross_analyst = AgentSpec(
        role="Cross-Domain Analyst",
        task=(
            "You have access to outputs from a quantum computing researcher, "
            "fusion energy researcher, AI/ML researcher, and cybersecurity analyst. "
            "Identify the TOP 3 cross-domain synergies where breakthroughs in one "
            "field could accelerate another. For example: quantum computing + "
            "cryptography, AI + fusion simulation, etc. Be specific and quantitative."
        ),
        depends_on=[
            quantum_expert.agent_id,
            fusion_expert.agent_id,
            ai_expert.agent_id,
            cyber_expert.agent_id,
        ],
        priority=2,
    )

    # Layer 2: Final strategic advisor (depends on analyst)
    strategist = AgentSpec(
        role="Technology Strategist",
        task=(
            "Based on the cross-domain analysis, draft a 5-year strategic "
            "investment thesis for a $10B technology fund. Allocate percentage "
            "weights to quantum, fusion, AI, and cybersecurity sectors. "
            "Use the calculate tool to verify your allocation sums to 100%. "
            "Justify each allocation with evidence from the researchers."
        ),
        depends_on=[cross_analyst.agent_id],
        tools=["calculate"],
        priority=1,
    )

    plan = SwarmPlan(
        goal="Multi-domain technology analysis and investment thesis",
        agents=[quantum_expert, fusion_expert, ai_expert, cyber_expert, cross_analyst, strategist],
        strategy="Layer 0: parallel domain research → Layer 1: cross-domain synthesis → Layer 2: strategic output",
    )

    errors = plan.validate()
    if errors:
        print(f"  Plan validation errors: {errors}")
        return None

    result = await Swarm(
        plan.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan,
        tool_registry=registry,
        verbose=True,
    )

    print_result(result, "MANUAL + tools")
    return result


# ═════════════════════════════════════════════════════════════════
# Stage 3: ITERATIVE mode with budget tracking + sub-swarms
# ═════════════════════════════════════════════════════════════════

async def stage_iterative_with_budget() -> SwarmResult:
    """Iterative re-planning with global budget limits and sub-swarm spawning."""

    banner("Stage 3: ITERATIVE + Budget Tracking + Sub-Swarms")

    budget = BudgetTracker(
        token_budget=500_000,    # 500K token ceiling
        max_llm_calls=50,        # max 50 LLM calls total
        max_agents=15,           # max 15 agents (including sub-swarm agents)
    )

    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_agents=8,
        max_parallel=4,

        # Phase 2: Sub-swarms
        enable_sub_swarms=True,
        sub_swarm_max_depth=1,       # allow 1 level of nesting
        sub_swarm_max_agents=3,      # up to 3 agents per sub-swarm

        # Phase 3: Adaptive
        adaptive_rate_limit=True,
        rate_limit_rpm=30,
        circuit_breaker_enabled=True,

        # Phase 4: Context
        enable_context_pruning=True,
        context_window_tokens=1500,
        context_summary_tokens=150,

        # Phase 5
        enable_plan_critique=False,   # skip critique for speed
        enable_agent_killing=True,
        agent_kill_threshold=5.0,

        # Re-planning
        allow_replan=True,
        replan_max=1,                 # 1 re-plan round

        # Output
        save_json=False,
        save_markdown=False,
        stream_to_terminal=True,
    )

    result = await Swarm(
        "Design a next-generation cybersecurity operations center (SOC) for a "
        "Fortune 500 company. The SOC must handle: real-time threat detection "
        "using AI/ML, automated incident response playbooks, threat intelligence "
        "fusion from multiple feeds, compliance with SOC 2 / ISO 27001 / NIST CSF, "
        "and a 24/7 staffing model with burnout mitigation. Provide architecture "
        "diagrams (in text), technology stack recommendations, staffing plan with "
        "cost estimates, and a 12-month implementation roadmap.",
        config=config,
        mode=SwarmMode.ITERATIVE,
        budget=budget,
        verbose=True,
    )

    print_result(result, "ITERATIVE + budget + sub-swarms")

    # Print budget stats
    stats = budget.stats
    print("  Budget consumption:")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    return result


# ═════════════════════════════════════════════════════════════════
# Stage 4: Low-level orchestrator with all bells and whistles
# ═════════════════════════════════════════════════════════════════

async def stage_low_level_orchestrator() -> SwarmResult:
    """Direct Orchestrator usage with custom callbacks and full observability."""

    banner("Stage 4: Low-Level Orchestrator + Custom Callbacks")

    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_agents=6,
        max_parallel=3,
        adaptive_rate_limit=True,
        rate_limit_rpm=20,
        circuit_breaker_enabled=True,
        enable_plan_critique=True,
        enable_agent_killing=True,
        agent_kill_threshold=5.0,
        enable_context_pruning=True,
        context_window_tokens=1000,
        context_summary_tokens=150,
        save_json=False,
        save_markdown=False,
        stream_to_terminal=False,  # we'll handle output ourselves
    )

    budget = BudgetTracker(max_llm_calls=40, max_agents=10)

    # Custom event tracking
    events: list[dict] = []
    t0 = time.time()

    async def on_start(agent: AgentSpec):
        elapsed = time.time() - t0
        events.append({"t": elapsed, "event": "start", "role": agent.role})
        print(f"  [{elapsed:6.1f}s] ▶ Starting: {agent.role}")

    async def on_done(result: AgentResult):
        elapsed = time.time() - t0
        icon = "✓" if result.status == AgentStatus.COMPLETED else "✗"
        events.append({
            "t": elapsed,
            "event": "done",
            "role": result.role,
            "status": result.status.value,
            "duration": result.duration,
        })
        print(f"  [{elapsed:6.1f}s] {icon} Finished: {result.role} ({result.duration:.1f}s)")

    async def on_retry(agent: AgentSpec, attempt: int, error: Exception):
        elapsed = time.time() - t0
        events.append({"t": elapsed, "event": "retry", "role": agent.role, "attempt": attempt})
        print(f"  [{elapsed:6.1f}s] ↻ Retry #{attempt}: {agent.role} — {error}")

    async def on_synthesis():
        elapsed = time.time() - t0
        print(f"  [{elapsed:6.1f}s] ⚙ Synthesizing final report...")

    async with Orchestrator(config=config, budget=budget) as orc:
        # Decompose
        plan = await orc.decompose(
            "Create a comprehensive comparison of container orchestration platforms "
            "in 2025: Kubernetes, Nomad, Docker Swarm, and emerging alternatives. "
            "Cover: architecture differences, scaling characteristics, security models, "
            "ecosystem maturity, operational complexity, and cost at scale. "
            "Include a decision framework for choosing between them.",
            SwarmMode.AUTO,
        )

        print(f"\n  Plan: {len(plan.agents)} agents, strategy: {plan.strategy[:100]}")
        for i, a in enumerate(plan.agents):
            deps = [plan.agents[j].role for j in range(len(plan.agents))
                    if plan.agents[j].agent_id in a.depends_on]
            dep_str = f" (waits for: {', '.join(deps)})" if deps else ""
            print(f"    [{i}] {a.role} — priority {a.priority}{dep_str}")

        # Execute
        result = await orc.run(
            goal=plan.goal,
            plan=plan,
            on_start=on_start,
            on_done=on_done,
            on_retry=on_retry,
            on_synthesis_start=on_synthesis,
        )

    print_result(result, "low-level orchestrator")

    # Print event timeline
    print("  Event timeline:")
    for e in events:
        print(f"    {e['t']:6.1f}s  {e['event']:>6}  {e['role']}")

    # Budget stats
    print(f"\n  Budget: {budget.stats}")

    return result


# ═════════════════════════════════════════════════════════════════
# Main: run all stages sequentially
# ═════════════════════════════════════════════════════════════════

async def main():
    print(f"\n{'━' * 70}")
    print("  REVENGINEER SWARM — FULL FEATURE TEST")
    print(f"  Testing all capabilities across Phases 0-5")
    print(f"{'━' * 70}")

    total_start = time.time()
    results: dict[str, SwarmResult | None] = {}

    stages = [
        ("Stage 1: AUTO + Critique + Pruning", stage_auto_with_features),
        ("Stage 2: MANUAL + Tools + Deps",     stage_manual_with_tools),
        ("Stage 3: ITERATIVE + Budget + Subs", stage_iterative_with_budget),
        ("Stage 4: Low-Level Orchestrator",    stage_low_level_orchestrator),
    ]

    for name, fn in stages:
        try:
            results[name] = await fn()
        except KeyboardInterrupt:
            print(f"\n  ⚠ {name} interrupted by user")
            results[name] = None
        except Exception as e:
            print(f"\n  ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = None

    # ─── Final Summary ──────────────────────────────────────────
    total_time = time.time() - total_start

    banner("FINAL SUMMARY")
    print(f"  Total wall time: {total_time:.1f}s\n")

    for name, result in results.items():
        if result:
            print(
                f"  ✓ {name:40} "
                f"{result.duration:6.1f}s  "
                f"{len(result.successful):2d}/{len(result.agent_results):2d} agents  "
                f"{result.total_tokens:6d} tokens"
            )
        else:
            print(f"  ✗ {name:40} FAILED")

    total_agents = sum(len(r.agent_results) for r in results.values() if r)
    total_ok = sum(len(r.successful) for r in results.values() if r)
    total_tokens = sum(r.total_tokens for r in results.values() if r)

    print(f"\n  Totals: {total_ok}/{total_agents} agents succeeded, {total_tokens} tokens consumed")
    print(f"  Average stage time: {total_time / len(stages):.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
