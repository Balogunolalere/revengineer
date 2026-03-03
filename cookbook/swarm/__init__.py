"""
Revengineer Swarm — parallel AI agent orchestration.

Usage:
    from cookbook.swarm import Swarm

    result = await Swarm("research quantum computing breakthroughs in 2026")
    print(result.synthesis)

Or with full control:
    from cookbook.swarm import Orchestrator, SwarmConfig, SwarmMode

    config = SwarmConfig(api_base="http://localhost:8000/v1", max_agents=20)
    async with Orchestrator(config) as orc:
        result = await orc.run("your complex task", mode=SwarmMode.AUTO)
"""

from .models import (
    AgentSpec,
    AgentResult,
    AgentStatus,
    SwarmPlan,
    SwarmResult,
    SwarmMode,
)
from .config import SwarmConfig
from .engine import SwarmEngine, RateLimiter
from .orchestrator import Orchestrator
from .renderer import SwarmRenderer


async def Swarm(
    goal: str,
    *,
    config: SwarmConfig | None = None,
    mode: SwarmMode = SwarmMode.AUTO,
    plan: SwarmPlan | None = None,
    max_agents: int = 0,
    verbose: bool = False,
    save: bool = True,
) -> SwarmResult:
    """
    One-liner swarm execution.

    Args:
        goal:       What the swarm should accomplish
        config:     Optional SwarmConfig (defaults from env)
        mode:       AUTO (LLM decomposes), MANUAL (you provide plan), ITERATIVE (re-plans)
        plan:       Pre-built SwarmPlan for MANUAL mode
        max_agents: Override max number of agents (0 = use config default)
        verbose:    Show detailed agent output in terminal
        save:       Save markdown + JSON output files

    Returns:
        SwarmResult with .synthesis (final report) and .agent_results
    """
    cfg = config or SwarmConfig.from_env()
    if max_agents > 0:
        cfg.max_agents = max_agents
    renderer = SwarmRenderer(stream=cfg.stream_to_terminal, verbose=verbose)

    async with Orchestrator(cfg) as orc:
        # Decompose
        if plan is None:
            plan = await orc.decompose(goal, mode)
        renderer.on_plan_ready(plan)

        # Run agents + synthesize (synthesis_start callback fires inside orc.run)
        result = await orc.run(
            goal=goal,
            mode=mode,
            plan=plan,
            on_start=renderer.on_agent_start,
            on_done=renderer.on_agent_done,
            on_retry=renderer.on_agent_retry,
            on_synthesis_start=renderer.on_synthesis_start,
        )

        renderer.on_complete(result)

        # Save output files
        if save:
            import re
            slug = re.sub(r'[^\w]+', '_', goal[:50]).strip('_').lower()
            ts = __import__('time').strftime('%Y%m%d_%H%M%S')
            base = f"swarm_{slug}_{ts}"
            out_dir = cfg.output_dir or "."

            if cfg.save_markdown:
                renderer.save_markdown(result, f"{out_dir}/{base}.md")
            if cfg.save_json:
                renderer.save_json(result, f"{out_dir}/{base}.json")

    return result


__all__ = [
    "Swarm",
    "Orchestrator",
    "SwarmEngine",
    "SwarmConfig",
    "SwarmPlan",
    "SwarmResult",
    "SwarmMode",
    "SwarmRenderer",
    "AgentSpec",
    "AgentResult",
    "AgentStatus",
    "RateLimiter",
]
