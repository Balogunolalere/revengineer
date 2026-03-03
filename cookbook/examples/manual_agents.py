#!/usr/bin/env python3
"""
Example: Manual swarm — define your own agents with custom roles,
tasks, and dependency graph.

This gives you full control over:
- Which agents run
- What each agent does
- Who depends on whom
- Which model each agent uses

Prerequisites:
    1. Start your DeepSeek API proxy:  python deepseek_api.py
    2. Run this script:                python cookbook/examples/manual_agents.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cookbook.swarm import (
    Swarm, SwarmConfig, SwarmMode,
    SwarmPlan, AgentSpec,
)


async def main():
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=5,
        output_dir=".",
    )

    # Define agents manually — full control
    researcher = AgentSpec(
        role="Security Researcher",
        task=(
            "Research the top 10 most critical CVEs discovered in 2025-2026. "
            "For each, provide: CVE ID, severity score, affected software, "
            "and a brief description of the vulnerability."
        ),
        tools=["search"],  # this agent uses the search model
        priority=5,
    )

    analyst = AgentSpec(
        role="Trend Analyst",
        task=(
            "Analyze cybersecurity trends in 2025-2026. Focus on: "
            "most targeted industries, common attack vectors, "
            "rise of AI-powered attacks, and defense improvements."
        ),
        tools=["search"],
        priority=3,
    )

    # This agent DEPENDS on the researcher finishing first
    advisor = AgentSpec(
        role="Security Advisor",
        task=(
            "Based on the Security Researcher's findings, create an "
            "actionable security hardening checklist for a mid-size "
            "tech company. Prioritize by impact and ease of implementation."
        ),
        depends_on=[researcher.agent_id],  # waits for researcher
        priority=1,
    )

    # Build the plan
    plan = SwarmPlan(
        goal="Comprehensive cybersecurity analysis and recommendations for 2025-2026",
        agents=[researcher, analyst, advisor],
        strategy="Parallel research + analysis, then dependent advisory synthesis",
    )

    # Validate before running
    errors = plan.validate()
    if errors:
        print(f"Plan errors: {errors}")
        return

    result = await Swarm(
        plan.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(main())
