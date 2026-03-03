#!/usr/bin/env python3
"""
Example: Competitive analysis swarm — research multiple companies
in parallel, then synthesize into a strategic report.

Prerequisites:
    1. Start your DeepSeek API proxy:  python deepseek_api.py
    2. Run this script:                python cookbook/examples/competitive_analysis.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cookbook.swarm import Swarm, SwarmConfig


async def main():
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=6,
        default_model="deepseek-search",   # all agents use search by default
        output_dir=".",
    )

    # AUTO mode: the orchestrator decomposes this into agents for us
    result = await Swarm(
        "Perform a competitive analysis of the top AI code assistants "
        "in 2025: GitHub Copilot, Cursor, Windsurf (Codeium), Tabnine, "
        "and Amazon Q Developer. For each, research: pricing tiers, "
        "supported IDEs, key features, model backends, market share "
        "estimates, and recent funding/acquisitions. Then produce a "
        "comparison matrix and strategic recommendations for a startup "
        "choosing a code assistant for their team of 20 developers.",
        config=config,
        max_agents=8,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(main())
