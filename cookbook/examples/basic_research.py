#!/usr/bin/env python3
"""
Example: Basic swarm usage — one line to research a topic.

This spawns multiple specialized agents in parallel, each tackling
a different aspect of the research, then synthesizes a final report.

Prerequisites:
    1. Start your DeepSeek API proxy:  python deepseek_api.py
    2. Run this script:                python cookbook/examples/basic_research.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cookbook.swarm import Swarm, SwarmConfig


async def main():
    # Optionally customize config
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_agents=10,
        max_parallel=5,
        save_markdown=True,
        save_json=True,
        output_dir=".",
    )

    result = await Swarm(
        "What are the most important breakthroughs in AI in 2025-2026? "
        "Cover: new model architectures, open-source developments, "
        "agentic AI advances, and real-world deployment trends.",
        config=config,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"✅ Swarm completed in {result.duration:.1f}s")
    print(f"   {len(result.successful)} agents succeeded, {len(result.failed)} failed")
    print(f"{'='*60}\n")
    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(main())
