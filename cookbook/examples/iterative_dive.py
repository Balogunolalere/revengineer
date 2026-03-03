#!/usr/bin/env python3
"""
Example: Iterative deep-dive swarm — starts with a broad question,
the orchestrator reviews the first round's output, identifies gaps,
and spawns additional agents to fill them.

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Run this script:                uv run python cookbook/examples/iterative_dive.py
"""

import asyncio

from cookbook.swarm import Swarm, SwarmConfig, SwarmMode


async def main():
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=5,
        orchestrator_model="deepseek-chat",
        default_model="deepseek-search",
        output_dir=".",
    )

    result = await Swarm(
        "I want to build a high-frequency trading system in Rust. "
        "Research: (1) the lowest-latency network I/O libraries in Rust "
        "(io_uring, DPDK bindings, etc.), (2) lock-free data structures "
        "for order books, (3) FIX protocol implementations in Rust, "
        "(4) regulatory requirements for HFT in the US and EU, "
        "(5) realistic latency benchmarks for Rust vs C++ vs Java in "
        "financial applications. Go deep on each.",
        config=config,
        mode=SwarmMode.ITERATIVE,  # will do 2 rounds
        max_agents=12,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(f"Total time: {result.duration:.1f}s")
    print(f"Agents used: {len(result.agent_results)}")
    print(f"\n{result.synthesis}")


if __name__ == "__main__":
    asyncio.run(main())
