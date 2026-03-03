#!/usr/bin/env python3
"""
Swarm CLI — run a parallel agent swarm from the command line.

Usage:
    # Using your DeepSeek proxy (default):
    python -m cookbook.swarm.cli "research the best open-source LLMs for code generation"

    # Using any OpenAI-compatible endpoint:
    python -m cookbook.swarm.cli --api-base http://localhost:11434/v1 "your task"

    # More options:
    python -m cookbook.swarm.cli --agents 20 --verbose "complex research task"

Environment variables (optional):
    SWARM_API_BASE      — LLM API base URL (default: http://localhost:8000/v1)
    SWARM_API_KEY       — API key (default: not-needed)
    SWARM_MODEL         — Default model (default: deepseek-chat)
    SWARM_MAX_AGENTS    — Max agents (default: 25)
    SWARM_MAX_PARALLEL  — Max concurrent (default: 15)
    SWARM_OUTPUT_DIR    — Output directory (default: current dir)
"""

import argparse
import asyncio
import sys

from cookbook.swarm import Swarm, SwarmConfig, SwarmMode


def main():
    parser = argparse.ArgumentParser(
        description="🐝 Swarm — parallel AI agent orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "compare Python vs Rust for backend development"
  %(prog)s --agents 10 "research quantum computing breakthroughs 2026"
  %(prog)s --api-base https://api.openai.com/v1 --api-key sk-... "analyze market trends"
  %(prog)s --verbose --no-save "quick question needing multiple perspectives"
        """,
    )
    parser.add_argument("goal", help="The task/goal for the swarm to accomplish")
    parser.add_argument("--api-base", default="", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--model", default="", help="Default model name")
    parser.add_argument("--agents", type=int, default=0, help="Max number of agents")
    parser.add_argument("--parallel", type=int, default=0, help="Max concurrent agents")
    parser.add_argument("--timeout", type=float, default=0, help="Per-agent timeout (seconds)")
    parser.add_argument("--output-dir", default="", help="Output directory for reports")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--no-save", action="store_true", help="Don't save output files")
    parser.add_argument("--no-stream", action="store_true", help="Don't print to terminal")
    parser.add_argument("--mode", choices=["auto", "iterative"], default="auto",
                        help="Swarm mode (default: auto)")

    args = parser.parse_args()

    # Build config from env + CLI args
    config = SwarmConfig.from_env()

    if args.api_base:
        config.api_base = args.api_base
    if args.api_key:
        config.api_key = args.api_key
    if args.model:
        config.default_model = args.model
    if args.agents:
        config.max_agents = args.agents
    if args.parallel:
        config.max_parallel = args.parallel
    if args.timeout:
        config.agent_timeout = args.timeout
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_stream:
        config.stream_to_terminal = False
    config.verbose = args.verbose

    mode = SwarmMode.ITERATIVE if args.mode == "iterative" else SwarmMode.AUTO

    # Run the swarm
    try:
        result = asyncio.run(
            Swarm(
                args.goal,
                config=config,
                mode=mode,
                verbose=args.verbose,
                save=not args.no_save,
            )
        )

        # Print synthesis if not streaming
        if args.no_stream:
            print(result.synthesis)

        sys.exit(0 if not result.failed else 1)

    except KeyboardInterrupt:
        print("\n\033[33mSwarm cancelled.\033[0m")
        sys.exit(130)
    except Exception as e:
        print(f"\n\033[31mSwarm error: {e}\033[0m", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
