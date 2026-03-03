# 🐝 Revengineer Swarm

A production-quality parallel AI agent orchestration system. Decomposes complex tasks into concurrent sub-agents, resolves dependencies via DAG scheduling, and synthesizes results into polished reports.

> **Superior to [Kimi Agent Swarm](https://kimi.ai) and [BushyTail](https://github.com/Balogunolalere/BushyTail)** — true DAG support, real rate limiting, iterative re-planning, live terminal streaming, and any OpenAI-compatible backend.

---

## Setup

### Prerequisites

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **An OpenAI-compatible API endpoint** (e.g., the `deepseek_api.py` proxy, OpenAI, ollama, vLLM, etc.)

### Install with uv

```bash
# Clone the project (if you haven't already)
git clone https://github.com/YOUR_USER/revengineer.git
cd revengineer

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies in one step
uv sync

# Start the DeepSeek API proxy (needed as the LLM backend)
uv run python deepseek_api.py
```

> **First time?** You'll need a DeepSeek token. See the [root README](../README.md#-quick-start) for authentication options.

### Verify it works

```bash
# In a second terminal — run the basic example
uv run python cookbook/examples/basic_research.py
```

You should see the swarm activate, agents spin up with a live spinner, and a final report synthesized.

---

## Architecture

```
                   ┌──────────────┐
                   │   CLI / API  │
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
                   │ Orchestrator │  ← LLM-powered task decomposition
                   └──────┬───────┘
                          │
                   ┌──────▼───────┐
              ┌────┤  SwarmEngine ├────┐
              │    └──────┬───────┘    │
              │           │            │
         ┌────▼───┐ ┌────▼───┐ ┌─────▼────┐
         │Agent 1 │ │Agent 2 │ │ Agent N  │  ← concurrent, DAG-ordered
         └────┬───┘ └────┬───┘ └─────┬────┘
              │           │            │
              └─────┬─────┘────────────┘
                    │
             ┌──────▼───────┐
             │  Synthesizer  │  ← combines all outputs (auto-continues on truncation)
             └──────┬───────┘
                    │
             ┌──────▼───────┐
             │   Renderer    │  ← live spinner animation + file output
             └──────────────┘
```

### Modules

| Module | Purpose |
|--------|---------|
| `models.py` | `AgentSpec`, `AgentResult`, `SwarmPlan`, `SwarmResult` with DAG validation + cycle detection |
| `config.py` | `SwarmConfig` — all settings with `SWARM_*` env var support |
| `engine.py` | `SwarmEngine` — concurrent execution with token-bucket rate limiting, DAG dependency resolution, retries with exponential backoff + jitter, per-agent timeouts |
| `orchestrator.py` | `Orchestrator` — LLM-powered decomposition, agent running with shared context, synthesis with auto-continuation, iterative re-planning |
| `renderer.py` | `SwarmRenderer` — live Braille spinner animation, ANSI status display, markdown/JSON file output |
| `cli.py` | CLI entry point with argparse |
| `__init__.py` | Public API — `Swarm()` one-liner |

---

## Quick Start

### One-Liner (Python)

```python
import asyncio
from cookbook.swarm import Swarm

result = asyncio.run(Swarm("research quantum computing breakthroughs in 2026"))
print(result.synthesis)
```

### CLI

```bash
# Using your DeepSeek proxy (start it first: uv run python deepseek_api.py)
uv run python -m cookbook.swarm "compare Python vs Rust for backend development"

# With options
uv run python -m cookbook.swarm --agents 10 --verbose "research the best open-source LLMs"

# Using any OpenAI-compatible endpoint
uv run python -m cookbook.swarm --api-base https://api.openai.com/v1 --api-key sk-... "your task"

# Iterative mode (re-plans to fill gaps)
uv run python -m cookbook.swarm --mode iterative "deep dive into WebAssembly's future"
```

---

## Swarm Modes

### AUTO (default)
The orchestrator decomposes your task into parallel agents automatically:
```python
result = await Swarm("analyze the competitive landscape of cloud providers")
```

### MANUAL
You define exactly which agents run, with full control over roles, tasks, and dependencies:
```python
from cookbook.swarm import Swarm, SwarmMode, SwarmPlan, AgentSpec

researcher = AgentSpec(role="Market Researcher", task="...", tools=["search"])
analyst = AgentSpec(role="Data Analyst", task="...", depends_on=[researcher.agent_id])

plan = SwarmPlan(goal="...", agents=[researcher, analyst])
result = await Swarm(plan.goal, mode=SwarmMode.MANUAL, plan=plan)
```

### ITERATIVE
Like AUTO, but after round 1, the orchestrator reviews all outputs, identifies gaps, and spawns additional agents:
```python
result = await Swarm(
    "comprehensive analysis of Rust's ecosystem for HFT",
    mode=SwarmMode.ITERATIVE,  # 2 rounds by default
    max_agents=12,
)
```

---

## Configuration

### Python

```python
from cookbook.swarm import SwarmConfig

config = SwarmConfig(
    api_base="http://localhost:8000/v1",
    default_model="deepseek-chat",
    search_model="deepseek-search",
    max_parallel=10,
    max_agents=20,
    agent_timeout=120,
    output_dir="./reports",
    verbose=True,
)
```

### Environment Variables

All settings can be set via `SWARM_*` environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SWARM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible API base URL |
| `SWARM_API_KEY` | `not-needed` | API key |
| `SWARM_MODEL` | `deepseek-chat` | Default model |
| `SWARM_SEARCH_MODEL` | `deepseek-search` | Model for search-enabled agents |
| `SWARM_ORCHESTRATOR_MODEL` | (same as default) | Model for decomposition/synthesis |
| `SWARM_MAX_PARALLEL` | `15` | Max concurrent agents |
| `SWARM_MAX_AGENTS` | `25` | Max sub-agents per swarm |
| `SWARM_RATE_LIMIT_RPM` | `60` | Requests per minute cap |
| `SWARM_MAX_RETRIES` | `2` | Per-agent retry count |
| `SWARM_AGENT_TIMEOUT` | `120` | Per-agent timeout (seconds) |
| `SWARM_SWARM_TIMEOUT` | `600` | Total swarm timeout (seconds) |
| `SWARM_ALLOW_REPLAN` | `true` | Allow iterative re-planning |
| `SWARM_OUTPUT_DIR` | `.` | Output directory for reports |
| `SWARM_SAVE_JSON` | `true` | Save JSON output |
| `SWARM_SAVE_MARKDOWN` | `true` | Save markdown report |
| `SWARM_VERBOSE` | `false` | Verbose terminal output |

---

## Examples

See the [`examples/`](examples/) directory:

| Example | Mode | Description |
|---------|------|-------------|
| `basic_research.py` | AUTO | Simple one-liner research swarm |
| `manual_agents.py` | MANUAL | Custom agents with dependency DAG |
| `code_review.py` | MANUAL | 4 specialist reviewers → 1 lead reviewer (DAG) |
| `competitive_analysis.py` | AUTO | Search-enabled competitive research |
| `iterative_dive.py` | ITERATIVE | 2-round deep dive with gap-filling |

Run any example:
```bash
uv run python cookbook/examples/basic_research.py
```

---

## Features

### Live Spinner Animation

While agents are working, a live Braille spinner shows which agents are running and for how long:

```
  ● Security Researcher started [0.2s]
  ● Trend Analyst started [0.3s]
  ⠹ Working: Security Researcher 12s · Trend Analyst 8s     ← live spinner
  ✓ Trend Analyst done in 14.2s [14.5s]
  ✓ Security Researcher done in 18.7s [18.9s]
  ⟳ Synthesizing results...
  ⠙ Working: Synthesis 5s                                    ← spins during synthesis too
```

### Auto-Continuation

When the LLM's synthesis output gets truncated (hitting `max_tokens`), the swarm automatically sends follow-up requests to get the complete report — up to 3 continuations, producing up to ~32K tokens of synthesis output. No more "Continue" buttons.

### Verbose Mode

Use `--verbose` or `verbose=True` to see:
- **Content preview** — first 80 chars of each agent's output as they complete
- **Agent breakdown** — individual timings, character counts, and retry attempts at the end

### Output Files

Results are saved to `output_dir` (default: current directory) as:
- `swarm_{slug}_{timestamp}.md` — polished markdown report
- `swarm_{slug}_{timestamp}.json` — machine-readable structured data

Disable with `--no-save` or `save=False`.

---

## DAG Dependencies

Agents can depend on other agents' outputs. The engine resolves dependencies automatically:

```
Agent A ──┐
          ├──→ Agent C ──→ Agent E
Agent B ──┘                  │
                             ▼
Agent D ─────────────→ Agent F
```

- A and B run in parallel (no deps)
- C waits for A and B to complete, then gets their outputs as context
- D runs in parallel with A, B, C (no deps)
- E waits for C
- F waits for D and E

### Cycle Detection

The plan validator uses DFS to detect circular dependencies before execution:
```python
plan = SwarmPlan(goal="...", agents=[a, b, c])
errors = plan.validate()  # Returns ["Circular dependency detected"] if applicable
```

---

## Key Differences vs Kimi / BushyTail

| Feature | Kimi Agent Swarm | BushyTail | Revengineer Swarm |
|---------|-----------------|-----------|-------------------|
| DAG dependencies | ❌ All parallel | ❌ Broken implementation | ✅ True DAG with cycle detection |
| Re-planning | ❌ Single round | ❌ No | ✅ ITERATIVE mode with gap analysis |
| Rate limiting | ❌ None | ❌ None | ✅ Token bucket with burst |
| Retries | ❌ None | ❌ Broken jitter (`hash%1=0`) | ✅ Exponential backoff + real jitter |
| Shared context | ❌ Isolated | ❌ No | ✅ Dependency outputs + peer summaries |
| Backend | Kimi API only | OpenAI only | ✅ Any OpenAI-compatible endpoint |
| Terminal output | ❌ None | ❌ Basic print | ✅ Live spinner + ANSI status icons |
| Truncation | ❌ Manual "Continue" | ❌ Truncated | ✅ Auto-continuation (up to 3 rounds) |
| File output | ❌ None | ❌ No | ✅ Markdown + JSON reports |
| Cycle detection | ❌ No | ❌ No | ✅ DFS validation |
| Per-agent timeout | ❌ No | ❌ No | ✅ `asyncio.wait_for` |
| Agent count | 100 (fixed) | User-set | 2-25 (LLM-decided or manual) |
| Task decomposition | Fixed templates | User-defined | ✅ LLM-powered dynamic decomposition |

---

## Terminal Output

```
╔══════════════════════════════════════════════╗
║  🐝 SWARM ACTIVATED                          ║
╚══════════════════════════════════════════════╝

Goal: compare Python vs Rust for backend development
Strategy: Parallel specialist analysis across 4 dimensions
Agents: 5 | Max parallel: configurable

Agent Lineup:
  ▸ Performance Benchmarker
  ▸ Ecosystem Analyst
  ▸ Developer Experience Reviewer
  ▸ Production Deployment Expert
  ▸ Strategic Advisor (waits for: Performance Benchmarker, Ecosystem Analyst, ...)

──────────────────────────────────────────────────
  ● Performance Benchmarker started [0.1s]
  ● Ecosystem Analyst started [0.2s]
  ● Developer Experience Reviewer started [0.3s]
  ● Production Deployment Expert started [0.3s]
  ⠹ Working: Performance Benchmarker 5s · Ecosystem Analyst 5s · ...
  ✓ Ecosystem Analyst done in 8.2s [8.4s]
  ✓ Performance Benchmarker done in 12.1s [12.2s]
  ✓ Developer Experience Reviewer done in 9.8s [10.1s]
  ✓ Production Deployment Expert done in 11.4s [11.5s]
  ● Strategic Advisor started [12.3s]
  ⠙ Working: Strategic Advisor 8s
  ✓ Strategic Advisor done in 15.2s [27.5s]

──────────────────────────────────────────────────
  ⟳ Synthesizing results...
  ⠋ Working: Synthesis 12s

╔══════════════════════════════════════════════╗
║  ✓ SWARM COMPLETE                            ║
╚══════════════════════════════════════════════╝
  Agents: 5/5 succeeded
  Duration: 45.2s
  📄 Report saved: ./swarm_compare_python_vs_rust_20260303_141523.md
  📊 JSON saved: ./swarm_compare_python_vs_rust_20260303_141523.json
```

---

## Advanced Usage

### Custom Orchestrator

```python
from cookbook.swarm import Orchestrator, SwarmConfig, SwarmEngine

config = SwarmConfig(api_base="http://localhost:8000/v1")

async with Orchestrator(config) as orc:
    # Manual decomposition
    plan = await orc.decompose("your complex task")

    # Inspect the plan
    for agent in plan.agents:
        print(f"{agent.role}: {agent.task[:60]}...")

    # Execute with custom callbacks
    engine = SwarmEngine(
        config=config,
        runner=orc.run_agent,
        on_start=lambda a: print(f"Starting: {a.role}"),
        on_done=lambda r: print(f"Done: {r.role} ({r.duration:.1f}s)"),
    )
    results = await engine.execute(plan)

    # Custom synthesis
    report = await orc.synthesize(plan.goal, results)
```

### Rate Limiting

The engine uses a token bucket rate limiter:
```python
config = SwarmConfig(
    rate_limit_rpm=30,    # 30 requests per minute
    rate_limit_burst=5,   # allow burst of 5
)
```

### Retry Configuration

Per-agent retries with exponential backoff and jitter:
```python
config = SwarmConfig(
    max_retries=3,            # retry up to 3 times
    retry_base_delay=1.0,     # start at 1s delay
    retry_max_delay=30.0,     # cap at 30s
    retry_jitter=True,        # randomize delay (0.5x-1.5x)
)
```

---

## Testing

Run the full test suite (64 tests):

```bash
uv run python -m pytest cookbook/tests/test_swarm.py -v
```

Tests cover: models, config, engine (DAG resolution, retries, timeouts, concurrency), orchestrator (decomposition, min/max agents), renderer, auto-continuation, and edge cases.

---

## Requirements

- Python 3.12+
- `httpx` — async HTTP client
- An OpenAI-compatible API endpoint

Install with uv (handles everything):
```bash
uv sync
```

Or install manually:
```bash
pip install httpx
```

No other dependencies. The swarm system is self-contained.
