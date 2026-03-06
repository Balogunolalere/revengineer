#!/usr/bin/env python3
"""
Example: Self-Building Search Tool

Agents bootstrap their own search engine from scratch, then use it.

Phase 1 — BUILD (one swarm run):
  • Data Collector     — gathers real files from this repo as documents
  • Search Engine Dev  — writes a TF-IDF search module in pure Python
  • Index Builder      — indexes the collected documents using the engine
  • Build Verifier     — runs test queries to prove the search works

Phase 2 — USE (second swarm run):
  • A fresh set of agents uses the newly built search tool (dynamically
    registered) to answer a real question about the codebase.

The agents write ALL the search code themselves — nothing is pre-built.

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Run this script:                uv run python cookbook/examples/self_build_search.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
import shlex
import sys
import tempfile

from cookbook.swarm import (
    Swarm,
    SwarmConfig,
    SwarmMode,
    SwarmPlan,
    AgentSpec,
    AgentStatus,
    ToolRegistry,
    ToolDef,
)

# ── Scratch directory — agents write everything here ─────────────

SCRATCH = tempfile.mkdtemp(prefix="swarm_search_")

MAX_OUTPUT = 16_384

# ── Allowed shell commands ───────────────────────────────────────

ALLOWED_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
    "date", "pwd", "which", "file", "stat", "sort", "uniq",
    "python3", "git", "tree",
})

BLOCKED_PATTERNS = re.compile(
    r"rm\s|rmdir|mkfs|dd\s|>\s*/dev|chmod|chown|sudo|su\s|"
    r"nc\s|ncat|eval\s|exec\s|reboot|shutdown|systemctl|"
    r"kill\s|pkill|&&\s*rm|;\s*rm|\|\s*rm",
    re.IGNORECASE,
)


def _is_safe(cmd: str) -> tuple[bool, str]:
    s = cmd.strip()
    if not s:
        return False, "Empty"
    if BLOCKED_PATTERNS.search(s):
        return False, f"Blocked: {s!r}"
    try:
        parts = shlex.split(s)
    except ValueError as e:
        return False, str(e)
    if os.path.basename(parts[0]) not in ALLOWED_COMMANDS:
        return False, f"{parts[0]!r} not allowed"
    return True, ""


# ── Tool implementations ────────────────────────────────────────

async def shell_exec(command: str = "", **_kw: object) -> str:
    ok, reason = _is_safe(command)
    if not ok:
        return f"[DENIED] {reason}"
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    return out.decode(errors="replace")[:MAX_OUTPUT] or "(no output)"


async def read_file_tool(path: str = "", **_kw: object) -> str:
    resolved = os.path.realpath(path)
    if not os.path.isfile(resolved):
        return f"[ERROR] Not found: {path}"
    with open(resolved, errors="replace") as f:
        return f.read(MAX_OUTPUT)


async def write_file_tool(filename: str = "", content: str = "", **_kw: object) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(filename))
    if not safe:
        return "[ERROR] Invalid filename"
    dest = os.path.join(SCRATCH, safe)
    with open(dest, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars → {dest}"


async def python_exec(code: str = "", **_kw: object) -> str:
    tmp = os.path.join(SCRATCH, "_run.py")
    with open(tmp, "w") as f:
        f.write(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=SCRATCH,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        return out.decode(errors="replace")[:MAX_OUTPUT] or "(no output)"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ── Build tool registry ─────────────────────────────────────────

def build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDef(
        name="shell_exec",
        description="Run an allow-listed shell command (ls, cat, grep, find, git, wc, python3, etc.).",
        parameters={"command": "Shell command to execute"},
        fn=shell_exec, safe=True,
    ))
    reg.register(ToolDef(
        name="read_file",
        description="Read a local file (up to 16 KB).",
        parameters={"path": "File path (absolute or relative)"},
        fn=read_file_tool, safe=True,
    ))
    reg.register(ToolDef(
        name="write_file",
        description=f"Write content to a file in the scratch dir ({SCRATCH}).",
        parameters={"filename": "Filename to create", "content": "Text content"},
        fn=write_file_tool, safe=True,
    ))
    reg.register(ToolDef(
        name="python_exec",
        description=f"Execute Python code (cwd={SCRATCH}). Use for building and testing the search engine. Print output to stdout.",
        parameters={"code": "Python source code"},
        fn=python_exec, safe=False,
    ))
    return reg


# ═════════════════════════════════════════════════════════════════
# Phase 1: BUILD — agents create a search engine from scratch
# ═════════════════════════════════════════════════════════════════

def phase1_plan() -> SwarmPlan:
    collector = AgentSpec(
        role="Data Collector",
        task=(
            f"Collect Python source files from this project to serve as searchable documents. "
            f"Use shell_exec to run: find . -name '*.py' -path '*/cookbook/*' | head -20 "
            f"Then use read_file on at least 6 different .py files (pick from cookbook/swarm/ "
            f"and cookbook/examples/). For EACH file you read, use write_file to save it as "
            f"'doc_<number>.txt' (e.g. doc_1.txt, doc_2.txt, ...) in the scratch dir. "
            f"At the start of each doc file, add a header line: '# SOURCE: <original_path>'. "
            f"Finally, use write_file to create 'manifest.txt' listing all doc filenames, one per line."
        ),
        tools=["shell_exec", "read_file", "write_file"],
        system_prompt=(
            "You are a data engineer. Your job is to gather source files and save them "
            "as numbered documents in the scratch directory. Be systematic."
        ),
        priority=5,
    )

    engine_dev = AgentSpec(
        role="Search Engine Developer",
        task=(
            f"Write a complete TF-IDF search engine in pure Python (no external libraries). "
            f"Use write_file to create 'search_engine.py' in the scratch dir ({SCRATCH}). "
            f"The module MUST define these functions:\n\n"
            f"  1. build_index(doc_dir: str) -> dict\n"
            f"     - Reads all 'doc_*.txt' files from doc_dir\n"
            f"     - Tokenizes text (split on non-alphanumeric, lowercase)\n"
            f"     - Computes TF-IDF scores\n"
            f"     - Returns an index dict with keys: 'documents', 'tfidf', 'vocab'\n"
            f"     - 'documents' maps doc_id -> {{'path': str, 'text': str}}\n\n"
            f"  2. search(index: dict, query: str, top_k: int = 5) -> list[dict]\n"
            f"     - Each result: {{'doc_id': str, 'path': str, 'score': float, 'snippet': str}}\n"
            f"     - Score results by cosine similarity of query TF-IDF vs document TF-IDF\n"
            f"     - 'snippet' should be the first 200 chars of the document text\n\n"
            f"  3. save_index(index: dict, path: str) and load_index(path: str) -> dict\n"
            f"     - Serialize/deserialize the index to/from JSON\n\n"
            f"After writing the file, use python_exec to import it and run a basic smoke test:\n"
            f"  import search_engine; print('Module loaded:', dir(search_engine))\n\n"
            f"Fix any import errors before finishing."
        ),
        tools=["write_file", "python_exec"],
        system_prompt=(
            "You are a search systems engineer. Write clean, working Python code. "
            "The search_engine.py module must be importable and have all required functions. "
            "Use only the Python standard library (math, json, os, re, collections). "
            "Test that it imports correctly."
        ),
        priority=5,
    )

    indexer = AgentSpec(
        role="Index Builder",
        task=(
            f"Build the search index from the collected documents. "
            f"Use python_exec to run code that:\n"
            f"  1. Imports search_engine from {SCRATCH}/search_engine.py\n"
            f"  2. Calls build_index('{SCRATCH}') to index all doc_*.txt files\n"
            f"  3. Prints how many documents were indexed and the vocab size\n"
            f"  4. Calls save_index(index, '{SCRATCH}/index.json') to persist it\n"
            f"  5. Verifies the saved file exists and prints its size\n\n"
            f"If there are errors, read the search_engine.py file to debug, "
            f"then fix and retry. The index MUST be saved successfully."
        ),
        tools=["python_exec", "read_file", "shell_exec"],
        depends_on=[collector.agent_id, engine_dev.agent_id],
        system_prompt=(
            "You are an indexing specialist. Use the search engine module to "
            "build and save the index. Debug any issues. Confirm success with evidence."
        ),
        priority=3,
    )

    verifier = AgentSpec(
        role="Build Verifier",
        task=(
            f"Verify the search engine works end-to-end. Use python_exec to:\n"
            f"  1. Load the saved index from '{SCRATCH}/index.json'\n"
            f"  2. Run at least 5 different search queries:\n"
            f"     - 'SwarmConfig'\n"
            f"     - 'agent timeout'\n"
            f"     - 'async def execute'\n"
            f"     - 'tool registry'\n"
            f"     - 'rate limit'\n"
            f"  3. For each query, print: query, number of results, top result path and score\n"
            f"  4. Confirm that results are returned (non-empty) for at least 3 queries\n"
            f"  5. Print a final verdict: PASS or FAIL with reasons\n\n"
            f"The search engine lives at {SCRATCH}/search_engine.py. "
            f"Import it with: sys.path.insert(0, '{SCRATCH}'); import search_engine"
        ),
        tools=["python_exec", "read_file"],
        depends_on=[indexer.agent_id],
        system_prompt=(
            "You are a QA engineer. Rigorously test the search engine with diverse queries. "
            "Report results clearly with pass/fail status."
        ),
        priority=1,
    )

    return SwarmPlan(
        goal=(
            "Build a working TF-IDF search engine from scratch: collect documents "
            "from this project, write the search module, build the index, and verify "
            "it returns correct results for test queries."
        ),
        agents=[collector, engine_dev, indexer, verifier],
        strategy=(
            "Layer 0: Data Collector and Search Engine Developer work in parallel. "
            "Layer 1: Index Builder combines their outputs. "
            "Layer 2: Build Verifier confirms everything works."
        ),
    )


# ═════════════════════════════════════════════════════════════════
# Phase 2: USE — dynamically register the agent-built tool
# ═════════════════════════════════════════════════════════════════

def try_load_search_tool() -> ToolDef | None:
    """Attempt to import the agent-written search engine and wrap it as a tool."""
    engine_path = os.path.join(SCRATCH, "search_engine.py")
    index_path = os.path.join(SCRATCH, "index.json")

    if not os.path.isfile(engine_path) or not os.path.isfile(index_path):
        return None

    # Dynamically import the agent-written module
    spec = importlib.util.spec_from_file_location("search_engine", engine_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [!] Failed to import agent-built search engine: {e}")
        return None

    # Load the index the agents built
    try:
        index = mod.load_index(index_path)
    except Exception as e:
        print(f"  [!] Failed to load agent-built index: {e}")
        return None

    doc_count = len(index.get("documents", {}))
    print(f"  [+] Loaded agent-built search engine: {doc_count} documents indexed")

    # Wrap as an async tool function
    async def agent_search(query: str = "", top_k: str = "5", **_kw: object) -> str:
        try:
            results = mod.search(index, query, int(top_k))
            if not results:
                return f"No results for: {query}"
            lines = [f"Search results for '{query}' ({len(results)} hits):"]
            for i, r in enumerate(results, 1):
                lines.append(
                    f"  {i}. [{r.get('score', 0):.3f}] {r.get('path', '?')}"
                    f"\n     {r.get('snippet', '')[:120]}..."
                )
            return "\n".join(lines)
        except Exception as e:
            return f"[ERROR] Search failed: {e}"

    return ToolDef(
        name="codebase_search",
        description=(
            "Search the project codebase using the TF-IDF engine that was built "
            "by the previous swarm. Returns ranked results with scores and snippets."
        ),
        parameters={
            "query": "Search query (keywords or code fragments)",
            "top_k": "(optional) number of results, default 5",
        },
        fn=agent_search,
        safe=True,
    )


def phase2_plan() -> SwarmPlan:
    researcher = AgentSpec(
        role="Architecture Researcher",
        task=(
            "Using the codebase_search tool, investigate how the swarm system works. "
            "Search for: 'SwarmEngine execute', 'agent killing', 'rate limit adaptive', "
            "'circuit breaker', 'tool registry', 'BudgetTracker'. "
            "For each search, note the top files and scores. "
            "Produce a summary of the system's key architectural components."
        ),
        tools=["codebase_search"],
        system_prompt=(
            "You are a software architect. Use the codebase_search tool to explore "
            "and map the system architecture. Cite specific files from search results."
        ),
        priority=5,
    )

    feature_mapper = AgentSpec(
        role="Feature Mapper",
        task=(
            "Using codebase_search, find and catalog all major features. "
            "Search for: 'sub_swarm', 'context window pruning', 'plan critique', "
            "'few shot', 'ToolDef', 'arsenal bridge'. "
            "List each feature with the file it lives in and a one-line description."
        ),
        tools=["codebase_search"],
        system_prompt=(
            "You are a technical analyst. Search the codebase systematically "
            "and map features to their source files."
        ),
        priority=4,
    )

    reporter = AgentSpec(
        role="Architecture Reporter",
        task=(
            "Synthesize the findings from the Architecture Researcher and Feature Mapper "
            "into a clear architectural overview document. Include:\n"
            "1. System overview and purpose\n"
            "2. Key components and their files\n"
            "3. Feature catalog\n"
            "4. How tools/agents interact\n"
            "The report should demonstrate that the self-built search tool is working."
        ),
        depends_on=[researcher.agent_id, feature_mapper.agent_id],
        system_prompt="You are a technical writer. Write a clear, organized report.",
        priority=1,
    )

    return SwarmPlan(
        goal=(
            "Use the agent-built search tool to analyze and document the swarm "
            "system's architecture, proving the self-built tool works in practice."
        ),
        agents=[researcher, feature_mapper, reporter],
        strategy=(
            "Layer 0: Two researchers search the codebase in parallel using "
            "the agent-built search tool. Layer 1: Reporter synthesizes."
        ),
    )


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

async def main() -> None:
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=4,
        output_dir=".",
        enable_agent_killing=True,
        agent_kill_threshold=5.0,
        agent_kill_min_time=120.0,
        max_tool_calls_per_agent=20,
        tool_timeout=60.0,
    )

    print("\n" + "=" * 70)
    print("  SELF-BUILDING SEARCH TOOL")
    print("  Phase 1: Agents build a TF-IDF search engine from scratch")
    print("  Phase 2: Agents use the tool they built to explore the codebase")
    print(f"  Scratch dir: {SCRATCH}")
    print("=" * 70)

    # ── Phase 1: BUILD ───────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("  PHASE 1: Building the search engine...")
    print(f"{'─' * 70}\n")

    build_result = await Swarm(
        goal=phase1_plan().goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=phase1_plan(),
        tool_registry=build_tools(),
        save=True,
    )

    print(f"\n{'─' * 70}")
    print("  Phase 1 Results:")
    for r in build_result.agent_results:
        sym = "✓" if r.status == AgentStatus.COMPLETED else "✗"
        chars = len(r.content) if r.content else 0
        print(f"    {sym} {r.role}: {r.duration:.1f}s ({chars} chars)")
        if r.error:
            print(f"      ERROR: {r.error}")

    # Check what exists in scratch
    files = os.listdir(SCRATCH)
    print(f"\n  Files in scratch dir ({len(files)}):")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(SCRATCH, f))
        print(f"    {f} ({size:,} bytes)")

    # ── Phase 2: USE ─────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("  PHASE 2: Loading agent-built search tool...")
    print(f"{'─' * 70}\n")

    search_tool = try_load_search_tool()
    if search_tool is None:
        print("  [✗] Could not load the agent-built search engine.")
        print("      Phase 1 may not have produced working code.")
        print("      Check the scratch dir for debug info.")
        return

    # Build a new registry with the agent-built tool
    phase2_registry = ToolRegistry()
    phase2_registry.register(search_tool)

    print(f"\n  Running Phase 2 swarm with agent-built search tool...\n")

    use_result = await Swarm(
        goal=phase2_plan().goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=phase2_plan(),
        tool_registry=phase2_registry,
        save=True,
    )

    # ── Final report ─────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("  FINAL RESULTS")
    print(f"{'=' * 70}")

    print("\n  Phase 1 (Build):")
    for r in build_result.agent_results:
        sym = "✓" if r.status == AgentStatus.COMPLETED else "✗"
        print(f"    {sym} {r.role}: {r.duration:.1f}s")

    print(f"\n  Phase 2 (Use):")
    for r in use_result.agent_results:
        sym = "✓" if r.status == AgentStatus.COMPLETED else "✗"
        chars = len(r.content) if r.content else 0
        print(f"    {sym} {r.role}: {r.duration:.1f}s ({chars} chars)")

    print(f"\n  Phase 1 time: {build_result.duration:.1f}s")
    print(f"  Phase 2 time: {use_result.duration:.1f}s")
    print(f"  Total time:   {build_result.duration + use_result.duration:.1f}s")

    if use_result.synthesis:
        preview = use_result.synthesis[:600]
        print(f"\n{'─' * 50}")
        print("  Architecture report preview:\n")
        for line in preview.splitlines():
            print(f"  {line}")
        print(f"  ...")
        print(f"{'─' * 50}")

    print(f"\n  All files: {SCRATCH}/")
    print()


if __name__ == "__main__":
    asyncio.run(main())
