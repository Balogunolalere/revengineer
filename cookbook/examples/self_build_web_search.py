#!/usr/bin/env python3
"""
Example: Self-Building Web Search Tool

Agents build their own web search tool from scratch, verify it works,
then use it to research a real topic.

Phase 1 — BUILD (one swarm run):
  • Web Search Developer  — writes a Python module that searches the web
                            using DuckDuckGo's HTML lite interface (no API
                            keys needed). Parses results with stdlib only.
  • Test Harness Builder  — writes a test script that exercises the module
  • Build Verifier        — runs the tests and confirms the tool works

Phase 2 — USE (second swarm run):
  • A fresh set of research agents uses the agent-built web_search tool
    (dynamically loaded and registered) to research a real question and
    produce a report.

The agents write ALL the search code themselves using only Python stdlib.

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Internet access (for DuckDuckGo queries)
    3. Run this script:                uv run python cookbook/examples/self_build_web_search.py
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import re
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

SCRATCH = tempfile.mkdtemp(prefix="swarm_websearch_")

MAX_OUTPUT = 16_384


# ── Tool implementations ────────────────────────────────────────

async def write_file_tool(filename: str = "", content: str = "", **_kw: object) -> str:
    """Write content to a file in the scratch directory."""
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(filename))
    if not safe:
        return "[ERROR] Invalid filename"
    dest = os.path.join(SCRATCH, safe)
    with open(dest, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} chars → {dest}"


async def read_file_tool(path: str = "", **_kw: object) -> str:
    """Read a file from the scratch directory."""
    # Allow both bare filenames and full paths
    if os.sep not in path and not path.startswith("."):
        resolved = os.path.join(SCRATCH, path)
    else:
        resolved = os.path.realpath(path)
    if not os.path.isfile(resolved):
        return f"[ERROR] Not found: {path}"
    with open(resolved, errors="replace") as f:
        return f.read(MAX_OUTPUT)


async def python_exec(code: str = "", **_kw: object) -> str:
    """Execute a Python snippet in a subprocess (cwd = scratch dir)."""
    tmp = os.path.join(SCRATCH, "_run.py")
    with open(tmp, "w") as f:
        f.write(code)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=SCRATCH,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        return out.decode(errors="replace")[:MAX_OUTPUT] or "(no output)"
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


async def list_scratch(**_kw: object) -> str:
    """List all files in the scratch directory with sizes."""
    try:
        files = sorted(os.listdir(SCRATCH))
    except OSError as e:
        return f"[ERROR] {e}"
    if not files:
        return "(empty directory)"
    lines = []
    for f in files:
        p = os.path.join(SCRATCH, f)
        size = os.path.getsize(p)
        lines.append(f"  {f} ({size:,} bytes)")
    return f"Scratch dir: {SCRATCH}\n" + "\n".join(lines)


# ── Build tool registry ─────────────────────────────────────────

def build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDef(
        name="write_file",
        description=f"Write content to a file in the scratch directory ({SCRATCH}).",
        parameters={"filename": "Filename to create", "content": "Text content to write"},
        fn=write_file_tool, safe=True,
    ))
    reg.register(ToolDef(
        name="read_file",
        description="Read a file from the scratch directory.",
        parameters={"path": "Filename or full path"},
        fn=read_file_tool, safe=True,
    ))
    reg.register(ToolDef(
        name="python_exec",
        description=(
            f"Execute Python code in a subprocess (cwd={SCRATCH}). "
            "Has full internet access and stdlib. Print output to stdout. "
            "Use this to test the modules you write."
        ),
        parameters={"code": "Python source code to execute"},
        fn=python_exec, safe=False,
    ))
    reg.register(ToolDef(
        name="list_scratch",
        description="List all files in the scratch directory with their sizes.",
        parameters={},
        fn=list_scratch, safe=True,
    ))
    return reg


# ═════════════════════════════════════════════════════════════════
# Phase 1: BUILD — agents create a web search module from scratch
# ═════════════════════════════════════════════════════════════════

def phase1_plan() -> SwarmPlan:
    developer = AgentSpec(
        role="Web Search Developer",
        task=(
            f"Write a complete web search module in pure Python (only stdlib — "
            f"use urllib.request, urllib.parse, html.parser, json, re). "
            f"Use write_file to create 'web_search.py' in the scratch dir.\n\n"
            f"The module MUST implement these functions:\n\n"
            f"  1. search(query: str, num_results: int = 5) -> list[dict]\n"
            f"     - Sends an HTTP request to DuckDuckGo's HTML Lite interface:\n"
            f"       POST to https://lite.duckduckgo.com/lite/ with form data: q=<query>\n"
            f"     - Sets a realistic User-Agent header (e.g. Mozilla/5.0 ...)\n"
            f"     - Parses the HTML response to extract search results\n"
            f"     - Each result dict has keys: 'title', 'url', 'snippet'\n"
            f"     - Returns up to num_results results\n"
            f"     - Handles errors gracefully (returns empty list + prints error)\n\n"
            f"  2. fetch_page(url: str, max_chars: int = 8000) -> str\n"
            f"     - Fetches a URL and returns the text content\n"
            f"     - Strips HTML tags to return plain text\n"
            f"     - Respects max_chars limit\n"
            f"     - Has a 15-second timeout\n"
            f"     - Returns error string on failure\n\n"
            f"  3. search_and_summarize(query: str, num_results: int = 3) -> str\n"
            f"     - Calls search(), then for each result calls fetch_page()\n"
            f"     - Returns a formatted string with titles, URLs, and page excerpts\n\n"
            f"DuckDuckGo lite returns a simple HTML table. Each result row has:\n"
            f"  - A link (<a> tag) with the result URL and title\n"
            f"  - A snippet in the next table row\n"
            f"Parse using html.parser.HTMLParser or regex.\n\n"
            f"After writing the file, verify it imports:\n"
            f"  python_exec: import web_search; print(dir(web_search))\n\n"
            f"Fix any syntax or import errors before finishing."
        ),
        tools=["write_file", "python_exec", "read_file"],
        system_prompt=(
            "You are a Python developer specializing in web scraping. "
            "Write clean, robust code using only the Python standard library. "
            "The module must handle network errors, HTML parsing edge cases, "
            "and timeouts gracefully. Test that it imports successfully."
        ),
        priority=5,
    )

    test_builder = AgentSpec(
        role="Test Harness Builder",
        task=(
            f"Write a comprehensive test script for the web search module. "
            f"Use write_file to create 'test_web_search.py' in the scratch dir.\n\n"
            f"The test script should:\n"
            f"  1. Import web_search from the scratch dir\n"
            f"  2. Test search('python programming') — expect non-empty results\n"
            f"  3. Test search('latest AI news 2026') — expect non-empty results\n"
            f"  4. Test fetch_page on a known URL (e.g. https://example.com)\n"
            f"  5. Test error handling: fetch_page('https://thisdoesnotexist12345.invalid')\n"
            f"  6. Test search_and_summarize('quantum computing')\n"
            f"  7. Print clear PASS/FAIL for each test\n"
            f"  8. Print a final summary: X/Y tests passed\n\n"
            f"At the top of the script, add:\n"
            f"  import sys; sys.path.insert(0, '{SCRATCH}')\n\n"
            f"Make the script print detailed output so we can diagnose failures."
        ),
        tools=["write_file", "read_file"],
        depends_on=[developer.agent_id],
        system_prompt=(
            "You are a QA engineer. Write a thorough test script that validates "
            "every function in the web search module. Print clear results."
        ),
        priority=3,
    )

    verifier = AgentSpec(
        role="Build Verifier",
        task=(
            f"Run the test suite and verify the web search module works.\n\n"
            f"Steps:\n"
            f"  1. Use list_scratch to see what files exist\n"
            f"  2. Use python_exec to run the test script:\n"
            f"     import sys; sys.path.insert(0, '{SCRATCH}')\n"
            f"     exec(open('{SCRATCH}/test_web_search.py').read())\n"
            f"  3. If tests fail, use read_file to inspect web_search.py\n"
            f"  4. If there are bugs, use write_file to fix web_search.py, then re-run tests\n"
            f"  5. Keep iterating until at least 4 out of 6 tests pass\n"
            f"  6. Report final verdict: PASS (with details) or FAIL (with reasons)\n\n"
            f"You MUST actually run the tests — don't just read the code."
        ),
        tools=["python_exec", "read_file", "write_file", "list_scratch"],
        depends_on=[test_builder.agent_id],
        system_prompt=(
            "You are a build engineer. Run the tests, debug failures, fix issues, "
            "and re-test until the module works. Be persistent — iterate on fixes."
        ),
        priority=1,
    )

    return SwarmPlan(
        goal=(
            "Build a working web search tool from scratch using only Python stdlib. "
            "The module should search DuckDuckGo, fetch pages, and extract text. "
            "Write tests and verify everything works end-to-end."
        ),
        agents=[developer, test_builder, verifier],
        strategy=(
            "Layer 0: Developer writes the web search module. "
            "Layer 1: Test builder writes the test suite. "
            "Layer 2: Verifier runs tests, debugs, and fixes until it works."
        ),
    )


# ═════════════════════════════════════════════════════════════════
# Phase 2: USE — dynamically load the agent-built web search tool
# ═════════════════════════════════════════════════════════════════

def try_load_web_search_tool() -> ToolDef | None:
    """Import the agent-written web search module and wrap it as a tool."""
    module_path = os.path.join(SCRATCH, "web_search.py")
    if not os.path.isfile(module_path):
        return None

    spec = importlib.util.spec_from_file_location("web_search", module_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print(f"  [!] Failed to import agent-built web search: {e}")
        return None

    has_search = hasattr(mod, "search")
    has_fetch = hasattr(mod, "fetch_page")
    has_summarize = hasattr(mod, "search_and_summarize")
    print(f"  [+] Module loaded — search={has_search}, fetch_page={has_fetch}, "
          f"search_and_summarize={has_summarize}")

    if not has_search:
        return None

    async def web_search_tool(query: str = "", num_results: str = "5", **_kw: object) -> str:
        """Async wrapper around the agent-built synchronous search function."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, lambda: mod.search(query, int(num_results))
            )
            if not results:
                return f"No results found for: {query}"
            lines = [f"Web search results for '{query}' ({len(results)} hits):\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "(no title)")
                url = r.get("url", "")
                snippet = r.get("snippet", "")[:200]
                lines.append(f"  {i}. {title}\n     {url}\n     {snippet}\n")
            return "\n".join(lines)
        except Exception as e:
            return f"[ERROR] Web search failed: {e}"

    async def web_fetch_tool(url: str = "", max_chars: str = "6000", **_kw: object) -> str:
        """Async wrapper around the agent-built fetch_page function."""
        if not has_fetch:
            return "[ERROR] fetch_page not available in the module"
        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None, lambda: mod.fetch_page(url, int(max_chars))
            )
            return text or "(empty page)"
        except Exception as e:
            return f"[ERROR] Fetch failed: {e}"

    # We return the search tool; we'll register fetch separately
    # Store fetch on the module for the caller to pick up
    mod._fetch_tool_def = ToolDef(
        name="web_fetch",
        description=(
            "Fetch a web page URL and return its plain-text content "
            "(HTML tags stripped). Built by the previous swarm's agents."
        ),
        parameters={
            "url": "Full URL to fetch (http:// or https://)",
            "max_chars": "(optional) Max characters to return, default 6000",
        },
        fn=web_fetch_tool,
        safe=True,
    ) if has_fetch else None

    return ToolDef(
        name="web_search",
        description=(
            "Search the web using DuckDuckGo. Returns ranked results with "
            "titles, URLs, and snippets. Built from scratch by the previous "
            "swarm's agents using only Python stdlib."
        ),
        parameters={
            "query": "Search query string",
            "num_results": "(optional) Number of results, default 5",
        },
        fn=web_search_tool,
        safe=True,
    ), mod  # return tuple


def phase2_plan() -> SwarmPlan:
    researcher_1 = AgentSpec(
        role="AI Trends Researcher",
        task=(
            "Using the web_search tool, research the most significant AI developments "
            "in early 2026. Search for at least 4 different queries:\n"
            "  - 'AI breakthroughs 2026'\n"
            "  - 'large language model advances 2026'\n"
            "  - 'AGI progress 2026'\n"
            "  - 'AI regulation news 2026'\n"
            "For the most interesting results, use web_fetch to read the full pages.\n"
            "Compile a summary of key findings with sources."
        ),
        tools=["web_search", "web_fetch"],
        system_prompt=(
            "You are a technology researcher. Use the web search and fetch tools "
            "to find current information. Always cite your sources with URLs."
        ),
        priority=5,
    )

    researcher_2 = AgentSpec(
        role="Open Source AI Researcher",
        task=(
            "Using the web_search tool, research the state of open-source AI:\n"
            "  - 'open source AI models 2026'\n"
            "  - 'DeepSeek latest model'\n"
            "  - 'llama open source 2026'\n"
            "  - 'AI open source vs closed source debate'\n"
            "Use web_fetch on the most relevant results for details.\n"
            "Focus on: key players, model capabilities, licensing, and trends."
        ),
        tools=["web_search", "web_fetch"],
        system_prompt=(
            "You are an open-source technology analyst. Search the web for "
            "current information about open-source AI. Cite all sources."
        ),
        priority=5,
    )

    reporter = AgentSpec(
        role="Report Writer",
        task=(
            "Synthesize the findings from both researchers into a comprehensive "
            "report on the state of AI in 2026. Include:\n"
            "1. Executive summary\n"
            "2. Major breakthroughs and developments\n"
            "3. Open-source landscape\n"
            "4. Regulatory developments\n"
            "5. Outlook and predictions\n\n"
            "Cite specific sources from the researchers' findings. Note that "
            "all research was done using a web search tool that the agents "
            "built from scratch in the previous phase."
        ),
        depends_on=[researcher_1.agent_id, researcher_2.agent_id],
        system_prompt=(
            "You are a technology journalist. Write a polished, well-sourced "
            "report synthesizing research from multiple analysts."
        ),
        priority=1,
    )

    return SwarmPlan(
        goal=(
            "Use the agent-built web search tool to research and report on "
            "the current state of AI in 2026, proving the self-built tool works."
        ),
        agents=[researcher_1, researcher_2, reporter],
        strategy=(
            "Layer 0: Two researchers search the web in parallel. "
            "Layer 1: Reporter synthesizes into a final report."
        ),
    )


# ═════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════

async def main() -> None:
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=3,
        output_dir=".",
        enable_agent_killing=True,
        agent_kill_threshold=5.0,
        agent_kill_min_time=120.0,
        max_tool_calls_per_agent=20,
        tool_timeout=60.0,
        tool_agent_timeout=900.0,   # 15 min per tool-using agent
        swarm_timeout=3600.0,       # 1 hour total (agents run sequentially)
    )

    print("\n" + "=" * 70)
    print("  SELF-BUILDING WEB SEARCH TOOL")
    print("  Phase 1: Agents build a DuckDuckGo search module from scratch")
    print("  Phase 2: Agents use the tool they built to research real topics")
    print(f"  Scratch dir: {SCRATCH}")
    print("=" * 70)

    # ── Phase 1: BUILD ───────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("  PHASE 1: Building the web search module...")
    print(f"{'─' * 70}\n")

    plan1 = phase1_plan()
    build_result = await Swarm(
        goal=plan1.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan1,
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

    # Show scratch contents
    files = sorted(os.listdir(SCRATCH))
    print(f"\n  Files in scratch dir ({len(files)}):")
    for f in files:
        size = os.path.getsize(os.path.join(SCRATCH, f))
        print(f"    {f} ({size:,} bytes)")

    # ── Phase 2: USE ─────────────────────────────────────────────

    print(f"\n{'─' * 70}")
    print("  PHASE 2: Loading agent-built web search tool...")
    print(f"{'─' * 70}\n")

    loaded = try_load_web_search_tool()
    if loaded is None:
        print("  [✗] Could not load the agent-built web search module.")
        print("      Phase 1 may not have produced working code.")
        print("      Check scratch dir for debug info.")
        return

    search_tool_def, mod = loaded

    # Build Phase 2 registry with the agent-created tools
    phase2_registry = ToolRegistry()
    phase2_registry.register(search_tool_def)
    if hasattr(mod, "_fetch_tool_def") and mod._fetch_tool_def is not None:
        phase2_registry.register(mod._fetch_tool_def)
        print("  [+] Registered: web_search, web_fetch")
    else:
        print("  [+] Registered: web_search (fetch_page not available)")

    print(f"\n  Running Phase 2 swarm with agent-built web search...\n")

    plan2 = phase2_plan()
    use_result = await Swarm(
        goal=plan2.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan2,
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
        preview = use_result.synthesis[:800]
        print(f"\n{'─' * 50}")
        print("  Report preview:\n")
        for line in preview.splitlines():
            print(f"  {line}")
        print(f"  ...")
        print(f"{'─' * 50}")

    print(f"\n  All files: {SCRATCH}/")
    print()


if __name__ == "__main__":
    asyncio.run(main())
