#!/usr/bin/env python3
"""
Example: Computer Use — agents that interact with the local machine.

Demonstrates swarm agents using ToolRegistry to perform real computer
operations: running shell commands, reading/writing files, fetching
web pages, and executing Python snippets. A planning agent decomposes
the task, worker agents use tools to gather data, and a reporter
synthesizes findings.

Tools provided:
  - shell_exec   — run a shell command and capture stdout/stderr
  - read_file    — read a local file's contents
  - write_file   — write text to a local file
  - python_exec  — execute a Python snippet in a subprocess
  - http_get     — fetch a URL and return the body (first 8 KB)

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Run this script:                uv run python cookbook/examples/computer_use.py
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import sys
import tempfile

from cookbook.swarm import (
    Swarm,
    SwarmConfig,
    SwarmMode,
    SwarmPlan,
    AgentSpec,
    ToolRegistry,
    ToolDef,
)


# ── Allowed-list for shell commands ──────────────────────────────

ALLOWED_COMMANDS = frozenset({
    "ls", "cat", "head", "tail", "wc", "grep", "find", "echo",
    "date", "whoami", "hostname", "uname", "uptime", "df", "du",
    "free", "env", "printenv", "ps", "id", "pwd", "which",
    "file", "stat", "sha256sum", "md5sum", "sort", "uniq",
    "tr", "cut", "awk", "sed", "diff", "tree", "pip", "python3",
    "git",
})

# Patterns that are never allowed regardless of base command
BLOCKED_PATTERNS = re.compile(
    r"rm\s|rmdir|mkfs|dd\s|>\s*/dev|chmod\s|chown\s|sudo\s|su\s|"
    r"nc\s|ncat|curl.*-[dX]|wget.*-O|eval\s|exec\s|reboot|shutdown|"
    r"systemctl|kill\s|pkill|&&\s*rm|;\s*rm|\|\s*rm",
    re.IGNORECASE,
)

MAX_OUTPUT_BYTES = 16_384  # 16 KB cap on tool output


def _is_command_safe(cmd: str) -> tuple[bool, str]:
    """Validate a shell command against the allow-list."""
    stripped = cmd.strip()
    if not stripped:
        return False, "Empty command"
    if BLOCKED_PATTERNS.search(stripped):
        return False, f"Blocked pattern detected in: {stripped!r}"
    try:
        parts = shlex.split(stripped)
    except ValueError as e:
        return False, f"Unparseable command: {e}"
    base = os.path.basename(parts[0])
    if base not in ALLOWED_COMMANDS:
        return False, f"Command {base!r} is not in the allowed list"
    return True, ""


# ── Tool implementations ────────────────────────────────────────

async def shell_exec(command: str = "", **_kw: object) -> str:
    """Run a shell command (allow-listed) and return combined output."""
    ok, reason = _is_command_safe(command)
    if not ok:
        return f"[DENIED] {reason}"
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=os.getcwd(),
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    text = stdout.decode(errors="replace")[:MAX_OUTPUT_BYTES]
    return text or "(no output)"


async def read_file_tool(path: str = "", **_kw: object) -> str:
    """Read a local file (max 16 KB)."""
    resolved = os.path.realpath(path)
    if not os.path.isfile(resolved):
        return f"[ERROR] File not found: {path}"
    try:
        with open(resolved, "r", errors="replace") as f:
            return f.read(MAX_OUTPUT_BYTES)
    except OSError as e:
        return f"[ERROR] {e}"


WRITE_DIR = tempfile.mkdtemp(prefix="swarm_cuse_")


async def write_file_tool(filename: str = "", content: str = "", **_kw: object) -> str:
    """Write content to a file inside the scratch directory."""
    safe_name = re.sub(r"[^a-zA-Z0-9._-]", "_", os.path.basename(filename))
    if not safe_name:
        return "[ERROR] Invalid filename"
    dest = os.path.join(WRITE_DIR, safe_name)
    try:
        with open(dest, "w") as f:
            f.write(content)
        return f"Wrote {len(content)} chars to {dest}"
    except OSError as e:
        return f"[ERROR] {e}"


async def python_exec(code: str = "", **_kw: object) -> str:
    """Execute a Python snippet in an isolated subprocess."""
    # Write to temp file, run with python3
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=WRITE_DIR,
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        text = stdout.decode(errors="replace")[:MAX_OUTPUT_BYTES]
        return text or "(no output)"
    finally:
        os.unlink(tmp_path)


async def http_get(url: str = "", **_kw: object) -> str:
    """Fetch a URL via curl and return the first 8 KB of the body."""
    if not url.startswith(("http://", "https://")):
        return "[ERROR] URL must start with http:// or https://"
    proc = await asyncio.create_subprocess_exec(
        "curl", "-sS", "-L", "--max-time", "15",
        "--max-filesize", "65536", url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)
    if proc.returncode != 0:
        return f"[ERROR] curl failed: {stderr.decode(errors='replace')[:500]}"
    return stdout.decode(errors="replace")[:8192]


# ── Build the registry ──────────────────────────────────────────

def build_computer_tools() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(ToolDef(
        name="shell_exec",
        description=(
            "Execute a shell command on the local machine. "
            "Only allow-listed commands (ls, cat, grep, find, git, etc.) are permitted."
        ),
        parameters={"command": "The shell command to run"},
        fn=shell_exec,
        safe=True,
    ))
    registry.register(ToolDef(
        name="read_file",
        description="Read the contents of a local file (up to 16 KB).",
        parameters={"path": "Absolute or relative file path"},
        fn=read_file_tool,
        safe=True,
    ))
    registry.register(ToolDef(
        name="write_file",
        description="Write text content to a file in the scratch directory.",
        parameters={
            "filename": "Name of the file to create",
            "content": "Text content to write",
        },
        fn=write_file_tool,
        safe=True,
    ))
    registry.register(ToolDef(
        name="python_exec",
        description=(
            "Execute a Python code snippet and return stdout. "
            "Use for calculations, data processing, or quick scripts."
        ),
        parameters={"code": "Python source code to execute"},
        fn=python_exec,
        safe=False,
    ))
    registry.register(ToolDef(
        name="http_get",
        description="Fetch a URL via HTTP GET and return the response body (up to 8 KB).",
        parameters={"url": "Full URL starting with http:// or https://"},
        fn=http_get,
        safe=True,
    ))

    return registry


# ── Main example ────────────────────────────────────────────────

async def main() -> None:
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=4,
        output_dir=".",
        enable_agent_killing=True,
        agent_kill_threshold=5.0,
        agent_kill_min_time=60.0,
        max_tool_calls_per_agent=15,
        tool_timeout=30.0,
    )

    registry = build_computer_tools()

    # ── Define agents with computer-use tools ────────────────────

    sysinfo = AgentSpec(
        role="System Profiler",
        task=(
            "Gather a profile of this machine. Use the shell_exec tool to run: "
            "uname -a, hostname, whoami, uptime, free -h, df -h, python3 --version, "
            "and git --version. Compile the results into a clean summary table."
        ),
        tools=["shell_exec"],
        system_prompt=(
            "You are a Linux system administrator. Use the shell_exec tool "
            "to run commands and report findings. Format output as a clean "
            "Markdown table."
        ),
        priority=5,
    )

    project_scanner = AgentSpec(
        role="Project Analyzer",
        task=(
            "Analyze the current project directory. Use shell_exec to run: "
            "ls -la, find . -name '*.py' | head -30, wc -l cookbook/swarm/*.py, "
            "and git log --oneline -10. Then read_file on pyproject.toml to "
            "extract project metadata. Produce a structured project overview."
        ),
        tools=["shell_exec", "read_file"],
        system_prompt=(
            "You are a software engineer analyzing a codebase. Use tools to "
            "inspect the project structure, count lines of code, check git "
            "history, and read config files. Be thorough and organized."
        ),
        priority=5,
    )

    code_analyzer = AgentSpec(
        role="Code Metrics Calculator",
        task=(
            "Calculate code metrics for the cookbook/swarm/ directory. "
            "Use python_exec to write and run a script that: "
            "(1) counts total .py files, (2) counts total lines of code, "
            "(3) counts classes and functions (grep for 'class ' and 'def '), "
            "(4) finds the largest file by line count. "
            "Present results as a formatted summary."
        ),
        tools=["python_exec", "shell_exec"],
        system_prompt=(
            "You are a code metrics specialist. Write Python scripts and "
            "use shell commands to calculate precise code statistics. "
            "Always verify your results."
        ),
        priority=4,
    )

    dependency_checker = AgentSpec(
        role="Dependency Auditor",
        task=(
            "Audit the project's Python dependencies. Read requirements.txt "
            "and pyproject.toml to list all dependencies. Use shell_exec "
            "to run 'pip list' and check installed versions. Flag any "
            "mismatches between required and installed versions."
        ),
        tools=["shell_exec", "read_file"],
        system_prompt=(
            "You are a dependency management expert. Read config files and "
            "compare declared vs installed dependencies. Highlight any "
            "version mismatches or missing packages."
        ),
        priority=3,
    )

    report_writer = AgentSpec(
        role="Technical Report Writer",
        task=(
            "Using the findings from all other agents, compile a comprehensive "
            "technical report about this machine and project. Include sections: "
            "1) System Overview, 2) Project Structure, 3) Code Metrics, "
            "4) Dependency Status, 5) Recommendations. "
            "Use write_file to save the report as 'project_report.md'."
        ),
        tools=["write_file"],
        depends_on=[
            sysinfo.agent_id,
            project_scanner.agent_id,
            code_analyzer.agent_id,
            dependency_checker.agent_id,
        ],
        system_prompt=(
            "You are a technical writer. Synthesize findings from multiple "
            "sources into a polished Markdown report. Be concise but thorough."
        ),
        priority=1,
    )

    plan = SwarmPlan(
        goal=(
            "Perform a comprehensive computer-use audit of this machine "
            "and the current project. Gather system info, analyze the codebase, "
            "calculate metrics, audit dependencies, and produce a unified report."
        ),
        agents=[sysinfo, project_scanner, code_analyzer, dependency_checker, report_writer],
        strategy=(
            "Layer 0: Four parallel agents each use computer tools to gather data. "
            "Layer 1: Report writer synthesizes everything and writes the output file."
        ),
    )

    print("\n" + "=" * 70)
    print("  COMPUTER USE EXAMPLE")
    print("  Agents will use tools to interact with this machine")
    print(f"  Scratch directory: {WRITE_DIR}")
    print("=" * 70 + "\n")

    result = await Swarm(
        goal=plan.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan,
        tool_registry=registry,
        save=True,
    )

    # ── Print results ────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    for r in result.agent_results:
        status = "✓" if r.status.value == "completed" else "✗"
        chars = len(r.content) if r.content else 0
        print(f"  {status} {r.role}: {r.duration:.1f}s ({chars} chars, attempt {r.attempt})")
        if r.error:
            print(f"    ERROR: {r.error}")

    print(f"\n  Synthesis: {len(result.synthesis)} chars")
    print(f"  Total time: {result.duration:.1f}s")
    print(f"\n  Output saved to {WRITE_DIR}/")

    # Show a snippet of the synthesis
    if result.synthesis:
        preview = result.synthesis[:500]
        print(f"\n{'─' * 50}")
        print(f"  Report preview:\n")
        print(f"  {preview}...")
        print(f"{'─' * 50}")


if __name__ == "__main__":
    asyncio.run(main())
