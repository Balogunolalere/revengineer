"""
Live terminal renderer and file output for swarm execution.

Features:
- Real-time agent status display with colors
- Progress tracking
- Final report formatting
- Markdown and JSON file output
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from typing import TextIO

from .models import (
    AgentSpec, AgentResult, AgentStatus, SwarmPlan, SwarmResult,
)

# ANSI colors
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

STATUS_ICONS = {
    AgentStatus.PENDING: f"{DIM}○{RESET}",
    AgentStatus.WAITING: f"{YELLOW}◌{RESET}",
    AgentStatus.RUNNING: f"{CYAN}●{RESET}",
    AgentStatus.COMPLETED: f"{GREEN}✓{RESET}",
    AgentStatus.FAILED: f"{RED}✗{RESET}",
    AgentStatus.RETRYING: f"{YELLOW}↻{RESET}",
}

AGENT_COLORS = [CYAN, MAGENTA, BLUE, GREEN, YELLOW]


class SwarmRenderer:
    """Renders swarm progress to terminal and saves output to files."""

    def __init__(self, stream: bool = True, verbose: bool = False):
        self.stream = stream
        self.verbose = verbose
        self._start_time: float = 0
        self._agent_status: dict[str, AgentStatus] = {}
        self._agent_roles: dict[str, str] = {}
        self._log_lines: list[str] = []

    def _elapsed(self) -> str:
        e = time.time() - self._start_time
        return f"{e:.1f}s"

    def _color_for(self, idx: int) -> str:
        return AGENT_COLORS[idx % len(AGENT_COLORS)]

    def _print(self, msg: str):
        if self.stream:
            print(msg, flush=True)
        self._log_lines.append(_strip_ansi(msg))

    # ----- Event handlers (called by engine) -----

    def on_plan_ready(self, plan: SwarmPlan):
        self._start_time = time.time()
        self._print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
        self._print(f"{BOLD}{CYAN}║  🐝 SWARM ACTIVATED                          ║{RESET}")
        self._print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}")
        self._print(f"\n{BOLD}Goal:{RESET} {plan.goal}")
        if plan.strategy:
            self._print(f"{DIM}Strategy: {plan.strategy}{RESET}")
        self._print(f"{DIM}Agents: {len(plan.agents)} | Max parallel: configurable{RESET}")

        # Show agent lineup
        self._print(f"\n{BOLD}Agent Lineup:{RESET}")
        for i, agent in enumerate(plan.agents):
            color = self._color_for(i)
            deps = ""
            if agent.depends_on:
                dep_roles = []
                for dep_id in agent.depends_on:
                    dep_agent = plan.get_agent(dep_id)
                    dep_roles.append(dep_agent.role if dep_agent else dep_id)
                deps = f" {DIM}(waits for: {', '.join(dep_roles)}){RESET}"
            self._print(f"  {color}▸ {agent.role}{RESET}{deps}")
            self._agent_status[agent.agent_id] = AgentStatus.PENDING
            self._agent_roles[agent.agent_id] = agent.role

        self._print(f"\n{DIM}{'─' * 50}{RESET}")

    def on_agent_start(self, agent: AgentSpec):
        self._agent_status[agent.agent_id] = AgentStatus.RUNNING
        icon = STATUS_ICONS[AgentStatus.RUNNING]
        self._print(
            f"  {icon} {BOLD}{agent.role}{RESET} "
            f"{DIM}started{RESET} [{self._elapsed()}]"
        )

    def on_agent_done(self, result: AgentResult):
        self._agent_status[result.agent_id] = result.status
        if result.status == AgentStatus.COMPLETED:
            icon = STATUS_ICONS[AgentStatus.COMPLETED]
            content_preview = result.content[:80].replace("\n", " ")
            self._print(
                f"  {icon} {BOLD}{result.role}{RESET} "
                f"{GREEN}done{RESET} in {result.duration:.1f}s "
                f"[{self._elapsed()}]"
            )
            if self.verbose:
                self._print(f"     {DIM}{content_preview}...{RESET}")
        else:
            icon = STATUS_ICONS[AgentStatus.FAILED]
            self._print(
                f"  {icon} {BOLD}{result.role}{RESET} "
                f"{RED}FAILED{RESET}: {result.error} [{self._elapsed()}]"
            )

    def on_agent_retry(self, agent: AgentSpec, attempt: int, error: Exception):
        icon = STATUS_ICONS[AgentStatus.RETRYING]
        self._print(
            f"  {icon} {BOLD}{agent.role}{RESET} "
            f"{YELLOW}retrying{RESET} (attempt {attempt + 1}): {error} "
            f"[{self._elapsed()}]"
        )

    def on_synthesis_start(self):
        self._print(f"\n{DIM}{'─' * 50}{RESET}")
        self._print(f"  {CYAN}⟳{RESET} {BOLD}Synthesizing results...{RESET}")

    def on_complete(self, result: SwarmResult):
        succeeded = len(result.successful)
        failed = len(result.failed)
        total = len(result.agent_results)

        self._print(f"\n{BOLD}{GREEN}╔══════════════════════════════════════════════╗{RESET}")
        self._print(f"{BOLD}{GREEN}║  ✓ SWARM COMPLETE                            ║{RESET}")
        self._print(f"{BOLD}{GREEN}╚══════════════════════════════════════════════╝{RESET}")
        self._print(
            f"  Agents: {GREEN}{succeeded}/{total} succeeded{RESET}"
            + (f" | {RED}{failed} failed{RESET}" if failed else "")
        )
        self._print(f"  Duration: {BOLD}{result.duration:.1f}s{RESET}")

        # Show individual agent timings
        if self.verbose:
            self._print(f"\n{BOLD}Agent Breakdown:{RESET}")
            for r in sorted(result.agent_results, key=lambda x: x.started_at or 0):
                icon = STATUS_ICONS.get(r.status, "?")
                self._print(
                    f"  {icon} {r.role}: {r.duration:.1f}s "
                    f"({len(r.content)} chars, attempt {r.attempt})"
                )

    # ----- File output -----

    def save_markdown(self, result: SwarmResult, path: str):
        """Save the final report as a markdown file."""
        content = _build_markdown(result)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        self._print(f"  {DIM}📄 Report saved: {path}{RESET}")

    def save_json(self, result: SwarmResult, path: str):
        """Save the raw SwarmResult as JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        self._print(f"  {DIM}📊 JSON saved: {path}{RESET}")


# ----- Helpers -----

def _strip_ansi(text: str) -> str:
    return re.sub(r'\033\[[0-9;]*m', '', text)


def _build_markdown(result: SwarmResult) -> str:
    """Build a full markdown report from a SwarmResult."""
    lines = [
        f"# Swarm Report",
        f"",
        f"**Goal:** {result.goal}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Duration:** {result.duration:.1f}s",
        f"**Agents:** {len(result.successful)}/{len(result.agent_results)} succeeded",
        f"**Mode:** {result.mode.value}",
        f"",
    ]

    if result.plan.strategy:
        lines.extend([
            f"## Strategy",
            f"",
            f"{result.plan.strategy}",
            f"",
        ])

    lines.extend([
        f"---",
        f"",
        f"## Synthesized Report",
        f"",
        result.synthesis,
        f"",
        f"---",
        f"",
        f"## Individual Agent Outputs",
        f"",
    ])

    for i, r in enumerate(result.agent_results, 1):
        status = "✓" if r.status == AgentStatus.COMPLETED else "✗"
        lines.extend([
            f"### {status} Agent {i}: {r.role}",
            f"",
            f"- **Task:** {r.task}",
            f"- **Status:** {r.status.value}",
            f"- **Duration:** {r.duration:.1f}s",
            f"- **Attempt:** {r.attempt}",
            f"",
        ])
        if r.content:
            lines.extend([r.content, f""])
        if r.error:
            lines.extend([f"> ⚠️ Error: {r.error}", f""])

    lines.extend([
        f"---",
        f"",
        f"*Generated by Revengineer Swarm*",
    ])

    return "\n".join(lines)
