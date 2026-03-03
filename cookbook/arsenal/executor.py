"""
Tool executor — runs security tools with scope enforcement, evidence
capture, timeout handling, and optional Docker isolation.

Every tool execution goes through:
  1. Scope check (ScopeEnforcer)
  2. Command sanitization
  3. Execution (subprocess or Docker)
  4. Output capture + evidence creation
  5. Audit logging
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.models import (
    Evidence, EvidenceType, Mission, ToolResult, ToolCategory,
)
from cookbook.arsenal.scope import ScopeEnforcer, ScopeViolation

logger = logging.getLogger("arsenal.executor")


# Dangerous patterns that should NEVER be in a command
BLOCKED_PATTERNS = [
    "rm -rf /",
    "mkfs.",
    "dd if=/dev/zero",
    ":(){:|:&};:",     # fork bomb
    "> /dev/sda",
    "chmod -R 777 /",
    "|sh",              # pipe to shell
    "| sh",
    "|bash",
    "| bash",
    "eval(",
    "$(curl",
    "$(wget",
]


@dataclass
class ToolSpec:
    """Registration for a security tool."""
    name: str
    category: ToolCategory
    binary: str            # binary name or path
    description: str = ""
    safe: bool = True      # True = non-destructive (recon/scan only)
    requires_root: bool = False
    parse_fn: Callable[[str], dict[str, Any]] | None = None   # output parser
    default_args: list[str] = field(default_factory=list)
    available: bool = False  # set at runtime after checking binary exists


class ToolExecutor:
    """
    Executes security tools with full safety controls.

    All execution flows through scope enforcement, and all output
    is captured as forensic evidence.
    """

    def __init__(
        self,
        config: ArsenalConfig,
        mission: Mission,
        scope_enforcer: ScopeEnforcer,
    ):
        self.config = config
        self.mission = mission
        self.scope = scope_enforcer
        self._tools: dict[str, ToolSpec] = {}
        self._semaphore = asyncio.Semaphore(config.max_parallel_tools)
        self._execution_count = 0

    def register_tool(self, spec: ToolSpec) -> None:
        """Register a tool. Checks if binary is available."""
        if self.config.use_docker:
            spec.available = True  # assume tools are in the Docker image
        else:
            spec.available = shutil.which(spec.binary) is not None
        self._tools[spec.name] = spec
        if spec.available:
            logger.info(f"Tool registered: {spec.name} ({spec.binary})")
        else:
            logger.warning(f"Tool registered but NOT FOUND: {spec.name} ({spec.binary})")

    def get_tool(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def list_tools(self, category: ToolCategory | None = None) -> list[ToolSpec]:
        """List available tools, optionally filtered by category."""
        tools = [t for t in self._tools.values() if t.available]
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: t.name)

    async def execute(
        self,
        tool_name: str,
        args: list[str],
        target: str = "",
        timeout: float | None = None,
        parse: bool = True,
    ) -> ToolResult:
        """
        Execute a security tool with full safety controls.

        Args:
            tool_name: registered tool name
            args: command-line arguments
            target: primary target (for scope checking)
            timeout: override per-tool timeout
            parse: whether to run the tool's parser on output

        Returns:
            ToolResult with output, evidence, and parsed data
        """
        spec = self._tools.get(tool_name)
        if not spec:
            return ToolResult(
                tool_name=tool_name, command="",
                success=False, error=f"Unknown tool: {tool_name}",
            )

        if not spec.available:
            return ToolResult(
                tool_name=tool_name, command="",
                success=False, error=f"Tool not available: {spec.binary}",
            )

        # Safe mode check
        if self.config.safe_mode and not spec.safe:
            return ToolResult(
                tool_name=tool_name, command="",
                success=False,
                error=f"Tool '{tool_name}' blocked: safe_mode is ON and tool is destructive",
            )

        # Build command
        cmd_parts = [spec.binary] + spec.default_args + args
        command = shlex.join(cmd_parts)

        # Sanitize command
        violation = self._sanitize_command(command)
        if violation:
            self.mission.log("command_blocked", f"{tool_name}: {violation}")
            return ToolResult(
                tool_name=tool_name, command=command,
                success=False, error=f"Command blocked: {violation}",
            )

        # Scope enforcement
        if target:
            try:
                self.scope.enforce(target, context=tool_name)
            except ScopeViolation as e:
                return ToolResult(
                    tool_name=tool_name, command=command,
                    success=False, error=str(e),
                )

        # Also check the command for embedded targets
        cmd_violations = self.scope.check_command(command, context=tool_name)
        if cmd_violations:
            return ToolResult(
                tool_name=tool_name, command=command,
                success=False,
                error=f"Command targets out-of-scope hosts: {cmd_violations}",
            )

        # Dry run
        if self.config.dry_run:
            self.mission.log("dry_run", f"{tool_name}: {command}")
            return ToolResult(
                tool_name=tool_name, command=command,
                success=True, stdout=f"[DRY RUN] Would execute: {command}",
            )

        # Confirmation gate
        if self.config.require_confirmation:
            logger.info(f"CONFIRMATION REQUIRED: {command}")
            # In a real deployment, this would pause for human approval
            # For now, we proceed (the CLI layer handles confirmation)

        # Execute
        self.mission.log("tool_executing", f"{tool_name}: {command}")
        timeout = timeout or self.config.tool_timeout

        async with self._semaphore:
            self._execution_count += 1
            t0 = time.time()

            try:
                if self.config.use_docker:
                    result = await self._execute_docker(command, timeout)
                else:
                    result = await self._execute_local(command, timeout)
            except asyncio.TimeoutError:
                duration = time.time() - t0
                self.mission.log("tool_timeout", f"{tool_name} after {duration:.1f}s")
                return ToolResult(
                    tool_name=tool_name, command=command,
                    success=False, duration=duration,
                    error=f"Timeout after {duration:.1f}s",
                )
            except Exception as e:
                duration = time.time() - t0
                self.mission.log("tool_error", f"{tool_name}: {e}")
                return ToolResult(
                    tool_name=tool_name, command=command,
                    success=False, duration=duration,
                    error=str(e),
                )

            duration = time.time() - t0
            stdout, stderr, exit_code = result

        # Build result
        tool_result = ToolResult(
            tool_name=tool_name,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=duration,
            success=(exit_code == 0),
        )

        # Parse output if parser exists
        if parse and spec.parse_fn and tool_result.success:
            try:
                tool_result.parsed = spec.parse_fn(stdout)
            except Exception as e:
                logger.warning(f"Parse error for {tool_name}: {e}")
                tool_result.parsed = {"parse_error": str(e)}

        # Create evidence
        evidence = Evidence(
            evidence_type=EvidenceType.TOOL_OUTPUT,
            tool_name=tool_name,
            command=command,
            raw_output=stdout + ("\n--- STDERR ---\n" + stderr if stderr else ""),
            parsed_data=tool_result.parsed,
            target=target,
            duration=duration,
        )
        evidence_id = self.mission.add_evidence(evidence)
        tool_result.evidence = evidence

        # Save raw output to file
        if self.config.save_raw_output:
            self._save_output(tool_name, evidence_id, stdout, stderr)

        self.mission.log(
            "tool_completed",
            f"{tool_name}: exit={exit_code} duration={duration:.1f}s "
            f"output={len(stdout)} bytes",
        )

        return tool_result

    async def _execute_local(
        self, command: str, timeout: float,
    ) -> tuple[str, str, int]:
        """Execute a command locally via subprocess."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._safe_env(),
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
            proc.returncode or 0,
        )

    async def _execute_docker(
        self, command: str, timeout: float,
    ) -> tuple[str, str, int]:
        """Execute a command inside the Docker attack lab container."""
        evidence_mount = ""
        if self.config.mount_evidence:
            edir = self.config.get_evidence_dir()
            edir.mkdir(parents=True, exist_ok=True)
            evidence_mount = f"-v {edir}:/evidence"

        docker_cmd = (
            f"docker run --rm "
            f"--network {self.config.docker_network} "
            f"--memory {self.config.docker_memory} "
            f"--cpus {self.config.docker_cpus} "
            f"--security-opt no-new-privileges "
            f"--cap-drop ALL "
            f"--cap-add NET_RAW "                # needed for nmap SYN scan
            f"{evidence_mount} "
            f"{self.config.docker_image} "
            f"sh -c {shlex.quote(command)}"
        )

        proc = await asyncio.create_subprocess_shell(
            docker_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
        return (
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
            proc.returncode or 0,
        )

    def _sanitize_command(self, command: str) -> str | None:
        """Check for dangerous command patterns. Returns violation or None."""
        cmd_lower = command.lower()
        for pattern in BLOCKED_PATTERNS:
            if pattern.lower() in cmd_lower:
                return f"Blocked pattern: {pattern}"

        # Check for shell injection attempts
        if any(c in command for c in [";", "&&", "||", "`", "$("]):
            # Allow these ONLY if they're part of the tool's expected format
            # Simple heuristic: block if it looks like chaining
            if ";" in command and command.count(";") > 1:
                return "Multiple command separators detected"

        return None

    def _safe_env(self) -> dict[str, str]:
        """Build a sanitized environment for subprocess execution."""
        env = dict(os.environ)
        # Remove sensitive variables
        for key in list(env.keys()):
            if any(s in key.upper() for s in [
                "PASSWORD", "SECRET", "TOKEN", "KEY", "CREDENTIAL",
                "AWS_", "GITHUB_", "DOCKER_",
            ]):
                del env[key]
        return env

    def _save_output(
        self, tool_name: str, evidence_id: str,
        stdout: str, stderr: str,
    ) -> None:
        """Save tool output to the evidence directory."""
        edir = self.config.get_evidence_dir()
        edir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{tool_name}_{evidence_id}_{ts}"

        if stdout:
            (edir / f"{base}.stdout.txt").write_text(stdout)
        if stderr:
            (edir / f"{base}.stderr.txt").write_text(stderr)

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "total_executions": self._execution_count,
            "registered_tools": len(self._tools),
            "available_tools": len([t for t in self._tools.values() if t.available]),
            "scope_stats": self.scope.stats,
        }
