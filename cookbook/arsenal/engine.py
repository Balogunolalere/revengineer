"""
Mission Execution Engine — orchestrates the full assessment pipeline.

Runs operators in sequence through the kill chain:
  1. Scope Validation  — verify authorization, sign scope
  2. Reconnaissance    — OSINT, DNS, subdomain enum
  3. Enumeration       — port scanning, service detection
  4. Vulnerability     — vuln scanning, analysis
  5. Reporting         — synthesis, attack chains, remediation

Each phase feeds its results into the next. The engine maintains
the Mission object as the single source of truth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.models import (
    Finding, Mission, MissionPhase, MissionStatus,
    Scope, ScopeRule, ScopeAction, Target,
)
from cookbook.arsenal.scope import ScopeEnforcer
from cookbook.arsenal.executor import ToolExecutor
from cookbook.arsenal.tools import SecurityTools
from cookbook.arsenal.operators import (
    ReconOperator, ScanOperator, VulnOperator, AnalysisOperator,
    OperatorResult,
)

logger = logging.getLogger("arsenal.engine")


class MissionEngine:
    """
    Top-level engine that runs a complete security assessment.

    Usage:
        config = ArsenalConfig(...)
        engine = MissionEngine(config)

        # Define scope
        engine.add_scope_rule("allow", "domain", "*.example.com")
        engine.add_scope_rule("allow", "cidr", "10.0.0.0/24")
        engine.add_scope_rule("deny", "ip", "10.0.0.1")  # exclude prod DB

        # Add targets
        engine.add_target("example.com")

        # Run
        result = await engine.execute()
    """

    def __init__(
        self,
        config: ArsenalConfig | None = None,
        mission_name: str = "",
        operator: str = "",
    ):
        self.config = config or ArsenalConfig()
        self.mission = Mission(
            name=mission_name or "Arsenal Assessment",
            operator=operator or "arsenal-auto",
        )
        self._scope_enforcer: ScopeEnforcer | None = None
        self._executor: ToolExecutor | None = None
        self._tools: SecurityTools | None = None
        self._on_progress: Callable | None = None
        self._on_phase_start: Callable | None = None
        self._on_phase_end: Callable | None = None
        self._on_finding: Callable | None = None
        self._phase_results: dict[str, OperatorResult] = {}

    # ── Configuration ─────────────────────────────────────────────

    def add_scope_rule(
        self,
        action: str,      # "allow" or "deny"
        target_type: str,  # "ip", "cidr", "domain", "port", "url"
        value: str,
        note: str = "",
    ) -> MissionEngine:
        """Add a scope rule. Returns self for chaining."""
        self.mission.scope.rules.append(ScopeRule(
            action=ScopeAction(action),
            target_type=target_type,
            value=value,
            note=note,
        ))
        return self

    def add_target(self, host: str, ports: list[int] | None = None) -> MissionEngine:
        """Add an initial target. Returns self for chaining."""
        self.mission.targets.append(Target(
            host=host,
            ports=ports or [],
        ))
        return self

    def load_scope(self, path: str) -> MissionEngine:
        """Load scope from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        self.mission.scope = Scope.from_dict(data)
        return self

    def set_callbacks(
        self,
        on_progress: Callable | None = None,
        on_phase_start: Callable | None = None,
        on_phase_end: Callable | None = None,
        on_finding: Callable | None = None,
    ) -> MissionEngine:
        """Set event callbacks. Returns self for chaining."""
        self._on_progress = on_progress
        self._on_phase_start = on_phase_start
        self._on_phase_end = on_phase_end
        self._on_finding = on_finding
        return self

    # ── Execution ─────────────────────────────────────────────────

    async def execute(
        self,
        phases: list[MissionPhase] | None = None,
    ) -> Mission:
        """
        Execute the full assessment mission.

        Args:
            phases: specific phases to run (default: all phases in order)

        Returns:
            The completed Mission object with all findings and evidence.
        """
        if phases is None:
            phases = [
                MissionPhase.SCOPE_VALIDATION,
                MissionPhase.RECONNAISSANCE,
                MissionPhase.ENUMERATION,
                MissionPhase.VULNERABILITY_ANALYSIS,
                MissionPhase.REPORTING,
            ]

        self.mission.started_at = time.time()
        self.mission.status = MissionStatus.RUNNING
        self.mission.log("mission_start", f"Phases: {[p.value for p in phases]}")

        try:
            # Phase 0: Validate & sign scope
            if MissionPhase.SCOPE_VALIDATION in phases:
                await self._phase_scope_validation()

            # Initialize executor and tools
            self._scope_enforcer = ScopeEnforcer(self.mission)
            self._executor = ToolExecutor(self.config, self.mission, self._scope_enforcer)
            self._tools = SecurityTools(self._executor, self.config)

            # Run assessment phases
            if MissionPhase.RECONNAISSANCE in phases:
                await self._phase_recon()

            if MissionPhase.ENUMERATION in phases:
                await self._phase_scan()

            if MissionPhase.VULNERABILITY_ANALYSIS in phases:
                await self._phase_vuln()

            if MissionPhase.REPORTING in phases:
                await self._phase_analysis()

            self.mission.status = MissionStatus.COMPLETED

        except asyncio.TimeoutError:
            self.mission.status = MissionStatus.FAILED
            self.mission.log("mission_timeout", f"Exceeded {self.config.mission_timeout}s")

        except Exception as e:
            self.mission.status = MissionStatus.FAILED
            self.mission.log("mission_error", str(e))
            logger.exception(f"Mission failed: {e}")

        finally:
            self.mission.finished_at = time.time()
            self.mission.log(
                "mission_end",
                f"Status: {self.mission.status.value}, "
                f"Duration: {self.mission.duration:.1f}s, "
                f"Findings: {len(self.mission.findings)}, "
                f"Evidence: {len(self.mission.evidence)}",
            )

            # Save artifacts
            if self.config.save_report:
                await self._save_artifacts()

        return self.mission

    # ── Phase Implementations ─────────────────────────────────────

    async def _phase_scope_validation(self) -> None:
        """Validate and sign the scope."""
        self.mission.phase = MissionPhase.SCOPE_VALIDATION
        self._emit_phase_start(MissionPhase.SCOPE_VALIDATION)

        if not self.mission.scope.rules:
            if self.config.require_scope_approval:
                raise ValueError(
                    "No scope rules defined. Add rules with add_scope_rule() "
                    "or load from file with load_scope()"
                )
            else:
                self.mission.log("scope_warning", "No scope rules — operating unrestricted")

        # Sign the scope
        sig = self.mission.scope.sign()
        self.mission.log("scope_signed", f"Scope hash: {sig[:16]}...")

        # Validate time bounds
        if self.mission.scope.valid_until and not self.mission.scope.is_valid():
            raise ValueError("Scope authorization has expired")

        # Validate all initial targets are in scope
        enforcer = ScopeEnforcer(self.mission)
        for target in self.mission.targets:
            if not enforcer.check_target(target.host, "initial_target"):
                raise ValueError(
                    f"Initial target '{target.host}' is not within the approved scope"
                )

        self.mission.status = MissionStatus.SCOPE_APPROVED
        self.mission.log("scope_validated", f"{len(self.mission.scope.rules)} rules, "
                         f"{len(self.mission.targets)} initial targets")
        self._emit_phase_end(MissionPhase.SCOPE_VALIDATION)

    async def _phase_recon(self) -> None:
        """Run the reconnaissance operator."""
        self.mission.phase = MissionPhase.RECONNAISSANCE
        self._emit_phase_start(MissionPhase.RECONNAISSANCE)

        async with ReconOperator(self.config, self.mission, self._tools) as op:
            result = await asyncio.wait_for(
                op.run(on_progress=self._on_progress),
                timeout=self.config.mission_timeout / 4,
            )

        self._phase_results["recon"] = result
        self.mission.log(
            "recon_complete",
            f"Discovered {len(result.targets_discovered)} targets, "
            f"{result.tool_calls_made} tool calls",
        )
        self._emit_phase_end(MissionPhase.RECONNAISSANCE)

    async def _phase_scan(self) -> None:
        """Run the scanning operator."""
        self.mission.phase = MissionPhase.ENUMERATION
        self._emit_phase_start(MissionPhase.ENUMERATION)

        async with ScanOperator(self.config, self.mission, self._tools) as op:
            result = await asyncio.wait_for(
                op.run(on_progress=self._on_progress),
                timeout=self.config.mission_timeout / 3,
            )

        self._phase_results["scan"] = result
        self.mission.log(
            "scan_complete",
            f"{result.tool_calls_made} scans, {len(self.mission.findings)} findings so far",
        )
        self._emit_phase_end(MissionPhase.ENUMERATION)

    async def _phase_vuln(self) -> None:
        """Run the vulnerability operator."""
        self.mission.phase = MissionPhase.VULNERABILITY_ANALYSIS
        self._emit_phase_start(MissionPhase.VULNERABILITY_ANALYSIS)

        async with VulnOperator(self.config, self.mission, self._tools) as op:
            result = await asyncio.wait_for(
                op.run(on_progress=self._on_progress),
                timeout=self.config.mission_timeout / 3,
            )

        self._phase_results["vuln"] = result
        self.mission.log(
            "vuln_complete",
            f"{result.tool_calls_made} scans, {len(self.mission.findings)} total findings",
        )
        self._emit_phase_end(MissionPhase.VULNERABILITY_ANALYSIS)

    async def _phase_analysis(self) -> None:
        """Run the analysis operator for final reporting."""
        self.mission.phase = MissionPhase.REPORTING
        self._emit_phase_start(MissionPhase.REPORTING)

        async with AnalysisOperator(self.config, self.mission, self._tools) as op:
            result = await asyncio.wait_for(
                op.run(on_progress=self._on_progress),
                timeout=self.config.mission_timeout / 4,
            )

        self._phase_results["analysis"] = result
        self.mission.log("analysis_complete", f"Duration: {result.duration:.1f}s")
        self._emit_phase_end(MissionPhase.REPORTING)

    # ── Output ────────────────────────────────────────────────────

    async def _save_artifacts(self) -> None:
        """Save mission artifacts to disk."""
        out_dir = Path(self.config.output_dir) if self.config.output_dir else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"arsenal_{self.mission.mission_id}_{ts}"

        # JSON report
        json_path = out_dir / f"{base}.json"
        with open(json_path, "w") as f:
            json.dump(self.mission.to_dict(), f, indent=2, default=str)
        self.mission.log("artifact_saved", str(json_path))

        # Markdown report
        if self.config.report_format in ("markdown", "all"):
            md_path = out_dir / f"{base}.md"
            md_content = self._generate_markdown_report()
            md_path.write_text(md_content)
            self.mission.log("artifact_saved", str(md_path))

        logger.info(f"Artifacts saved to {out_dir}")

    def _generate_markdown_report(self) -> str:
        """Generate a markdown assessment report."""
        m = self.mission
        stats = m.stats
        lines = [
            f"# Security Assessment Report",
            f"",
            f"**Mission:** {m.name}",
            f"**ID:** {m.mission_id}",
            f"**Operator:** {m.operator}",
            f"**Status:** {m.status.value}",
            f"**Duration:** {m.duration:.1f} seconds",
            f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(m.started_at))}",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"| Severity | Count |",
            f"|----------|-------|",
            f"| Critical | {stats.get('critical', 0)} |",
            f"| High     | {stats.get('high', 0)} |",
            f"| Medium   | {stats.get('medium', 0)} |",
            f"| Low      | {stats.get('low', 0)} |",
            f"| Info     | {stats.get('info', 0)} |",
            f"| **Total**| **{sum(stats.values())}** |",
            f"",
            f"## Scope",
            f"",
            f"| Action | Type | Value | Note |",
            f"|--------|------|-------|------|",
        ]

        for r in m.scope.rules:
            lines.append(f"| {r.action.value} | {r.target_type} | `{r.value}` | {r.note} |")

        lines.extend([
            f"",
            f"## Targets Assessed",
            f"",
        ])

        for t in m.targets:
            svcs = ", ".join(f"{p}:{s}" for p, s in t.services.items())
            lines.append(f"- **{t.label}** — Ports: {t.ports or 'N/A'} — Services: {svcs or 'N/A'}")

        lines.extend([
            f"",
            f"## Findings",
            f"",
        ])

        # Group by severity
        for sev in [s for s in Severity]:
            sev_findings = [f for f in m.findings if f.severity == sev and not f.false_positive]
            if not sev_findings:
                continue
            lines.append(f"### {sev.value.upper()} ({len(sev_findings)})")
            lines.append("")

            for i, f in enumerate(sev_findings, 1):
                lines.extend([
                    f"#### {i}. {f.title}",
                    f"",
                    f"- **Target:** `{f.target}`" + (f":{f.port}" if f.port else ""),
                    f"- **Category:** {f.category.value}",
                    f"- **CVSS:** {f.cvss_score}" if f.cvss_score else "",
                ])
                if f.cve_ids:
                    lines.append(f"- **CVEs:** {', '.join(f.cve_ids)}")
                if f.mitre_attack:
                    lines.append(f"- **MITRE ATT&CK:** {', '.join(f.mitre_attack)}")
                lines.extend([
                    f"",
                    f"{f.description}",
                    f"",
                ])
                if f.remediation:
                    lines.append(f"**Remediation:** {f.remediation}")
                    lines.append("")

        # Analysis section
        analysis = self._phase_results.get("analysis")
        if analysis and analysis.raw_analysis:
            lines.extend([
                f"## Analysis",
                f"",
                analysis.raw_analysis,
                f"",
            ])

        # Evidence chain
        lines.extend([
            f"## Evidence Chain",
            f"",
            f"| ID | Type | Tool | Target | SHA-256 (first 16) |",
            f"|----|------|------|--------|-------------------|",
        ])
        for e in m.evidence:
            if e.evidence_type != EvidenceType.LLM_ANALYSIS:
                lines.append(
                    f"| {e.evidence_id} | {e.evidence_type.value} | {e.tool_name} "
                    f"| {e.target} | `{e.sha256[:16]}...` |"
                )

        lines.extend([
            f"",
            f"## Audit Log",
            f"",
            f"<details>",
            f"<summary>Full audit trail ({len(m.audit_log)} entries)</summary>",
            f"",
            f"```",
        ])
        for entry in m.audit_log:
            ts = time.strftime(
                "%H:%M:%S",
                time.localtime(entry.get("timestamp", 0)),
            )
            lines.append(
                f"[{ts}] [{entry.get('phase', '')}] "
                f"{entry.get('action', '')}: {entry.get('detail', '')}"
            )
        lines.extend([
            f"```",
            f"",
            f"</details>",
            f"",
            f"---",
            f"*Generated by Arsenal — Autonomous Offensive Security Platform*",
        ])

        return "\n".join(lines)

    # ── Callbacks ─────────────────────────────────────────────────

    def _emit_phase_start(self, phase: MissionPhase) -> None:
        if self._on_phase_start:
            try:
                self._on_phase_start(phase)
            except Exception:
                pass

    def _emit_phase_end(self, phase: MissionPhase) -> None:
        if self._on_phase_end:
            try:
                self._on_phase_end(phase)
            except Exception:
                pass


# Import for report generation
from cookbook.arsenal.models import EvidenceType, Severity
