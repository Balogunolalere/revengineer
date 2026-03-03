#!/usr/bin/env python3
"""
Example: Run a full Arsenal security assessment.

This demonstrates the programmatic API — for CLI usage, see:
    uv run python -m cookbook.arsenal assess --target example.com

Prerequisites:
    1. Start the DeepSeek API proxy:  uv run python deepseek_api.py
    2. Build the Docker lab:          uv run python -m cookbook.arsenal lab --build
    3. Run this script:               uv run python cookbook/examples/arsenal_assess.py

For dry-run mode (no real tool execution):
    ARSENAL_DRY_RUN=true uv run python cookbook/examples/arsenal_assess.py
"""

import asyncio
import sys
import time

from cookbook.arsenal import (
    ArsenalConfig, MissionEngine, MissionPhase,
    Scope, ScopeRule, Finding,
)
from cookbook.arsenal.models import ScopeAction


async def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "scanme.nmap.org"

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  Arsenal Security Assessment             ║")
    print(f"║  Target: {target:<32s}║")
    print(f"╚══════════════════════════════════════════╝")

    # ── Configure ──
    config = ArsenalConfig(
        api_base="http://localhost:8000/v1",
        dry_run="--dry-run" in sys.argv,
        safe_mode=True,                   # non-destructive only
        use_docker=False,                 # set True if lab is built
        output_dir=".",
        max_parallel_tools=3,
        tool_timeout=120,
        mission_timeout=1800,             # 30 minutes
        scan_rate_limit=50,               # be gentle
        verbose=True,
    )

    # ── Build engine ──
    engine = MissionEngine(
        config=config,
        mission_name=f"Security Assessment: {target}",
        operator="arsenal-example",
    )

    # ── Define scope ──
    # In production: load from a signed scope file
    #   engine.load_scope("signed_scope.json")
    #
    # For this example, auto-generate scope:
    engine.add_scope_rule("allow", "domain", target, note="Primary target")
    engine.add_scope_rule("allow", "domain", f"*.{target}", note="Subdomains")

    # If target looks like an IP:
    if any(c.isdigit() for c in target.split(".")[0]):
        engine.add_scope_rule("allow", "ip", target)

    # Add target
    engine.add_target(target)

    # ── Callbacks ──
    def on_phase_start(phase: MissionPhase):
        icons = {
            MissionPhase.SCOPE_VALIDATION: "🔒",
            MissionPhase.RECONNAISSANCE: "🔍",
            MissionPhase.ENUMERATION: "📡",
            MissionPhase.VULNERABILITY_ANALYSIS: "💉",
            MissionPhase.REPORTING: "📄",
        }
        print(f"\n{'─'*50}")
        print(f"  {icons.get(phase, '▶')}  {phase.value.upper()}")
        print(f"{'─'*50}")

    def on_phase_end(phase: MissionPhase):
        print(f"  ✓ {phase.value} complete")

    def on_progress(phase, iteration, max_iter):
        print(f"  [{phase.value}] Iteration {iteration}/{max_iter}")

    engine.set_callbacks(
        on_phase_start=on_phase_start,
        on_phase_end=on_phase_end,
        on_progress=on_progress,
    )

    # ── Execute ──
    mission = await engine.execute()

    # ── Results ──
    print(f"\n{'='*50}")
    stats = mission.stats
    total = sum(stats.values())
    print(f"MISSION COMPLETE: {mission.status.value}")
    print(f"  Duration:  {mission.duration:.1f}s")
    print(f"  Targets:   {len(mission.targets)}")
    print(f"  Evidence:  {len(mission.evidence)} items")
    print(f"  Findings:  {total}")
    for sev, count in stats.items():
        if count > 0:
            print(f"    {sev.upper()}: {count}")
    print(f"  Audit log: {len(mission.audit_log)} entries")
    print(f"{'='*50}")

    # Print top findings
    critical_high = [
        f for f in mission.findings
        if f.severity.value in ("critical", "high") and not f.false_positive
    ]
    if critical_high:
        print(f"\n🚨 CRITICAL/HIGH FINDINGS:")
        for f in critical_high[:10]:
            print(f"  [{f.severity.value.upper()}] {f.title}")
            print(f"    Target: {f.target}:{f.port}" if f.port else f"    Target: {f.target}")
            if f.cve_ids:
                print(f"    CVEs: {', '.join(f.cve_ids)}")


if __name__ == "__main__":
    asyncio.run(main())
