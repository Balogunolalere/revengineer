#!/usr/bin/env python3
"""
Example: Arsenal Auto-Attack — autonomous security assessment with real tools.

Uses SwarmMode.AUTO to let the LLM decompose a security assessment, then
agents execute real Arsenal tools (nmap, nuclei, subfinder, etc.) against
the target within a scope-enforced, audited pipeline.

Architecture:
  1. User provides a target (domain, IP, or CIDR)
  2. A Mission + Scope is built to authorize the target
  3. Arsenal SecurityTools are registered into the swarm ToolRegistry
  4. AUTO mode decomposes the assessment into agents that call real tools
  5. Results are synthesized into a final security report

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Ensure security tools are installed (nmap, nuclei, subfinder, etc.)
       OR have Docker with the arsenal-lab image.
    3. Run this script:
         uv run python cookbook/examples/arsenal_attack.py scanme.nmap.org

    ⚠️  Only run against targets you have explicit authorization to test.

Usage:
    # Single domain
    uv run python cookbook/examples/arsenal_attack.py scanme.nmap.org

    # IP address
    uv run python cookbook/examples/arsenal_attack.py 45.33.32.156

    # CIDR range
    uv run python cookbook/examples/arsenal_attack.py 10.0.0.0/24

    # With scope type override (default: auto-detect)
    uv run python cookbook/examples/arsenal_attack.py --scope-type domain example.com
"""

import asyncio
import ipaddress
import re
import sys
import time

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.executor import ToolExecutor
from cookbook.arsenal.models import (
    Mission, MissionStatus, MissionPhase,
    Scope, ScopeRule, ScopeAction,
)
from cookbook.arsenal.scope import ScopeEnforcer
from cookbook.arsenal.tools import SecurityTools

from cookbook.swarm import (
    Swarm, SwarmConfig, SwarmMode, ToolRegistry,
)
from cookbook.swarm.arsenal_bridge import register_arsenal_tools


# ── Scope helpers ─────────────────────────────────────────────────


def detect_target_type(target: str) -> str:
    """Auto-detect whether the target is an IP, CIDR, or domain."""
    try:
        ipaddress.ip_address(target)
        return "ip"
    except ValueError:
        pass
    try:
        ipaddress.ip_network(target, strict=False)
        return "cidr"
    except ValueError:
        pass
    if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?(\.[a-zA-Z]{2,})+$', target):
        return "domain"
    return "domain"  # fallback


def build_scope(target: str, target_type: str) -> Scope:
    """Build a signed scope that authorizes the given target."""
    rules = []

    if target_type == "domain":
        # Allow the domain and all subdomains
        rules.append(ScopeRule(
            action=ScopeAction.ALLOW,
            target_type="domain",
            value=f"*.{target}",
            note=f"Wildcard subdomain scope for {target}",
        ))
        rules.append(ScopeRule(
            action=ScopeAction.ALLOW,
            target_type="domain",
            value=target,
            note=f"Primary domain {target}",
        ))
    elif target_type == "ip":
        rules.append(ScopeRule(
            action=ScopeAction.ALLOW,
            target_type="ip",
            value=target,
            note=f"Single IP target {target}",
        ))
    elif target_type == "cidr":
        rules.append(ScopeRule(
            action=ScopeAction.ALLOW,
            target_type="cidr",
            value=target,
            note=f"CIDR range {target}",
        ))

    # Allow all ports on in-scope targets
    rules.append(ScopeRule(
        action=ScopeAction.ALLOW,
        target_type="port",
        value="1-65535",
        note="All ports authorized for in-scope targets",
    ))

    # Allow URL access for in-scope targets
    rules.append(ScopeRule(
        action=ScopeAction.ALLOW,
        target_type="url",
        value=f"*{target}*",
        note=f"URL access for {target}",
    ))

    scope = Scope(
        rules=rules,
        authorized_by="arsenal_attack.py (automated scope)",
        authorization_ref="CLI invocation",
        valid_from=time.time(),
        valid_until=time.time() + 3600 * 4,  # 4-hour window
    )
    scope.sign()
    return scope


def build_mission(target: str, target_type: str, scope: Scope) -> Mission:
    """Build an Arsenal mission for the target."""
    return Mission(
        name=f"Auto-Attack: {target}",
        description=(
            f"Automated security assessment of {target} ({target_type}) "
            f"using swarm-orchestrated Arsenal tools."
        ),
        scope=scope,
        status=MissionStatus.SCOPE_APPROVED,
        phase=MissionPhase.RECONNAISSANCE,
        operator="arsenal_attack.py",
    )


# ── Main ──────────────────────────────────────────────────────────


async def main():
    # Parse CLI args
    args = sys.argv[1:]
    scope_type_override = None

    if "--scope-type" in args:
        idx = args.index("--scope-type")
        if idx + 1 < len(args):
            scope_type_override = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if not args:
        print("Usage: uv run python cookbook/examples/arsenal_attack.py [--scope-type domain|ip|cidr] <target>")
        print("\nExamples:")
        print("  uv run python cookbook/examples/arsenal_attack.py scanme.nmap.org")
        print("  uv run python cookbook/examples/arsenal_attack.py 45.33.32.156")
        print("  uv run python cookbook/examples/arsenal_attack.py 10.0.0.0/24")
        sys.exit(1)

    target = args[0]
    target_type = scope_type_override or detect_target_type(target)

    print(f"{'='*60}")
    print(f"  ARSENAL AUTO-ATTACK")
    print(f"  Target:     {target}")
    print(f"  Type:       {target_type}")
    print(f"{'='*60}")

    # ── 1. Build scope + mission ─────────────────────────────────

    scope = build_scope(target, target_type)
    mission = build_mission(target, target_type, scope)

    print(f"  Mission:    {mission.mission_id}")
    print(f"  Scope:      {len(scope.rules)} rules, signed={bool(scope.signature_hash)}")
    print(f"  Valid for:  4 hours")
    print(f"{'='*60}\n")

    # ── 2. Initialize Arsenal ────────────────────────────────────

    arsenal_config = ArsenalConfig(
        safe_mode=True,          # non-destructive tools only
        use_docker=False,        # use host tools (set True for Docker isolation)
        tool_timeout=120.0,
        scan_rate_limit=100,
    )

    scope_enforcer = ScopeEnforcer(mission)
    executor = ToolExecutor(arsenal_config, mission, scope_enforcer)
    security_tools = SecurityTools(executor, arsenal_config)

    # ── 3. Register Arsenal tools into swarm ─────────────────────

    tool_registry = ToolRegistry()
    register_arsenal_tools(tool_registry, security_tools)

    print(f"Registered {len(tool_registry._tools)} Arsenal tools into swarm\n")

    # ── 4. Run the swarm ─────────────────────────────────────────

    swarm_config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=3,
        agent_timeout=300,
        tool_agent_timeout=900,    # tool agents get more time
        swarm_timeout=3600,        # 1 hour total
        max_retries=3,
        max_tokens=4096,
        output_dir=".",
    )

    goal = (
        f"Perform a comprehensive security assessment of the target: {target}\n\n"
        f"Target type: {target_type}\n\n"
        f"You have access to the following real security tools:\n"
        f"  - nmap_scan: Port scanning and service detection\n"
        f"  - nuclei_scan: Template-based vulnerability scanning\n"
        f"  - subfinder_enum: Passive subdomain enumeration\n"
        f"  - httpx_probe: HTTP probing with tech detection\n"
        f"  - dns_lookup: DNS record queries\n"
        f"  - whois_lookup: WHOIS registration info\n"
        f"  - nikto_scan: Web server vulnerability scanner\n"
        f"  - whatweb_scan: Web technology fingerprinting\n"
        f"  - testssl_scan: TLS/SSL analysis\n"
        f"  - curl_request: HTTP requests for banner grabbing\n\n"
        f"Execute a methodical assessment:\n"
        f"  Phase 1 — Reconnaissance: Use subfinder, dns_lookup, whois_lookup to map the attack surface\n"
        f"  Phase 2 — Enumeration: Use nmap_scan, httpx_probe, whatweb_scan to enumerate services and tech\n"
        f"  Phase 3 — Vulnerability Analysis: Use nuclei_scan, nikto_scan, testssl_scan to find vulnerabilities\n"
        f"  Phase 4 — Synthesis: Analyze all findings, assess risk, and produce a prioritized report\n\n"
        f"IMPORTANT: Each agent MUST call its assigned tools using the tool-calling format. "
        f"Do not just describe what you would do — actually execute the tools and analyze real output.\n\n"
        f"Produce a final security assessment report with:\n"
        f"  - Executive summary\n"
        f"  - Attack surface map\n"
        f"  - Discovered services and technologies\n"
        f"  - Vulnerabilities found (rated by severity)\n"
        f"  - Remediation recommendations\n"
    )

    result = await Swarm(
        goal,
        config=swarm_config,
        mode=SwarmMode.AUTO,
        tool_registry=tool_registry,
        verbose=True,
    )

    # ── 5. Print results ─────────────────────────────────────────

    print(f"\n{'='*60}")
    print(f"  ASSESSMENT COMPLETE")
    print(f"  Agents: {len(result.successful)} succeeded, {len(result.failed)} failed")
    print(f"  Duration: {result.duration:.1f}s")
    print(f"  Mission: {mission.mission_id}")
    print(f"{'='*60}\n")

    # Print mission stats
    if mission.findings:
        print(f"Arsenal Findings: {len(mission.findings)}")
        for sev, count in mission.stats.items():
            if count > 0:
                print(f"  {sev.upper()}: {count}")
        print()

    if mission.evidence:
        print(f"Evidence Records: {len(mission.evidence)}")
        print()

    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(main())
