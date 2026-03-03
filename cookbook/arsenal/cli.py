"""
Arsenal CLI — command-line interface for security assessments.

Usage:
    # Quick scan with auto-scope from target
    uv run python -m cookbook.arsenal scan example.com

    # Full assessment with scope file
    uv run python -m cookbook.arsenal assess --scope scope.json --target 10.0.0.0/24

    # Generate Docker attack lab
    uv run python -m cookbook.arsenal lab --build

    # Dry run (plan but don't execute)
    uv run python -m cookbook.arsenal assess --target example.com --dry-run

    # Recon only
    uv run python -m cookbook.arsenal recon example.com

    # Export scope template
    uv run python -m cookbook.arsenal scope --export scope.json --target example.com
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.engine import MissionEngine
from cookbook.arsenal.lab import AttackLab
from cookbook.arsenal.models import MissionPhase, Scope, ScopeRule, ScopeAction


# ANSI colors
class C:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


BANNER = f"""{C.RED}{C.BOLD}
    ╔═══════════════════════════════════════════╗
    ║   █████╗ ██████╗ ███████╗███████╗███╗   ██╗ ║
    ║  ██╔══██╗██╔══██╗██╔════╝██╔════╝████╗  ██║ ║
    ║  ███████║██████╔╝███████╗█████╗  ██╔██╗ ██║ ║
    ║  ██╔══██║██╔══██╗╚════██║██╔══╝  ██║╚██╗██║ ║
    ║  ██║  ██║██║  ██║███████║███████╗██║ ╚████║  ║
    ║  ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝║
    ║     Autonomous Offensive Security Platform    ║
    ╚═══════════════════════════════════════════╝
{C.RESET}"""


PHASE_ICONS = {
    MissionPhase.SCOPE_VALIDATION: f"{C.BLUE}🔒 SCOPE VALIDATION{C.RESET}",
    MissionPhase.RECONNAISSANCE: f"{C.CYAN}🔍 RECONNAISSANCE{C.RESET}",
    MissionPhase.ENUMERATION: f"{C.YELLOW}📡 ENUMERATION{C.RESET}",
    MissionPhase.VULNERABILITY_ANALYSIS: f"{C.RED}💉 VULNERABILITY ANALYSIS{C.RESET}",
    MissionPhase.REPORTING: f"{C.GREEN}📄 REPORTING{C.RESET}",
}


def on_phase_start(phase: MissionPhase):
    icon = PHASE_ICONS.get(phase, phase.value)
    print(f"\n{'─'*60}")
    print(f"  {icon}")
    print(f"{'─'*60}")


def on_phase_end(phase: MissionPhase):
    print(f"  {C.DIM}Phase complete{C.RESET}")


def on_progress(phase: MissionPhase, iteration: int, max_iter: int):
    print(f"  {C.DIM}[{phase.value}] Iteration {iteration}/{max_iter}{C.RESET}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arsenal",
        description="Arsenal — Autonomous Offensive Security Platform",
    )
    sub = parser.add_subparsers(dest="command", help="Command to run")

    # ── assess ──
    assess = sub.add_parser("assess", help="Run a full security assessment")
    assess.add_argument("--target", "-t", action="append", default=[],
                        help="Target host/IP/domain (repeatable)")
    assess.add_argument("--scope", "-s", help="Path to scope JSON file")
    assess.add_argument("--name", "-n", default="", help="Mission name")
    assess.add_argument("--operator", default="", help="Operator name")
    assess.add_argument("--dry-run", action="store_true", help="Plan but don't execute tools")
    assess.add_argument("--safe-mode", action="store_true", default=True,
                        help="Non-destructive tools only (default)")
    assess.add_argument("--no-docker", action="store_true", help="Run tools locally (no Docker)")
    assess.add_argument("--output", "-o", default=".", help="Output directory")
    assess.add_argument("--verbose", "-v", action="store_true")

    # ── scan ──
    scan = sub.add_parser("scan", help="Quick scan a target")
    scan.add_argument("target", help="Target to scan")
    scan.add_argument("--ports", "-p", default="", help="Port range (e.g. 1-1000)")
    scan.add_argument("--no-docker", action="store_true")
    scan.add_argument("--output", "-o", default=".")
    scan.add_argument("--verbose", "-v", action="store_true")

    # ── recon ──
    recon = sub.add_parser("recon", help="Reconnaissance only")
    recon.add_argument("target", help="Target domain")
    recon.add_argument("--no-docker", action="store_true")
    recon.add_argument("--output", "-o", default=".")
    recon.add_argument("--verbose", "-v", action="store_true")

    # ── lab ──
    lab = sub.add_parser("lab", help="Manage the Docker attack lab")
    lab.add_argument("--build", action="store_true", help="Build the Docker image")
    lab.add_argument("--check", action="store_true", help="Check tool availability")
    lab.add_argument("--compose", action="store_true", help="Generate docker-compose.yml")
    lab.add_argument("--teardown", action="store_true", help="Remove lab containers/network")

    # ── scope ──
    scope_cmd = sub.add_parser("scope", help="Manage scope")
    scope_cmd.add_argument("--export", help="Export scope template to file")
    scope_cmd.add_argument("--target", "-t", action="append", default=[],
                           help="Targets to include in scope template")
    scope_cmd.add_argument("--validate", help="Validate a scope file")

    return parser


async def cmd_assess(args: argparse.Namespace) -> None:
    """Run a full assessment."""
    config = ArsenalConfig(
        dry_run=args.dry_run,
        safe_mode=args.safe_mode,
        use_docker=not args.no_docker,
        output_dir=args.output,
        verbose=args.verbose,
    )

    engine = MissionEngine(
        config=config,
        mission_name=args.name or f"Assessment of {', '.join(args.target)}",
        operator=args.operator,
    )

    # Load scope
    if args.scope:
        engine.load_scope(args.scope)
    else:
        # Auto-generate scope from targets
        for t in args.target:
            engine.add_scope_rule("allow", "domain", f"*.{t}" if "." in t else t)
            engine.add_scope_rule("allow", "domain", t)
            engine.add_scope_rule("allow", "ip", t)

    # Add targets
    for t in args.target:
        engine.add_target(t)

    if not args.target:
        print(f"{C.RED}Error: No targets specified. Use --target/-t{C.RESET}")
        sys.exit(1)

    engine.set_callbacks(
        on_phase_start=on_phase_start,
        on_phase_end=on_phase_end,
        on_progress=on_progress,
    )

    # Setup Docker lab if needed
    if config.use_docker:
        lab = AttackLab(config)
        if not await lab.is_ready():
            print(f"{C.YELLOW}Building Docker attack lab (first run)...{C.RESET}")
            await lab.setup()

    # Execute
    mission = await engine.execute()

    # Summary
    print(f"\n{'='*60}")
    stats = mission.stats
    print(f"{C.BOLD}ASSESSMENT COMPLETE{C.RESET}")
    print(f"  Status: {mission.status.value}")
    print(f"  Duration: {mission.duration:.1f}s")
    print(f"  Targets: {len(mission.targets)}")
    print(f"  Findings: {sum(stats.values())}")
    print(f"    {C.RED}Critical: {stats.get('critical', 0)}{C.RESET}")
    print(f"    {C.RED}High:     {stats.get('high', 0)}{C.RESET}")
    print(f"    {C.YELLOW}Medium:   {stats.get('medium', 0)}{C.RESET}")
    print(f"    Low:      {stats.get('low', 0)}")
    print(f"    Info:     {stats.get('info', 0)}")
    print(f"  Evidence: {len(mission.evidence)} items")
    print(f"{'='*60}")


async def cmd_scan(args: argparse.Namespace) -> None:
    """Quick scan a target."""
    config = ArsenalConfig(
        use_docker=not args.no_docker,
        output_dir=args.output,
        verbose=args.verbose,
    )

    engine = MissionEngine(
        config=config,
        mission_name=f"Quick scan: {args.target}",
    )
    engine.add_scope_rule("allow", "domain", f"*.{args.target}")
    engine.add_scope_rule("allow", "domain", args.target)
    engine.add_scope_rule("allow", "ip", args.target)
    engine.add_target(args.target)

    engine.set_callbacks(
        on_phase_start=on_phase_start,
        on_phase_end=on_phase_end,
        on_progress=on_progress,
    )

    await engine.execute(phases=[
        MissionPhase.SCOPE_VALIDATION,
        MissionPhase.ENUMERATION,
        MissionPhase.REPORTING,
    ])


async def cmd_recon(args: argparse.Namespace) -> None:
    """Recon only."""
    config = ArsenalConfig(
        use_docker=not args.no_docker,
        output_dir=args.output,
        verbose=args.verbose,
    )

    engine = MissionEngine(
        config=config,
        mission_name=f"Recon: {args.target}",
    )
    engine.add_scope_rule("allow", "domain", f"*.{args.target}")
    engine.add_scope_rule("allow", "domain", args.target)
    engine.add_target(args.target)

    engine.set_callbacks(
        on_phase_start=on_phase_start,
        on_phase_end=on_phase_end,
        on_progress=on_progress,
    )

    await engine.execute(phases=[
        MissionPhase.SCOPE_VALIDATION,
        MissionPhase.RECONNAISSANCE,
        MissionPhase.REPORTING,
    ])


async def cmd_lab(args: argparse.Namespace) -> None:
    """Manage Docker attack lab."""
    config = ArsenalConfig()
    lab = AttackLab(config)

    if args.build:
        print(f"{C.CYAN}Building Docker attack lab...{C.RESET}")
        await lab.setup()
        print(f"{C.GREEN}Lab built successfully!{C.RESET}")

        print(f"\n{C.BOLD}Checking tools:{C.RESET}")
        tools = await lab.check_tools()
        for tool, available in sorted(tools.items()):
            icon = f"{C.GREEN}✓{C.RESET}" if available else f"{C.RED}✗{C.RESET}"
            print(f"  {icon} {tool}")

    elif args.check:
        if not await lab.is_ready():
            print(f"{C.RED}Lab not built. Run: arsenal lab --build{C.RESET}")
            return
        tools = await lab.check_tools()
        for tool, available in sorted(tools.items()):
            icon = f"{C.GREEN}✓{C.RESET}" if available else f"{C.RED}✗{C.RESET}"
            print(f"  {icon} {tool}")

    elif args.compose:
        path = lab.write_compose()
        print(f"Compose file: {path}")

    elif args.teardown:
        await lab.teardown()
        print(f"{C.GREEN}Lab cleaned up.{C.RESET}")

    else:
        status = "READY" if await lab.is_ready() else "NOT BUILT"
        print(f"Lab status: {status}")
        print(f"  Image: {config.docker_image}")
        print(f"  Network: {config.docker_network}")
        print(f"  Memory: {config.docker_memory}")
        print(f"  CPUs: {config.docker_cpus}")


async def cmd_scope(args: argparse.Namespace) -> None:
    """Manage scope."""
    if args.export:
        scope = Scope(
            authorized_by="<YOUR NAME>",
            authorization_ref="<TICKET/CONTRACT REF>",
            valid_from=time.time(),
            valid_until=time.time() + 7 * 24 * 3600,  # 1 week
        )
        for t in args.target:
            scope.rules.append(ScopeRule(
                action=ScopeAction.ALLOW,
                target_type="domain",
                value=t,
                note="Primary target",
            ))
            scope.rules.append(ScopeRule(
                action=ScopeAction.ALLOW,
                target_type="domain",
                value=f"*.{t}",
                note="Subdomains",
            ))

        scope.sign()
        data = scope.to_dict()

        with open(args.export, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Scope template exported: {args.export}")
        print(f"Signature: {scope.signature_hash[:16]}...")
        print(f"\n{C.YELLOW}Edit the file to customize rules before use.{C.RESET}")

    elif args.validate:
        with open(args.validate) as f:
            data = json.load(f)
        scope = Scope.from_dict(data)
        print(f"Rules: {len(scope.rules)}")
        print(f"Authorized by: {scope.authorized_by}")
        print(f"Valid: {scope.is_valid()}")
        if scope.signature_hash:
            scope2 = Scope.from_dict(data)
            computed = scope2.sign()
            match = computed == scope.signature_hash
            icon = f"{C.GREEN}✓{C.RESET}" if match else f"{C.RED}✗ TAMPERED{C.RESET}"
            print(f"Integrity: {icon}")
        for r in scope.rules:
            print(f"  {r.action.value:5s} {r.target_type:8s} {r.value} {C.DIM}{r.note}{C.RESET}")


def main():
    print(BANNER)
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    cmd_map = {
        "assess": cmd_assess,
        "scan": cmd_scan,
        "recon": cmd_recon,
        "lab": cmd_lab,
        "scope": cmd_scope,
    }

    fn = cmd_map.get(args.command)
    if fn:
        asyncio.run(fn(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
