"""
Arsenal — Autonomous Offensive Security Platform

A defense-grade cybersecurity assessment framework that:
  - Executes real security tools (nmap, nuclei, subfinder, etc.)
  - Enforces legal scope with cryptographic authorization
  - Maintains forensic-grade evidence chains
  - Uses LLM agents to plan, execute, and adapt attack strategies
  - Runs in isolated Docker attack labs
  - Produces MoD/government-grade assessment reports

Usage:
    from cookbook.arsenal import Arsenal

    async with Arsenal(target="example.com", authorization="signed_auth.json") as ops:
        result = await ops.execute()
"""

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.models import (
    Target, Finding, Severity, FindingCategory,
    Mission, MissionPhase, MissionStatus,
    Evidence, EvidenceType,
    Scope, ScopeRule,
    ToolResult,
)
from cookbook.arsenal.engine import MissionEngine

__all__ = [
    "ArsenalConfig",
    "Target", "Finding", "Severity", "FindingCategory",
    "Mission", "MissionPhase", "MissionStatus",
    "Evidence", "EvidenceType",
    "Scope", "ScopeRule",
    "ToolResult",
    "MissionEngine",
]
