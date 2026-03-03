"""
Arsenal data models — defense-grade offensive security primitives.

Every object here is designed for forensic chain-of-custody:
  - Immutable evidence records with SHA-256 hashes
  - Scope enforcement at the type level
  - CVSS 3.1 scoring for findings
  - Full audit trail with timestamps
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv4Network, IPv6Address, IPv6Network
from pathlib import Path
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────


class Severity(str, Enum):
    """CVSS 3.1 qualitative severity ratings."""
    CRITICAL = "critical"   # 9.0 - 10.0
    HIGH = "high"           # 7.0 - 8.9
    MEDIUM = "medium"       # 4.0 - 6.9
    LOW = "low"             # 0.1 - 3.9
    INFO = "info"           # 0.0


class FindingCategory(str, Enum):
    """Finding categories aligned with OWASP/MITRE."""
    INJECTION = "injection"
    BROKEN_AUTH = "broken_authentication"
    SENSITIVE_DATA = "sensitive_data_exposure"
    XXE = "xml_external_entity"
    BROKEN_ACCESS = "broken_access_control"
    MISCONFIG = "security_misconfiguration"
    XSS = "cross_site_scripting"
    DESERIALIZATION = "insecure_deserialization"
    VULN_COMPONENTS = "vulnerable_components"
    LOGGING = "insufficient_logging"
    SSRF = "server_side_request_forgery"
    CRYPTO = "cryptographic_failure"
    NETWORK = "network_security"
    RCE = "remote_code_execution"
    PRIVESC = "privilege_escalation"
    INFO_DISCLOSURE = "information_disclosure"
    DOS = "denial_of_service"
    SUPPLY_CHAIN = "supply_chain"
    DEFAULT_CREDS = "default_credentials"
    OTHER = "other"


class MissionPhase(str, Enum):
    """Kill-chain aligned assessment phases."""
    SCOPE_VALIDATION = "scope_validation"
    RECONNAISSANCE = "reconnaissance"
    ENUMERATION = "enumeration"
    VULNERABILITY_ANALYSIS = "vulnerability_analysis"
    EXPLOITATION = "exploitation"
    POST_EXPLOITATION = "post_exploitation"
    REPORTING = "reporting"


class MissionStatus(str, Enum):
    PENDING = "pending"
    SCOPE_APPROVED = "scope_approved"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class EvidenceType(str, Enum):
    TOOL_OUTPUT = "tool_output"
    SCREENSHOT = "screenshot"
    PACKET_CAPTURE = "packet_capture"
    LOG = "log"
    FILE = "file"
    LLM_ANALYSIS = "llm_analysis"
    MANUAL_NOTE = "manual_note"


class ToolCategory(str, Enum):
    RECON = "reconnaissance"
    SCAN = "scanning"
    ENUM = "enumeration"
    VULN = "vulnerability"
    EXPLOIT = "exploitation"
    POST = "post_exploitation"
    UTIL = "utility"


# ── Scope ─────────────────────────────────────────────────────────


class ScopeAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass
class ScopeRule:
    """A single scope rule — allow or deny a target pattern."""
    action: ScopeAction
    target_type: str     # "ip", "cidr", "domain", "port", "url"
    value: str           # the pattern (e.g. "10.0.0.0/24", "*.example.com")
    note: str = ""       # justification

    def matches(self, target: str, target_type: str) -> bool:
        """Check if this rule matches a given target."""
        if self.target_type != target_type:
            return False

        if self.target_type == "cidr":
            try:
                net = IPv4Network(self.value, strict=False)
                addr = IPv4Address(target)
                return addr in net
            except (ValueError, TypeError):
                try:
                    net6 = IPv6Network(self.value, strict=False)
                    addr6 = IPv6Address(target)
                    return addr6 in net6
                except (ValueError, TypeError):
                    return False

        if self.target_type == "domain":
            if self.value.startswith("*."):
                suffix = self.value[1:]  # ".example.com"
                return target == self.value[2:] or target.endswith(suffix)
            return target == self.value

        if self.target_type == "port":
            if "-" in self.value:
                lo, hi = self.value.split("-", 1)
                return int(lo) <= int(target) <= int(hi)
            return str(target) == self.value

        return target == self.value


@dataclass
class Scope:
    """
    Authorization scope — cryptographically signed boundary.

    Rules are evaluated in order; first match wins.
    Default deny if no rule matches (safe by default).
    """
    rules: list[ScopeRule] = field(default_factory=list)
    authorized_by: str = ""         # who signed off
    authorization_ref: str = ""     # reference document / ticket
    valid_from: float = 0.0
    valid_until: float = 0.0
    signature_hash: str = ""        # SHA-256 of the serialized scope

    def is_valid(self) -> bool:
        """Check if scope is currently valid (time-bound)."""
        now = time.time()
        if self.valid_from and now < self.valid_from:
            return False
        if self.valid_until and now > self.valid_until:
            return False
        return True

    def check(self, target: str, target_type: str) -> bool:
        """
        Check if a target is in scope.
        Returns True if allowed, False if denied or no match.
        """
        if not self.is_valid():
            return False
        for rule in self.rules:
            if rule.matches(target, target_type):
                return rule.action == ScopeAction.ALLOW
        return False  # default deny

    def sign(self) -> str:
        """Compute and store a SHA-256 hash of the scope rules."""
        blob = json.dumps(
            [{"action": r.action.value, "type": r.target_type, "value": r.value}
             for r in self.rules],
            sort_keys=True,
        ).encode()
        self.signature_hash = hashlib.sha256(blob).hexdigest()
        return self.signature_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [
                {"action": r.action.value, "type": r.target_type,
                 "value": r.value, "note": r.note}
                for r in self.rules
            ],
            "authorized_by": self.authorized_by,
            "authorization_ref": self.authorization_ref,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "signature_hash": self.signature_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scope:
        rules = [
            ScopeRule(
                action=ScopeAction(r["action"]),
                target_type=r["type"],
                value=r["value"],
                note=r.get("note", ""),
            )
            for r in data.get("rules", [])
        ]
        return cls(
            rules=rules,
            authorized_by=data.get("authorized_by", ""),
            authorization_ref=data.get("authorization_ref", ""),
            valid_from=data.get("valid_from", 0.0),
            valid_until=data.get("valid_until", 0.0),
            signature_hash=data.get("signature_hash", ""),
        )


# ── Target ────────────────────────────────────────────────────────


@dataclass
class Target:
    """A host, network, or service under assessment."""
    host: str                      # IP or hostname
    ports: list[int] = field(default_factory=list)
    services: dict[int, str] = field(default_factory=dict)  # port → service banner
    os_guess: str = ""
    hostnames: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    first_seen: float = field(default_factory=time.time)

    @property
    def label(self) -> str:
        if self.hostnames:
            return f"{self.hostnames[0]} ({self.host})"
        return self.host


# ── Evidence ──────────────────────────────────────────────────────


@dataclass
class Evidence:
    """
    A single piece of forensic evidence.

    Every evidence record is hashed for chain-of-custody integrity.
    """
    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    evidence_type: EvidenceType = EvidenceType.TOOL_OUTPUT
    tool_name: str = ""
    command: str = ""              # exact command executed
    raw_output: str = ""
    parsed_data: dict[str, Any] = field(default_factory=dict)
    target: str = ""               # what this was run against
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0          # execution time
    sha256: str = ""               # hash of raw_output for integrity
    file_path: str = ""            # if evidence is a file

    def compute_hash(self) -> str:
        self.sha256 = hashlib.sha256(self.raw_output.encode()).hexdigest()
        return self.sha256

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "type": self.evidence_type.value,
            "tool": self.tool_name,
            "command": self.command,
            "target": self.target,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "sha256": self.sha256,
            "raw_output_length": len(self.raw_output),
            "parsed_data": self.parsed_data,
            "file_path": self.file_path,
        }


# ── Tool Result ───────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Result from executing a security tool."""
    tool_name: str
    command: str
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    success: bool = True
    parsed: dict[str, Any] = field(default_factory=dict)
    evidence: Evidence | None = None
    error: str = ""

    @property
    def output(self) -> str:
        return self.stdout or self.stderr


# ── Finding ───────────────────────────────────────────────────────


@dataclass
class Finding:
    """
    A security finding — the core output unit.

    Includes CVSS scoring, evidence chain, and remediation guidance.
    """
    finding_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = ""
    description: str = ""
    severity: Severity = Severity.INFO
    category: FindingCategory = FindingCategory.OTHER
    cvss_score: float = 0.0
    cvss_vector: str = ""
    target: str = ""
    port: int = 0
    service: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    cve_ids: list[str] = field(default_factory=list)
    cwe_ids: list[str] = field(default_factory=list)
    mitre_attack: list[str] = field(default_factory=list)  # T-codes
    remediation: str = ""
    references: list[str] = field(default_factory=list)
    false_positive: bool = False
    verified: bool = False
    phase: MissionPhase = MissionPhase.VULNERABILITY_ANALYSIS
    raw_tool_output: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "target": self.target,
            "port": self.port,
            "service": self.service,
            "evidence_ids": self.evidence_ids,
            "cve_ids": self.cve_ids,
            "cwe_ids": self.cwe_ids,
            "mitre_attack": self.mitre_attack,
            "remediation": self.remediation,
            "references": self.references,
            "false_positive": self.false_positive,
            "verified": self.verified,
            "phase": self.phase.value,
            "timestamp": self.timestamp,
        }


# ── Mission ───────────────────────────────────────────────────────


@dataclass
class Mission:
    """
    A complete assessment mission — the top-level container.

    Contains scope, targets, findings, evidence, and full audit trail.
    """
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str = ""
    description: str = ""
    scope: Scope = field(default_factory=Scope)
    status: MissionStatus = MissionStatus.PENDING
    phase: MissionPhase = MissionPhase.SCOPE_VALIDATION
    targets: list[Target] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0
    operator: str = ""     # human operator name
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if self.started_at and self.finished_at:
            return self.finished_at - self.started_at
        if self.started_at:
            return time.time() - self.started_at
        return 0.0

    def log(self, action: str, detail: str = "", phase: str = ""):
        """Append to the immutable audit log."""
        self.audit_log.append({
            "timestamp": time.time(),
            "action": action,
            "detail": detail,
            "phase": phase or self.phase.value,
            "status": self.status.value,
        })

    def add_evidence(self, evidence: Evidence) -> str:
        """Add evidence and compute its integrity hash."""
        evidence.compute_hash()
        self.evidence.append(evidence)
        self.log("evidence_added", f"{evidence.tool_name}: {evidence.evidence_id}")
        return evidence.evidence_id

    def add_finding(self, finding: Finding) -> str:
        """Add a security finding."""
        self.findings.append(finding)
        self.log("finding_added",
                 f"[{finding.severity.value.upper()}] {finding.title}")
        return finding.finding_id

    @property
    def stats(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            if not f.false_positive:
                counts[f.severity.value] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "phase": self.phase.value,
            "scope": self.scope.to_dict(),
            "targets": [
                {"host": t.host, "ports": t.ports, "services": t.services,
                 "os_guess": t.os_guess, "hostnames": t.hostnames}
                for t in self.targets
            ],
            "findings": [f.to_dict() for f in self.findings],
            "evidence": [e.to_dict() for e in self.evidence],
            "audit_log": self.audit_log,
            "stats": self.stats,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration": self.duration,
            "operator": self.operator,
        }
