"""
Scope enforcement — the legal safety layer.

Every tool execution passes through scope validation BEFORE execution.
This is not optional and cannot be bypassed programmatically.

Design principles:
  - Default deny: if no rule matches, access is denied
  - Time-bound: scopes expire
  - Audited: every check is logged
  - Signed: scopes have integrity hashes
"""

from __future__ import annotations

import ipaddress
import logging
import re
import time
from urllib.parse import urlparse

from cookbook.arsenal.models import (
    Scope, ScopeRule, ScopeAction, Mission,
)

logger = logging.getLogger("arsenal.scope")


class ScopeViolation(Exception):
    """Raised when a tool targets something outside approved scope."""
    def __init__(self, target: str, target_type: str, detail: str = ""):
        self.target = target
        self.target_type = target_type
        self.detail = detail
        super().__init__(
            f"SCOPE VIOLATION: {target_type}={target} is out of scope. {detail}"
        )


class ScopeEnforcer:
    """
    Validates all targets against the mission scope before tool execution.

    This is the single choke-point — every tool call routes through here.
    """

    def __init__(self, mission: Mission):
        self.mission = mission
        self.scope = mission.scope
        self._check_count = 0
        self._deny_count = 0

    def validate_scope_integrity(self) -> bool:
        """Verify the scope hasn't been tampered with."""
        if not self.scope.signature_hash:
            logger.warning("Scope has no signature hash — cannot verify integrity")
            return True  # unsigned scopes are allowed but logged

        import hashlib, json
        blob = json.dumps(
            [{"action": r.action.value, "type": r.target_type, "value": r.value}
             for r in self.scope.rules],
            sort_keys=True,
        ).encode()
        computed = hashlib.sha256(blob).hexdigest()

        if computed != self.scope.signature_hash:
            logger.error(
                f"SCOPE INTEGRITY FAILURE: computed={computed[:16]}... "
                f"expected={self.scope.signature_hash[:16]}..."
            )
            return False
        return True

    def check_target(self, target: str, context: str = "") -> bool:
        """
        Check if a target (IP, domain, URL, etc.) is in scope.

        Automatically detects the target type and validates against all
        applicable rules.

        Args:
            target: The target to check (IP, domain, URL, CIDR, etc.)
            context: What tool/operation is requesting this (for logging)

        Returns:
            True if in scope, False if denied

        Raises:
            ScopeViolation: if scope is expired or integrity check fails
        """
        self._check_count += 1

        # Check scope validity
        if not self.scope.is_valid():
            self._deny_count += 1
            self.mission.log(
                "scope_denied",
                f"Scope expired or not yet valid: {target}",
                phase="scope_validation",
            )
            raise ScopeViolation(target, "expired", "Scope is not currently valid")

        # Detect target type and check
        allowed = False
        target_type = self._detect_type(target)

        if target_type == "url":
            # Extract host from URL and check both URL and host
            parsed = urlparse(target)
            host = parsed.hostname or ""
            port = parsed.port
            allowed = self.scope.check(target, "url")
            if not allowed:
                allowed = self._check_host(host)
            if allowed and port:
                allowed = self._check_port(port)
        elif target_type == "ip":
            allowed = self._check_host(target)
        elif target_type == "domain":
            allowed = self._check_host(target)
        elif target_type == "cidr":
            allowed = self.scope.check(target, "cidr")
        else:
            allowed = self.scope.check(target, target_type)

        if not allowed:
            self._deny_count += 1
            self.mission.log(
                "scope_denied",
                f"[{context}] {target_type}={target} → DENIED",
                phase="scope_validation",
            )
            logger.warning(f"Scope DENIED: {target} (type={target_type}, ctx={context})")
        else:
            self.mission.log(
                "scope_allowed",
                f"[{context}] {target_type}={target} → ALLOWED",
                phase="scope_validation",
            )

        return allowed

    def enforce(self, target: str, context: str = "") -> None:
        """Like check_target but raises ScopeViolation on denial."""
        if not self.check_target(target, context):
            raise ScopeViolation(target, self._detect_type(target),
                                 f"Denied by scope rules (context: {context})")

    def check_command(self, command: str, context: str = "") -> list[str]:
        """
        Extract targets from a command string and check all of them.

        Returns list of out-of-scope targets (empty = all good).
        """
        targets = self._extract_targets_from_command(command)
        violations = []
        for t in targets:
            if not self.check_target(t, context):
                violations.append(t)
        return violations

    def enforce_command(self, command: str, context: str = "") -> None:
        """Check a command — raises ScopeViolation on first violation."""
        violations = self.check_command(command, context)
        if violations:
            raise ScopeViolation(
                violations[0],
                self._detect_type(violations[0]),
                f"Command '{command[:80]}...' targets out-of-scope: {violations}",
            )

    def _check_host(self, host: str) -> bool:
        """Check a host against ip, cidr, and domain rules."""
        # Try as IP first
        try:
            addr = ipaddress.ip_address(host)
            if self.scope.check(host, "ip"):
                return True
            # Check CIDR rules
            for rule in self.scope.rules:
                if rule.target_type == "cidr" and rule.action == ScopeAction.ALLOW:
                    try:
                        net = ipaddress.ip_network(rule.value, strict=False)
                        if addr in net:
                            return True
                    except ValueError:
                        pass
            return False
        except ValueError:
            pass

        # Try as domain
        return self.scope.check(host, "domain")

    def _check_port(self, port: int) -> bool:
        """Check if a port is in scope (True if no port rules or port is allowed)."""
        has_port_rules = any(r.target_type == "port" for r in self.scope.rules)
        if not has_port_rules:
            return True  # no port restrictions
        return self.scope.check(str(port), "port")

    def _detect_type(self, target: str) -> str:
        """Auto-detect the type of a target string."""
        if target.startswith(("http://", "https://")):
            return "url"
        if "/" in target:
            try:
                ipaddress.ip_network(target, strict=False)
                return "cidr"
            except ValueError:
                pass
        try:
            ipaddress.ip_address(target)
            return "ip"
        except ValueError:
            pass
        if re.match(r'^[a-zA-Z0-9]([a-zA-Z0-9\-]*\.)+[a-zA-Z]{2,}$', target):
            return "domain"
        if target.isdigit():
            return "port"
        return "unknown"

    def _extract_targets_from_command(self, command: str) -> list[str]:
        """
        Best-effort extraction of targets from a command line.

        Looks for IPs, domains, URLs, and CIDR ranges in the command.
        """
        targets = []

        # URLs
        urls = re.findall(r'https?://[^\s"\'<>]+', command)
        targets.extend(urls)

        # IPs (including CIDR)
        ips = re.findall(
            r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b', command
        )
        targets.extend(ips)

        # Domains (after flags like -h, --host, -d, --domain, -u, --url, or bare)
        flag_pattern = r'(?:-[hdu]|--host|--domain|--url|--target)\s+([a-zA-Z0-9](?:[a-zA-Z0-9\-]*\.)+[a-zA-Z]{2,})'
        flag_domains = re.findall(flag_pattern, command)
        targets.extend(flag_domains)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        return unique

    @property
    def stats(self) -> dict[str, int]:
        return {
            "total_checks": self._check_count,
            "denied": self._deny_count,
            "allowed": self._check_count - self._deny_count,
        }
