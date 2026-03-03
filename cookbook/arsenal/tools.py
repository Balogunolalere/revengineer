"""
Security tool wrappers — structured interfaces for real offensive tools.

Each tool wrapper:
  1. Registers the tool with the executor
  2. Builds correct argument vectors
  3. Parses structured output (XML, JSON, or text)
  4. Returns typed results

Supported tools:
  - nmap           (port scanning, service/OS detection)
  - nuclei         (vulnerability scanning)
  - subfinder      (subdomain enumeration)
  - httpx          (HTTP probing)
  - dig            (DNS enumeration)
  - whois          (domain/IP ownership)
  - nikto          (web server scanning)
  - whatweb        (web technology fingerprinting)
  - ffuf           (web fuzzing / directory brute-force)
  - testssl.sh     (TLS/SSL analysis)
  - curl           (HTTP requests / banner grabbing)
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.executor import ToolExecutor, ToolSpec
from cookbook.arsenal.models import (
    Finding, Severity, FindingCategory,
    Target, ToolCategory, ToolResult,
)

logger = logging.getLogger("arsenal.tools")


# ══════════════════════════════════════════════════════════════════
#  PARSERS — extract structured data from tool output
# ══════════════════════════════════════════════════════════════════


def parse_nmap_xml(output: str) -> dict[str, Any]:
    """Parse nmap XML output (-oX -)."""
    result: dict[str, Any] = {"hosts": [], "scan_info": {}}

    try:
        root = ET.fromstring(output)
    except ET.ParseError:
        # Fall back to grep-parsing text output
        return parse_nmap_text(output)

    # Scan metadata
    scan_info = root.find("scaninfo")
    if scan_info is not None:
        result["scan_info"] = scan_info.attrib

    for host_el in root.findall("host"):
        host_data: dict[str, Any] = {
            "status": "",
            "addresses": [],
            "hostnames": [],
            "ports": [],
            "os_matches": [],
        }

        # Status
        status = host_el.find("status")
        if status is not None:
            host_data["status"] = status.get("state", "")

        # Addresses
        for addr in host_el.findall("address"):
            host_data["addresses"].append({
                "addr": addr.get("addr", ""),
                "type": addr.get("addrtype", ""),
            })

        # Hostnames
        hostnames_el = host_el.find("hostnames")
        if hostnames_el is not None:
            for hn in hostnames_el.findall("hostname"):
                host_data["hostnames"].append(hn.get("name", ""))

        # Ports
        ports_el = host_el.find("ports")
        if ports_el is not None:
            for port_el in ports_el.findall("port"):
                port_data: dict[str, Any] = {
                    "port": int(port_el.get("portid", 0)),
                    "protocol": port_el.get("protocol", "tcp"),
                    "state": "",
                    "service": "",
                    "version": "",
                    "product": "",
                    "extra": "",
                }
                state = port_el.find("state")
                if state is not None:
                    port_data["state"] = state.get("state", "")
                svc = port_el.find("service")
                if svc is not None:
                    port_data["service"] = svc.get("name", "")
                    port_data["product"] = svc.get("product", "")
                    port_data["version"] = svc.get("version", "")
                    port_data["extra"] = svc.get("extrainfo", "")
                host_data["ports"].append(port_data)

        # OS detection
        os_el = host_el.find("os")
        if os_el is not None:
            for match in os_el.findall("osmatch"):
                host_data["os_matches"].append({
                    "name": match.get("name", ""),
                    "accuracy": match.get("accuracy", ""),
                })

        result["hosts"].append(host_data)

    return result


def parse_nmap_text(output: str) -> dict[str, Any]:
    """Fallback parser for nmap text output."""
    result: dict[str, Any] = {"hosts": [], "raw": output}
    current_host: dict[str, Any] | None = None

    for line in output.splitlines():
        line = line.strip()

        # New host
        m = re.match(r'Nmap scan report for (.+)', line)
        if m:
            if current_host:
                result["hosts"].append(current_host)
            target = m.group(1)
            ip_match = re.search(r'\((\d+\.\d+\.\d+\.\d+)\)', target)
            current_host = {
                "addresses": [{"addr": ip_match.group(1) if ip_match else target, "type": "ipv4"}],
                "hostnames": [target.split(" (")[0]] if "(" in target else [],
                "ports": [],
                "status": "up",
            }

        # Ports
        m = re.match(r'(\d+)/(tcp|udp)\s+(\S+)\s+(.*)', line)
        if m and current_host:
            current_host["ports"].append({
                "port": int(m.group(1)),
                "protocol": m.group(2),
                "state": m.group(3),
                "service": m.group(4).strip(),
            })

    if current_host:
        result["hosts"].append(current_host)

    return result


def parse_nuclei_jsonl(output: str) -> dict[str, Any]:
    """Parse nuclei JSONL output (-jsonl)."""
    findings: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            findings.append({
                "template_id": obj.get("template-id", obj.get("templateID", "")),
                "name": obj.get("info", {}).get("name", obj.get("name", "")),
                "severity": obj.get("info", {}).get("severity", obj.get("severity", "info")),
                "type": obj.get("type", ""),
                "host": obj.get("host", obj.get("matched-at", "")),
                "matched_at": obj.get("matched-at", ""),
                "extracted_results": obj.get("extracted-results", []),
                "curl_command": obj.get("curl-command", ""),
                "description": obj.get("info", {}).get("description", ""),
                "reference": obj.get("info", {}).get("reference", []),
                "tags": obj.get("info", {}).get("tags", []),
                "matcher_name": obj.get("matcher-name", ""),
            })
        except json.JSONDecodeError:
            continue

    return {
        "findings": findings,
        "total": len(findings),
        "by_severity": _count_by(findings, "severity"),
    }


def parse_subfinder_text(output: str) -> dict[str, Any]:
    """Parse subfinder output (one subdomain per line)."""
    subs = [line.strip() for line in output.strip().splitlines()
            if line.strip() and not line.startswith("[")]
    return {"subdomains": subs, "total": len(subs)}


def parse_httpx_jsonl(output: str) -> dict[str, Any]:
    """Parse httpx JSONL output (-json)."""
    results: list[dict[str, Any]] = []
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            results.append({
                "url": obj.get("url", ""),
                "status_code": obj.get("status_code", obj.get("status-code", 0)),
                "title": obj.get("title", ""),
                "tech": obj.get("tech", []),
                "content_length": obj.get("content_length", obj.get("content-length", 0)),
                "webserver": obj.get("webserver", ""),
                "content_type": obj.get("content_type", obj.get("content-type", "")),
                "host": obj.get("host", ""),
                "tls": obj.get("tls", {}),
                "cdn": obj.get("cdn", False),
                "method": obj.get("method", ""),
            })
        except json.JSONDecodeError:
            continue

    return {"results": results, "total": len(results)}


def parse_dig(output: str) -> dict[str, Any]:
    """Parse dig output."""
    records: list[dict[str, str]] = []
    section = ""
    for line in output.splitlines():
        line = line.strip()
        if line.startswith(";; ANSWER SECTION"):
            section = "answer"
        elif line.startswith(";; AUTHORITY SECTION"):
            section = "authority"
        elif line.startswith(";; ADDITIONAL SECTION"):
            section = "additional"
        elif line.startswith(";;") or line.startswith(";") or not line:
            continue
        elif section:
            parts = line.split()
            if len(parts) >= 5:
                records.append({
                    "name": parts[0],
                    "ttl": parts[1],
                    "class": parts[2],
                    "type": parts[3],
                    "value": " ".join(parts[4:]),
                    "section": section,
                })
    return {"records": records, "total": len(records)}


def parse_whois(output: str) -> dict[str, Any]:
    """Parse whois output into key-value pairs."""
    data: dict[str, str] = {}
    for line in output.splitlines():
        if ":" in line and not line.startswith("%") and not line.startswith("#"):
            key, _, val = line.partition(":")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip()
            if key and val:
                data[key] = val
    return {"whois": data}


def parse_nikto(output: str) -> dict[str, Any]:
    """Parse nikto text output."""
    findings: list[dict[str, str]] = []
    server_info: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("+ Server:"):
            server_info["server"] = line.split(":", 1)[1].strip()
        elif line.startswith("+ "):
            # Strip the leading "+ " and OSVDB references
            msg = line[2:]
            osvdb = ""
            m = re.match(r'(OSVDB-\d+):\s*(.*)', msg)
            if m:
                osvdb = m.group(1)
                msg = m.group(2)
            findings.append({"message": msg, "osvdb": osvdb})
    return {
        "server_info": server_info,
        "findings": findings,
        "total": len(findings),
    }


def parse_whatweb(output: str) -> dict[str, Any]:
    """Parse whatweb JSON output (--log-json=-)."""
    results: list[dict[str, Any]] = []
    try:
        data = json.loads(output)
        if isinstance(data, list):
            for entry in data:
                results.append({
                    "target": entry.get("target", ""),
                    "plugins": entry.get("plugins", {}),
                    "status": entry.get("http_status", ""),
                })
        return {"results": results, "total": len(results)}
    except json.JSONDecodeError:
        # Text output fallback
        return {"raw": output}


def parse_ffuf_json(output: str) -> dict[str, Any]:
    """Parse ffuf JSON output (-of json)."""
    try:
        data = json.loads(output)
        results = []
        for r in data.get("results", []):
            results.append({
                "url": r.get("url", ""),
                "status": r.get("status", 0),
                "length": r.get("length", 0),
                "words": r.get("words", 0),
                "lines": r.get("lines", 0),
                "input": r.get("input", {}),
                "redirectlocation": r.get("redirectlocation", ""),
            })
        return {
            "results": results,
            "total": len(results),
            "commandline": data.get("commandline", ""),
        }
    except json.JSONDecodeError:
        return {"raw": output}


def parse_testssl(output: str) -> dict[str, Any]:
    """Parse testssl.sh JSON output (--jsonfile -)."""
    try:
        data = json.loads(output)
        if isinstance(data, list):
            findings: list[dict[str, str]] = []
            for item in data:
                findings.append({
                    "id": item.get("id", ""),
                    "severity": item.get("severity", ""),
                    "finding": item.get("finding", ""),
                })
            return {"findings": findings, "total": len(findings)}
        return data
    except json.JSONDecodeError:
        return {"raw": output}


def _count_by(items: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        val = item.get(key, "unknown")
        counts[val] = counts.get(val, 0) + 1
    return counts


# ══════════════════════════════════════════════════════════════════
#  TOOL WRAPPERS — high-level interfaces to security tools
# ══════════════════════════════════════════════════════════════════


class SecurityTools:
    """
    High-level interface to all security tools.

    Provides structured methods for each tool, building correct
    arguments and returning parsed results.
    """

    def __init__(self, executor: ToolExecutor, config: ArsenalConfig):
        self.executor = executor
        self.config = config
        self._register_all()

    def _register_all(self) -> None:
        """Register all supported tools."""
        tools = [
            ToolSpec("nmap", ToolCategory.SCAN, config.nmap_path,
                     "Network mapper — port scanning and service detection",
                     safe=True, requires_root=True, parse_fn=parse_nmap_xml,
                     default_args=["-oX", "-"]),

            ToolSpec("nuclei", ToolCategory.VULN, config.nuclei_path,
                     "Fast vulnerability scanner with template-based detection",
                     safe=True, parse_fn=parse_nuclei_jsonl,
                     default_args=["-jsonl", "-silent"]),

            ToolSpec("subfinder", ToolCategory.RECON, config.subfinder_path,
                     "Passive subdomain enumeration",
                     safe=True, parse_fn=parse_subfinder_text,
                     default_args=["-silent"]),

            ToolSpec("httpx", ToolCategory.ENUM, config.httpx_path,
                     "HTTP probing and technology detection",
                     safe=True, parse_fn=parse_httpx_jsonl,
                     default_args=["-json", "-silent"]),

            ToolSpec("dig", ToolCategory.RECON, config.dig_path,
                     "DNS query tool",
                     safe=True, parse_fn=parse_dig),

            ToolSpec("whois", ToolCategory.RECON, config.whois_path,
                     "Domain/IP WHOIS lookups",
                     safe=True, parse_fn=parse_whois),

            ToolSpec("nikto", ToolCategory.VULN, config.nikto_path,
                     "Web server vulnerability scanner",
                     safe=True, parse_fn=parse_nikto),

            ToolSpec("whatweb", ToolCategory.ENUM, config.whatweb_path,
                     "Web technology fingerprinting",
                     safe=True, parse_fn=parse_whatweb,
                     default_args=["--log-json=-"]),

            ToolSpec("ffuf", ToolCategory.ENUM, config.ffuf_path,
                     "Web fuzzer — directory and parameter brute-forcing",
                     safe=True, parse_fn=parse_ffuf_json,
                     default_args=["-of", "json", "-o", "-"]),

            ToolSpec("testssl", ToolCategory.SCAN, config.testssl_path,
                     "TLS/SSL cipher and vulnerability analysis",
                     safe=True, parse_fn=parse_testssl,
                     default_args=["--jsonfile", "-"]),

            ToolSpec("curl", ToolCategory.UTIL, config.curl_path,
                     "HTTP client for banner grabbing and API testing",
                     safe=True,
                     default_args=["-s", "-S", "-L", "-m", "30"]),
        ]

        for spec in tools:
            spec.binary = getattr(self.config, f"{spec.name}_path", spec.binary)
            self.executor.register_tool(spec)

    # ── Nmap ──────────────────────────────────────────────────────

    async def nmap_scan(
        self,
        target: str,
        ports: str = "",
        scan_type: str = "default",
        scripts: str = "",
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """
        Run an nmap scan.

        scan_type: "default", "quick", "full", "stealth", "service", "os", "vuln"
        """
        args: list[str] = []

        if scan_type == "quick":
            args.extend(["-T4", "-F"])
        elif scan_type == "full":
            args.extend(["-sS", "-sV", "-sC", "-O", "-p-", "-T4"])
        elif scan_type == "stealth":
            args.extend(["-sS", "-T2", "--max-rate", str(self.config.scan_rate_limit)])
        elif scan_type == "service":
            args.extend(["-sV", "-sC"])
        elif scan_type == "os":
            args.extend(["-O", "--osscan-guess"])
        elif scan_type == "vuln":
            args.extend(["--script", "vuln", "-sV"])
        else:
            args.extend(["-sV", "-T4"])

        if ports:
            args.extend(["-p", ports])

        if scripts:
            args.extend(["--script", scripts])

        if self.config.scan_rate_limit:
            args.extend(["--max-rate", str(self.config.scan_rate_limit)])

        if extra_args:
            args.extend(extra_args)

        args.append(target)

        return await self.executor.execute("nmap", args, target=target)

    # ── Nuclei ────────────────────────────────────────────────────

    async def nuclei_scan(
        self,
        target: str,
        templates: str = "",
        severity: str = "",
        tags: str = "",
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """Run a nuclei vulnerability scan."""
        args = ["-u", target]

        if templates:
            args.extend(["-t", templates])
        if severity:
            args.extend(["-severity", severity])
        if tags:
            args.extend(["-tags", tags])
        if self.config.scan_rate_limit:
            args.extend(["-rate-limit", str(self.config.scan_rate_limit)])

        if extra_args:
            args.extend(extra_args)

        return await self.executor.execute("nuclei", args, target=target)

    # ── Subfinder ─────────────────────────────────────────────────

    async def subfinder_enum(
        self,
        domain: str,
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """Enumerate subdomains passively."""
        args = ["-d", domain]
        if extra_args:
            args.extend(extra_args)
        return await self.executor.execute("subfinder", args, target=domain)

    # ── HTTPX ─────────────────────────────────────────────────────

    async def httpx_probe(
        self,
        targets: list[str],
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """Probe HTTP services from a list of targets."""
        # httpx reads from stdin, we pass via -u for single or -l for lists
        if len(targets) == 1:
            args = ["-u", targets[0]]
        else:
            args = ["-u", ",".join(targets)]

        args.extend(["-title", "-tech-detect", "-status-code", "-cdn"])

        if extra_args:
            args.extend(extra_args)

        return await self.executor.execute(
            "httpx", args, target=targets[0] if targets else "",
        )

    # ── DNS ───────────────────────────────────────────────────────

    async def dns_lookup(
        self,
        domain: str,
        record_type: str = "ANY",
        nameserver: str = "",
    ) -> ToolResult:
        """Perform DNS lookups."""
        args = [domain, record_type]
        if nameserver:
            args.append(f"@{nameserver}")
        args.append("+noall")
        args.append("+answer")
        return await self.executor.execute("dig", args, target=domain)

    async def dns_zone_transfer(self, domain: str, ns: str = "") -> ToolResult:
        """Attempt DNS zone transfer (AXFR)."""
        args = ["axfr", domain]
        if ns:
            args.append(f"@{ns}")
        return await self.executor.execute("dig", args, target=domain)

    # ── WHOIS ─────────────────────────────────────────────────────

    async def whois_lookup(self, target: str) -> ToolResult:
        """WHOIS lookup for domain or IP."""
        return await self.executor.execute("whois", [target], target=target)

    # ── Nikto ─────────────────────────────────────────────────────

    async def nikto_scan(
        self,
        target: str,
        port: int = 80,
        ssl: bool = False,
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """Run nikto web vulnerability scanner."""
        args = ["-h", target, "-p", str(port)]
        if ssl:
            args.append("-ssl")
        if extra_args:
            args.extend(extra_args)
        return await self.executor.execute("nikto", args, target=target)

    # ── WhatWeb ───────────────────────────────────────────────────

    async def whatweb_scan(self, target: str) -> ToolResult:
        """Fingerprint web technologies."""
        return await self.executor.execute("whatweb", [target], target=target)

    # ── FFUF ──────────────────────────────────────────────────────

    async def ffuf_fuzz(
        self,
        url: str,
        wordlist: str = "/usr/share/wordlists/dirb/common.txt",
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """Directory/parameter fuzzing."""
        args = ["-u", url, "-w", wordlist]
        if self.config.scan_rate_limit:
            args.extend(["-rate", str(self.config.scan_rate_limit)])
        if extra_args:
            args.extend(extra_args)
        return await self.executor.execute("ffuf", args, target=url)

    # ── TestSSL ───────────────────────────────────────────────────

    async def testssl_scan(self, target: str) -> ToolResult:
        """Analyze TLS/SSL configuration."""
        return await self.executor.execute("testssl", [target], target=target)

    # ── Curl ──────────────────────────────────────────────────────

    async def curl_request(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        data: str = "",
        extra_args: list[str] | None = None,
    ) -> ToolResult:
        """Make HTTP requests for banner grabbing / API testing."""
        args = ["-X", method, "-i"]  # -i includes headers in output
        if headers:
            for k, v in headers.items():
                args.extend(["-H", f"{k}: {v}"])
        if data:
            args.extend(["-d", data])
        if extra_args:
            args.extend(extra_args)
        args.append(url)
        return await self.executor.execute("curl", args, target=url)

    # ── Utility ───────────────────────────────────────────────────

    def available_tools(self) -> list[dict[str, str]]:
        """List all available tools with descriptions."""
        return [
            {
                "name": t.name,
                "category": t.category.value,
                "description": t.description,
                "safe": str(t.safe),
                "available": str(t.available),
            }
            for t in self.executor.list_tools()
        ]

    def tool_help(self, tool_name: str) -> str:
        """Get help text for a tool (for LLM context)."""
        help_map = {
            "nmap": (
                "nmap — Network exploration and security auditing.\n"
                "Methods: nmap_scan(target, ports='', scan_type='default|quick|full|stealth|service|os|vuln', scripts='')\n"
                "Examples:\n"
                "  nmap_scan('10.0.0.1', scan_type='quick')  # Fast top ports\n"
                "  nmap_scan('10.0.0.1', ports='1-1000', scan_type='service')  # Service detection\n"
                "  nmap_scan('10.0.0.1', scan_type='vuln')  # NSE vuln scripts\n"
            ),
            "nuclei": (
                "nuclei — Template-based vulnerability scanner.\n"
                "Methods: nuclei_scan(target, templates='', severity='', tags='')\n"
                "Examples:\n"
                "  nuclei_scan('https://target.com')  # All templates\n"
                "  nuclei_scan('https://target.com', severity='critical,high')  # Critical+High only\n"
                "  nuclei_scan('https://target.com', tags='cve')  # CVE templates\n"
            ),
            "subfinder": (
                "subfinder — Passive subdomain enumeration.\n"
                "Methods: subfinder_enum(domain)\n"
                "Example: subfinder_enum('example.com')\n"
            ),
            "httpx": (
                "httpx — HTTP probing with tech detection.\n"
                "Methods: httpx_probe(targets)\n"
                "Example: httpx_probe(['sub1.example.com', 'sub2.example.com'])\n"
            ),
            "dig": (
                "dig — DNS queries.\n"
                "Methods: dns_lookup(domain, record_type='ANY'), dns_zone_transfer(domain)\n"
                "Examples:\n"
                "  dns_lookup('example.com', 'MX')  # MX records\n"
                "  dns_zone_transfer('example.com', 'ns1.example.com')  # AXFR attempt\n"
            ),
            "whois": (
                "whois — Domain/IP registration info.\n"
                "Methods: whois_lookup(target)\n"
            ),
            "nikto": (
                "nikto — Web server scanner (CGI, misconfigs, etc.).\n"
                "Methods: nikto_scan(target, port=80, ssl=False)\n"
            ),
            "whatweb": (
                "whatweb — Web technology fingerprinting.\n"
                "Methods: whatweb_scan(target)\n"
            ),
            "ffuf": (
                "ffuf — Web fuzzer for directory/file discovery.\n"
                "Methods: ffuf_fuzz(url, wordlist='/usr/share/wordlists/dirb/common.txt')\n"
                "Example: ffuf_fuzz('https://target.com/FUZZ')\n"
            ),
            "testssl": (
                "testssl.sh — TLS/SSL configuration analysis.\n"
                "Methods: testssl_scan(target)\n"
            ),
            "curl": (
                "curl — HTTP requests and banner grabbing.\n"
                "Methods: curl_request(url, method='GET', headers={}, data='')\n"
            ),
        }
        return help_map.get(tool_name, f"No help available for '{tool_name}'")


# ══════════════════════════════════════════════════════════════════
#  FINDING GENERATORS — convert tool output into Finding objects
# ══════════════════════════════════════════════════════════════════


def findings_from_nuclei(result: ToolResult, target: str = "") -> list[Finding]:
    """Convert nuclei results into Finding objects."""
    findings = []
    severity_map = {
        "critical": Severity.CRITICAL,
        "high": Severity.HIGH,
        "medium": Severity.MEDIUM,
        "low": Severity.LOW,
        "info": Severity.INFO,
    }

    for item in result.parsed.get("findings", []):
        sev = severity_map.get(item.get("severity", "info"), Severity.INFO)
        finding = Finding(
            title=item.get("name", item.get("template_id", "Unknown")),
            description=item.get("description", ""),
            severity=sev,
            target=item.get("host", target),
            category=FindingCategory.OTHER,
            evidence_ids=[result.evidence.evidence_id] if result.evidence else [],
            references=item.get("reference", []),
            phase=MissionPhase.VULNERABILITY_ANALYSIS,
        )
        findings.append(finding)

    return findings


def findings_from_nmap(result: ToolResult, target: str = "") -> list[Finding]:
    """Generate findings from nmap results (open ports, weak services, etc.)."""
    findings = []

    for host in result.parsed.get("hosts", []):
        host_addr = host.get("addresses", [{}])[0].get("addr", target)
        for port_info in host.get("ports", []):
            if port_info.get("state") != "open":
                continue

            port = port_info.get("port", 0)
            service = port_info.get("service", "")
            product = port_info.get("product", "")
            version = port_info.get("version", "")

            # Flag interesting findings
            severity = Severity.INFO
            category = FindingCategory.NETWORK
            title = f"Open port {port}/{port_info.get('protocol', 'tcp')}: {service}"

            # Risky services
            risky = {
                "telnet": (Severity.HIGH, "Unencrypted remote access"),
                "ftp": (Severity.MEDIUM, "FTP — often unencrypted"),
                "microsoft-ds": (Severity.MEDIUM, "SMB/CIFS exposed"),
                "ms-sql-s": (Severity.HIGH, "MSSQL exposed to network"),
                "mysql": (Severity.MEDIUM, "MySQL exposed to network"),
                "redis": (Severity.HIGH, "Redis often has no auth"),
                "mongodb": (Severity.HIGH, "MongoDB often has no auth"),
                "vnc": (Severity.HIGH, "VNC remote access exposed"),
            }

            if service in risky:
                severity, note = risky[service]
                title = f"{note} — {host_addr}:{port}"
                category = FindingCategory.MISCONFIG

            findings.append(Finding(
                title=title,
                description=f"Service: {product} {version}".strip(),
                severity=severity,
                category=category,
                target=host_addr,
                port=port,
                service=f"{product} {version}".strip() or service,
                evidence_ids=[result.evidence.evidence_id] if result.evidence else [],
                phase=MissionPhase.ENUMERATION,
            ))

    return findings


# Import MissionPhase here to avoid circular imports in findings generators
from cookbook.arsenal.models import MissionPhase
