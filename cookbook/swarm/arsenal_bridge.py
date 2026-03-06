"""
Arsenal bridge — adapts Arsenal's SecurityTools into swarm ToolDef entries.

This lets swarm agents invoke real security tools (nmap, nuclei, etc.)
through the scope-enforced, Docker-isolated Arsenal execution pipeline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from cookbook.swarm.tool_registry import ToolDef, ToolRegistry

log = logging.getLogger(__name__)

# Tool metadata for the swarm registry (name, description, parameters)
_ARSENAL_TOOLS: list[dict[str, Any]] = [
    {
        "name": "nmap_scan",
        "description": "Port scanning and service detection via nmap",
        "parameters": {
            "target": "IP, hostname, or CIDR to scan",
            "ports": "(optional) port range, e.g. '1-1000' or '22,80,443'",
            "scan_type": "(optional) default|quick|full|stealth|service|os|vuln",
        },
        "safe": True,
    },
    {
        "name": "nuclei_scan",
        "description": "Template-based vulnerability scanner",
        "parameters": {
            "target": "URL to scan (e.g. https://target.com)",
            "severity": "(optional) critical,high,medium,low,info",
            "tags": "(optional) template tags like 'cve', 'xss'",
        },
        "safe": True,
    },
    {
        "name": "subfinder_enum",
        "description": "Passive subdomain enumeration",
        "parameters": {"domain": "Domain to enumerate subdomains for"},
        "safe": True,
    },
    {
        "name": "httpx_probe",
        "description": "HTTP probing with technology detection, status codes, CDN detection",
        "parameters": {"targets": "List of URLs/hosts to probe"},
        "safe": True,
    },
    {
        "name": "dns_lookup",
        "description": "DNS record lookup via dig",
        "parameters": {
            "domain": "Domain to query",
            "record_type": "(optional) A, AAAA, MX, TXT, NS, CNAME, ANY (default: ANY)",
        },
        "safe": True,
    },
    {
        "name": "whois_lookup",
        "description": "WHOIS registration info for domain or IP",
        "parameters": {"target": "Domain or IP to look up"},
        "safe": True,
    },
    {
        "name": "nikto_scan",
        "description": "Web server vulnerability scanner (CGI, misconfigs)",
        "parameters": {
            "target": "Hostname or IP",
            "port": "(optional) port number, default 80",
            "ssl": "(optional) true/false for HTTPS",
        },
        "safe": True,
    },
    {
        "name": "whatweb_scan",
        "description": "Web technology fingerprinting",
        "parameters": {"target": "URL to fingerprint"},
        "safe": True,
    },
    {
        "name": "ffuf_fuzz",
        "description": "Directory/file fuzzing via ffuf",
        "parameters": {
            "url": "URL with FUZZ keyword (e.g. https://target.com/FUZZ)",
            "wordlist": "(optional) wordlist path",
        },
        "safe": False,
    },
    {
        "name": "testssl_scan",
        "description": "TLS/SSL configuration analysis",
        "parameters": {"target": "host:port to test"},
        "safe": True,
    },
    {
        "name": "curl_request",
        "description": "HTTP request for banner grabbing / API testing",
        "parameters": {
            "url": "URL to request",
            "method": "(optional) GET, POST, PUT, etc.",
            "headers": "(optional) dict of headers",
            "data": "(optional) request body",
        },
        "safe": True,
    },
]


def _format_tool_result(result: Any) -> str:
    """Convert an Arsenal ToolResult to a string for the LLM."""
    if not result.success:
        return f"[TOOL FAILED] {result.error or 'Unknown error'}\nCommand: {result.command}"

    parts = [f"[TOOL OK] Command: {result.command}"]
    if result.parsed:
        parts.append(f"Parsed output:\n{json.dumps(result.parsed, indent=2, default=str)[:4000]}")
    elif result.stdout:
        parts.append(f"Output:\n{result.stdout[:4000]}")
    if result.stderr:
        parts.append(f"Stderr:\n{result.stderr[:1000]}")
    return "\n".join(parts)


def register_arsenal_tools(
    registry: ToolRegistry,
    security_tools: Any,  # SecurityTools instance
) -> None:
    """Register all Arsenal security tools into a swarm ToolRegistry.

    Args:
        registry: The swarm ToolRegistry to populate.
        security_tools: An initialized SecurityTools instance from Arsenal.
    """
    # Map tool names to SecurityTools method names
    dispatch = {
        "nmap_scan": security_tools.nmap_scan,
        "nuclei_scan": security_tools.nuclei_scan,
        "subfinder_enum": security_tools.subfinder_enum,
        "httpx_probe": security_tools.httpx_probe,
        "dns_lookup": security_tools.dns_lookup,
        "whois_lookup": security_tools.whois_lookup,
        "nikto_scan": security_tools.nikto_scan,
        "whatweb_scan": security_tools.whatweb_scan,
        "ffuf_fuzz": security_tools.ffuf_fuzz,
        "testssl_scan": security_tools.testssl_scan,
        "curl_request": security_tools.curl_request,
    }

    for tool_meta in _ARSENAL_TOOLS:
        name = tool_meta["name"]
        method = dispatch.get(name)
        if method is None:
            continue

        async def _wrap(fn=method, **kwargs: Any) -> str:
            result = await fn(**kwargs)
            return _format_tool_result(result)

        registry.register(ToolDef(
            name=name,
            description=tool_meta["description"],
            parameters=tool_meta["parameters"],
            fn=_wrap,
            safe=tool_meta.get("safe", True),
        ))

    log.info(f"Registered {len(dispatch)} Arsenal tools into swarm registry")
