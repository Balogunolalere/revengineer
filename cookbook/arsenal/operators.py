"""
LLM Operator Agents — autonomous security operators powered by LLM reasoning.

Each operator specializes in a phase of the kill chain:
  - ReconOperator: OSINT, subdomain enum, DNS, WHOIS
  - ScanOperator: port scanning, service detection
  - VulnOperator: vulnerability scanning, analysis
  - ExploitOperator: exploitation planning (POC only in safe mode)
  - AnalysisOperator: synthesizes findings, correlates data

Operators decide:
  1. WHICH tools to run (and with what arguments)
  2. HOW to interpret results
  3. WHAT to do next based on findings
  4. WHEN to escalate or pivot

They output structured tool calls that the MissionEngine executes
through the scope-enforced ToolExecutor.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.models import (
    Evidence, EvidenceType, Finding, Mission, MissionPhase,
    Severity, Target, ToolResult,
)
from cookbook.arsenal.tools import SecurityTools, findings_from_nmap, findings_from_nuclei

logger = logging.getLogger("arsenal.operators")


# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS — operator personas
# ══════════════════════════════════════════════════════════════════

OPERATOR_SYSTEM = """You are an autonomous security assessment operator. You have access to real security tools and must conduct a thorough, methodical assessment.

CRITICAL RULES:
1. You MUST stay within the authorized scope. Never target anything outside scope.
2. You operate tools by returning JSON tool calls. The engine executes them.
3. Be thorough but efficient — don't waste scans on unlikely vectors.
4. Document everything — every finding, every observation.
5. Think like an attacker but act like a professional.

SCOPE: {scope_summary}
TARGET: {target_summary}
PHASE: {phase}

Available tools:
{tool_help}
"""

RECON_SYSTEM = """You are the RECON operator — your job is passive and active reconnaissance.

Your workflow:
1. WHOIS lookup on the target domain/IP
2. DNS enumeration (A, AAAA, MX, TXT, NS, CNAME, SOA records)
3. Subdomain enumeration (subfinder)
4. HTTP probing of discovered subdomains (httpx)
5. Technology fingerprinting (whatweb)
6. Attempt DNS zone transfer (unlikely but check)

Output a comprehensive attack surface map.

Return your tool calls as a JSON array:
[
  {"tool": "whois_lookup", "args": {"target": "example.com"}},
  {"tool": "dns_lookup", "args": {"domain": "example.com", "record_type": "ANY"}},
  ...
]

After each batch of results, analyze what you found and decide the next batch.
When done, return {"status": "complete", "summary": "...", "targets_discovered": [...]}
"""

SCAN_SYSTEM = """You are the SCAN operator — your job is active port scanning and service detection.

Your workflow:
1. Quick scan of each target (top ports, service detection)
2. Based on results, deep scan interesting hosts
3. For web services: run whatweb + nikto
4. For TLS services: run testssl
5. Full port scan (-p-) on the most interesting hosts

Prioritize targets that recon flagged as interesting.

Return tool calls as JSON:
[
  {"tool": "nmap_scan", "args": {"target": "10.0.0.1", "scan_type": "quick"}},
  {"tool": "nmap_scan", "args": {"target": "10.0.0.2", "ports": "1-65535", "scan_type": "full"}},
  ...
]
"""

VULN_SYSTEM = """You are the VULN operator — vulnerability scanning and analysis.

Your workflow:
1. Run nuclei against all discovered web targets
2. Run nmap vuln scripts against hosts with interesting services
3. Test for common misconfigurations
4. Check for default credentials
5. SSL/TLS vulnerability assessment

For each vulnerability found:
- Assess real-world exploitability
- Rate severity (CVSS 3.1)
- Check for public exploits
- Write remediation guidance

Return tool calls or findings:
[
  {"tool": "nuclei_scan", "args": {"target": "https://target.com", "severity": "critical,high,medium"}},
  {"tool": "nmap_scan", "args": {"target": "10.0.0.1", "scan_type": "vuln"}},
  ...
]
"""

ANALYSIS_SYSTEM = """You are the ANALYSIS operator — you synthesize all findings into a coherent assessment.

You receive all evidence and findings from previous phases. Your job:
1. Correlate findings across tools (e.g., open port + vuln = attack chain)
2. Eliminate false positives
3. Build attack chains (initial access → lateral movement → objective)
4. Prioritize findings by real-world risk (not just CVSS)
5. Write remediation guidance for each finding
6. Produce the executive summary

Output format:
{
  "status": "complete",
  "executive_summary": "...",
  "risk_rating": "critical|high|medium|low",
  "attack_chains": [...],
  "false_positives": [...],
  "remediation_priority": [...],
  "findings_updated": [...]
}
"""


# ══════════════════════════════════════════════════════════════════
#  OPERATOR BASE CLASS
# ══════════════════════════════════════════════════════════════════


@dataclass
class OperatorResult:
    """Result from an operator's work."""
    phase: MissionPhase
    summary: str = ""
    targets_discovered: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    raw_analysis: str = ""
    duration: float = 0.0
    tool_calls_made: int = 0
    errors: list[str] = field(default_factory=list)


class Operator:
    """
    Base class for LLM-powered security operators.

    An operator:
      1. Gets context (scope, targets, previous findings)
      2. Asks the LLM what tools to run
      3. Executes tools via SecurityTools (scope-enforced)
      4. Feeds results back to the LLM for analysis
      5. Repeats until the LLM says "complete"
    """

    def __init__(
        self,
        config: ArsenalConfig,
        mission: Mission,
        tools: SecurityTools,
        phase: MissionPhase,
        system_prompt: str,
        max_iterations: int = 10,
    ):
        self.config = config
        self.mission = mission
        self.tools = tools
        self.phase = phase
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._http = httpx.AsyncClient(
            base_url=self.config.api_base,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            timeout=httpx.Timeout(120.0),
        )
        return self

    async def __aexit__(self, *args):
        if self._http:
            await self._http.aclose()

    async def _chat(self, messages: list[dict], **kwargs) -> str:
        """Send a chat completion request to the LLM."""
        payload = {
            "model": kwargs.get("model", self.config.model),
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        for attempt in range(3):
            try:
                resp = await self._http.post("/chat/completions", json=payload)
                if resp.status_code >= 500:
                    await _backoff(attempt)
                    continue
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.HTTPError, KeyError) as e:
                if attempt == 2:
                    raise
                await _backoff(attempt)
        return ""

    def _build_context(self, extra: str = "") -> str:
        """Build context string for the operator."""
        scope_summary = ", ".join(
            f"{r.action.value} {r.target_type}={r.value}"
            for r in self.mission.scope.rules[:20]
        )
        target_summary = ", ".join(
            t.label for t in self.mission.targets[:20]
        )
        tool_help = "\n".join(
            self.tools.tool_help(t["name"])
            for t in self.tools.available_tools()
        )

        ctx = OPERATOR_SYSTEM.format(
            scope_summary=scope_summary or "No scope defined",
            target_summary=target_summary or "See task description",
            phase=self.phase.value,
            tool_help=tool_help,
        )
        if extra:
            ctx += f"\n\n{extra}"
        return ctx

    def _build_findings_context(self) -> str:
        """Summarize existing findings for the operator."""
        if not self.mission.findings:
            return "No findings yet."
        lines = [f"Existing findings ({len(self.mission.findings)}):\n"]
        for f in self.mission.findings:
            lines.append(
                f"  [{f.severity.value.upper()}] {f.title} — {f.target}"
                + (f":{f.port}" if f.port else "")
            )
        return "\n".join(lines)

    def _build_targets_context(self) -> str:
        """Summarize known targets."""
        if not self.mission.targets:
            return "No targets discovered yet."
        lines = ["Known targets:\n"]
        for t in self.mission.targets:
            svcs = ", ".join(f"{p}:{s}" for p, s in t.services.items())
            lines.append(f"  {t.label} — ports: {t.ports or 'unknown'} — services: {svcs or 'unknown'}")
        return "\n".join(lines)

    async def _execute_tool_calls(
        self, tool_calls: list[dict[str, Any]],
    ) -> list[ToolResult]:
        """Execute a batch of tool calls from the LLM."""
        results = []
        for call in tool_calls:
            tool_name = call.get("tool", "")
            args = call.get("args", {})

            self.mission.log(
                "operator_tool_call",
                f"{self.phase.value}: {tool_name}({json.dumps(args)[:200]})",
                phase=self.phase.value,
            )

            try:
                result = await self._dispatch_tool(tool_name, args)
                results.append(result)

                # Auto-generate findings from certain tools
                if result.success and result.parsed:
                    if tool_name == "nmap_scan":
                        for f in findings_from_nmap(result, args.get("target", "")):
                            self.mission.add_finding(f)
                    elif tool_name == "nuclei_scan":
                        for f in findings_from_nuclei(result, args.get("target", "")):
                            self.mission.add_finding(f)

            except Exception as e:
                logger.error(f"Tool call failed: {tool_name}: {e}")
                results.append(ToolResult(
                    tool_name=tool_name, command="",
                    success=False, error=str(e),
                ))

        return results

    async def _dispatch_tool(
        self, tool_name: str, args: dict[str, Any],
    ) -> ToolResult:
        """Route a tool call to the correct SecurityTools method."""
        dispatch = {
            "nmap_scan": lambda: self.tools.nmap_scan(**args),
            "nuclei_scan": lambda: self.tools.nuclei_scan(**args),
            "subfinder_enum": lambda: self.tools.subfinder_enum(**args),
            "httpx_probe": lambda: self.tools.httpx_probe(**args),
            "dns_lookup": lambda: self.tools.dns_lookup(**args),
            "dns_zone_transfer": lambda: self.tools.dns_zone_transfer(**args),
            "whois_lookup": lambda: self.tools.whois_lookup(**args),
            "nikto_scan": lambda: self.tools.nikto_scan(**args),
            "whatweb_scan": lambda: self.tools.whatweb_scan(**args),
            "ffuf_fuzz": lambda: self.tools.ffuf_fuzz(**args),
            "testssl_scan": lambda: self.tools.testssl_scan(**args),
            "curl_request": lambda: self.tools.curl_request(**args),
        }

        fn = dispatch.get(tool_name)
        if not fn:
            return ToolResult(
                tool_name=tool_name, command="",
                success=False, error=f"Unknown tool method: {tool_name}",
            )
        return await fn()

    def _extract_tool_calls(self, llm_response: str) -> list[dict[str, Any]]:
        """Extract JSON tool calls from LLM response."""
        # Try to find a JSON array in the response
        for pattern in [
            r'\[[\s\S]*?\{[\s\S]*?"tool"[\s\S]*?\}[\s\S]*?\]',
            r'\{[\s\S]*?"tool"[\s\S]*?\}',
        ]:
            matches = re.findall(pattern, llm_response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return [parsed]
                    if isinstance(parsed, list):
                        return [p for p in parsed if isinstance(p, dict) and "tool" in p]
                except json.JSONDecodeError:
                    continue

        # Try markdown fenced blocks
        fenced = re.findall(r'```(?:json)?\s*\n([\s\S]*?)```', llm_response)
        for block in fenced:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list):
                    return [p for p in parsed if isinstance(p, dict) and "tool" in p]
            except json.JSONDecodeError:
                continue

        return []

    def _extract_completion(self, llm_response: str) -> dict[str, Any] | None:
        """Check if the LLM is signaling completion."""
        for pattern in [
            r'\{[\s\S]*?"status"\s*:\s*"complete"[\s\S]*?\}',
        ]:
            matches = re.findall(pattern, llm_response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if parsed.get("status") == "complete":
                        return parsed
                except json.JSONDecodeError:
                    continue
        return None

    async def run(self, task: str = "", on_progress: Any = None) -> OperatorResult:
        """
        Execute the operator's full workflow.

        The operator iteratively:
          1. Asks the LLM what to do next
          2. Executes tool calls
          3. Feeds results back
          4. Repeats until complete or max_iterations
        """
        t0 = time.time()
        result = OperatorResult(phase=self.phase)
        self.mission.phase = self.phase
        self.mission.log("operator_start", f"Phase: {self.phase.value}")

        messages = [
            {"role": "system", "content": self._build_context(self.system_prompt)},
            {"role": "user", "content": self._build_task_prompt(task)},
        ]

        for iteration in range(self.max_iterations):
            logger.info(f"[{self.phase.value}] Iteration {iteration + 1}/{self.max_iterations}")

            if on_progress:
                on_progress(self.phase, iteration + 1, self.max_iterations)

            # Ask LLM
            try:
                llm_response = await self._chat(messages)
            except Exception as e:
                result.errors.append(f"LLM error: {e}")
                break

            messages.append({"role": "assistant", "content": llm_response})

            # Store LLM analysis as evidence
            evidence = Evidence(
                evidence_type=EvidenceType.LLM_ANALYSIS,
                tool_name=f"operator_{self.phase.value}",
                raw_output=llm_response,
                target="analysis",
            )
            self.mission.add_evidence(evidence)

            # Check for completion
            completion = self._extract_completion(llm_response)
            if completion:
                result.summary = completion.get("summary", llm_response[:500])
                result.raw_analysis = llm_response

                # Extract discovered targets
                for td in completion.get("targets_discovered", []):
                    if isinstance(td, str):
                        result.targets_discovered.append({"host": td})
                    elif isinstance(td, dict):
                        result.targets_discovered.append(td)
                break

            # Extract and execute tool calls
            tool_calls = self._extract_tool_calls(llm_response)
            if not tool_calls:
                # No tool calls and no completion — ask LLM to be more specific
                messages.append({
                    "role": "user",
                    "content": (
                        "Please either:\n"
                        "1. Return tool calls as a JSON array, OR\n"
                        "2. Return {\"status\": \"complete\", \"summary\": \"...\"} if done.\n\n"
                        "Current findings:\n" + self._build_findings_context()
                    ),
                })
                continue

            # Execute tools
            tool_results = await self._execute_tool_calls(tool_calls)
            result.tool_results.extend(tool_results)
            result.tool_calls_made += len(tool_calls)

            # Build results summary for LLM
            results_summary = self._summarize_results(tool_results)
            messages.append({
                "role": "user",
                "content": (
                    f"Tool results from iteration {iteration + 1}:\n\n"
                    f"{results_summary}\n\n"
                    f"Current findings:\n{self._build_findings_context()}\n\n"
                    f"Known targets:\n{self._build_targets_context()}\n\n"
                    "Analyze these results and decide: run more tools or complete."
                ),
            })

        else:
            # Max iterations reached
            result.summary = f"Max iterations ({self.max_iterations}) reached"
            result.errors.append("Hit max iteration limit")

        result.duration = time.time() - t0
        result.findings = list(self.mission.findings)  # snapshot
        self.mission.log(
            "operator_complete",
            f"{self.phase.value}: {result.tool_calls_made} tool calls, "
            f"{len(result.findings)} findings, {result.duration:.1f}s",
        )

        return result

    def _build_task_prompt(self, task: str) -> str:
        """Build the initial task prompt."""
        parts = [task or f"Conduct a {self.phase.value} assessment."]
        parts.append(f"\nKnown targets:\n{self._build_targets_context()}")
        parts.append(f"\nExisting findings:\n{self._build_findings_context()}")
        parts.append(
            "\nProceed with your first batch of tool calls. "
            "Return a JSON array of tool calls."
        )
        return "\n".join(parts)

    def _summarize_results(self, results: list[ToolResult]) -> str:
        """Summarize tool results for the LLM (truncated to fit context)."""
        parts = []
        for r in results:
            status = "SUCCESS" if r.success else "FAILED"
            output = r.output[:3000]  # truncate long output
            if len(r.output) > 3000:
                output += f"\n... [truncated, {len(r.output)} total bytes]"

            parts.append(
                f"=== {r.tool_name} ({status}, {r.duration:.1f}s) ===\n"
                f"Command: {r.command}\n"
                f"Output:\n{output}\n"
            )

            if r.parsed:
                parsed_str = json.dumps(r.parsed, indent=2)[:2000]
                parts.append(f"Parsed data:\n{parsed_str}\n")

            if r.error:
                parts.append(f"Error: {r.error}\n")

        return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════
#  SPECIALIZED OPERATORS
# ══════════════════════════════════════════════════════════════════


class ReconOperator(Operator):
    """Reconnaissance operator — OSINT, DNS, subdomain enum."""

    def __init__(self, config: ArsenalConfig, mission: Mission, tools: SecurityTools):
        super().__init__(
            config, mission, tools,
            phase=MissionPhase.RECONNAISSANCE,
            system_prompt=RECON_SYSTEM,
            max_iterations=8,
        )

    async def run(self, task: str = "", on_progress: Any = None) -> OperatorResult:
        result = await super().run(
            task or "Perform comprehensive reconnaissance on all in-scope targets.",
            on_progress,
        )

        # Update mission targets with discovered subdomains/hosts
        for td in result.targets_discovered:
            host = td.get("host", "") if isinstance(td, dict) else str(td)
            if host and not any(t.host == host for t in self.mission.targets):
                self.mission.targets.append(Target(host=host))
                self.mission.log("target_discovered", host, phase="reconnaissance")

        return result


class ScanOperator(Operator):
    """Scanning operator — port scanning, service detection."""

    def __init__(self, config: ArsenalConfig, mission: Mission, tools: SecurityTools):
        super().__init__(
            config, mission, tools,
            phase=MissionPhase.ENUMERATION,
            system_prompt=SCAN_SYSTEM,
            max_iterations=10,
        )

    async def run(self, task: str = "", on_progress: Any = None) -> OperatorResult:
        result = await super().run(
            task or "Scan all known targets for open ports and services.",
            on_progress,
        )

        # Update targets with discovered services
        for tr in result.tool_results:
            if tr.tool_name == "nmap" and tr.parsed:
                for host_data in tr.parsed.get("hosts", []):
                    addr = host_data.get("addresses", [{}])[0].get("addr", "")
                    if not addr:
                        continue
                    # Find or create target
                    target = next(
                        (t for t in self.mission.targets if t.host == addr), None
                    )
                    if not target:
                        target = Target(host=addr)
                        self.mission.targets.append(target)

                    for port_info in host_data.get("ports", []):
                        if port_info.get("state") == "open":
                            port = port_info.get("port", 0)
                            if port and port not in target.ports:
                                target.ports.append(port)
                            svc = port_info.get("service", "")
                            if port and svc:
                                target.services[port] = svc

                    for os_info in host_data.get("os_matches", []):
                        if not target.os_guess:
                            target.os_guess = os_info.get("name", "")

        return result


class VulnOperator(Operator):
    """Vulnerability operator — vuln scanning and analysis."""

    def __init__(self, config: ArsenalConfig, mission: Mission, tools: SecurityTools):
        super().__init__(
            config, mission, tools,
            phase=MissionPhase.VULNERABILITY_ANALYSIS,
            system_prompt=VULN_SYSTEM,
            max_iterations=8,
        )


class AnalysisOperator(Operator):
    """Analysis operator — correlates findings, builds attack chains, writes report."""

    def __init__(self, config: ArsenalConfig, mission: Mission, tools: SecurityTools):
        super().__init__(
            config, mission, tools,
            phase=MissionPhase.REPORTING,
            system_prompt=ANALYSIS_SYSTEM,
            max_iterations=3,
        )

    async def run(self, task: str = "", on_progress: Any = None) -> OperatorResult:
        # The analysis operator doesn't run tools — it just analyzes
        return await super().run(
            task or "Synthesize all findings into a comprehensive security assessment.",
            on_progress,
        )


# ── Utility ───────────────────────────────────────────────────────

async def _backoff(attempt: int) -> None:
    import asyncio
    delay = min(2 ** attempt, 30)
    await asyncio.sleep(delay)
