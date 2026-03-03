# 🛡️ Arsenal — Autonomous Offensive Security Platform

> Real security tools. LLM-driven decision making. Full legal scope enforcement.

Arsenal is a defense-grade autonomous security assessment platform that orchestrates real offensive tools (nmap, nuclei, subfinder, etc.) with LLM-powered operators, cryptographically signed scoping, Docker isolation, and forensic evidence chains.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │              Mission Engine              │
                    │    scope → recon → scan → vuln → report │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │           LLM Operators                  │
                    │  ReconOp │ ScanOp │ VulnOp │ AnalysisOp │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │           Security Tools                 │
                    │  nmap │ nuclei │ subfinder │ httpx │ ... │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │         Scope Enforcer (MANDATORY)       │
                    │  SHA-256 signed │ time-bound │ default deny│
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────┐
                    │          Tool Executor                    │
                    │  sanitize → scope → docker/local → evidence│
                    └─────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
cd revengineer
uv sync   # installs the package + dependencies
```

### 2. Build the Attack Lab (Docker)

```bash
# Build the Docker image with all security tools pre-installed
uv run arsenal lab --build

# Verify tools are available
uv run arsenal lab --check
```

This builds an Ubuntu 24.04 image with nmap, nuclei, subfinder, httpx, ffuf, nikto, whatweb, testssl.sh, dig, and whois.

### 3. Run an Assessment

```bash
# Dry-run mode (plans but doesn't execute tools)
uv run arsenal assess --target scanme.nmap.org --dry-run

# Real assessment with Docker isolation
uv run arsenal assess --target example.com --scope scope.json

# Quick scan (just recon + port scan)
uv run arsenal scan --target 10.0.0.1

# Recon only (subdomains, DNS, WHOIS)
uv run arsenal recon --target example.com
```

### 4. Programmatic API

```python
import asyncio
from cookbook.arsenal.config import ArsenalConfig
from cookbook.arsenal.engine import MissionEngine

async def run():
    config = ArsenalConfig(dry_run=True, use_docker=False)
    engine = MissionEngine(config)

    engine.add_scope_rule("allow", "domain", "*.example.com")
    engine.add_scope_rule("allow", "cidr", "10.0.0.0/24")
    engine.add_target("example.com")

    mission = await engine.execute()
    print(f"Status: {mission.status.value}")
    print(f"Findings: {len(mission.findings)}")
    print(f"Evidence: {len(mission.evidence)}")

asyncio.run(run())
```

---

## Scope Enforcement

Arsenal uses a **default-deny** scope model. Every tool execution is validated against the scope before running. Out-of-scope targets are blocked immediately.

### Scope Rules

```json
{
  "rules": [
    {"action": "allow", "type": "domain", "value": "*.target.com", "note": "Primary target per SOW §3.1"},
    {"action": "allow", "type": "cidr", "value": "10.0.0.0/24", "note": "Internal lab network"},
    {"action": "deny", "type": "ip", "value": "10.0.0.99", "note": "Production DB — DO NOT TOUCH"},
    {"action": "allow", "type": "port", "value": "1-65535", "note": "All ports authorized"}
  ],
  "authorized_by": "Jane Smith, CISO",
  "authorization_ref": "SOW-2025-042",
  "valid_from": 1720000000,
  "valid_until": 1721000000
}
```

### Safety Features

| Feature | Description |
|---------|-------------|
| **Default deny** | No rule match → blocked |
| **Time-bound** | Scopes expire automatically |
| **SHA-256 signed** | Tamper detection on scope rules |
| **Command extraction** | IPs/domains/URLs extracted from commands and validated |
| **Blocked patterns** | Fork bombs, `rm -rf`, pipe-to-shell, etc. always blocked |
| **Safe mode** | Only non-destructive tools allowed (default: ON) |
| **Environment sanitization** | Secrets stripped from subprocess env vars |

### Generate / validate scope files

```bash
# Export a scope template
uv run arsenal scope --export scope.json

# Validate a scope file
uv run arsenal scope --validate scope.json
```

---

## Security Tools

| Tool | Category | Description |
|------|----------|-------------|
| **nmap** | Scanning | Port scanning, service/OS detection, NSE scripts |
| **nuclei** | Vulnerability | Template-based vulnerability scanning (10k+ templates) |
| **subfinder** | Recon | Passive subdomain enumeration |
| **httpx** | Enumeration | HTTP probing, tech detection, CDN detection |
| **dig** | Recon | DNS lookups, zone transfer attempts |
| **whois** | Recon | Domain/IP registration data |
| **nikto** | Vulnerability | Web server misconfiguration scanning |
| **whatweb** | Enumeration | Web technology fingerprinting |
| **ffuf** | Enumeration | Directory/parameter fuzzing |
| **testssl.sh** | Scanning | TLS/SSL cipher and vulnerability analysis |
| **curl** | Utility | HTTP requests, banner grabbing |

All tools have structured output parsers (XML, JSONL, text) that extract typed data for LLM analysis.

---

## LLM Operators

Each kill-chain phase has a specialized LLM operator that iteratively decides what tools to run:

| Operator | Phase | Responsibility |
|----------|-------|---------------|
| **ReconOperator** | Reconnaissance | Discover targets, subdomains, DNS records, WHOIS data |
| **ScanOperator** | Enumeration | Port scanning, service detection, tech fingerprinting |
| **VulnOperator** | Vulnerability Analysis | Run nuclei, nikto, testssl against discovered services |
| **AnalysisOperator** | Reporting | Analyze all findings, correlate evidence, write report |

Each operator:
1. Receives context (scope, targets, prior findings)
2. Asks the LLM what tools to run
3. Executes tools via the scope-enforced executor
4. Feeds results back to the LLM
5. Repeats until the LLM decides the phase is complete

---

## Docker Attack Lab

Arsenal isolates all tool execution inside a Docker container:

```bash
# Build the lab image
uv run arsenal lab --build

# Generate docker-compose.yml for manual use
uv run arsenal lab --compose

# Check which tools are available
uv run arsenal lab --check

# Tear down lab resources
uv run arsenal lab --teardown
```

**Container security:**
- `--cap-drop ALL --cap-add NET_RAW` (minimal capabilities)
- `--security-opt no-new-privileges`
- Memory/CPU limits (configurable)
- Isolated Docker network
- Evidence directory mounted for output

---

## Configuration

All settings configurable via `ArsenalConfig` or `ARSENAL_*` environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `ARSENAL_API_BASE` | LLM API endpoint | `http://localhost:8000/v1` |
| `ARSENAL_MODEL` | Default LLM model | `deepseek-chat` |
| `ARSENAL_DRY_RUN` | Plan but don't execute | `false` |
| `ARSENAL_SAFE_MODE` | Block destructive tools | `true` |
| `ARSENAL_USE_DOCKER` | Run tools in Docker | `true` |
| `ARSENAL_SCAN_RATE_LIMIT` | Max scan probes/sec | `100` |
| `ARSENAL_TEMPERATURE` | LLM temperature | `0.3` |
| `ARSENAL_TOOL_TIMEOUT` | Per-tool timeout (sec) | `300` |
| `ARSENAL_MISSION_TIMEOUT` | Total mission timeout | `3600` |
| `ARSENAL_OUTPUT_DIR` | Artifact output directory | `.` (cwd) |

---

## Output

Arsenal generates:

- **Markdown report** — Executive summary, findings by severity, evidence chain, audit log
- **JSON mission file** — Full structured data for programmatic consumption
- **Raw evidence** — Tool outputs with SHA-256 integrity hashes

Example report structure:
```
arsenal_mission_abc123/
├── mission_abc123.json      # Full structured mission data
├── mission_abc123.md        # Markdown report
└── evidence/
    ├── nmap_e1a2b3_20250101_120000.stdout.txt
    ├── nuclei_f4d5e6_20250101_120100.stdout.txt
    └── subfinder_a7b8c9_20250101_120200.stdout.txt
```

---

## Testing

```bash
# Run all tests (222 total: 158 arsenal + 64 swarm)
uv run python -m pytest cookbook/tests/ -v

# Arsenal tests only
uv run python -m pytest cookbook/tests/test_arsenal.py -v

# Quick smoke test
uv run python -c "from cookbook.arsenal import MissionEngine, ArsenalConfig; print('OK')"
```

Test coverage includes:
- All 20 data model classes/enums
- Scope enforcement (allow/deny/CIDR/wildcard/URL/port/time-bound/integrity)
- Tool executor pipeline (sanitization, scope, dry-run, safe-mode, evidence)
- All 11 output parsers (nmap XML/text, nuclei, subfinder, httpx, dig, whois, nikto, whatweb, ffuf, testssl)
- Finding generators (nuclei → findings, nmap → findings with severity elevation)
- Integration tests (scope → executor → evidence → audit log)

---

## Module Reference

| Module | Lines | Description |
|--------|-------|-------------|
| `models.py` | ~460 | Enums, scope rules, targets, evidence, findings, missions |
| `config.py` | ~130 | Configuration with env var support |
| `scope.py` | ~265 | Scope enforcement with SHA-256 integrity |
| `executor.py` | ~380 | Tool execution with safety controls |
| `tools.py` | ~820 | 11 tool wrappers with parsers |
| `operators.py` | ~430 | 4 LLM operator agents |
| `engine.py` | ~330 | Mission orchestration engine |
| `lab.py` | ~200 | Docker attack lab |
| `cli.py` | ~290 | Command-line interface |

---

## ⚠️ Legal Disclaimer

Arsenal is designed for **authorized security assessments only**. Always obtain written authorization before scanning or testing any system. The scope enforcement system is a safety net, not a substitute for proper legal authorization.

Unauthorized access to computer systems is illegal. Use responsibly.
