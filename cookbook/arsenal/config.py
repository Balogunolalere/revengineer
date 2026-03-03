"""
Arsenal configuration.

Supports environment variables with ARSENAL_ prefix, .env files,
and direct instantiation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ArsenalConfig:
    """All configuration for Arsenal operations."""

    # ── LLM Backend ──
    api_base: str = "http://localhost:8000/v1"
    api_key: str = "not-needed"
    model: str = "deepseek-chat"
    search_model: str = "deepseek-search"
    reasoning_model: str = "deepseek-reasoner"
    planning_model: str = ""       # model for mission planning (empty = use default)
    temperature: float = 0.3       # lower temp for security work — precision > creativity
    max_tokens: int = 4096

    # ── Execution ──
    max_parallel_tools: int = 3    # max concurrent tool executions
    tool_timeout: float = 300.0    # per-tool timeout (seconds)
    mission_timeout: float = 3600.0  # total mission timeout (1 hour)

    # ── Safety ──
    require_scope_approval: bool = True   # must have signed scope before execution
    dry_run: bool = False                 # plan but don't execute tools
    safe_mode: bool = True                # limit to non-destructive tools only
    require_confirmation: bool = False    # pause before each tool execution
    max_exploitation_depth: int = 1       # how deep exploit chains can go (1 = POC only)

    # ── Docker Lab ──
    use_docker: bool = True               # run tools inside Docker container
    docker_image: str = "arsenal-lab:latest"
    docker_network: str = "arsenal-net"
    docker_timeout: float = 600.0         # container lifetime
    docker_memory: str = "2g"
    docker_cpus: str = "2"
    mount_evidence: bool = True           # mount evidence dir into container

    # ── Tool Paths (host or container) ──
    nmap_path: str = "nmap"
    nuclei_path: str = "nuclei"
    subfinder_path: str = "subfinder"
    httpx_path: str = "httpx"
    ffuf_path: str = "ffuf"
    nikto_path: str = "nikto"
    whatweb_path: str = "whatweb"
    dig_path: str = "dig"
    whois_path: str = "whois"
    curl_path: str = "curl"
    testssl_path: str = "testssl.sh"
    wpscan_path: str = "wpscan"

    # ── Output ──
    output_dir: str = ""                  # base directory for mission artifacts
    evidence_dir: str = ""                # separate evidence directory (empty = output_dir/evidence)
    save_raw_output: bool = True
    save_parsed: bool = True
    save_report: bool = True
    report_format: str = "markdown"       # markdown, html, json, pdf

    # ── Rate Limiting ──
    rate_limit_rpm: int = 30              # LLM requests per minute
    scan_rate_limit: int = 100            # max packets/requests per second for scans
    scan_delay: float = 0.0               # delay between scan probes (seconds)

    # ── Logging ──
    verbose: bool = False
    log_level: str = "INFO"
    log_file: str = ""                    # empty = stderr only

    def _out_dir(self) -> Path:
        return Path(self.output_dir) if self.output_dir else Path.cwd()

    def get_evidence_dir(self) -> Path:
        if self.evidence_dir:
            return Path(self.evidence_dir)
        return self._out_dir() / "evidence"

    @classmethod
    def from_env(cls) -> ArsenalConfig:
        """Load config from ARSENAL_* environment variables."""
        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"ARSENAL_{key}", os.environ.get(key, default))

        return cls(
            api_base=_get("API_BASE", "http://localhost:8000/v1"),
            api_key=_get("API_KEY", "not-needed"),
            model=_get("MODEL", "deepseek-chat"),
            search_model=_get("SEARCH_MODEL", "deepseek-search"),
            reasoning_model=_get("REASONING_MODEL", "deepseek-reasoner"),
            planning_model=_get("PLANNING_MODEL", ""),
            temperature=float(_get("TEMPERATURE", "0.3")),
            max_tokens=int(_get("MAX_TOKENS", "4096")),
            max_parallel_tools=int(_get("MAX_PARALLEL_TOOLS", "3")),
            tool_timeout=float(_get("TOOL_TIMEOUT", "300")),
            mission_timeout=float(_get("MISSION_TIMEOUT", "3600")),
            require_scope_approval=_get("REQUIRE_SCOPE", "true").lower() in ("true", "1"),
            dry_run=_get("DRY_RUN", "false").lower() in ("true", "1"),
            safe_mode=_get("SAFE_MODE", "true").lower() in ("true", "1"),
            require_confirmation=_get("REQUIRE_CONFIRM", "false").lower() in ("true", "1"),
            max_exploitation_depth=int(_get("MAX_EXPLOIT_DEPTH", "1")),
            use_docker=_get("USE_DOCKER", "true").lower() in ("true", "1"),
            docker_image=_get("DOCKER_IMAGE", "arsenal-lab:latest"),
            docker_timeout=float(_get("DOCKER_TIMEOUT", "600")),
            output_dir=_get("OUTPUT_DIR", ""),
            evidence_dir=_get("EVIDENCE_DIR", ""),
            save_raw_output=_get("SAVE_RAW", "true").lower() in ("true", "1"),
            save_report=_get("SAVE_REPORT", "true").lower() in ("true", "1"),
            report_format=_get("REPORT_FORMAT", "markdown"),
            rate_limit_rpm=int(_get("RATE_LIMIT_RPM", "30")),
            scan_rate_limit=int(_get("SCAN_RATE_LIMIT", "100")),
            verbose=_get("VERBOSE", "false").lower() in ("true", "1"),
            log_level=_get("LOG_LEVEL", "INFO"),
            log_file=_get("LOG_FILE", ""),
        )
