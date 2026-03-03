"""
Swarm configuration.

Supports environment variables, .env files, and direct instantiation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class SwarmConfig:
    """All configuration for the swarm."""

    # --- LLM backend ---
    api_base: str = ""            # OpenAI-compatible base URL
    api_key: str = "not-needed"   # API key (most local proxies don't need one)
    default_model: str = "deepseek-chat"
    search_model: str = "deepseek-search"
    reasoning_model: str = "deepseek-reasoner"
    orchestrator_model: str = ""  # model for decomposition/synthesis (empty = use default)

    # --- Concurrency ---
    max_parallel: int = 15        # max concurrent agents
    rate_limit_rpm: int = 60      # requests per minute cap
    rate_limit_burst: int = 10    # burst allowance

    # --- Retries ---
    max_retries: int = 2
    retry_base_delay: float = 1.0
    retry_max_delay: float = 30.0
    retry_jitter: bool = True

    # --- Timeouts ---
    agent_timeout: float = 120.0  # per-agent timeout (seconds)
    swarm_timeout: float = 600.0  # total swarm timeout (seconds)

    # --- Orchestrator ---
    max_agents: int = 25          # cap on sub-agents per swarm
    min_agents: int = 2           # minimum decomposition
    allow_replan: bool = True     # allow orchestrator to add agents mid-run
    replan_max: int = 2           # max re-plans allowed

    # --- Output ---
    output_dir: str = ""          # directory for file output (empty = cwd)
    save_json: bool = True        # save SwarmResult as JSON
    save_markdown: bool = True    # save report as markdown
    stream_to_terminal: bool = True

    # --- Advanced ---
    temperature: float = 0.7
    max_tokens: int = 4096
    verbose: bool = False

    @classmethod
    def from_env(cls) -> SwarmConfig:
        """Load config from environment variables (with SWARM_ prefix)."""
        def _get(key: str, default: str = "") -> str:
            return os.environ.get(f"SWARM_{key}", os.environ.get(key, default))

        return cls(
            api_base=_get("API_BASE", "http://localhost:8000/v1"),
            api_key=_get("API_KEY", "not-needed"),
            default_model=_get("MODEL", "deepseek-chat"),
            search_model=_get("SEARCH_MODEL", "deepseek-search"),
            reasoning_model=_get("REASONING_MODEL", "deepseek-reasoner"),
            orchestrator_model=_get("ORCHESTRATOR_MODEL", ""),
            max_parallel=int(_get("MAX_PARALLEL", "15")),
            rate_limit_rpm=int(_get("RATE_LIMIT_RPM", "60")),
            max_retries=int(_get("MAX_RETRIES", "2")),
            agent_timeout=float(_get("AGENT_TIMEOUT", "120")),
            swarm_timeout=float(_get("SWARM_TIMEOUT", "600")),
            max_agents=int(_get("MAX_AGENTS", "25")),
            allow_replan=_get("ALLOW_REPLAN", "true").lower() in ("true", "1", "yes"),
            output_dir=_get("OUTPUT_DIR", ""),
            save_json=_get("SAVE_JSON", "true").lower() in ("true", "1", "yes"),
            save_markdown=_get("SAVE_MARKDOWN", "true").lower() in ("true", "1", "yes"),
            stream_to_terminal=_get("STREAM", "true").lower() in ("true", "1", "yes"),
            temperature=float(_get("TEMPERATURE", "0.7")),
            max_tokens=int(_get("MAX_TOKENS", "4096")),
            verbose=_get("VERBOSE", "false").lower() in ("true", "1", "yes"),
        )
