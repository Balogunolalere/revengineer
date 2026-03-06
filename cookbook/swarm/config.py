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
    max_tokens: int = 8192
    verbose: bool = False

    # --- Adaptive concurrency ---
    adaptive_rate_limit: bool = True     # auto-adjust rate on errors/429s
    rate_limit_min_rpm: int = 5          # floor for adaptive rate
    rate_limit_backoff_factor: float = 0.5   # multiply rate by this on error
    rate_limit_recovery_factor: float = 1.1  # multiply rate by this on success streak
    rate_limit_recovery_streak: int = 5      # consecutive successes to trigger recovery

    # --- Circuit breaker ---
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5   # consecutive failures to trip open
    circuit_breaker_cooldown: float = 30.0   # seconds in OPEN before HALF_OPEN

    # --- Tool use ---
    max_tool_calls_per_agent: int = 20   # cap tool iterations per agent
    tool_timeout: float = 120.0          # per-tool-call timeout (seconds)
    enable_reflection: bool = False      # agent self-review before submitting
    tool_agent_timeout: float = 600.0    # timeout for agents that use tools

    # --- Smarter decomposition ---
    enable_plan_critique: bool = False   # LLM reviews plan before execution
    enable_agent_killing: bool = True    # cancel agents exceeding kill threshold
    agent_kill_threshold: float = 3.0    # kill agents taking > N× median peer duration
    agent_kill_min_time: float = 30.0    # absolute floor — never kill before this many seconds

    # --- Budget ---
    token_budget: int = 0                # total token cap (0 = unlimited)
    max_llm_calls: int = 0              # total LLM call cap (0 = unlimited)

    # --- Sub-swarms ---
    enable_sub_swarms: bool = False      # allow agents to spawn sub-swarms
    sub_swarm_max_depth: int = 2         # max nesting depth
    sub_swarm_max_agents: int = 5        # max agents per sub-swarm

    # --- Context window ---
    context_window_tokens: int = 0       # max tokens of context per agent (0 = unlimited)
    context_summary_tokens: int = 300    # target summary length in tokens
    enable_context_pruning: bool = False # enable token-aware context pruning

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
            max_tokens=int(_get("MAX_TOKENS", "8192")),
            verbose=_get("VERBOSE", "false").lower() in ("true", "1", "yes"),
            adaptive_rate_limit=_get("ADAPTIVE_RATE", "true").lower() in ("true", "1", "yes"),
            rate_limit_min_rpm=int(_get("RATE_LIMIT_MIN_RPM", "5")),
            rate_limit_backoff_factor=float(_get("RATE_BACKOFF_FACTOR", "0.5")),
            rate_limit_recovery_factor=float(_get("RATE_RECOVERY_FACTOR", "1.1")),
            rate_limit_recovery_streak=int(_get("RATE_RECOVERY_STREAK", "5")),
            circuit_breaker_enabled=_get("CIRCUIT_BREAKER", "true").lower() in ("true", "1", "yes"),
            circuit_breaker_threshold=int(_get("CB_THRESHOLD", "5")),
            circuit_breaker_cooldown=float(_get("CB_COOLDOWN", "30")),
            max_tool_calls_per_agent=int(_get("MAX_TOOL_CALLS", "20")),
            tool_timeout=float(_get("TOOL_TIMEOUT", "120")),
            enable_reflection=_get("ENABLE_REFLECTION", "false").lower() in ("true", "1", "yes"),
            tool_agent_timeout=float(_get("TOOL_AGENT_TIMEOUT", "600")),
            enable_plan_critique=_get("PLAN_CRITIQUE", "false").lower() in ("true", "1", "yes"),
            enable_agent_killing=_get("AGENT_KILLING", "true").lower() in ("true", "1", "yes"),
            agent_kill_threshold=float(_get("AGENT_KILL_THRESHOLD", "3.0")),
            agent_kill_min_time=float(_get("AGENT_KILL_MIN_TIME", "30.0")),
            token_budget=int(_get("TOKEN_BUDGET", "0")),
            max_llm_calls=int(_get("MAX_LLM_CALLS", "0")),
            enable_sub_swarms=_get("SUB_SWARMS", "false").lower() in ("true", "1", "yes"),
            sub_swarm_max_depth=int(_get("SUB_SWARM_MAX_DEPTH", "2")),
            sub_swarm_max_agents=int(_get("SUB_SWARM_MAX_AGENTS", "5")),
            context_window_tokens=int(_get("CONTEXT_WINDOW_TOKENS", "0")),
            context_summary_tokens=int(_get("CONTEXT_SUMMARY_TOKENS", "300")),
            enable_context_pruning=_get("CONTEXT_PRUNING", "false").lower() in ("true", "1", "yes"),
        )
