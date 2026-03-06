"""
Tool registry for swarm agents.

Provides a registry of callable tools that agents can invoke mid-execution.
Uses text-based tool calling (JSON in LLM responses) matching the pattern
already proven in Arsenal's Operator system.

Tools are registered as async callables with metadata for LLM prompting.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

log = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Definition of a tool available to swarm agents."""
    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)  # param_name -> description
    fn: Callable[..., Awaitable[str]] | None = None  # async (kwargs) -> str result
    safe: bool = True  # non-destructive?


class ToolRegistry:
    """Registry of tools available to swarm agents."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        """Register a tool definition with its callable."""
        self._tools[tool.name] = tool

    def register_fn(
        self,
        name: str,
        fn: Callable[..., Awaitable[str]],
        description: str = "",
        parameters: dict[str, str] | None = None,
        safe: bool = True,
    ) -> None:
        """Convenience: register a tool from a function."""
        self._tools[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters or {},
            fn=fn,
            safe=safe,
        )

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def available(self, filter_names: list[str] | None = None) -> list[ToolDef]:
        """Return available tools, optionally filtered by name list."""
        if filter_names is None:
            return list(self._tools.values())
        return [t for name in filter_names if (t := self._tools.get(name))]

    def build_tool_prompt(self, filter_names: list[str] | None = None) -> str:
        """Build a tool description block for injection into agent system prompts."""
        tools = self.available(filter_names)
        if not tools:
            return ""

        lines = ["## Available Tools", ""]
        lines.append("CRITICAL: You MUST call tools by outputting a JSON array. Do NOT describe what you would do — actually call the tool.")
        lines.append("")
        lines.append("Format — output ONLY this JSON block (no prose before it):")
        lines.append('```json')
        lines.append('[{"tool": "tool_name", "args": {"param": "value"}}]')
        lines.append('```')
        lines.append("")
        lines.append("Rules:")
        lines.append("- Your FIRST response MUST be a JSON tool call. Never skip calling tools.")
        lines.append("- After receiving tool results, either call more tools OR give your final answer.")
        lines.append("- NEVER fabricate tool results. NEVER pretend you called a tool.")
        lines.append("- If you need to call a tool for each item in a list, call them one at a time or batch them in a JSON array.")
        lines.append("")

        for tool in tools:
            params_str = ""
            if tool.parameters:
                params = ", ".join(
                    f"`{k}`: {v}" for k, v in tool.parameters.items()
                )
                params_str = f"  Parameters: {params}"
            lines.append(f"- **{tool.name}**: {tool.description}")
            if params_str:
                lines.append(params_str)

        return "\n".join(lines)

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        timeout: float = 120.0,
    ) -> str:
        """Execute a tool by name. Returns result string or error message."""
        tool = self._tools.get(tool_name)
        if not tool:
            return f"[ERROR] Unknown tool: {tool_name}"
        if not tool.fn:
            return f"[ERROR] Tool '{tool_name}' has no callable registered"

        import asyncio
        try:
            result = await asyncio.wait_for(tool.fn(**args), timeout=timeout)
            return result
        except asyncio.TimeoutError:
            return f"[ERROR] Tool '{tool_name}' timed out after {timeout:.0f}s"
        except Exception as e:
            return f"[ERROR] Tool '{tool_name}' failed: {e}"


# ── JSON extraction from LLM responses ──────────────────────────


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract JSON tool call arrays from LLM text response.

    Looks for JSON arrays containing objects with a "tool" key.
    Handles raw JSON, markdown-fenced blocks, and embedded JSON.
    """
    # Try markdown fenced blocks first (most reliable delimiter)
    fenced = re.findall(r'```(?:json)?\s*\n?([\s\S]*?)```', text)
    for block in fenced:
        parsed = _try_parse_tools(block.strip())
        if parsed:
            return parsed

    # Try to find a JSON array with "tool" key in raw text
    for pattern in [
        r'\[[\s\S]*?\{[\s\S]*?"tool"[\s\S]*?\}[\s\S]*?\]',
    ]:
        matches = re.findall(pattern, text)
        for match in matches:
            parsed = _try_parse_tools(match)
            if parsed:
                return parsed

    # Try single JSON objects — use brace-depth matching for nested objects
    for obj_str in _extract_json_objects(text):
        if '"tool"' in obj_str:
            parsed = _try_parse_tools(obj_str)
            if parsed:
                return parsed

    return []


def _extract_json_objects(text: str) -> list[str]:
    """Extract top-level JSON objects from text using brace-depth counting."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            start = i
            in_string = False
            escape = False
            while i < len(text):
                ch = text[i]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"' and not escape:
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            results.append(text[start:i + 1])
                            break
                i += 1
        i += 1
    return results


def _try_parse_tools(text: str) -> list[dict[str, Any]] | None:
    """Try to parse text as a tool call list."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "tool" in parsed:
            return [parsed]
        if isinstance(parsed, list):
            tools = [p for p in parsed if isinstance(p, dict) and "tool" in p]
            if tools:
                return tools
    except json.JSONDecodeError:
        pass
    return None
