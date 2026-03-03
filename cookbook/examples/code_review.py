#!/usr/bin/env python3
"""
Example: Code review swarm — multiple specialized reviewers
analyze code in parallel, then a lead produces a unified report.

Prerequisites:
    1. Start your DeepSeek API proxy:  uv run python deepseek_api.py
    2. Run this script:                uv run python cookbook/examples/code_review.py
"""

import asyncio

from cookbook.swarm import (
    Swarm, SwarmConfig, SwarmMode,
    SwarmPlan, AgentSpec,
)


SAMPLE_CODE = '''
import hashlib, subprocess, json, os, re

class AuthManager:
    def __init__(self):
        self.users = {}
        self.sessions = {}

    def register(self, username, password):
        # Store password directly
        self.users[username] = password
        return True

    def login(self, username, password):
        if username in self.users and self.users[username] == password:
            import random
            session_id = str(random.randint(1000, 9999))
            self.sessions[session_id] = username
            return session_id
        return None

    def run_report(self, session_id, report_type):
        if session_id not in self.sessions:
            return "Unauthorized"
        # Generate report
        cmd = f"generate_report --type {report_type}"
        result = subprocess.run(cmd, shell=True, capture_output=True)
        return result.stdout.decode()

    def get_user_data(self, user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        # execute query...
        return query

    def export_data(self, path):
        data = json.dumps(self.users)
        with open(path, 'w') as f:
            f.write(data)
        return f"Exported to {path}"
'''


async def main():
    config = SwarmConfig(
        api_base="http://localhost:8000/v1",
        max_parallel=4,
        output_dir=".",
    )

    # Each reviewer specializes in a different area
    security = AgentSpec(
        role="Security Reviewer",
        task=f"Review this Python code for security vulnerabilities. "
             f"Check for: injection attacks, hardcoded secrets, insecure "
             f"crypto, command injection, path traversal, etc.\n\n```python\n{SAMPLE_CODE}\n```",
        system_prompt="You are a senior application security engineer. "
                      "Focus exclusively on security issues. Rate each "
                      "finding as Critical/High/Medium/Low.",
        priority=5,
    )

    architecture = AgentSpec(
        role="Architecture Reviewer",
        task=f"Review this Python code for architectural issues. "
             f"Check for: SOLID violations, separation of concerns, "
             f"testability, dependency management, error handling.\n\n```python\n{SAMPLE_CODE}\n```",
        system_prompt="You are a senior software architect. Focus on "
                      "design patterns, maintainability, and scalability.",
        priority=3,
    )

    performance = AgentSpec(
        role="Performance Reviewer",
        task=f"Review this Python code for performance issues. "
             f"Check for: unnecessary allocations, O(n) lookups that "
             f"could be O(1), blocking calls, resource leaks.\n\n```python\n{SAMPLE_CODE}\n```",
        system_prompt="You are a performance engineer. Focus on "
                      "efficiency, resource usage, and scalability.",
        priority=2,
    )

    style = AgentSpec(
        role="Style Reviewer",
        task=f"Review this Python code for style and best practices. "
             f"Check for: PEP 8, type hints, docstrings, naming "
             f"conventions, Pythonic idioms.\n\n```python\n{SAMPLE_CODE}\n```",
        system_prompt="You are a Python style guide expert. Focus on "
                      "readability, documentation, and idiomatic Python.",
        priority=1,
    )

    # Lead reviewer depends on ALL others
    lead = AgentSpec(
        role="Lead Reviewer",
        task="Produce a unified code review report combining all "
             "specialist reviews. Organize by severity, deduplicate "
             "findings, and add a prioritized remediation plan.",
        depends_on=[
            security.agent_id,
            architecture.agent_id,
            performance.agent_id,
            style.agent_id,
        ],
        system_prompt="You are a tech lead producing a final code review. "
                      "Synthesize all specialist reviews into one clear, "
                      "actionable report with severity ratings.",
        priority=0,
    )

    plan = SwarmPlan(
        goal=f"Comprehensive code review of AuthManager class",
        agents=[security, architecture, performance, style, lead],
        strategy="Parallel specialist reviews → unified lead report",
    )

    result = await Swarm(
        plan.goal,
        config=config,
        mode=SwarmMode.MANUAL,
        plan=plan,
        verbose=True,
    )

    print(f"\n{'='*60}")
    print(result.synthesis)


if __name__ == "__main__":
    asyncio.run(main())
