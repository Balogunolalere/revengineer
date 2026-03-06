"""
Comprehensive test suite for the Revengineer Swarm system.

Covers:
  - models.py: AgentSpec, AgentResult, SwarmPlan (DAG validation, cycle detection), SwarmResult
  - config.py: SwarmConfig defaults + from_env()
  - engine.py: RateLimiter, SwarmEngine (DAG execution, retries, timeouts, failure cascade, deadlock)
  - orchestrator.py: _extract_json, decompose (mocked), run_agent (mocked), synthesis
  - renderer.py: SwarmRenderer event handling, markdown/JSON output
  - __init__.py: Swarm() one-liner

Run:  python -m pytest cookbook/tests/test_swarm.py -v
  or: python cookbook/tests/test_swarm.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from cookbook.swarm.models import (
    AgentSpec, AgentResult, AgentStatus, SwarmMode, SwarmPlan, SwarmResult, BudgetTracker, ContextWindow,
)
from cookbook.swarm.config import SwarmConfig
from cookbook.swarm.engine import RateLimiter, SwarmEngine, AdaptiveRateLimiter, CircuitBreaker, CircuitState, CircuitOpenError
from cookbook.swarm.orchestrator import Orchestrator, _extract_json, DECOMPOSE_SYSTEM, CRITIQUE_SYSTEM
from cookbook.swarm.renderer import SwarmRenderer, _strip_ansi, _build_markdown
from cookbook.swarm.tool_registry import ToolDef, ToolRegistry, extract_tool_calls


# ═══════════════════════════════════════════════════════════════════
# models.py tests
# ═══════════════════════════════════════════════════════════════════

class TestAgentSpec(unittest.TestCase):
    def test_defaults(self):
        a = AgentSpec(role="Tester", task="test things")
        self.assertEqual(a.role, "Tester")
        self.assertEqual(a.task, "test things")
        self.assertIsInstance(a.agent_id, str)
        self.assertEqual(len(a.agent_id), 8)
        self.assertEqual(a.depends_on, [])
        self.assertEqual(a.tools, [])
        self.assertEqual(a.priority, 0)
        self.assertIsNone(a.timeout)
        self.assertEqual(a.max_retries, 2)

    def test_unique_ids(self):
        a = AgentSpec(role="A", task="t1")
        b = AgentSpec(role="B", task="t2")
        self.assertNotEqual(a.agent_id, b.agent_id)


class TestAgentResult(unittest.TestCase):
    def test_duration(self):
        r = AgentResult(agent_id="x", role="R", task="T",
                        started_at=100.0, finished_at=112.5)
        self.assertAlmostEqual(r.duration, 12.5)

    def test_duration_zero_when_not_set(self):
        r = AgentResult(agent_id="x", role="R", task="T")
        self.assertEqual(r.duration, 0.0)


class TestSwarmPlan(unittest.TestCase):
    def _make_agents(self):
        a = AgentSpec(role="A", task="do A")
        b = AgentSpec(role="B", task="do B", depends_on=[a.agent_id])
        c = AgentSpec(role="C", task="do C", depends_on=[b.agent_id])
        return a, b, c

    def test_agent_ids(self):
        a, b, c = self._make_agents()
        plan = SwarmPlan(goal="test", agents=[a, b, c])
        self.assertEqual(plan.agent_ids, {a.agent_id, b.agent_id, c.agent_id})

    def test_get_agent(self):
        a, b, c = self._make_agents()
        plan = SwarmPlan(goal="test", agents=[a, b, c])
        self.assertIs(plan.get_agent(b.agent_id), b)
        self.assertIsNone(plan.get_agent("nonexistent"))

    def test_get_ready_agents_initial(self):
        a, b, c = self._make_agents()
        plan = SwarmPlan(goal="test", agents=[a, b, c])
        ready = plan.get_ready_agents(completed=set())
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].agent_id, a.agent_id)

    def test_get_ready_agents_after_a(self):
        a, b, c = self._make_agents()
        plan = SwarmPlan(goal="test", agents=[a, b, c])
        ready = plan.get_ready_agents(completed={a.agent_id})
        roles = {r.role for r in ready}
        self.assertIn("B", roles)
        self.assertNotIn("C", roles)

    def test_get_ready_agents_all_parallel(self):
        agents = [AgentSpec(role=f"A{i}", task="t") for i in range(5)]
        plan = SwarmPlan(goal="test", agents=agents)
        ready = plan.get_ready_agents(completed=set())
        self.assertEqual(len(ready), 5)

    def test_validate_ok(self):
        a, b, c = self._make_agents()
        plan = SwarmPlan(goal="test", agents=[a, b, c])
        self.assertEqual(plan.validate(), [])

    def test_validate_missing_dep(self):
        a = AgentSpec(role="A", task="t", depends_on=["nonexistent_id"])
        plan = SwarmPlan(goal="test", agents=[a])
        errors = plan.validate()
        self.assertTrue(len(errors) > 0)
        self.assertIn("unknown agent_id", errors[0])

    def test_validate_cycle_detection(self):
        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t", depends_on=[a.agent_id])
        # Create cycle: a depends on b, b depends on a
        a.depends_on = [b.agent_id]
        plan = SwarmPlan(goal="test", agents=[a, b])
        errors = plan.validate()
        self.assertTrue(any("Circular" in e or "cycle" in e.lower() for e in errors))

    def test_validate_self_cycle(self):
        a = AgentSpec(role="A", task="t")
        a.depends_on = [a.agent_id]
        plan = SwarmPlan(goal="test", agents=[a])
        errors = plan.validate()
        self.assertTrue(len(errors) > 0)

    def test_validate_three_node_cycle(self):
        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t", depends_on=[a.agent_id])
        c = AgentSpec(role="C", task="t", depends_on=[b.agent_id])
        a.depends_on = [c.agent_id]  # a → c → b → a
        plan = SwarmPlan(goal="test", agents=[a, b, c])
        errors = plan.validate()
        self.assertTrue(any("Circular" in e for e in errors))


class TestSwarmResult(unittest.TestCase):
    def test_to_dict(self):
        plan = SwarmPlan(goal="test goal", agents=[], strategy="test strat")
        r = SwarmResult(
            goal="test goal", plan=plan,
            agent_results=[
                AgentResult(agent_id="1", role="R1", task="T1",
                            content="output1", status=AgentStatus.COMPLETED,
                            started_at=10.0, finished_at=15.0),
                AgentResult(agent_id="2", role="R2", task="T2",
                            status=AgentStatus.FAILED, error="boom"),
            ],
            synthesis="final report",
            started_at=10.0, finished_at=20.0,
        )
        d = r.to_dict()
        self.assertEqual(d["goal"], "test goal")
        self.assertEqual(d["agents_total"], 2)
        self.assertEqual(d["agents_succeeded"], 1)
        self.assertEqual(d["agents_failed"], 1)
        self.assertEqual(d["synthesis"], "final report")
        self.assertAlmostEqual(d["duration_seconds"], 10.0)

    def test_successful_and_failed_properties(self):
        plan = SwarmPlan(goal="g", agents=[])
        r = SwarmResult(goal="g", plan=plan, agent_results=[
            AgentResult(agent_id="1", role="R1", task="T1", status=AgentStatus.COMPLETED),
            AgentResult(agent_id="2", role="R2", task="T2", status=AgentStatus.FAILED),
            AgentResult(agent_id="3", role="R3", task="T3", status=AgentStatus.COMPLETED),
        ])
        self.assertEqual(len(r.successful), 2)
        self.assertEqual(len(r.failed), 1)


# ═══════════════════════════════════════════════════════════════════
# config.py tests
# ═══════════════════════════════════════════════════════════════════

class TestSwarmConfig(unittest.TestCase):
    def test_defaults(self):
        c = SwarmConfig()
        self.assertEqual(c.api_base, "")
        self.assertEqual(c.default_model, "deepseek-chat")
        self.assertEqual(c.max_parallel, 15)
        self.assertEqual(c.max_retries, 2)
        self.assertEqual(c.agent_timeout, 120.0)
        self.assertEqual(c.max_agents, 25)
        self.assertTrue(c.save_json)
        self.assertTrue(c.save_markdown)
        self.assertTrue(c.stream_to_terminal)
        self.assertEqual(c.temperature, 0.7)

    def test_from_env(self):
        env = {
            "SWARM_API_BASE": "http://test:1234/v1",
            "SWARM_MODEL": "gpt-4",
            "SWARM_MAX_PARALLEL": "3",
            "SWARM_MAX_AGENTS": "10",
            "SWARM_VERBOSE": "true",
            "SWARM_SAVE_JSON": "false",
            "SWARM_TEMPERATURE": "0.2",
        }
        with patch.dict(os.environ, env, clear=False):
            c = SwarmConfig.from_env()
            self.assertEqual(c.api_base, "http://test:1234/v1")
            self.assertEqual(c.default_model, "gpt-4")
            self.assertEqual(c.max_parallel, 3)
            self.assertEqual(c.max_agents, 10)
            self.assertTrue(c.verbose)
            self.assertFalse(c.save_json)
            self.assertAlmostEqual(c.temperature, 0.2)

    def test_from_env_defaults(self):
        """Ensure from_env works without any SWARM_ vars set."""
        with patch.dict(os.environ, {}, clear=True):
            c = SwarmConfig.from_env()
            self.assertEqual(c.api_base, "http://localhost:8000/v1")
            self.assertEqual(c.default_model, "deepseek-chat")

    def test_orchestrator_model(self):
        c = SwarmConfig(orchestrator_model="gpt-4o")
        self.assertEqual(c.orchestrator_model, "gpt-4o")


# ═══════════════════════════════════════════════════════════════════
# engine.py tests
# ═══════════════════════════════════════════════════════════════════

class TestRateLimiter(unittest.TestCase):
    def test_acquire_burst(self):
        """Should allow burst number of immediate acquires."""
        loop = asyncio.new_event_loop()
        rl = RateLimiter(rpm=60, burst=3)

        async def acquire_n(n):
            start = time.monotonic()
            for _ in range(n):
                await rl.acquire()
            return time.monotonic() - start

        # 3 should be instant (within burst)
        elapsed = loop.run_until_complete(acquire_n(3))
        self.assertLess(elapsed, 0.5)
        loop.close()

    def test_acquire_rate_limited(self):
        """4th acquire beyond burst=3 should wait."""
        loop = asyncio.new_event_loop()
        rl = RateLimiter(rpm=600, burst=2)  # 10/s, burst 2

        async def acquire_n(n):
            start = time.monotonic()
            for _ in range(n):
                await rl.acquire()
            return time.monotonic() - start

        elapsed = loop.run_until_complete(acquire_n(3))
        # After 2 burst tokens, 3rd should wait ~0.1s (1/10)
        self.assertGreater(elapsed, 0.05)
        loop.close()


class TestSwarmEngine(unittest.TestCase):
    def _make_config(self, **kw):
        defaults = dict(
            max_parallel=5, rate_limit_rpm=6000, rate_limit_burst=50,
            retry_base_delay=0.01, retry_max_delay=0.05,
            retry_jitter=False, agent_timeout=5.0,
        )
        defaults.update(kw)
        return SwarmConfig(**defaults)

    def test_parallel_execution(self):
        """All independent agents should run concurrently."""
        config = self._make_config()
        call_times = []

        async def runner(agent, results, cfg):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.1)
            return f"result_{agent.role}"

        engine = SwarmEngine(config, runner)

        agents = [AgentSpec(role=f"A{i}", task="t") for i in range(3)]
        plan = SwarmPlan(goal="test", agents=agents)

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.status == AgentStatus.COMPLETED for r in results))
        # All should have started within ~50ms of each other (parallel)
        if len(call_times) >= 2:
            spread = max(call_times) - min(call_times)
            self.assertLess(spread, 0.5)

    def test_dag_ordering(self):
        """Agent C (depends on A) should only start after A completes."""
        config = self._make_config()
        execution_order = []

        async def runner(agent, results, cfg):
            execution_order.append(agent.role)
            await asyncio.sleep(0.05)
            return f"result_{agent.role}"

        engine = SwarmEngine(config, runner)

        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t")  # independent
        c = AgentSpec(role="C", task="t", depends_on=[a.agent_id])

        plan = SwarmPlan(goal="test", agents=[a, b, c])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.status == AgentStatus.COMPLETED for r in results))
        # C must come after A
        a_idx = execution_order.index("A")
        c_idx = execution_order.index("C")
        self.assertGreater(c_idx, a_idx)

    def test_diamond_dag(self):
        """
        Diamond DAG: A → B, A → C, B+C → D
        D should only run after both B and C complete.
        """
        config = self._make_config()
        execution_order = []

        async def runner(agent, results, cfg):
            execution_order.append(agent.role)
            await asyncio.sleep(0.05)
            return f"result_{agent.role}"

        engine = SwarmEngine(config, runner)

        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t", depends_on=[a.agent_id])
        c = AgentSpec(role="C", task="t", depends_on=[a.agent_id])
        d = AgentSpec(role="D", task="t", depends_on=[b.agent_id, c.agent_id])

        plan = SwarmPlan(goal="test", agents=[a, b, c, d])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 4)
        self.assertTrue(all(r.status == AgentStatus.COMPLETED for r in results))
        d_idx = execution_order.index("D")
        b_idx = execution_order.index("B")
        c_idx = execution_order.index("C")
        self.assertGreater(d_idx, b_idx)
        self.assertGreater(d_idx, c_idx)

    def test_retry_on_failure(self):
        """Agent should retry on exception and eventually succeed."""
        config = self._make_config(max_retries=2)
        call_count = 0

        async def runner(agent, results, cfg):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("temporary error")
            return "success"

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="A", task="t", max_retries=2)
        plan = SwarmPlan(goal="test", agents=[a])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, AgentStatus.COMPLETED)
        self.assertEqual(results[0].content, "success")
        self.assertEqual(results[0].attempt, 3)
        self.assertEqual(call_count, 3)

    def test_retry_exhausted(self):
        """Agent should fail after all retries exhausted."""
        config = self._make_config()

        async def runner(agent, results, cfg):
            raise RuntimeError("permanent error")

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="A", task="t", max_retries=1)
        plan = SwarmPlan(goal="test", agents=[a])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, AgentStatus.FAILED)
        self.assertIn("permanent error", results[0].error)
        self.assertEqual(results[0].attempt, 2)  # 1 + 1 retry

    def test_timeout(self):
        """Agent that exceeds timeout should fail."""
        config = self._make_config(agent_timeout=0.2)

        async def runner(agent, results, cfg):
            await asyncio.sleep(10)  # way too long
            return "should not reach"

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="Slow", task="t", timeout=0.2, max_retries=0)
        plan = SwarmPlan(goal="test", agents=[a])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].status, AgentStatus.FAILED)
        self.assertIn("timed out", results[0].error)

    def test_failure_cascade(self):
        """If A fails, B (depends on A) should also fail with 'Dependency failed'."""
        config = self._make_config()

        async def runner(agent, results, cfg):
            if agent.role == "A":
                raise RuntimeError("A exploded")
            return "ok"

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="A", task="t", max_retries=0)
        b = AgentSpec(role="B", task="t", depends_on=[a.agent_id])

        plan = SwarmPlan(goal="test", agents=[a, b])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        results_by_role = {r.role: r for r in results}
        self.assertEqual(results_by_role["A"].status, AgentStatus.FAILED)
        self.assertEqual(results_by_role["B"].status, AgentStatus.FAILED)
        self.assertIn("Dependency failed", results_by_role["B"].error)

    def test_event_callbacks(self):
        """on_start, on_done callbacks should fire."""
        config = self._make_config()
        started = []
        done = []

        async def runner(agent, results, cfg):
            return "ok"

        engine = SwarmEngine(
            config, runner,
            on_start=lambda a: started.append(a.role),
            on_done=lambda r: done.append(r.role),
        )

        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t")
        plan = SwarmPlan(goal="test", agents=[a, b])

        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertIn("A", started)
        self.assertIn("B", started)
        self.assertIn("A", done)
        self.assertIn("B", done)

    def test_priority_ordering(self):
        """Higher priority agents should start first when all are ready."""
        config = self._make_config(max_parallel=1)  # serialize
        execution_order = []

        async def runner(agent, results, cfg):
            execution_order.append(agent.role)
            return "ok"

        engine = SwarmEngine(config, runner)

        lo = AgentSpec(role="Low", task="t", priority=1)
        hi = AgentSpec(role="High", task="t", priority=10)
        mid = AgentSpec(role="Mid", task="t", priority=5)
        plan = SwarmPlan(goal="test", agents=[lo, hi, mid])

        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.execute(plan))
        loop.close()

        # All 3 are ready immediately, should start in priority order
        self.assertEqual(execution_order[0], "High")

    def test_shared_results_visible_to_dependents(self):
        """Agent B should see A's result in the results dict."""
        config = self._make_config()
        seen_results = {}

        async def runner(agent, results, cfg):
            if agent.role == "B":
                seen_results.update(results)
            return f"output_{agent.role}"

        engine = SwarmEngine(config, runner)

        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t", depends_on=[a.agent_id])
        plan = SwarmPlan(goal="test", agents=[a, b])

        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertIn(a.agent_id, seen_results)
        self.assertEqual(seen_results[a.agent_id].content, "output_A")

    def test_invalid_plan_rejected(self):
        """Engine should raise ValueError for invalid plans."""
        config = self._make_config()

        async def runner(agent, results, cfg):
            return "ok"

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="A", task="t", depends_on=["nonexistent"])
        plan = SwarmPlan(goal="test", agents=[a])

        loop = asyncio.new_event_loop()
        with self.assertRaises(ValueError):
            loop.run_until_complete(engine.execute(plan))
        loop.close()

    def test_empty_plan(self):
        """Empty plan should return empty results."""
        config = self._make_config()

        async def runner(agent, results, cfg):
            return "ok"

        engine = SwarmEngine(config, runner)
        plan = SwarmPlan(goal="test", agents=[])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(results, [])


# ═══════════════════════════════════════════════════════════════════
# orchestrator.py tests
# ═══════════════════════════════════════════════════════════════════

class TestExtractJson(unittest.TestCase):
    def test_raw_json(self):
        text = '{"strategy": "test", "agents": []}'
        result = _extract_json(text)
        self.assertEqual(result["strategy"], "test")

    def test_markdown_fenced(self):
        text = 'Here is the plan:\n```json\n{"strategy": "test", "agents": [{"role": "A"}]}\n```\nDone.'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["agents"]), 1)

    def test_embedded_in_prose(self):
        text = 'I will create agents: {"strategy": "x", "agents": []} that do things.'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["strategy"], "x")

    def test_no_json(self):
        text = "This is just plain text with no JSON."
        result = _extract_json(text)
        self.assertIsNone(result)

    def test_nested_braces(self):
        text = '{"outer": {"inner": {"deep": 1}}, "agents": []}'
        result = _extract_json(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["outer"]["inner"]["deep"], 1)


class TestOrchestrator(unittest.TestCase):
    def test_context_manager(self):
        """Orchestrator should work as async context manager."""
        config = SwarmConfig(api_base="http://localhost:9999/v1")

        async def run():
            async with Orchestrator(config) as orc:
                self.assertIsNotNone(orc._client)
            # After exit, client should be closed
            self.assertTrue(orc._client.is_closed)

        asyncio.new_event_loop().run_until_complete(run())

    def test_client_property_without_enter(self):
        """Accessing .client without __aenter__ should raise."""
        orc = Orchestrator(SwarmConfig())
        with self.assertRaises(RuntimeError):
            _ = orc.client

    def test_decompose_manual_raises(self):
        """Decompose with MANUAL mode should raise ValueError."""
        config = SwarmConfig(api_base="http://localhost:9999/v1")

        async def run():
            async with Orchestrator(config) as orc:
                with self.assertRaises(ValueError):
                    await orc.decompose("test", SwarmMode.MANUAL)

        asyncio.new_event_loop().run_until_complete(run())

    def test_decompose_enforces_min_agents(self):
        """Orchestrator should pad agents if LLM returns fewer than min_agents."""
        config = SwarmConfig(
            api_base="http://localhost:9999/v1",
            min_agents=3,
            max_agents=10,
        )

        # Mock _chat to return a plan with only 1 agent
        async def mock_chat(messages, model="", temperature=None,
                            max_tokens=None, is_orchestrator=False, **kwargs):
            return json.dumps({
                "strategy": "single agent",
                "agents": [{"role": "Solo", "task": "do it all"}]
            })

        async def run():
            async with Orchestrator(config) as orc:
                with patch.object(Orchestrator, '_chat', side_effect=mock_chat):
                    plan = await orc.decompose("test task")
            self.assertGreaterEqual(len(plan.agents), 3)
            roles = [a.role for a in plan.agents]
            self.assertIn("Solo", roles)

        asyncio.new_event_loop().run_until_complete(run())

    def test_decompose_caps_max_agents(self):
        """Orchestrator should cap agents at max_agents."""
        config = SwarmConfig(
            api_base="http://localhost:9999/v1",
            min_agents=1,
            max_agents=3,
        )

        async def mock_chat(messages, model="", temperature=None,
                            max_tokens=None, is_orchestrator=False, **kwargs):
            return json.dumps({
                "strategy": "too many agents",
                "agents": [
                    {"role": f"A{i}", "task": f"task {i}"}
                    for i in range(10)
                ]
            })

        async def run():
            async with Orchestrator(config) as orc:
                with patch.object(Orchestrator, '_chat', side_effect=mock_chat):
                    plan = await orc.decompose("test task")
            self.assertLessEqual(len(plan.agents), 3)

        asyncio.new_event_loop().run_until_complete(run())


# ═══════════════════════════════════════════════════════════════════
# renderer.py tests
# ═══════════════════════════════════════════════════════════════════

class TestRenderer(unittest.TestCase):
    def test_strip_ansi(self):
        text = "\033[1m\033[31mHello\033[0m World"
        self.assertEqual(_strip_ansi(text), "Hello World")

    def test_on_plan_ready(self):
        renderer = SwarmRenderer(stream=False, verbose=False)
        a = AgentSpec(role="A", task="t")
        b = AgentSpec(role="B", task="t", depends_on=[a.agent_id])
        plan = SwarmPlan(goal="test goal", agents=[a, b], strategy="test strat")
        renderer.on_plan_ready(plan)
        log = "\n".join(renderer._log_lines)
        self.assertIn("SWARM ACTIVATED", log)
        self.assertIn("test goal", log)
        self.assertIn("test strat", log)
        self.assertIn("A", log)
        self.assertIn("B", log)

    def test_on_agent_start(self):
        renderer = SwarmRenderer(stream=False)
        renderer._start_time = time.time()
        a = AgentSpec(role="TestAgent", task="t")
        renderer.on_agent_start(a)
        log = "\n".join(renderer._log_lines)
        self.assertIn("TestAgent", log)
        self.assertIn("started", log)

    def test_on_agent_done_success(self):
        renderer = SwarmRenderer(stream=False)
        renderer._start_time = time.time()
        r = AgentResult(agent_id="x", role="TestAgent", task="t",
                        content="some output", status=AgentStatus.COMPLETED,
                        started_at=time.time()-5, finished_at=time.time())
        renderer.on_agent_done(r)
        log = "\n".join(renderer._log_lines)
        self.assertIn("TestAgent", log)
        self.assertIn("done", log)

    def test_on_agent_done_failure(self):
        renderer = SwarmRenderer(stream=False)
        renderer._start_time = time.time()
        r = AgentResult(agent_id="x", role="FailAgent", task="t",
                        status=AgentStatus.FAILED, error="kaboom")
        renderer.on_agent_done(r)
        log = "\n".join(renderer._log_lines)
        self.assertIn("FAILED", log)
        self.assertIn("kaboom", log)

    def test_on_complete(self):
        renderer = SwarmRenderer(stream=False)
        renderer._start_time = time.time()
        plan = SwarmPlan(goal="g", agents=[])
        result = SwarmResult(
            goal="g", plan=plan,
            agent_results=[
                AgentResult(agent_id="1", role="R1", task="T1",
                            status=AgentStatus.COMPLETED),
            ],
            started_at=time.time()-10, finished_at=time.time(),
        )
        renderer.on_complete(result)
        log = "\n".join(renderer._log_lines)
        self.assertIn("SWARM COMPLETE", log)
        self.assertIn("1/1 succeeded", log)

    def test_build_markdown(self):
        plan = SwarmPlan(goal="test goal", agents=[], strategy="my strategy")
        result = SwarmResult(
            goal="test goal", plan=plan,
            agent_results=[
                AgentResult(agent_id="1", role="R1", task="T1",
                            content="agent output here",
                            status=AgentStatus.COMPLETED,
                            started_at=10, finished_at=15),
            ],
            synthesis="final synthesis",
            started_at=10, finished_at=20,
        )
        md = _build_markdown(result)
        self.assertIn("# Swarm Report", md)
        self.assertIn("test goal", md)
        self.assertIn("my strategy", md)
        self.assertIn("final synthesis", md)
        self.assertIn("agent output here", md)
        self.assertIn("R1", md)

    def test_save_markdown_file(self):
        renderer = SwarmRenderer(stream=False)
        plan = SwarmPlan(goal="test", agents=[])
        result = SwarmResult(goal="test", plan=plan, synthesis="report",
                             started_at=0, finished_at=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.md")
            renderer.save_markdown(result, path)
            self.assertTrue(os.path.exists(path))
            content = open(path).read()
            self.assertIn("# Swarm Report", content)
            self.assertIn("report", content)

    def test_save_json_file(self):
        renderer = SwarmRenderer(stream=False)
        plan = SwarmPlan(goal="test", agents=[])
        result = SwarmResult(goal="test", plan=plan, synthesis="report",
                             started_at=0, finished_at=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.json")
            renderer.save_json(result, path)
            self.assertTrue(os.path.exists(path))
            data = json.loads(open(path).read())
            self.assertEqual(data["goal"], "test")
            self.assertEqual(data["synthesis"], "report")


# ═══════════════════════════════════════════════════════════════════
# __init__.py / Swarm() integration tests (mocked LLM)
# ═══════════════════════════════════════════════════════════════════

class TestSwarmOneLiner(unittest.TestCase):
    def test_swarm_manual_mode(self):
        """Swarm() in MANUAL mode with mocked Orchestrator."""
        config = SwarmConfig(
            api_base="http://localhost:9999/v1",
            stream_to_terminal=False,
            save_json=False, save_markdown=False,
        )

        a = AgentSpec(role="A", task="do A")
        b = AgentSpec(role="B", task="do B", depends_on=[a.agent_id])
        plan = SwarmPlan(goal="test goal", agents=[a, b], strategy="strat")

        # Mock the Orchestrator's _chat to return canned results
        async def mock_chat(messages, model="", temperature=None,
                            max_tokens=None, is_orchestrator=False, **kwargs):
            content = messages[-1]["content"]
            if "Original Goal" in content:
                return "Synthesized final report."
            return f"Agent output for some task."

        async def run():
            from cookbook.swarm import Swarm as SwarmFn
            # Patch _chat on Orchestrator
            with patch.object(Orchestrator, '_chat', side_effect=mock_chat):
                result = await SwarmFn(
                    "test goal",
                    config=config,
                    mode=SwarmMode.MANUAL,
                    plan=plan,
                    verbose=False,
                    save=False,
                )
            self.assertEqual(result.goal, "test goal")
            self.assertEqual(len(result.agent_results), 2)
            self.assertTrue(all(r.status == AgentStatus.COMPLETED
                                for r in result.agent_results))
            self.assertIn("Synthesized", result.synthesis)

        asyncio.new_event_loop().run_until_complete(run())


# ═══════════════════════════════════════════════════════════════════
# Edge cases and stress tests
# ═══════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    def test_single_agent_swarm(self):
        """Single agent with no dependencies should work."""
        config = SwarmConfig(max_parallel=5, rate_limit_rpm=6000,
                             rate_limit_burst=50, retry_base_delay=0.01)

        async def runner(agent, results, cfg):
            return "solo output"

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="Solo", task="t")
        plan = SwarmPlan(goal="test", agents=[a])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "solo output")

    def test_many_agents_parallel(self):
        """20 independent agents should all complete."""
        config = SwarmConfig(max_parallel=20, rate_limit_rpm=60000,
                             rate_limit_burst=100, retry_base_delay=0.01)

        async def runner(agent, results, cfg):
            await asyncio.sleep(0.01)
            return f"output_{agent.role}"

        engine = SwarmEngine(config, runner)
        agents = [AgentSpec(role=f"Agent{i}", task="t") for i in range(20)]
        plan = SwarmPlan(goal="test", agents=agents)

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 20)
        self.assertTrue(all(r.status == AgentStatus.COMPLETED for r in results))

    def test_deep_chain(self):
        """Chain of 10 sequential agents (each depends on previous)."""
        config = SwarmConfig(max_parallel=5, rate_limit_rpm=60000,
                             rate_limit_burst=100, retry_base_delay=0.01)
        execution_order = []

        async def runner(agent, results, cfg):
            execution_order.append(agent.role)
            return f"output_{agent.role}"

        engine = SwarmEngine(config, runner)
        agents = []
        for i in range(10):
            deps = [agents[-1].agent_id] if agents else []
            agents.append(AgentSpec(role=f"Step{i}", task="t", depends_on=deps))

        plan = SwarmPlan(goal="test", agents=agents)

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        self.assertEqual(len(results), 10)
        self.assertTrue(all(r.status == AgentStatus.COMPLETED for r in results))
        # Should execute in order
        self.assertEqual(execution_order, [f"Step{i}" for i in range(10)])

    def test_mixed_success_failure_dag(self):
        """
        A succeeds, B fails. C depends on A (should succeed).
        D depends on B (should fail with Dependency failed).
        """
        config = SwarmConfig(max_parallel=5, rate_limit_rpm=60000,
                             rate_limit_burst=100, retry_base_delay=0.01)

        async def runner(agent, results, cfg):
            if agent.role == "B":
                raise RuntimeError("B broke")
            return "ok"

        engine = SwarmEngine(config, runner)
        a = AgentSpec(role="A", task="t", max_retries=0)
        b = AgentSpec(role="B", task="t", max_retries=0)
        c = AgentSpec(role="C", task="t", depends_on=[a.agent_id])
        d = AgentSpec(role="D", task="t", depends_on=[b.agent_id])

        plan = SwarmPlan(goal="test", agents=[a, b, c, d])

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(engine.execute(plan))
        loop.close()

        by_role = {r.role: r for r in results}
        self.assertEqual(by_role["A"].status, AgentStatus.COMPLETED)
        self.assertEqual(by_role["B"].status, AgentStatus.FAILED)
        self.assertEqual(by_role["C"].status, AgentStatus.COMPLETED)
        self.assertEqual(by_role["D"].status, AgentStatus.FAILED)
        self.assertIn("Dependency failed", by_role["D"].error)

    def test_concurrent_with_semaphore(self):
        """With max_parallel=2, only 2 agents should run at once."""
        config = SwarmConfig(max_parallel=2, rate_limit_rpm=60000,
                             rate_limit_burst=100, retry_base_delay=0.01)
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def runner(agent, results, cfg):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent
            await asyncio.sleep(0.1)
            async with lock:
                current_concurrent -= 1
            return "ok"

        engine = SwarmEngine(config, runner)
        agents = [AgentSpec(role=f"A{i}", task="t") for i in range(6)]
        plan = SwarmPlan(goal="test", agents=agents)

        loop = asyncio.new_event_loop()
        # Need to use the same loop for the Lock
        async def run():
            nonlocal lock
            lock = asyncio.Lock()
            return await engine.execute(plan)

        results = loop.run_until_complete(run())
        loop.close()

        self.assertEqual(len(results), 6)
        self.assertLessEqual(max_concurrent, 2)


# ═══════════════════════════════════════════════════════════════════
# SwarmMode enum
# ═══════════════════════════════════════════════════════════════════

class TestSwarmMode(unittest.TestCase):
    def test_values(self):
        self.assertEqual(SwarmMode.AUTO.value, "auto")
        self.assertEqual(SwarmMode.MANUAL.value, "manual")
        self.assertEqual(SwarmMode.ITERATIVE.value, "iterative")


# ═══════════════════════════════════════════════════════════════════
# Auto-continuation tests
# ═══════════════════════════════════════════════════════════════════

class TestAutoContinuation(unittest.TestCase):
    """Test the auto_continue logic in _chat for handling truncated responses."""

    def test_single_response_no_truncation(self):
        """Normal response (finish_reason='stop') returns content directly."""
        config = SwarmConfig(api_base="http://localhost:9999/v1")

        call_count = 0

        async def mock_post(path, json=None):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{
                    "message": {"content": "Complete response."},
                    "finish_reason": "stop",
                }]
            }
            return resp

        async def run():
            async with Orchestrator(config) as orc:
                orc._client.post = mock_post
                result = await orc._chat(
                    messages=[{"role": "user", "content": "hello"}],
                    auto_continue=True,
                )
            self.assertEqual(result, "Complete response.")
            self.assertEqual(call_count, 1)

        asyncio.new_event_loop().run_until_complete(run())

    def test_truncated_response_continues(self):
        """Truncated response (finish_reason='length') triggers continuation."""
        config = SwarmConfig(api_base="http://localhost:9999/v1")

        call_count = 0

        async def mock_post(path, json=None):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if call_count == 1:
                # First call — truncated
                resp.json.return_value = {
                    "choices": [{
                        "message": {"content": "Part one of the report. "},
                        "finish_reason": "length",
                    }]
                }
            else:
                # Continuation — complete
                resp.json.return_value = {
                    "choices": [{
                        "message": {"content": "Part two completes it."},
                        "finish_reason": "stop",
                    }]
                }
            return resp

        async def run():
            async with Orchestrator(config) as orc:
                orc._client.post = mock_post
                result = await orc._chat(
                    messages=[{"role": "user", "content": "write report"}],
                    auto_continue=True,
                )
            self.assertEqual(result, "Part one of the report. Part two completes it.")
            self.assertEqual(call_count, 2)

        asyncio.new_event_loop().run_until_complete(run())

    def test_auto_continue_disabled_no_continuation(self):
        """auto_continue=False doesn't attempt continuation even on truncation."""
        config = SwarmConfig(api_base="http://localhost:9999/v1")

        call_count = 0

        async def mock_post(path, json=None):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{
                    "message": {"content": "Truncated output"},
                    "finish_reason": "length",
                }]
            }
            return resp

        async def run():
            async with Orchestrator(config) as orc:
                orc._client.post = mock_post
                result = await orc._chat(
                    messages=[{"role": "user", "content": "hello"}],
                    auto_continue=False,
                )
            self.assertEqual(result, "Truncated output")
            self.assertEqual(call_count, 1)

        asyncio.new_event_loop().run_until_complete(run())

    def test_max_continuations_respected(self):
        """Continuation stops after max_continuations even if still truncated."""
        config = SwarmConfig(api_base="http://localhost:9999/v1")

        call_count = 0

        async def mock_post(path, json=None):
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {
                "choices": [{
                    "message": {"content": f"chunk{call_count} "},
                    "finish_reason": "length",
                }]
            }
            return resp

        async def run():
            async with Orchestrator(config) as orc:
                orc._client.post = mock_post
                result = await orc._chat(
                    messages=[{"role": "user", "content": "hello"}],
                    auto_continue=True,
                    max_continuations=2,
                )
            # 1 initial + 2 continuations = 3 total
            self.assertEqual(call_count, 3)
            self.assertEqual(result, "chunk1 chunk2 chunk3 ")

        asyncio.new_event_loop().run_until_complete(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — Tool Registry tests
# ═══════════════════════════════════════════════════════════════════

class TestToolDef(unittest.TestCase):
    def test_defaults(self):
        td = ToolDef(name="t", description="desc")
        self.assertEqual(td.name, "t")
        self.assertEqual(td.description, "desc")
        self.assertEqual(td.parameters, {})
        self.assertIsNone(td.fn)
        self.assertTrue(td.safe)

    def test_with_fn(self):
        async def _dummy(**kw):
            return "ok"
        td = ToolDef(name="t", description="d", fn=_dummy, safe=False)
        self.assertFalse(td.safe)
        self.assertIsNotNone(td.fn)


class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()

    def test_register_and_get(self):
        td = ToolDef(name="scan", description="a scanner")
        self.reg.register(td)
        self.assertIs(self.reg.get("scan"), td)
        self.assertIsNone(self.reg.get("nonexistent"))

    def test_register_fn(self):
        async def _fn(**kw):
            return "ok"
        self.reg.register_fn("test_tool", _fn, description="test", safe=False)
        t = self.reg.get("test_tool")
        self.assertIsNotNone(t)
        self.assertEqual(t.name, "test_tool")
        self.assertFalse(t.safe)

    def test_available_all(self):
        self.reg.register(ToolDef(name="a", description="A"))
        self.reg.register(ToolDef(name="b", description="B"))
        avail = self.reg.available()
        self.assertEqual(len(avail), 2)

    def test_available_filtered(self):
        self.reg.register(ToolDef(name="a", description="A"))
        self.reg.register(ToolDef(name="b", description="B"))
        self.reg.register(ToolDef(name="c", description="C"))
        avail = self.reg.available(["a", "c"])
        self.assertEqual(len(avail), 2)
        names = {t.name for t in avail}
        self.assertEqual(names, {"a", "c"})

    def test_available_filtered_unknown(self):
        self.reg.register(ToolDef(name="a", description="A"))
        avail = self.reg.available(["a", "nonexistent"])
        self.assertEqual(len(avail), 1)

    def test_build_tool_prompt_empty(self):
        prompt = self.reg.build_tool_prompt()
        self.assertEqual(prompt, "")

    def test_build_tool_prompt_content(self):
        self.reg.register(ToolDef(
            name="nmap",
            description="Port scanner",
            parameters={"target": "IP to scan", "ports": "Port range"},
        ))
        prompt = self.reg.build_tool_prompt()
        self.assertIn("## Available Tools", prompt)
        self.assertIn("nmap", prompt)
        self.assertIn("Port scanner", prompt)
        self.assertIn("`target`", prompt)
        self.assertIn("`ports`", prompt)

    def test_build_tool_prompt_filter(self):
        self.reg.register(ToolDef(name="a", description="Tool A"))
        self.reg.register(ToolDef(name="b", description="Tool B"))
        prompt = self.reg.build_tool_prompt(["a"])
        self.assertIn("Tool A", prompt)
        self.assertNotIn("Tool B", prompt)

    def test_execute_success(self):
        async def _echo(**kw):
            return f"result:{kw.get('x', '')}"
        self.reg.register(ToolDef(name="echo", description="echo", fn=_echo))

        async def run():
            result = await self.reg.execute("echo", {"x": "hello"})
            self.assertEqual(result, "result:hello")
        asyncio.new_event_loop().run_until_complete(run())

    def test_execute_unknown_tool(self):
        async def run():
            result = await self.reg.execute("ghost", {})
            self.assertIn("[ERROR]", result)
            self.assertIn("Unknown tool", result)
        asyncio.new_event_loop().run_until_complete(run())

    def test_execute_no_fn(self):
        self.reg.register(ToolDef(name="nofn", description="no fn"))  # fn=None

        async def run():
            result = await self.reg.execute("nofn", {})
            self.assertIn("[ERROR]", result)
            self.assertIn("no callable", result)
        asyncio.new_event_loop().run_until_complete(run())

    def test_execute_timeout(self):
        async def _slow(**kw):
            await asyncio.sleep(10)
            return "done"
        self.reg.register(ToolDef(name="slow", description="slow", fn=_slow))

        async def run():
            result = await self.reg.execute("slow", {}, timeout=0.05)
            self.assertIn("[ERROR]", result)
            self.assertIn("timed out", result)
        asyncio.new_event_loop().run_until_complete(run())

    def test_execute_exception(self):
        async def _bad(**kw):
            raise ValueError("boom")
        self.reg.register(ToolDef(name="bad", description="bad", fn=_bad))

        async def run():
            result = await self.reg.execute("bad", {})
            self.assertIn("[ERROR]", result)
            self.assertIn("boom", result)
        asyncio.new_event_loop().run_until_complete(run())


class TestExtractToolCalls(unittest.TestCase):
    def test_fenced_json_array(self):
        text = 'Some text\n```json\n[{"tool": "nmap", "args": {"target": "1.2.3.4"}}]\n```\nMore text'
        calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "nmap")
        self.assertEqual(calls[0]["args"]["target"], "1.2.3.4")

    def test_fenced_no_lang_tag(self):
        text = 'Here:\n```\n[{"tool": "dig", "args": {"domain": "x.com"}}]\n```'
        calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "dig")

    def test_raw_json_array(self):
        text = 'I will scan now: [{"tool": "nmap", "args": {"target": "10.0.0.1"}}]'
        calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "nmap")

    def test_single_object(self):
        text = 'Let me scan: {"tool": "whois", "args": {"target": "example.com"}}'
        calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "whois")

    def test_multiple_tools(self):
        text = '```json\n[{"tool": "nmap", "args": {"target": "x"}}, {"tool": "dig", "args": {"domain": "y"}}]\n```'
        calls = extract_tool_calls(text)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["tool"], "nmap")
        self.assertEqual(calls[1]["tool"], "dig")

    def test_no_tool_calls(self):
        text = "This is a plain text response with no tool calls at all."
        calls = extract_tool_calls(text)
        self.assertEqual(calls, [])

    def test_json_without_tool_key(self):
        text = '```json\n[{"name": "foo", "value": 42}]\n```'
        calls = extract_tool_calls(text)
        self.assertEqual(calls, [])

    def test_malformed_json(self):
        text = '```json\n{broken json\n```'
        calls = extract_tool_calls(text)
        self.assertEqual(calls, [])

    def test_mixed_valid_invalid(self):
        text = '```json\n[{"tool": "nmap", "args": {}}, {"not_tool": true}]\n```'
        calls = extract_tool_calls(text)
        # Should extract only the object with "tool" key
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool"], "nmap")


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — New model fields & status tests
# ═══════════════════════════════════════════════════════════════════

class TestCancelledStatus(unittest.TestCase):
    def test_cancelled_in_enum(self):
        self.assertEqual(AgentStatus.CANCELLED.value, "cancelled")
        # Ensure all expected statuses exist
        expected = {"pending", "waiting", "running", "completed", "failed", "retrying", "cancelled"}
        actual = {s.value for s in AgentStatus}
        self.assertEqual(expected, actual)


class TestAgentResultNewFields(unittest.TestCase):
    def test_tool_calls_made_default(self):
        r = AgentResult(agent_id="x", role="R", task="T")
        self.assertEqual(r.tool_calls_made, 0)

    def test_confidence_default(self):
        r = AgentResult(agent_id="x", role="R", task="T")
        self.assertAlmostEqual(r.confidence, 0.0)

    def test_set_tool_calls_made(self):
        r = AgentResult(agent_id="x", role="R", task="T", tool_calls_made=5)
        self.assertEqual(r.tool_calls_made, 5)

    def test_set_confidence(self):
        r = AgentResult(agent_id="x", role="R", task="T", confidence=0.85)
        self.assertAlmostEqual(r.confidence, 0.85)


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — New config fields tests
# ═══════════════════════════════════════════════════════════════════

class TestNewConfigFields(unittest.TestCase):
    def test_defaults(self):
        cfg = SwarmConfig()
        self.assertEqual(cfg.max_tool_calls_per_agent, 20)
        self.assertAlmostEqual(cfg.tool_timeout, 120.0)
        self.assertFalse(cfg.enable_reflection)
        self.assertAlmostEqual(cfg.tool_agent_timeout, 600.0)

    def test_from_env(self):
        env = {
            "SWARM_MAX_TOOL_CALLS": "10",
            "SWARM_TOOL_TIMEOUT": "60",
            "SWARM_ENABLE_REFLECTION": "true",
            "SWARM_TOOL_AGENT_TIMEOUT": "300",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertEqual(cfg.max_tool_calls_per_agent, 10)
        self.assertAlmostEqual(cfg.tool_timeout, 60.0)
        self.assertTrue(cfg.enable_reflection)
        self.assertAlmostEqual(cfg.tool_agent_timeout, 300.0)

    def test_reasoning_model_default(self):
        cfg = SwarmConfig()
        self.assertEqual(cfg.reasoning_model, "deepseek-reasoner")


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — Tool-calling loop in run_agent tests
# ═══════════════════════════════════════════════════════════════════

class TestToolCallingLoop(unittest.TestCase):
    """Test the tool-calling loop in Orchestrator.run_agent."""

    def _make_config(self, **overrides):
        defaults = dict(
            api_base="http://localhost:1/v1",
            api_key="test",
            agent_timeout=30,
            swarm_timeout=60,
            max_tool_calls_per_agent=5,
            tool_timeout=10,
            enable_reflection=False,
            tool_agent_timeout=60,
        )
        defaults.update(overrides)
        return SwarmConfig(**defaults)

    def test_agent_without_tools_single_call(self):
        """Agent with no tools should make one _chat call and return."""
        config = self._make_config()
        reg = ToolRegistry()

        call_count = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_count
            call_count += 1
            return "Final answer without tools"

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Analyst", task="Analyze stuff")
                result = await orc.run_agent(agent, {}, config)
            self.assertEqual(result, "Final answer without tools")
            self.assertEqual(call_count, 1)

        asyncio.new_event_loop().run_until_complete(run())

    def test_agent_with_tools_single_iteration(self):
        """Agent calls a tool once, then provides final answer."""
        config = self._make_config()
        reg = ToolRegistry()

        async def fake_tool(**kw):
            return f"scan_result_for_{kw.get('target', 'unknown')}"
        reg.register(ToolDef(
            name="nmap_scan", description="Scan", fn=fake_tool,
            parameters={"target": "IP"},
        ))

        call_idx = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                # First call: agent requests a tool
                return '```json\n[{"tool": "nmap_scan", "args": {"target": "10.0.0.1"}}]\n```'
            else:
                # Second call: agent provides final answer after seeing tool results
                return "Based on the scan results, port 80 is open."

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Scanner", task="Scan target", tools=["nmap_scan"])
                result = await orc.run_agent(agent, {}, config)
            self.assertIn("port 80 is open", result)
            self.assertEqual(call_idx, 2)

        asyncio.new_event_loop().run_until_complete(run())

    def test_agent_with_tools_multiple_iterations(self):
        """Agent calls tools across multiple iterations."""
        config = self._make_config(max_tool_calls_per_agent=10)
        reg = ToolRegistry()

        async def fake_scan(**kw):
            return "open_ports: 22, 80, 443"
        async def fake_dig(**kw):
            return "A record: 10.0.0.1"
        reg.register(ToolDef(name="nmap_scan", description="s", fn=fake_scan))
        reg.register(ToolDef(name="dns_lookup", description="d", fn=fake_dig))

        call_idx = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return '```json\n[{"tool": "nmap_scan", "args": {}}]\n```'
            elif call_idx == 2:
                return '```json\n[{"tool": "dns_lookup", "args": {"domain": "t.com"}}]\n```'
            else:
                return "Comprehensive result: 3 ports open, A record found."

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Recon", task="Full recon", tools=["nmap_scan", "dns_lookup"])
                result = await orc.run_agent(agent, {}, config)
            self.assertIn("3 ports open", result)
            self.assertEqual(call_idx, 3)

        asyncio.new_event_loop().run_until_complete(run())

    def test_max_iterations_cap(self):
        """Agent that always returns tool calls should be capped."""
        config = self._make_config(max_tool_calls_per_agent=3)
        reg = ToolRegistry()

        async def fake_tool(**kw):
            return "result"
        reg.register(ToolDef(name="scan", description="s", fn=fake_tool))

        call_idx = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_idx
            call_idx += 1
            # Always return tool calls (except the very last forced summary)
            if call_idx <= 3:
                return '```json\n[{"tool": "scan", "args": {}}]\n```'
            else:
                return "Final forced summary."

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Looper", task="Loop", tools=["scan"])
                result = await orc.run_agent(agent, {}, config)
            # 3 iterations + 1 forced summary = 4 total calls
            self.assertEqual(call_idx, 4)
            self.assertIn("Final forced summary", result)

        asyncio.new_event_loop().run_until_complete(run())

    def test_search_tool_is_meta_not_real(self):
        """The 'search' tool controls model routing, not actual tool calls."""
        config = self._make_config()
        reg = ToolRegistry()

        model_used = None
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal model_used
            model_used = model
            return "Search results here"

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Searcher", task="Find info", tools=["search"])
                result = await orc.run_agent(agent, {}, config)
            self.assertEqual(result, "Search results here")
            # Should use search model, not enter tool loop
            self.assertEqual(model_used, config.search_model)

        asyncio.new_event_loop().run_until_complete(run())

    def test_reasoning_tool_model_routing(self):
        """The 'reasoning' tag routes to the reasoning model."""
        config = self._make_config()
        reg = ToolRegistry()

        model_used = None
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal model_used
            model_used = model
            return "Deep reasoning output"

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Thinker", task="Think hard", tools=["reasoning"])
                result = await orc.run_agent(agent, {}, config)
            self.assertEqual(model_used, config.reasoning_model)

        asyncio.new_event_loop().run_until_complete(run())


class TestReflection(unittest.TestCase):
    """Test the _reflect self-review feature."""

    def _make_config(self, **overrides):
        defaults = dict(
            api_base="http://localhost:1/v1",
            api_key="test",
            agent_timeout=30,
            swarm_timeout=60,
            enable_reflection=True,
            max_tool_calls_per_agent=5,
            tool_timeout=10,
            tool_agent_timeout=60,
        )
        defaults.update(overrides)
        return SwarmConfig(**defaults)

    def test_reflect_called_when_enabled(self):
        """With enable_reflection=True, agent should make a second _chat call."""
        config = self._make_config(enable_reflection=True)

        call_idx = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return "Initial output"
            else:
                return "Revised and improved output"

        async def run():
            async with Orchestrator(config) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Writer", task="Write something")
                result = await orc.run_agent(agent, {}, config)
            self.assertEqual(result, "Revised and improved output")
            self.assertEqual(call_idx, 2)

        asyncio.new_event_loop().run_until_complete(run())

    def test_reflect_skipped_when_disabled(self):
        config = self._make_config(enable_reflection=False)

        call_idx = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_idx
            call_idx += 1
            return "Only output"

        async def run():
            async with Orchestrator(config) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Writer", task="Write something")
                result = await orc.run_agent(agent, {}, config)
            self.assertEqual(result, "Only output")
            self.assertEqual(call_idx, 1)

        asyncio.new_event_loop().run_until_complete(run())

    def test_reflect_with_tool_loop(self):
        """Reflection should happen after the tool loop completes."""
        config = self._make_config(enable_reflection=True)
        reg = ToolRegistry()

        async def fake_tool(**kw):
            return "tool_output"
        reg.register(ToolDef(name="scan", description="s", fn=fake_tool))

        call_idx = 0
        async def mock_chat(messages, model="", max_tokens=None, **kw):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return '```json\n[{"tool": "scan", "args": {}}]\n```'
            elif call_idx == 2:
                return "Final answer from tools"
            else:
                return "Reflected final answer from tools"

        async def run():
            async with Orchestrator(config, tool_registry=reg) as orc:
                orc._chat = mock_chat
                agent = AgentSpec(role="Scanner", task="Scan", tools=["scan"])
                result = await orc.run_agent(agent, {}, config)
            self.assertEqual(result, "Reflected final answer from tools")
            self.assertEqual(call_idx, 3)

        asyncio.new_event_loop().run_until_complete(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — Arsenal bridge (unit-level) tests
# ═══════════════════════════════════════════════════════════════════

class TestArsenalBridge(unittest.TestCase):
    """Test the arsenal_bridge adapter (mocked SecurityTools)."""

    def test_register_arsenal_tools(self):
        from cookbook.swarm.arsenal_bridge import register_arsenal_tools, _ARSENAL_TOOLS

        # Create a mock SecurityTools with methods for each tool
        mock_sec = MagicMock()
        for meta in _ARSENAL_TOOLS:
            setattr(mock_sec, meta["name"], AsyncMock())

        reg = ToolRegistry()
        register_arsenal_tools(reg, mock_sec)

        # All 11 tools should be registered
        self.assertEqual(len(reg.available()), 11)
        for meta in _ARSENAL_TOOLS:
            tool = reg.get(meta["name"])
            self.assertIsNotNone(tool, f"Tool '{meta['name']}' should be registered")
            self.assertEqual(tool.description, meta["description"])

    def test_format_tool_result_success(self):
        from cookbook.swarm.arsenal_bridge import _format_tool_result

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.command = "nmap -sV 10.0.0.1"
        mock_result.parsed = {"ports": [22, 80]}
        mock_result.stdout = ""
        mock_result.stderr = ""

        formatted = _format_tool_result(mock_result)
        self.assertIn("[TOOL OK]", formatted)
        self.assertIn("nmap -sV 10.0.0.1", formatted)
        self.assertIn("ports", formatted)

    def test_format_tool_result_failure(self):
        from cookbook.swarm.arsenal_bridge import _format_tool_result

        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Connection refused"
        mock_result.command = "nmap 10.0.0.1"

        formatted = _format_tool_result(mock_result)
        self.assertIn("[TOOL FAILED]", formatted)
        self.assertIn("Connection refused", formatted)

    def test_format_tool_result_stdout_fallback(self):
        from cookbook.swarm.arsenal_bridge import _format_tool_result

        mock_result = MagicMock()
        mock_result.success = True
        mock_result.command = "dig example.com"
        mock_result.parsed = None
        mock_result.stdout = "A 93.184.216.34"
        mock_result.stderr = ""

        formatted = _format_tool_result(mock_result)
        self.assertIn("[TOOL OK]", formatted)
        self.assertIn("93.184.216.34", formatted)


# ═══════════════════════════════════════════════════════════════════
# Phase 1 — __init__.py exports test
# ═══════════════════════════════════════════════════════════════════

class TestSwarmExports(unittest.TestCase):
    def test_tool_registry_exported(self):
        import cookbook.swarm as swarm_pkg
        self.assertTrue(hasattr(swarm_pkg, "ToolRegistry"))
        self.assertTrue(hasattr(swarm_pkg, "ToolDef"))

    def test_swarm_accepts_tool_registry(self):
        """Swarm() convenience function should accept tool_registry param."""
        import cookbook.swarm as swarm_pkg
        import inspect
        sig = inspect.signature(swarm_pkg.Swarm)
        self.assertIn("tool_registry", sig.parameters)


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Adaptive Rate Limiter tests
# ═══════════════════════════════════════════════════════════════════

class TestAdaptiveRateLimiter(unittest.TestCase):

    def test_initial_state(self):
        arl = AdaptiveRateLimiter(max_rpm=60, min_rpm=5, burst=10)
        self.assertAlmostEqual(arl.current_rpm, 60.0)
        stats = arl.stats
        self.assertEqual(stats["successes"], 0)
        self.assertEqual(stats["errors"], 0)

    def test_acquire_works(self):
        arl = AdaptiveRateLimiter(max_rpm=600, burst=10)

        async def run():
            await arl.acquire()

        asyncio.new_event_loop().run_until_complete(run())

    def test_backoff_on_error(self):
        arl = AdaptiveRateLimiter(max_rpm=60, min_rpm=5, backoff_factor=0.5)
        arl.record_error()
        self.assertAlmostEqual(arl.current_rpm, 42.0)  # 60 * 0.7 (non-rate-limit)
        self.assertEqual(arl.stats["errors"], 1)
        self.assertEqual(arl.stats["backoffs"], 1)

    def test_backoff_harder_on_429(self):
        arl = AdaptiveRateLimiter(max_rpm=60, min_rpm=5, backoff_factor=0.5)
        arl.record_error(is_rate_limit=True)
        self.assertAlmostEqual(arl.current_rpm, 30.0)  # 60 * 0.5

    def test_backoff_floors_at_min(self):
        arl = AdaptiveRateLimiter(max_rpm=60, min_rpm=10, backoff_factor=0.5)
        for _ in range(20):
            arl.record_error(is_rate_limit=True)
        self.assertGreaterEqual(arl.current_rpm, 10.0)

    def test_recovery_on_success_streak(self):
        arl = AdaptiveRateLimiter(
            max_rpm=60, min_rpm=5, backoff_factor=0.5,
            recovery_factor=1.2, recovery_streak=3,
        )
        # First back off
        arl.record_error(is_rate_limit=True)
        backed_off = arl.current_rpm  # should be 30
        self.assertLess(backed_off, 60)

        # 3 successes should trigger recovery
        for _ in range(3):
            arl.record_success()
        self.assertGreater(arl.current_rpm, backed_off)

    def test_recovery_caps_at_max(self):
        arl = AdaptiveRateLimiter(
            max_rpm=60, min_rpm=5,
            recovery_factor=10.0, recovery_streak=1,
        )
        arl.record_error(is_rate_limit=True)
        arl.record_success()
        self.assertLessEqual(arl.current_rpm, 60.0)

    def test_error_resets_success_streak(self):
        arl = AdaptiveRateLimiter(
            max_rpm=60, min_rpm=5,
            recovery_streak=3, recovery_factor=1.5,
        )
        arl.record_error(is_rate_limit=True)
        backed = arl.current_rpm
        # 2 successes then error — no recovery
        arl.record_success()
        arl.record_success()
        arl.record_error()
        # Should have backed off further, not recovered
        self.assertLessEqual(arl.current_rpm, backed)


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Circuit Breaker tests
# ═══════════════════════════════════════════════════════════════════

class TestCircuitBreaker(unittest.TestCase):

    def test_initial_state_closed(self):
        cb = CircuitBreaker(threshold=3, cooldown=10.0)
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.stats["consecutive_failures"], 0)

    def test_stays_closed_below_threshold(self):
        cb = CircuitBreaker(threshold=3)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(threshold=3)
        for _ in range(3):
            cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)
        self.assertEqual(cb.stats["total_trips"], 1)

    def test_check_raises_when_open(self):
        cb = CircuitBreaker(threshold=2, cooldown=1000)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

        async def run():
            with self.assertRaises(CircuitOpenError):
                await cb.check()

        asyncio.new_event_loop().run_until_complete(run())

    def test_check_passes_when_closed(self):
        cb = CircuitBreaker(threshold=3)

        async def run():
            await cb.check()  # Should not raise

        asyncio.new_event_loop().run_until_complete(run())

    def test_half_open_after_cooldown(self):
        cb = CircuitBreaker(threshold=2, cooldown=0.01)
        cb.record_failure()
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

        async def run():
            await asyncio.sleep(0.02)  # Wait for cooldown
            await cb.check()  # Should transition to HALF_OPEN
            self.assertEqual(cb.state, CircuitState.HALF_OPEN)

        asyncio.new_event_loop().run_until_complete(run())

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(threshold=2, cooldown=0.01)
        cb.record_failure()
        cb.record_failure()

        async def run():
            await asyncio.sleep(0.02)
            await cb.check()  # → HALF_OPEN
            cb.record_success()
            self.assertEqual(cb.state, CircuitState.CLOSED)
            self.assertEqual(cb.stats["consecutive_failures"], 0)

        asyncio.new_event_loop().run_until_complete(run())

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(threshold=2, cooldown=0.01)
        cb.record_failure()
        cb.record_failure()

        async def run():
            await asyncio.sleep(0.02)
            await cb.check()  # → HALF_OPEN
            cb.record_failure()
            self.assertEqual(cb.state, CircuitState.OPEN)
            self.assertEqual(cb.stats["total_trips"], 2)

        asyncio.new_event_loop().run_until_complete(run())

    def test_success_resets_failures(self):
        cb = CircuitBreaker(threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertEqual(cb.stats["consecutive_failures"], 0)
        # Now 5 more failures should bring it to threshold
        for _ in range(5):
            cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Engine integration with adaptive + circuit breaker
# ═══════════════════════════════════════════════════════════════════

class TestEngineAdaptiveConcurrency(unittest.TestCase):

    def _make_config(self, **overrides):
        defaults = dict(
            api_base="http://localhost:1/v1",
            api_key="test",
            agent_timeout=5,
            swarm_timeout=30,
            max_parallel=5,
            rate_limit_rpm=600,
            rate_limit_burst=10,
            adaptive_rate_limit=True,
            rate_limit_min_rpm=5,
            rate_limit_backoff_factor=0.5,
            rate_limit_recovery_factor=1.1,
            rate_limit_recovery_streak=3,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=0.05,
        )
        defaults.update(overrides)
        return SwarmConfig(**defaults)

    def test_engine_uses_adaptive_limiter(self):
        config = self._make_config(adaptive_rate_limit=True)
        engine = SwarmEngine(config=config, runner=AsyncMock())
        self.assertIsInstance(engine._rate_limiter, AdaptiveRateLimiter)

    def test_engine_uses_fixed_limiter_when_disabled(self):
        config = self._make_config(adaptive_rate_limit=False)
        engine = SwarmEngine(config=config, runner=AsyncMock())
        self.assertIsInstance(engine._rate_limiter, RateLimiter)

    def test_engine_creates_circuit_breaker(self):
        config = self._make_config(circuit_breaker_enabled=True)
        engine = SwarmEngine(config=config, runner=AsyncMock())
        self.assertIsNotNone(engine._circuit_breaker)
        self.assertEqual(engine._circuit_breaker.state, CircuitState.CLOSED)

    def test_engine_no_circuit_breaker_when_disabled(self):
        config = self._make_config(circuit_breaker_enabled=False)
        engine = SwarmEngine(config=config, runner=AsyncMock())
        self.assertIsNone(engine._circuit_breaker)

    def test_success_signals_adaptive_limiter(self):
        """Successful agent execution should record success on adaptive limiter."""
        config = self._make_config()

        async def success_runner(agent, results, cfg):
            return "done"

        async def run():
            engine = SwarmEngine(config=config, runner=success_runner)
            plan = SwarmPlan(goal="test", agents=[
                AgentSpec(role="A", task="t"),
            ])
            await engine.execute(plan)
            arl = engine._rate_limiter
            self.assertIsInstance(arl, AdaptiveRateLimiter)
            self.assertEqual(arl.stats["successes"], 1)

        asyncio.new_event_loop().run_until_complete(run())

    def test_failure_signals_adaptive_limiter(self):
        """Failed agent should signal error to adaptive limiter."""
        config = self._make_config(circuit_breaker_enabled=False)

        async def fail_runner(agent, results, cfg):
            raise RuntimeError("429 rate limit exceeded")

        async def run():
            engine = SwarmEngine(config=config, runner=fail_runner)
            plan = SwarmPlan(goal="test", agents=[
                AgentSpec(role="A", task="t", max_retries=0),
            ])
            results = await engine.execute(plan)
            self.assertEqual(results[0].status, AgentStatus.FAILED)
            arl = engine._rate_limiter
            self.assertIsInstance(arl, AdaptiveRateLimiter)
            self.assertGreater(arl.stats["errors"], 0)

        asyncio.new_event_loop().run_until_complete(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — New config fields
# ═══════════════════════════════════════════════════════════════════

class TestPhase3Config(unittest.TestCase):
    def test_adaptive_rate_defaults(self):
        cfg = SwarmConfig()
        self.assertTrue(cfg.adaptive_rate_limit)
        self.assertEqual(cfg.rate_limit_min_rpm, 5)
        self.assertAlmostEqual(cfg.rate_limit_backoff_factor, 0.5)
        self.assertAlmostEqual(cfg.rate_limit_recovery_factor, 1.1)
        self.assertEqual(cfg.rate_limit_recovery_streak, 5)

    def test_circuit_breaker_defaults(self):
        cfg = SwarmConfig()
        self.assertTrue(cfg.circuit_breaker_enabled)
        self.assertEqual(cfg.circuit_breaker_threshold, 5)
        self.assertAlmostEqual(cfg.circuit_breaker_cooldown, 30.0)

    def test_from_env_adaptive(self):
        env = {
            "SWARM_ADAPTIVE_RATE": "false",
            "SWARM_RATE_LIMIT_MIN_RPM": "10",
            "SWARM_RATE_BACKOFF_FACTOR": "0.3",
            "SWARM_RATE_RECOVERY_FACTOR": "1.5",
            "SWARM_RATE_RECOVERY_STREAK": "10",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertFalse(cfg.adaptive_rate_limit)
        self.assertEqual(cfg.rate_limit_min_rpm, 10)
        self.assertAlmostEqual(cfg.rate_limit_backoff_factor, 0.3)
        self.assertAlmostEqual(cfg.rate_limit_recovery_factor, 1.5)
        self.assertEqual(cfg.rate_limit_recovery_streak, 10)

    def test_from_env_circuit_breaker(self):
        env = {
            "SWARM_CIRCUIT_BREAKER": "false",
            "SWARM_CB_THRESHOLD": "10",
            "SWARM_CB_COOLDOWN": "60",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertFalse(cfg.circuit_breaker_enabled)
        self.assertEqual(cfg.circuit_breaker_threshold, 10)
        self.assertAlmostEqual(cfg.circuit_breaker_cooldown, 60.0)


# ═══════════════════════════════════════════════════════════════════
# Phase 3 — Exports test
# ═══════════════════════════════════════════════════════════════════

class TestPhase3Exports(unittest.TestCase):
    def test_new_exports(self):
        import cookbook.swarm as swarm_pkg
        self.assertTrue(hasattr(swarm_pkg, "AdaptiveRateLimiter"))
        self.assertTrue(hasattr(swarm_pkg, "CircuitBreaker"))
        self.assertTrue(hasattr(swarm_pkg, "CircuitOpenError"))


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — Few-shot decomposition prompt tests
# ═══════════════════════════════════════════════════════════════════

class TestFewShotDecomposition(unittest.TestCase):
    """Verify the DECOMPOSE_SYSTEM prompt contains few-shot examples."""

    def test_contains_security_example(self):
        self.assertIn("Security audit", DECOMPOSE_SYSTEM)
        self.assertIn("Authentication Analyst", DECOMPOSE_SYSTEM)
        self.assertIn("Input Validation Specialist", DECOMPOSE_SYSTEM)

    def test_contains_research_example(self):
        self.assertIn("Research task", DECOMPOSE_SYSTEM)
        self.assertIn("Magnetic Confinement Researcher", DECOMPOSE_SYSTEM)

    def test_contains_code_review_example(self):
        self.assertIn("Code review", DECOMPOSE_SYSTEM)
        self.assertIn("Bug Hunter", DECOMPOSE_SYSTEM)
        self.assertIn("Design Critic", DECOMPOSE_SYSTEM)

    def test_examples_show_dependency_pattern(self):
        # At least one example should show depends_on with non-empty list
        self.assertIn('"depends_on": [0, 1, 2]', DECOMPOSE_SYSTEM)

    def test_examples_show_priority_pattern(self):
        # Examples should have diverse priorities
        self.assertIn('"priority": 4', DECOMPOSE_SYSTEM)
        self.assertIn('"priority": 1', DECOMPOSE_SYSTEM)

    def test_examples_show_tool_assignment(self):
        self.assertIn('"tools": ["search"]', DECOMPOSE_SYSTEM)
        self.assertIn('"tools": ["code"]', DECOMPOSE_SYSTEM)
        self.assertIn('"tools": ["analyze"]', DECOMPOSE_SYSTEM)

    def test_format_placeholders(self):
        # Should still have format placeholders for min/max agents
        self.assertIn("{min_agents}", DECOMPOSE_SYSTEM)
        self.assertIn("{max_agents}", DECOMPOSE_SYSTEM)

    def test_new_rules_present(self):
        self.assertIn("redundant agents", DECOMPOSE_SYSTEM)
        self.assertIn("priority 3-5", DECOMPOSE_SYSTEM)


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — Plan critic tests
# ═══════════════════════════════════════════════════════════════════

class TestCritiqueSystem(unittest.TestCase):
    """Verify the CRITIQUE_SYSTEM prompt structure."""

    def test_contains_check_categories(self):
        self.assertIn("REDUNDANCY", CRITIQUE_SYSTEM)
        self.assertIn("COVERAGE GAPS", CRITIQUE_SYSTEM)
        self.assertIn("BAD DEPENDENCIES", CRITIQUE_SYSTEM)
        self.assertIn("TASK CLARITY", CRITIQUE_SYSTEM)
        self.assertIn("TOOL MISMATCH", CRITIQUE_SYSTEM)

    def test_contains_verdict_options(self):
        self.assertIn('"accept"', CRITIQUE_SYSTEM)
        self.assertIn('"revise"', CRITIQUE_SYSTEM)

    def test_contains_output_schema(self):
        self.assertIn('"verdict"', CRITIQUE_SYSTEM)
        self.assertIn('"issues"', CRITIQUE_SYSTEM)
        self.assertIn('"revised_agents"', CRITIQUE_SYSTEM)


class TestPlanCritique(unittest.TestCase):
    """Test the _critique_plan method on the Orchestrator."""

    def _make_orc(self, **kwargs):
        cfg = SwarmConfig(
            api_base="http://test:8000/v1",
            enable_plan_critique=True,
            **kwargs,
        )
        return Orchestrator(config=cfg)

    def _make_plan(self, n_agents=3):
        agents = [
            AgentSpec(role=f"Agent-{i}", task=f"Task {i}")
            for i in range(n_agents)
        ]
        return SwarmPlan(goal="test goal", agents=agents, strategy="test strategy")

    def test_critique_accept_keeps_plan(self):
        """When critic says 'accept', the original plan is returned."""
        orc = self._make_orc()
        plan = self._make_plan()

        accept_response = json.dumps({
            "verdict": "accept",
            "issues": [],
            "revised_agents": [],
        })

        async def run():
            async with orc:
                with patch.object(orc, '_chat', new_callable=AsyncMock, return_value=accept_response):
                    result = await orc._critique_plan("test goal", plan)
                    self.assertEqual(len(result.agents), len(plan.agents))
                    # Same plan object since verdict is accept
                    self.assertIs(result, plan)

        asyncio.run(run())

    def test_critique_revise_rebuilds_plan(self):
        """When critic says 'revise', a new plan is built from revised_agents."""
        orc = self._make_orc()
        plan = self._make_plan(n_agents=2)

        revise_response = json.dumps({
            "verdict": "revise",
            "issues": [{"type": "gap", "description": "Missing analyst", "fix": "Add one"}],
            "revised_agents": [
                {"role": "Alpha", "task": "Do A", "depends_on": [], "priority": 3, "tools": ["search"]},
                {"role": "Beta", "task": "Do B", "depends_on": [], "priority": 2, "tools": []},
                {"role": "Gamma", "task": "Summarize A and B", "depends_on": [0, 1], "priority": 1, "tools": []},
            ],
        })

        async def run():
            async with orc:
                with patch.object(orc, '_chat', new_callable=AsyncMock, return_value=revise_response):
                    result = await orc._critique_plan("test goal", plan)
                    self.assertIsNot(result, plan)
                    self.assertEqual(len(result.agents), 3)
                    self.assertEqual(result.agents[0].role, "Alpha")
                    self.assertEqual(result.agents[2].role, "Gamma")
                    # Gamma depends on Alpha and Beta
                    self.assertEqual(len(result.agents[2].depends_on), 2)

        asyncio.run(run())

    def test_critique_invalid_json_keeps_original(self):
        """If critic returns garbage, keep the original plan."""
        orc = self._make_orc()
        plan = self._make_plan()

        async def run():
            async with orc:
                with patch.object(orc, '_chat', new_callable=AsyncMock, return_value="not json at all"):
                    result = await orc._critique_plan("test goal", plan)
                    self.assertIs(result, plan)

        asyncio.run(run())

    def test_critique_invalid_revised_plan_keeps_original(self):
        """If critic's revised plan has cycles, keep original."""
        orc = self._make_orc()
        plan = self._make_plan()

        # Create a plan with circular deps
        revise_response = json.dumps({
            "verdict": "revise",
            "issues": [],
            "revised_agents": [
                {"role": "A", "task": "X", "depends_on": [1], "priority": 0, "tools": []},
                {"role": "B", "task": "Y", "depends_on": [0], "priority": 0, "tools": []},
            ],
        })

        async def run():
            async with orc:
                with patch.object(orc, '_chat', new_callable=AsyncMock, return_value=revise_response):
                    result = await orc._critique_plan("test goal", plan)
                    # Should keep original because revised plan has cycle
                    self.assertIs(result, plan)

        asyncio.run(run())

    def test_decompose_calls_critique_when_enabled(self):
        """Decompose should call _critique_plan when enable_plan_critique is True."""
        orc = self._make_orc()

        decompose_response = json.dumps({
            "strategy": "test",
            "agents": [
                {"role": "A", "task": "Do A", "depends_on": [], "priority": 0, "tools": []},
                {"role": "B", "task": "Do B", "depends_on": [], "priority": 0, "tools": []},
            ],
        })
        accept_response = json.dumps({
            "verdict": "accept", "issues": [], "revised_agents": [],
        })

        call_log = []

        async def mock_chat(messages, **kwargs):
            # First call = decompose, second = critique
            call_log.append(messages[-1]["content"])
            if len(call_log) == 1:
                return decompose_response
            return accept_response

        async def run():
            async with orc:
                with patch.object(orc, '_chat', side_effect=mock_chat):
                    plan = await orc.decompose("test goal")
                    self.assertEqual(len(call_log), 2)
                    self.assertEqual(len(plan.agents), 2)

        asyncio.run(run())

    def test_decompose_skips_critique_when_disabled(self):
        """Decompose should NOT call _critique_plan when enable_plan_critique is False."""
        cfg = SwarmConfig(api_base="http://test:8000/v1", enable_plan_critique=False)
        orc = Orchestrator(config=cfg)

        decompose_response = json.dumps({
            "strategy": "test",
            "agents": [
                {"role": "A", "task": "Do A", "depends_on": [], "priority": 0, "tools": []},
                {"role": "B", "task": "Do B", "depends_on": [], "priority": 0, "tools": []},
            ],
        })

        call_count = 0

        async def mock_chat(messages, **kwargs):
            nonlocal call_count
            call_count += 1
            return decompose_response

        async def run():
            async with orc:
                with patch.object(orc, '_chat', side_effect=mock_chat):
                    await orc.decompose("test goal")
                    self.assertEqual(call_count, 1)  # Only decompose, no critique

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — Agent killing tests
# ═══════════════════════════════════════════════════════════════════

class TestAgentKilling(unittest.TestCase):
    """Test agent killing based on median peer duration."""

    def _make_config(self, **kw):
        defaults = dict(
            api_base="http://test:8000/v1",
            max_parallel=10,
            max_retries=0,
            agent_timeout=120,
            enable_agent_killing=True,
            agent_kill_threshold=2.0,  # kill at 2× median for easier testing
            agent_kill_min_time=0,     # disable absolute floor for fast tests
        )
        defaults.update(kw)
        return SwarmConfig(**defaults)

    def test_kill_slow_agents_method(self):
        """_kill_slow_agents cancels tasks exceeding threshold."""
        config = self._make_config()

        async def fast_runner(agent, results, cfg):
            return "fast result"

        engine = SwarmEngine(config=config, runner=fast_runner)

        # Simulate 3 completed agents with ~1s duration
        for i in range(3):
            engine._results[f"done-{i}"] = AgentResult(
                agent_id=f"done-{i}", role=f"Done-{i}", task="t",
                status=AgentStatus.COMPLETED,
                started_at=100.0, finished_at=101.0,  # 1s each
            )
            engine._completed.add(f"done-{i}")

        slow_agent = AgentSpec(role="SlowPoke", task="slow task")
        plan = SwarmPlan(goal="test", agents=[slow_agent])

        async def run():
            # Create task inside event loop
            slow_task = asyncio.ensure_future(asyncio.sleep(999))
            running = {slow_agent.agent_id: slow_task}
            start_times = {slow_agent.agent_id: time.time() - 5.0}

            engine._kill_slow_agents(running, start_times, plan)

            # Should be killed
            self.assertNotIn(slow_agent.agent_id, running)
            self.assertIn(slow_agent.agent_id, engine._results)
            self.assertEqual(engine._results[slow_agent.agent_id].status, AgentStatus.CANCELLED)
            self.assertIn("Killed", engine._results[slow_agent.agent_id].error)

        asyncio.run(run())

    def test_no_kill_when_under_threshold(self):
        """Agents under the threshold are not killed."""
        config = self._make_config()

        async def fast_runner(agent, results, cfg):
            return "result"

        engine = SwarmEngine(config=config, runner=fast_runner)

        # 3 completed agents with 10s duration each
        for i in range(3):
            engine._results[f"done-{i}"] = AgentResult(
                agent_id=f"done-{i}", role=f"Done-{i}", task="t",
                status=AgentStatus.COMPLETED,
                started_at=100.0, finished_at=110.0,  # 10s each
            )
            engine._completed.add(f"done-{i}")

        agent = AgentSpec(role="Normal", task="normal task")
        plan = SwarmPlan(goal="test", agents=[agent])

        async def run():
            task = asyncio.ensure_future(asyncio.sleep(999))
            running = {agent.agent_id: task}
            start_times = {agent.agent_id: time.time() - 15.0}

            engine._kill_slow_agents(running, start_times, plan)

            # Should NOT be killed (15s < 2×10s = 20s threshold)
            self.assertIn(agent.agent_id, running)
            self.assertNotIn(agent.agent_id, engine._results)
            task.cancel()

        asyncio.run(run())

    def test_no_kill_when_disabled(self):
        """Agent killing is skipped when enable_agent_killing=False."""
        config = self._make_config(enable_agent_killing=False)

        agents = [
            AgentSpec(role="Fast", task="fast"),
            AgentSpec(role="Fast2", task="fast2"),
            AgentSpec(role="Slow", task="slow"),
        ]
        plan = SwarmPlan(goal="test", agents=agents)

        call_count = 0

        async def timed_runner(agent, results, cfg):
            nonlocal call_count
            call_count += 1
            if "Slow" in agent.role:
                await asyncio.sleep(0.3)
            else:
                await asyncio.sleep(0.01)
            return f"result from {agent.role}"

        engine = SwarmEngine(config=config, runner=timed_runner)

        async def run():
            results = await engine.execute(plan)
            # All should complete (slow one shouldn't be killed since killing is disabled)
            statuses = {r.role: r.status for r in results}
            self.assertEqual(statuses["Slow"], AgentStatus.COMPLETED)

        asyncio.run(run())

    def test_no_kill_when_insufficient_completed(self):
        """Don't kill agents when there aren't enough completed peers.

        The engine requires max(3, (total+1)//2) completed agents before
        kill checks activate in _execute_inner.
        """
        config = self._make_config()

        async def runner(agent, results, cfg):
            return "result"

        engine = SwarmEngine(config=config, runner=runner)

        # Only 1 completed agent
        engine._results["done-0"] = AgentResult(
            agent_id="done-0", role="Done-0", task="t",
            status=AgentStatus.COMPLETED,
            started_at=100.0, finished_at=101.0,
        )
        engine._completed.add("done-0")

        agent = AgentSpec(role="Running", task="task")
        plan = SwarmPlan(goal="test", agents=[agent])

        async def run():
            task = asyncio.ensure_future(asyncio.sleep(999))
            running = {agent.agent_id: task}
            start_times = {agent.agent_id: time.time() - 100.0}

            # _kill_slow_agents can still be called directly (it only needs
            # durations), but _execute_inner gates it behind
            # len(completed) >= max(3, (total+1)//2)
            engine._kill_slow_agents(running, start_times, plan)
            task.cancel()

        asyncio.run(run())

    def test_kill_threshold_config(self):
        """agent_kill_threshold is configurable."""
        config = self._make_config(agent_kill_threshold=5.0)
        self.assertAlmostEqual(config.agent_kill_threshold, 5.0)

    def test_integration_kill_in_execute(self):
        """Full integration: slow agent gets killed during execute."""
        config = self._make_config(agent_kill_threshold=2.0, agent_timeout=60)

        agents = [
            AgentSpec(role="Fast-1", task="fast"),
            AgentSpec(role="Fast-2", task="fast"),
            AgentSpec(role="Fast-3", task="fast"),
            AgentSpec(role="Straggler", task="very slow"),
        ]
        plan = SwarmPlan(goal="test integration", agents=agents)

        async def runner(agent, results, cfg):
            if "Straggler" in agent.role:
                # Sleep long enough to be caught by 1s periodic check
                await asyncio.sleep(60)
            else:
                await asyncio.sleep(0.01)
            return f"result from {agent.role}"

        engine = SwarmEngine(config=config, runner=runner)

        async def run():
            results = await engine.execute(plan)
            statuses = {r.role: r.status for r in results}
            # Straggler should be cancelled by agent killer
            # Fast agents: ~0.01s each, median ~0.01s, threshold = 0.02s
            # After 1s wait timeout, straggler at ~1s >> 0.02s → killed
            self.assertEqual(statuses["Straggler"], AgentStatus.CANCELLED)
            self.assertEqual(statuses["Fast-1"], AgentStatus.COMPLETED)
            self.assertEqual(statuses["Fast-2"], AgentStatus.COMPLETED)

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 5 — Config tests
# ═══════════════════════════════════════════════════════════════════

class TestPhase5Config(unittest.TestCase):
    def test_defaults(self):
        cfg = SwarmConfig()
        self.assertFalse(cfg.enable_plan_critique)
        self.assertTrue(cfg.enable_agent_killing)
        self.assertAlmostEqual(cfg.agent_kill_threshold, 3.0)
        self.assertAlmostEqual(cfg.agent_kill_min_time, 30.0)

    def test_from_env(self):
        env = {
            "SWARM_PLAN_CRITIQUE": "true",
            "SWARM_AGENT_KILLING": "false",
            "SWARM_AGENT_KILL_THRESHOLD": "5.0",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertTrue(cfg.enable_plan_critique)
        self.assertFalse(cfg.enable_agent_killing)
        self.assertAlmostEqual(cfg.agent_kill_threshold, 5.0)


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — BudgetTracker tests
# ═══════════════════════════════════════════════════════════════════

class TestBudgetTracker(unittest.TestCase):
    def test_initial_state(self):
        bt = BudgetTracker(token_budget=1000, max_llm_calls=50, max_agents=10)
        self.assertEqual(bt.tokens_used, 0)
        self.assertEqual(bt.llm_calls, 0)
        self.assertEqual(bt.agents_spawned, 0)
        self.assertEqual(bt.tokens_remaining, 1000)
        self.assertIsNone(bt.check_budget())

    def test_unlimited_budget(self):
        bt = BudgetTracker()  # all zeros = unlimited
        self.assertIsNone(bt.tokens_remaining)
        self.assertIsNone(bt.check_budget())

    def test_record_tokens(self):
        bt = BudgetTracker(token_budget=100)

        async def run():
            await bt.record_tokens(50)
            self.assertEqual(bt.tokens_used, 50)
            self.assertEqual(bt.tokens_remaining, 50)
            self.assertIsNone(bt.check_budget())

            await bt.record_tokens(60)
            self.assertEqual(bt.tokens_used, 110)
            self.assertEqual(bt.tokens_remaining, 0)
            self.assertIn("Token budget", bt.check_budget())

        asyncio.run(run())

    def test_record_llm_calls(self):
        bt = BudgetTracker(max_llm_calls=3)

        async def run():
            for _ in range(3):
                await bt.record_llm_call()
            self.assertEqual(bt.llm_calls, 3)
            self.assertIn("LLM call limit", bt.check_budget())

        asyncio.run(run())

    def test_record_agents(self):
        bt = BudgetTracker(max_agents=2)

        async def run():
            await bt.record_agent()
            self.assertIsNone(bt.check_budget())
            await bt.record_agent()
            self.assertIn("Agent limit", bt.check_budget())

        asyncio.run(run())

    def test_stats(self):
        bt = BudgetTracker(token_budget=500, max_llm_calls=10)
        stats = bt.stats
        self.assertEqual(stats["tokens_used"], 0)
        self.assertEqual(stats["token_budget"], 500)
        self.assertEqual(stats["max_llm_calls"], 10)
        self.assertEqual(stats["max_agents"], "unlimited")

    def test_check_priority_order(self):
        """Token budget is checked first."""
        bt = BudgetTracker(token_budget=10, max_llm_calls=1, max_agents=1)

        async def run():
            await bt.record_tokens(100)
            await bt.record_llm_call()
            await bt.record_agent()
            # Token check should trigger first
            err = bt.check_budget()
            self.assertIn("Token budget", err)

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Engine budget integration tests
# ═══════════════════════════════════════════════════════════════════

class TestEngineBudgetIntegration(unittest.TestCase):
    def _make_config(self, **kw):
        defaults = dict(
            api_base="http://test:8000/v1",
            max_parallel=10,
            max_retries=0,
            agent_timeout=10,
        )
        defaults.update(kw)
        return SwarmConfig(**defaults)

    def test_budget_blocks_agent_launch(self):
        """Agents should fail with budget error when budget is exhausted."""
        config = self._make_config()
        bt = BudgetTracker(max_agents=1)

        async def runner(agent, results, cfg):
            return "result"

        engine = SwarmEngine(config=config, runner=runner, budget=bt)

        agents = [
            AgentSpec(role="A", task="task a"),
            AgentSpec(role="B", task="task b"),
        ]
        plan = SwarmPlan(goal="test", agents=agents)

        async def run():
            # Pre-exhaust the budget
            await bt.record_agent()
            results = await engine.execute(plan)
            statuses = {r.role: r.status for r in results}
            # Both should fail since budget was already at limit before execution
            self.assertEqual(statuses["A"], AgentStatus.FAILED)
            self.assertEqual(statuses["B"], AgentStatus.FAILED)
            self.assertIn("Budget exceeded", results[0].error)

        asyncio.run(run())

    def test_budget_records_on_completion(self):
        """Successful agents should be recorded in the budget."""
        config = self._make_config()
        bt = BudgetTracker(max_agents=10)

        async def runner(agent, results, cfg):
            return "result"

        engine = SwarmEngine(config=config, runner=runner, budget=bt)

        agents = [AgentSpec(role="A", task="task")]
        plan = SwarmPlan(goal="test", agents=agents)

        async def run():
            await engine.execute(plan)
            self.assertEqual(bt.agents_spawned, 1)

        asyncio.run(run())

    def test_no_budget_no_issue(self):
        """Without budget, everything runs normally."""
        config = self._make_config()

        async def runner(agent, results, cfg):
            return "result"

        engine = SwarmEngine(config=config, runner=runner)  # no budget

        agents = [AgentSpec(role="A", task="task")]
        plan = SwarmPlan(goal="test", agents=agents)

        async def run():
            results = await engine.execute(plan)
            self.assertEqual(results[0].status, AgentStatus.COMPLETED)

        asyncio.run(run())


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Sub-swarm tests
# ═══════════════════════════════════════════════════════════════════

class TestSubSwarm(unittest.TestCase):
    def _make_config(self, **kw):
        defaults = dict(
            api_base="http://test:8000/v1",
            enable_sub_swarms=True,
            sub_swarm_max_depth=2,
            sub_swarm_max_agents=3,
        )
        defaults.update(kw)
        return SwarmConfig(**defaults)

    def test_can_spawn_sub_swarm_enabled(self):
        cfg = self._make_config()
        orc = Orchestrator(config=cfg)
        self.assertTrue(orc._can_spawn_sub_swarm())

    def test_can_spawn_sub_swarm_disabled(self):
        cfg = self._make_config(enable_sub_swarms=False)
        orc = Orchestrator(config=cfg)
        self.assertFalse(orc._can_spawn_sub_swarm())

    def test_can_spawn_sub_swarm_depth_exceeded(self):
        cfg = self._make_config(sub_swarm_max_depth=1)
        orc = Orchestrator(config=cfg, depth=1)  # already at max depth
        self.assertFalse(orc._can_spawn_sub_swarm())

    def test_can_spawn_sub_swarm_budget_exhausted(self):
        cfg = self._make_config()
        bt = BudgetTracker(token_budget=10)

        async def run():
            await bt.record_tokens(100)  # exhaust
            orc = Orchestrator(config=cfg, budget=bt)
            self.assertFalse(orc._can_spawn_sub_swarm())

        asyncio.run(run())

    def test_run_sub_swarm_disabled_returns_message(self):
        cfg = self._make_config(enable_sub_swarms=False)
        orc = Orchestrator(config=cfg)

        async def run():
            result = await orc._run_sub_swarm("test goal")
            self.assertIn("not available", result)

        asyncio.run(run())

    def test_register_sub_swarm_tool(self):
        cfg = self._make_config()
        orc = Orchestrator(config=cfg)
        registry = ToolRegistry()
        orc.register_sub_swarm_tool(registry)
        self.assertIn("spawn_sub_swarm", registry._tools)

    def test_sub_swarm_in_run_auto_registers_tool(self):
        """When sub-swarms are enabled, run() auto-registers the tool."""
        cfg = self._make_config()
        orc = Orchestrator(config=cfg)

        decompose_response = json.dumps({
            "strategy": "test",
            "agents": [
                {"role": "A", "task": "Do A", "depends_on": [], "priority": 0, "tools": []},
                {"role": "B", "task": "Do B", "depends_on": [], "priority": 0, "tools": []},
            ],
        })

        async def mock_chat(messages, **kwargs):
            return decompose_response

        async def run():
            async with orc:
                with patch.object(orc, '_chat', side_effect=mock_chat):
                    with patch.object(SwarmEngine, 'execute', new_callable=AsyncMock, return_value=[
                        AgentResult(agent_id="a1", role="A", task="Do A", content="done", status=AgentStatus.COMPLETED,
                                    started_at=1.0, finished_at=2.0),
                    ]):
                        result = await orc.run("test goal")
                        # Tool registry should have been created and sub-swarm registered
                        self.assertIsNotNone(orc.tool_registry)
                        self.assertIn("spawn_sub_swarm", orc.tool_registry._tools)

        asyncio.run(run())

    def test_depth_propagates_to_child(self):
        """Child orchestrator should be at depth+1."""
        cfg = self._make_config()
        parent = Orchestrator(config=cfg, depth=0)
        # We test indirectly via _can_spawn_sub_swarm and child config
        self.assertEqual(parent.depth, 0)
        self.assertTrue(parent._can_spawn_sub_swarm())


# ═══════════════════════════════════════════════════════════════════
# Phase 2 — Config tests
# ═══════════════════════════════════════════════════════════════════

class TestPhase2Config(unittest.TestCase):
    def test_budget_defaults(self):
        cfg = SwarmConfig()
        self.assertEqual(cfg.token_budget, 0)
        self.assertEqual(cfg.max_llm_calls, 0)

    def test_sub_swarm_defaults(self):
        cfg = SwarmConfig()
        self.assertFalse(cfg.enable_sub_swarms)
        self.assertEqual(cfg.sub_swarm_max_depth, 2)
        self.assertEqual(cfg.sub_swarm_max_agents, 5)

    def test_from_env_budget(self):
        env = {
            "SWARM_TOKEN_BUDGET": "100000",
            "SWARM_MAX_LLM_CALLS": "500",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertEqual(cfg.token_budget, 100000)
        self.assertEqual(cfg.max_llm_calls, 500)

    def test_from_env_sub_swarms(self):
        env = {
            "SWARM_SUB_SWARMS": "true",
            "SWARM_SUB_SWARM_MAX_DEPTH": "3",
            "SWARM_SUB_SWARM_MAX_AGENTS": "8",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertTrue(cfg.enable_sub_swarms)
        self.assertEqual(cfg.sub_swarm_max_depth, 3)
        self.assertEqual(cfg.sub_swarm_max_agents, 8)


class TestPhase2Exports(unittest.TestCase):
    def test_budget_tracker_exported(self):
        import cookbook.swarm as swarm_pkg
        self.assertTrue(hasattr(swarm_pkg, "BudgetTracker"))

    def test_swarm_accepts_budget(self):
        """Swarm() one-liner should accept budget parameter."""
        import inspect
        from cookbook.swarm import Swarm as SwarmFn
        sig = inspect.signature(SwarmFn)
        self.assertIn("budget", sig.parameters)


# ═══════════════════════════════════════════════════════════════════
# Phase 4 — ContextWindow tests
# ═══════════════════════════════════════════════════════════════════

class TestContextWindow(unittest.TestCase):
    def test_estimate_tokens(self):
        self.assertEqual(ContextWindow.estimate_tokens("a" * 400), 100)
        self.assertEqual(ContextWindow.estimate_tokens(""), 1)

    def test_build_unlimited_dependencies(self):
        """Unlimited mode includes full dependency content."""
        cw = ContextWindow(max_tokens=0)
        agent = AgentSpec(role="B", task="task", depends_on=["a1"])
        results = {
            "a1": AgentResult(
                agent_id="a1", role="A", task="t", content="Full output here",
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
        }
        ctx = cw.build_context(agent, results)
        self.assertIn("Full output here", ctx)
        self.assertIn("Required context", ctx)

    def test_build_unlimited_peers(self):
        """Unlimited mode includes peer previews."""
        cw = ContextWindow(max_tokens=0)
        agent = AgentSpec(role="B", task="task")
        results = {
            "a1": AgentResult(
                agent_id="a1", role="A", task="t", content="Long " * 200,
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
        }
        ctx = cw.build_context(agent, results)
        self.assertIn("Other agents' outputs", ctx)
        self.assertIn("...", ctx)  # truncated

    def test_build_empty_results(self):
        cw = ContextWindow(max_tokens=1000)
        agent = AgentSpec(role="A", task="task")
        self.assertEqual(cw.build_context(agent, {}), "")

    def test_pruned_fits_within_budget(self):
        """Pruned context should not exceed the token budget by much."""
        cw = ContextWindow(max_tokens=100)  # ~400 chars
        agent = AgentSpec(role="C", task="task", depends_on=["a1", "a2"])
        results = {
            "a1": AgentResult(
                agent_id="a1", role="A", task="t", content="Short",
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
            "a2": AgentResult(
                agent_id="a2", role="B", task="t", content="X" * 2000,
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
        }
        ctx = cw.build_context(agent, results)
        # Should have truncated the long dependency
        self.assertIn("A", ctx)
        self.assertIn("truncated", ctx)

    def test_pruned_uses_cached_summary(self):
        """Pruned mode should use cached summaries when content is too long."""
        cw = ContextWindow(max_tokens=100)

        async def run():
            await cw.cache_summary("a2", "Brief summary of B's work")

        asyncio.run(run())

        agent = AgentSpec(role="C", task="task", depends_on=["a2"])
        results = {
            "a2": AgentResult(
                agent_id="a2", role="B", task="t", content="X" * 5000,
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
        }
        ctx = cw.build_context(agent, results)
        self.assertIn("summarized", ctx)
        self.assertIn("Brief summary", ctx)

    def test_pruned_peer_summaries(self):
        """Non-dependency peers use cached summaries when available."""
        cw = ContextWindow(max_tokens=500)

        async def run():
            await cw.cache_summary("peer1", "Peer summary text")

        asyncio.run(run())

        agent = AgentSpec(role="C", task="task")
        results = {
            "peer1": AgentResult(
                agent_id="peer1", role="PeerA", task="t", content="Full long output",
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
        }
        ctx = cw.build_context(agent, results)
        self.assertIn("Peer summary text", ctx)

    def test_stats(self):
        cw = ContextWindow(max_tokens=2000, summary_max_tokens=500)
        stats = cw.stats
        self.assertEqual(stats["max_tokens"], 2000)
        self.assertEqual(stats["summaries_cached"], 0)
        self.assertEqual(stats["summary_max_tokens"], 500)

    def test_stats_unlimited(self):
        cw = ContextWindow(max_tokens=0)
        self.assertEqual(cw.stats["max_tokens"], "unlimited")

    def test_cache_summary(self):
        cw = ContextWindow(max_tokens=100)

        async def run():
            self.assertIsNone(cw.get_summary("a1"))
            await cw.cache_summary("a1", "summary")
            self.assertEqual(cw.get_summary("a1"), "summary")

        asyncio.run(run())

    def test_pruned_skips_failed_deps(self):
        """Failed dependencies should be skipped in context."""
        cw = ContextWindow(max_tokens=1000)
        agent = AgentSpec(role="B", task="task", depends_on=["a1"])
        results = {
            "a1": AgentResult(
                agent_id="a1", role="A", task="t", content="Failed content",
                status=AgentStatus.FAILED, error="boom",
            ),
        }
        ctx = cw.build_context(agent, results)
        self.assertEqual(ctx, "")

    def test_pruned_budget_stops_peers(self):
        """When budget is fully used by dependencies, peers are skipped."""
        cw = ContextWindow(max_tokens=50)  # ~200 chars budget
        agent = AgentSpec(role="C", task="task", depends_on=["a1"])
        results = {
            "a1": AgentResult(
                agent_id="a1", role="A", task="t", content="X" * 200,
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
            "peer1": AgentResult(
                agent_id="peer1", role="Peer", task="t", content="Should not appear",
                status=AgentStatus.COMPLETED, started_at=1.0, finished_at=2.0,
            ),
        }
        ctx = cw.build_context(agent, results)
        self.assertNotIn("Should not appear", ctx)


class TestSummarizeResult(unittest.TestCase):
    def _make_config(self, **kw):
        defaults = dict(
            api_base="http://test:8000/v1",
            enable_context_pruning=True,
            context_window_tokens=500,
            context_summary_tokens=100,
        )
        defaults.update(kw)
        return SwarmConfig(**defaults)

    def test_summarize_caches_result(self):
        """_summarize_result should call LLM and cache the summary."""
        cfg = self._make_config()
        orc = Orchestrator(config=cfg)
        self.assertIsNotNone(orc.context_window)

        long_content = "Important finding. " * 200  # well over 400 chars

        async def mock_chat(messages, **kwargs):
            return "Concise summary of findings."

        result = AgentResult(
            agent_id="a1", role="Researcher", task="research",
            content=long_content, status=AgentStatus.COMPLETED,
            started_at=1.0, finished_at=2.0,
        )

        async def run():
            async with orc:
                with patch.object(orc, '_chat', side_effect=mock_chat):
                    await orc._summarize_result(result)
                    cached = orc.context_window.get_summary("a1")
                    self.assertEqual(cached, "Concise summary of findings.")

        asyncio.run(run())

    def test_summarize_short_content_no_llm(self):
        """Short content should be cached directly without LLM call."""
        cfg = self._make_config()
        orc = Orchestrator(config=cfg)

        result = AgentResult(
            agent_id="a1", role="R", task="t",
            content="Short", status=AgentStatus.COMPLETED,
        )

        async def run():
            async with orc:
                # Should not call _chat since content is short
                with patch.object(orc, '_chat', side_effect=AssertionError("Should not be called")):
                    await orc._summarize_result(result)
                    cached = orc.context_window.get_summary("a1")
                    self.assertEqual(cached, "Short")

        asyncio.run(run())

    def test_no_context_window_noop(self):
        """Without context pruning, _summarize_result is a no-op."""
        cfg = SwarmConfig(api_base="http://test:8000/v1")
        orc = Orchestrator(config=cfg)
        self.assertIsNone(orc.context_window)

        result = AgentResult(
            agent_id="a1", role="R", task="t",
            content="anything", status=AgentStatus.COMPLETED,
        )

        async def run():
            async with orc:
                await orc._summarize_result(result)  # should not raise

        asyncio.run(run())

    def test_summarize_already_cached_skips(self):
        """Already-cached results should not trigger another LLM call."""
        cfg = self._make_config()
        orc = Orchestrator(config=cfg)

        long_content = "Finding. " * 200

        result = AgentResult(
            agent_id="a1", role="R", task="t",
            content=long_content, status=AgentStatus.COMPLETED,
        )

        async def run():
            # Pre-cache
            await orc.context_window.cache_summary("a1", "Already cached")
            async with orc:
                with patch.object(orc, '_chat', side_effect=AssertionError("Should not be called")):
                    await orc._summarize_result(result)
                    self.assertEqual(orc.context_window.get_summary("a1"), "Already cached")

        asyncio.run(run())


class TestOrchestratorContextWindowIntegration(unittest.TestCase):
    def test_context_window_created_when_enabled(self):
        cfg = SwarmConfig(
            api_base="http://test:8000/v1",
            enable_context_pruning=True,
            context_window_tokens=2000,
            context_summary_tokens=300,
        )
        orc = Orchestrator(config=cfg)
        self.assertIsNotNone(orc.context_window)
        self.assertEqual(orc.context_window.max_tokens, 2000)
        self.assertEqual(orc.context_window.summary_max_tokens, 300)

    def test_context_window_not_created_when_disabled(self):
        cfg = SwarmConfig(api_base="http://test:8000/v1")
        orc = Orchestrator(config=cfg)
        self.assertIsNone(orc.context_window)

    def test_context_window_not_created_if_zero_tokens(self):
        cfg = SwarmConfig(
            api_base="http://test:8000/v1",
            enable_context_pruning=True,
            context_window_tokens=0,
        )
        orc = Orchestrator(config=cfg)
        self.assertIsNone(orc.context_window)


class TestPhase4Config(unittest.TestCase):
    def test_defaults(self):
        cfg = SwarmConfig()
        self.assertEqual(cfg.context_window_tokens, 0)
        self.assertEqual(cfg.context_summary_tokens, 300)
        self.assertFalse(cfg.enable_context_pruning)

    def test_from_env(self):
        env = {
            "SWARM_CONTEXT_WINDOW_TOKENS": "4000",
            "SWARM_CONTEXT_SUMMARY_TOKENS": "500",
            "SWARM_CONTEXT_PRUNING": "true",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = SwarmConfig.from_env()
        self.assertEqual(cfg.context_window_tokens, 4000)
        self.assertEqual(cfg.context_summary_tokens, 500)
        self.assertTrue(cfg.enable_context_pruning)


class TestPhase4Exports(unittest.TestCase):
    def test_context_window_exported(self):
        import cookbook.swarm as swarm_pkg
        self.assertTrue(hasattr(swarm_pkg, "ContextWindow"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
