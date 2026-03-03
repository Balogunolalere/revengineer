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
    AgentSpec, AgentResult, AgentStatus, SwarmMode, SwarmPlan, SwarmResult,
)
from cookbook.swarm.config import SwarmConfig
from cookbook.swarm.engine import RateLimiter, SwarmEngine
from cookbook.swarm.orchestrator import Orchestrator, _extract_json
from cookbook.swarm.renderer import SwarmRenderer, _strip_ansi, _build_markdown


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
        self.assertEqual(a.timeout, 120.0)
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
                            max_tokens=None, is_orchestrator=False):
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
                            max_tokens=None, is_orchestrator=False):
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
                            max_tokens=None, is_orchestrator=False):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
