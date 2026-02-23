"""Tests for ai_helper.agent."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from ai_helper.agent import Agent, AgentResult, AgentStep
from ai_helper.tools import ToolRegistry, ToolResult


def _make_registry_with_echo() -> ToolRegistry:
    """Return a minimal registry with a single echo tool for testing."""
    from ai_helper.tools import Tool, ToolParam
    reg = ToolRegistry(register_defaults=False)
    reg.register(Tool(
        name="echo",
        description="Echo the input text.",
        params=[ToolParam("text", "str", "text to echo")],
        handler=lambda text="": ToolResult("echo", True, text),
        category="test",
    ))
    return reg


# ---------------------------------------------------------------------------
# AgentStep / AgentResult
# ---------------------------------------------------------------------------

class TestAgentStep(unittest.TestCase):
    def _make_step(self, success=True):
        return AgentStep(
            step_number=1, thought="thinking",
            tool_name="echo", tool_args={"text": "hi"},
            result=ToolResult("echo", success, "hi" if success else "", error="fail" if not success else ""),
            elapsed_s=0.1,
        )

    def test_str_success(self):
        step = self._make_step(success=True)
        self.assertIn("✓", str(step))
        self.assertIn("echo", str(step))

    def test_str_failure(self):
        step = self._make_step(success=False)
        self.assertIn("✗", str(step))

    def test_agent_result_str(self):
        result = AgentResult(goal="do x", answer="done", steps=[self._make_step()])
        text = str(result)
        self.assertIn("do x", text)
        self.assertIn("done", text)


# ---------------------------------------------------------------------------
# _parse_action
# ---------------------------------------------------------------------------

class TestParseAction(unittest.TestCase):
    def test_valid_json(self):
        raw = '{"thought": "need info", "tool": "system_snapshot", "args": {}}'
        action = Agent._parse_action(raw)
        self.assertIsNotNone(action)
        self.assertEqual(action["tool"], "system_snapshot")

    def test_json_in_prose(self):
        raw = 'I will use the tool: {"thought": "x", "tool": "echo", "args": {"text": "hi"}}'
        action = Agent._parse_action(raw)
        self.assertIsNotNone(action)
        self.assertEqual(action["tool"], "echo")

    def test_json_in_code_fence(self):
        raw = "```json\n{\"thought\": \"ok\", \"tool\": \"finish\", \"args\": {\"answer\": \"done\"}}\n```"
        action = Agent._parse_action(raw)
        self.assertIsNotNone(action)
        self.assertEqual(action["tool"], "finish")

    def test_invalid_json_returns_none(self):
        self.assertIsNone(Agent._parse_action("not json at all"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(Agent._parse_action(""))


# ---------------------------------------------------------------------------
# Rule-based planner
# ---------------------------------------------------------------------------

class TestRuleBasedPlanner(unittest.TestCase):
    def _agent(self):
        reg = ToolRegistry(register_defaults=True)
        return Agent(registry=reg, ollama_url="http://localhost:19999")

    def test_cpu_routes_to_system_snapshot(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False):
            result = agent.execute("What is my CPU usage?")
        self.assertTrue(any(s.tool_name == "system_snapshot" for s in result.steps))

    def test_gpu_routes_to_gpu_stats(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False):
            result = agent.execute("Show me GPU VRAM usage")
        self.assertTrue(any(s.tool_name == "gpu_stats" for s in result.steps))

    def test_list_ai_apps(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False), \
             patch("ai_helper.ai_integrations._reachable", return_value=False):
            result = agent.execute("What AI apps are running?")
        self.assertTrue(any(s.tool_name == "list_ai_apps" for s in result.steps))

    def test_search_files_route(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False):
            result = agent.execute("Search for *.py files in my home directory")
        self.assertTrue(any(s.tool_name == "search_files" for s in result.steps))

    def test_list_processes_route(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False):
            result = agent.execute("What processes are running?")
        self.assertTrue(any(s.tool_name == "list_programs" for s in result.steps))

    def test_unknown_goal_falls_back_to_system_snapshot(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False):
            result = agent.execute("xyzzy_unknown_request_123")
        self.assertTrue(any(s.tool_name == "system_snapshot" for s in result.steps))
        self.assertIn("status", result.answer.lower())

    def test_result_has_answer(self):
        agent = self._agent()
        with patch.object(agent, "_ollama_available", return_value=False):
            result = agent.execute("Show system status")
        self.assertTrue(result.answer)
        self.assertIsInstance(result.total_elapsed_s, float)


# ---------------------------------------------------------------------------
# Ollama loop (mocked)
# ---------------------------------------------------------------------------

class TestOllamaLoop(unittest.TestCase):
    def _agent_with_mock_ollama(self, responses):
        """Build an agent whose Ollama client yields *responses* in sequence."""
        from ai_helper.ai_integrations import GenerateResult
        reg = ToolRegistry(register_defaults=True)
        agent = Agent(registry=reg, ollama_model="llama3", max_steps=5)

        call_count = [0]

        def fake_chat(self_client, model, messages, timeout=60.0):
            idx = min(call_count[0], len(responses) - 1)
            call_count[0] += 1
            return GenerateResult(
                model=model, prompt="", response=responses[idx], done=True
            )

        return agent, fake_chat

    def test_finish_action_sets_answer(self):
        finish_json = json.dumps({
            "thought": "done",
            "tool": "finish",
            "args": {"answer": "The answer is 42"},
        })
        agent, fake_chat = self._agent_with_mock_ollama([finish_json])

        with patch.object(agent, "_ollama_available", return_value=True), \
             patch("ai_helper.ai_integrations.OllamaClient.chat", fake_chat):
            result = agent.execute("What is the answer?")

        self.assertEqual(result.answer, "The answer is 42")
        self.assertEqual(len(result.steps), 0)  # finish on first reply = 0 tool steps

    def test_tool_call_then_finish(self):
        tool_json = json.dumps({
            "thought": "need system info",
            "tool": "system_snapshot",
            "args": {},
        })
        finish_json = json.dumps({
            "thought": "got it",
            "tool": "finish",
            "args": {"answer": "System looks good"},
        })
        agent, fake_chat = self._agent_with_mock_ollama([tool_json, finish_json])

        with patch.object(agent, "_ollama_available", return_value=True), \
             patch("ai_helper.ai_integrations.OllamaClient.chat", fake_chat):
            result = agent.execute("Check system health")

        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].tool_name, "system_snapshot")
        self.assertEqual(result.answer, "System looks good")

    def test_unparseable_reply_used_as_answer(self):
        from ai_helper.ai_integrations import GenerateResult
        agent = Agent(registry=ToolRegistry(register_defaults=False), max_steps=3)

        def fake_chat(self_client, model, messages, timeout=60.0):
            return GenerateResult(model=model, prompt="", response="Here is my answer in plain text.", done=True)

        with patch.object(agent, "_ollama_available", return_value=True), \
             patch("ai_helper.ai_integrations.OllamaClient.chat", fake_chat):
            result = agent.execute("What time is it?")

        self.assertIn("plain text", result.answer)

    def test_ollama_error_sets_answer(self):
        from ai_helper.ai_integrations import GenerateResult
        agent = Agent(registry=ToolRegistry(register_defaults=False), max_steps=3)

        def fake_chat(self_client, model, messages, timeout=60.0):
            return GenerateResult(model=model, prompt="", response="", done=False, error="connection refused")

        with patch.object(agent, "_ollama_available", return_value=True), \
             patch("ai_helper.ai_integrations.OllamaClient.chat", fake_chat):
            result = agent.execute("do something")

        self.assertIn("error", result.answer.lower())


# ---------------------------------------------------------------------------
# _extract_args
# ---------------------------------------------------------------------------

class TestExtractArgs(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(registry=ToolRegistry(register_defaults=False))

    def test_read_file_path_extraction(self):
        args = self.agent._extract_args("read_file", "Read the file /home/user/notes.txt")
        self.assertEqual(args["path"], "/home/user/notes.txt")

    def test_search_files_pattern(self):
        args = self.agent._extract_args("search_files", "search for *.py files")
        self.assertIn("*.py", args["query"])

    def test_run_program_command(self):
        args = self.agent._extract_args("run_program", "run notepad")
        self.assertEqual(args["command"], "notepad")


if __name__ == "__main__":
    unittest.main()
