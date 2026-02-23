"""Tests for ai_helper.tools."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from ai_helper.tools import Tool, ToolParam, ToolRegistry, ToolResult


# ---------------------------------------------------------------------------
# ToolParam / ToolResult / Tool
# ---------------------------------------------------------------------------

class TestToolParam(unittest.TestCase):
    def test_required_default(self):
        p = ToolParam("path", "str", "a path")
        self.assertTrue(p.required)

    def test_optional(self):
        p = ToolParam("ext", "str", "extension", required=False, default=".txt")
        self.assertFalse(p.required)
        self.assertEqual(p.default, ".txt")


class TestToolResult(unittest.TestCase):
    def test_str_success(self):
        r = ToolResult(tool_name="read_file", success=True, output="hello")
        self.assertEqual(str(r), "hello")

    def test_str_failure(self):
        r = ToolResult(tool_name="read_file", success=False, output="", error="not found")
        self.assertIn("ERROR", str(r))
        self.assertIn("not found", str(r))


class TestTool(unittest.TestCase):
    def _make_tool(self, handler=None):
        if handler is None:
            handler = lambda **kw: ToolResult("t", True, "ok")  # noqa: E731
        return Tool(
            name="t",
            description="test tool",
            params=[
                ToolParam("required_arg", "str", "required"),
                ToolParam("optional_arg", "str", "optional", required=False, default="x"),
            ],
            handler=handler,
        )

    def test_missing_required_returns_error(self):
        tool = self._make_tool()
        result = tool.invoke()
        self.assertFalse(result.success)
        self.assertIn("required_arg", result.error)

    def test_handler_called_with_kwargs(self):
        received = {}
        def handler(**kw):
            received.update(kw)
            return ToolResult("t", True, "done")
        tool = self._make_tool(handler)
        result = tool.invoke(required_arg="hello", optional_arg="world")
        self.assertTrue(result.success)
        self.assertEqual(received["required_arg"], "hello")

    def test_handler_exception_returns_error_result(self):
        def bad(**kw):
            raise ValueError("boom")
        tool = self._make_tool(bad)
        result = tool.invoke(required_arg="x")
        self.assertFalse(result.success)
        self.assertIn("ValueError", result.error)

    def test_describe_contains_name(self):
        tool = self._make_tool()
        self.assertIn("t", tool.describe())
        self.assertIn("required_arg", tool.describe())


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry(register_defaults=False)

    def _add_tool(self, name="echo", succeed=True):
        def handler(text: str = "") -> ToolResult:
            return ToolResult(name, succeed, text if succeed else "", error="" if succeed else "fail")
        self.reg.register(Tool(
            name=name, description="echo", category="test",
            params=[ToolParam("text", "str", "text", required=False)],
            handler=handler,
        ))

    def test_register_and_get(self):
        self._add_tool("mytool")
        self.assertIsNotNone(self.reg.get("mytool"))

    def test_get_unknown_returns_none(self):
        self.assertIsNone(self.reg.get("unknown"))

    def test_unregister(self):
        self._add_tool("del_me")
        self.assertTrue(self.reg.unregister("del_me"))
        self.assertIsNone(self.reg.get("del_me"))

    def test_unregister_unknown_returns_false(self):
        self.assertFalse(self.reg.unregister("ghost"))

    def test_invoke_known_tool(self):
        self._add_tool("echo")
        result = self.reg.invoke("echo", text="hello")
        self.assertTrue(result.success)
        self.assertEqual(result.output, "hello")

    def test_invoke_unknown_returns_error(self):
        result = self.reg.invoke("does_not_exist")
        self.assertFalse(result.success)
        self.assertIn("Unknown tool", result.error)

    def test_list_tools_by_category(self):
        self._add_tool("t1")
        self.reg.register(Tool(
            name="other", description="x", category="other",
            params=[], handler=lambda: ToolResult("other", True, ""),
        ))
        test_tools = self.reg.list_tools(category="test")
        self.assertEqual(len(test_tools), 1)
        self.assertEqual(test_tools[0].name, "t1")

    def test_describe_all_contains_tool_name(self):
        self._add_tool("echo")
        desc = self.reg.describe_all()
        self.assertIn("echo", desc)


# ---------------------------------------------------------------------------
# Built-in tools (smoke tests — mock side-effectful deps)
# ---------------------------------------------------------------------------

class TestBuiltinTools(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry(register_defaults=True)
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_file_success(self):
        p = self.root / "test.txt"
        p.write_text("hello file")
        result = self.reg.invoke("read_file", path=str(p))
        self.assertTrue(result.success)
        self.assertIn("hello file", result.output)

    def test_read_file_missing(self):
        result = self.reg.invoke("read_file", path=str(self.root / "ghost.txt"))
        self.assertFalse(result.success)

    def test_write_file(self):
        p = self.root / "out.txt"
        result = self.reg.invoke("write_file", path=str(p), content="written!")
        self.assertTrue(result.success)
        self.assertEqual(p.read_text(), "written!")

    def test_append_file(self):
        p = self.root / "log.txt"
        self.reg.invoke("append_file", path=str(p), content="line1\n")
        self.reg.invoke("append_file", path=str(p), content="line2\n")
        self.assertIn("line2", p.read_text())

    def test_search_files_finds_file(self):
        (self.root / "needle.py").write_text("content")
        result = self.reg.invoke("search_files", query="needle.py", root=str(self.root))
        self.assertTrue(result.success)
        self.assertIn("needle.py", result.output)

    def test_list_directory(self):
        (self.root / "sub").mkdir()
        result = self.reg.invoke("list_directory", path=str(self.root))
        self.assertTrue(result.success)
        self.assertIn("sub", result.output)

    def test_list_directory_missing(self):
        result = self.reg.invoke("list_directory", path=str(self.root / "nope"))
        self.assertFalse(result.success)

    def test_run_program_echo(self):
        result = self.reg.invoke("run_program", command="echo", args="hello")
        self.assertTrue(result.success)
        self.assertIn("hello", result.output)

    def test_list_programs(self):
        result = self.reg.invoke("list_programs")
        self.assertTrue(result.success)
        self.assertIn("processes", result.output.lower())

    def test_system_snapshot(self):
        result = self.reg.invoke("system_snapshot")
        self.assertTrue(result.success)
        self.assertIn("CPU", result.output)

    def test_gpu_stats_no_crash(self):
        # No GPU in CI — must not crash, just return gracefully
        result = self.reg.invoke("gpu_stats")
        self.assertIsInstance(result, ToolResult)

    def test_list_ai_apps_no_crash(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            result = self.reg.invoke("list_ai_apps")
        self.assertTrue(result.success)
        self.assertIn("Ollama", result.output)

    def test_ask_ollama_not_running(self):
        with patch("ai_helper.ai_integrations._reachable", return_value=False):
            result = self.reg.invoke("ask_ollama", prompt="hello")
        self.assertFalse(result.success)
        self.assertIn("not running", result.error)

    def test_list_ollama_models_not_running(self):
        with patch("ai_helper.ai_integrations._get", return_value=None):
            result = self.reg.invoke("list_ollama_models")
        self.assertTrue(result.success)  # graceful: returns empty message
        self.assertIn("No models", result.output)


if __name__ == "__main__":
    unittest.main()
