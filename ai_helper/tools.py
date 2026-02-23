"""Tool registry.

Every capability AI Helper has is exposed here as a *tool* — a named,
typed, self-describing callable.  The :class:`ToolRegistry` is the single
place the agent queries to discover what it can do and invokes to do it.

Built-in tools cover:
- File operations (read, write, search, list)
- Program execution (run any command)
- System information (CPU, memory, disk, processes)
- NVIDIA GPU statistics
- AI program discovery and Ollama inference
"""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class ToolParam:
    """Description of one parameter accepted by a :class:`Tool`."""
    name: str
    type: str           # "str" | "int" | "float" | "bool" | "list"
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolResult:
    """The outcome of invoking a :class:`Tool`."""
    tool_name: str
    success: bool
    output: str
    error: str = ""
    data: Any = None    # structured data for programmatic use

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"[ERROR from {self.tool_name}] {self.error}"


@dataclass
class Tool:
    """A named, typed capability that the agent can invoke."""
    name: str
    description: str
    params: List[ToolParam]
    handler: Callable[..., ToolResult]
    category: str = "general"

    def invoke(self, **kwargs: Any) -> ToolResult:
        """Validate required params then call the handler."""
        for p in self.params:
            if p.required and p.name not in kwargs:
                return ToolResult(
                    tool_name=self.name, success=False, output="",
                    error=f"Missing required parameter: {p.name!r}",
                )
        try:
            return self.handler(**kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool %r raised an exception", self.name)
            return ToolResult(
                tool_name=self.name, success=False, output="",
                error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )

    def describe(self) -> str:
        """Return a compact human-readable description for LLM prompts."""
        params_str = ", ".join(
            f"{p.name}: {p.type}{'?' if not p.required else ''}"
            for p in self.params
        )
        return f"{self.name}({params_str}) — {self.description}"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Register and invoke tools by name.

    Parameters
    ----------
    register_defaults:
        When ``True`` (default) all built-in AI Helper tools are
        registered automatically.
    """

    def __init__(self, register_defaults: bool = True) -> None:
        self._tools: Dict[str, Tool] = {}
        if register_defaults:
            _register_builtin_tools(self)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Add *tool* to the registry, replacing any existing tool with the same name."""
        self._tools[tool.name] = tool
        logger.debug("Registered tool %r", tool.name)

    def unregister(self, name: str) -> bool:
        """Remove a tool by name.  Returns ``True`` if it existed."""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    # ------------------------------------------------------------------
    # Lookup / listing
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[Tool]:
        """Return the named tool, or *None*."""
        return self._tools.get(name)

    def list_tools(self, category: Optional[str] = None) -> List[Tool]:
        """Return all registered tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return sorted(tools, key=lambda t: (t.category, t.name))

    def describe_all(self) -> str:
        """Return a formatted tool catalogue for use in LLM system prompts."""
        lines: List[str] = ["Available tools:"]
        for tool in self.list_tools():
            lines.append(f"  {tool.describe()}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(self, name: str, **kwargs: Any) -> ToolResult:
        """Invoke the named tool with *kwargs*.

        Returns an error :class:`ToolResult` if the tool is not found.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name, success=False, output="",
                error=f"Unknown tool: {name!r}. "
                      f"Available: {', '.join(sorted(self._tools))}",
            )
        return tool.invoke(**kwargs)


# ---------------------------------------------------------------------------
# Built-in tool handlers
# ---------------------------------------------------------------------------


def _ok(name: str, output: str, data: Any = None) -> ToolResult:
    return ToolResult(tool_name=name, success=True, output=output, data=data)


def _err(name: str, error: str) -> ToolResult:
    return ToolResult(tool_name=name, success=False, output="", error=error)


def _register_builtin_tools(reg: ToolRegistry) -> None:
    """Register all built-in tools into *reg*."""

    # ------------------------------------------------------------------ #
    # FILE TOOLS                                                           #
    # ------------------------------------------------------------------ #

    def _read_file(path: str) -> ToolResult:
        from .file_system import FileReader  # noqa: PLC0415
        try:
            content = FileReader().read(Path(path))
            return _ok("read_file", content, data={"path": path, "content": content})
        except (FileNotFoundError, IsADirectoryError) as exc:
            return _err("read_file", str(exc))

    reg.register(Tool(
        name="read_file",
        description="Read the full text content of any file on disk.",
        params=[ToolParam("path", "str", "Absolute or relative path to the file.")],
        handler=_read_file,
        category="files",
    ))

    def _write_file(path: str, content: str) -> ToolResult:
        from .file_system import FileWriter  # noqa: PLC0415
        written = FileWriter().write(Path(path), content)
        return _ok("write_file", f"Wrote {len(content)} characters to {written}",
                   data={"path": str(written)})

    reg.register(Tool(
        name="write_file",
        description="Write text content to a file, creating it if needed (backs up the original).",
        params=[
            ToolParam("path", "str", "Destination file path."),
            ToolParam("content", "str", "Text content to write."),
        ],
        handler=_write_file,
        category="files",
    ))

    def _append_file(path: str, content: str) -> ToolResult:
        from .file_system import FileWriter  # noqa: PLC0415
        FileWriter().append(Path(path), content)
        return _ok("append_file", f"Appended {len(content)} characters to {path}")

    reg.register(Tool(
        name="append_file",
        description="Append text to the end of a file without overwriting it.",
        params=[
            ToolParam("path", "str", "File path."),
            ToolParam("content", "str", "Text to append."),
        ],
        handler=_append_file,
        category="files",
    ))

    def _search_files(query: str = "*", root: str = "",
                      extensions: str = "", content: str = "") -> ToolResult:
        from .file_system import FileSearcher  # noqa: PLC0415
        searcher = FileSearcher(max_results=50)
        root_path = Path(root) if root else None
        ext_list = [e.strip() for e in extensions.split(",") if e.strip()] if extensions else None
        matches = searcher.search(
            name_pattern=query,
            root=root_path,
            extensions=ext_list,
            content_keyword=content,
        )
        if not matches:
            return _ok("search_files", "No files found matching the criteria.", data=[])
        lines = [str(m) for m in matches]
        return _ok("search_files", "\n".join(lines), data=[str(m.path) for m in matches])

    reg.register(Tool(
        name="search_files",
        description="Search for files by name pattern, extension or text content.",
        params=[
            ToolParam("query", "str", "Filename glob pattern, e.g. '*.py' or 'report*'.",
                      required=False, default="*"),
            ToolParam("root", "str", "Directory to search (default: home directory).",
                      required=False, default=""),
            ToolParam("extensions", "str",
                      "Comma-separated extensions to filter, e.g. '.py,.txt'.",
                      required=False, default=""),
            ToolParam("content", "str", "Keyword that must appear inside the file.",
                      required=False, default=""),
        ],
        handler=_search_files,
        category="files",
    ))

    def _list_directory(path: str = "") -> ToolResult:
        target = Path(path) if path else Path.home()
        if not target.exists():
            return _err("list_directory", f"Path does not exist: {target}")
        if target.is_file():
            return _err("list_directory", f"Path is a file, not a directory: {target}")
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = []
        for e in entries[:200]:
            tag = "/" if e.is_dir() else ""
            size = f"  {e.stat().st_size / 1024:.1f} KB" if e.is_file() else ""
            lines.append(f"  {e.name}{tag}{size}")
        return _ok("list_directory", f"Contents of {target}:\n" + "\n".join(lines),
                   data=[str(e) for e in entries])

    reg.register(Tool(
        name="list_directory",
        description="List the files and folders inside a directory.",
        params=[ToolParam("path", "str", "Directory path (default: home).",
                          required=False, default="")],
        handler=_list_directory,
        category="files",
    ))

    # ------------------------------------------------------------------ #
    # PROGRAM TOOLS                                                        #
    # ------------------------------------------------------------------ #

    def _run_program(command: str, args: str = "", input_data: str = "",
                     timeout: float = 30.0) -> ToolResult:
        from .program_interactor import ProgramInteractor  # noqa: PLC0415
        pi = ProgramInteractor(default_timeout=timeout)
        arg_list = args.split() if args else []
        result = pi.communicate(command, args=arg_list,
                                input_data=input_data or None,
                                timeout=timeout)
        if result.timed_out:
            return _err("run_program", f"Command timed out after {timeout}s")
        combined = (result.stdout + result.stderr).strip()
        success = result.returncode == 0
        if success:
            return _ok("run_program", combined or "(no output)",
                       data={"returncode": result.returncode, "stdout": result.stdout,
                             "stderr": result.stderr})
        return ToolResult(
            tool_name="run_program", success=False,
            output=combined,
            error=f"Exited with code {result.returncode}",
            data={"returncode": result.returncode},
        )

    reg.register(Tool(
        name="run_program",
        description="Run any installed program and return its output.",
        params=[
            ToolParam("command", "str", "The executable name or full path."),
            ToolParam("args", "str", "Space-separated arguments (optional).",
                      required=False, default=""),
            ToolParam("input_data", "str", "Text to send to the program's stdin.",
                      required=False, default=""),
            ToolParam("timeout", "float", "Seconds before giving up (default 30).",
                      required=False, default=30.0),
        ],
        handler=_run_program,
        category="programs",
    ))

    def _launch_program(command: str, args: str = "") -> ToolResult:
        from .program_interactor import ProgramInteractor  # noqa: PLC0415
        pi = ProgramInteractor()
        arg_list = args.split() if args else []
        result = pi.launch(command, args=arg_list, detach=True)
        if result.success:
            return _ok("launch_program", str(result), data={"pid": result.pid})
        return _err("launch_program", result.error)

    reg.register(Tool(
        name="launch_program",
        description="Launch an application in the background (detached, non-blocking).",
        params=[
            ToolParam("command", "str", "Executable name or path."),
            ToolParam("args", "str", "Space-separated arguments.", required=False, default=""),
        ],
        handler=_launch_program,
        category="programs",
    ))

    def _list_programs(name_filter: str = "") -> ToolResult:
        from .process_manager import ProcessManager  # noqa: PLC0415
        pm = ProcessManager()
        if name_filter:
            procs = pm.find_by_name(name_filter)
        else:
            procs = pm.list_processes()
        summary = pm.summary(procs)
        top = sorted(procs, key=lambda p: p.cpu_percent, reverse=True)[:20]
        lines = [f"  PID {p.pid:6d}  CPU {p.cpu_percent:5.1f}%  "
                 f"MEM {p.memory_mb:6.0f} MB  {p.name}"
                 for p in top]
        output = summary + "\n\nTop processes by CPU:\n" + "\n".join(lines)
        return _ok("list_programs", output, data=[{"pid": p.pid, "name": p.name} for p in top])

    reg.register(Tool(
        name="list_programs",
        description="List running programs / processes, optionally filtered by name.",
        params=[ToolParam("name_filter", "str", "Filter by program name (optional).",
                          required=False, default="")],
        handler=_list_programs,
        category="programs",
    ))

    # ------------------------------------------------------------------ #
    # SYSTEM TOOLS                                                         #
    # ------------------------------------------------------------------ #

    def _system_snapshot() -> ToolResult:
        from .monitor import SystemMonitor  # noqa: PLC0415
        mon = SystemMonitor()
        snap = mon.snapshot()
        text = mon.format_snapshot(snap)
        alerts = mon.alerts(snap)
        if alerts:
            text += "\n\nAlerts:\n" + "\n".join(f"  ⚠ {a}" for a in alerts)
        return _ok("system_snapshot", text, data={
            "cpu_percent": snap.cpu_percent,
            "memory_percent": snap.memory_percent,
        })

    reg.register(Tool(
        name="system_snapshot",
        description="Get current CPU, memory and disk usage with any active alerts.",
        params=[],
        handler=_system_snapshot,
        category="system",
    ))

    def _gpu_stats() -> ToolResult:
        from .gpu_monitor import GpuMonitor  # noqa: PLC0415
        gpu = GpuMonitor()
        snaps = gpu.snapshots()
        text = gpu.format_snapshots(snaps)
        alerts = gpu.alerts(snaps)
        if alerts:
            text += "\n\nAlerts:\n" + "\n".join(f"  ⚠ {a}" for a in alerts)
        return _ok("gpu_stats", text, data=[{
            "index": s.index, "name": s.name,
            "vram_percent": s.vram_percent, "temperature_c": s.temperature_c,
        } for s in snaps])

    reg.register(Tool(
        name="gpu_stats",
        description="Get NVIDIA GPU VRAM, temperature, utilisation and per-process memory.",
        params=[],
        handler=_gpu_stats,
        category="system",
    ))

    # ------------------------------------------------------------------ #
    # AI TOOLS                                                             #
    # ------------------------------------------------------------------ #

    def _list_ai_apps() -> ToolResult:
        from .ai_integrations import AIAppRegistry  # noqa: PLC0415
        registry = AIAppRegistry(timeout=2.0)
        statuses = registry.discover()
        text = registry.format_status(statuses)
        return _ok("list_ai_apps", text,
                   data=[{"name": s.name, "running": s.running} for s in statuses])

    reg.register(Tool(
        name="list_ai_apps",
        description="Discover all known AI programs (Ollama, ComfyUI, LM Studio, etc.) and show which are running.",
        params=[],
        handler=_list_ai_apps,
        category="ai",
    ))

    def _ask_ollama(prompt: str, model: str = "llama3",
                    url: str = "http://localhost:11434") -> ToolResult:
        from .ai_integrations import OllamaClient  # noqa: PLC0415
        client = OllamaClient(base_url=url)
        if not client.is_running():
            return _err("ask_ollama",
                        f"Ollama is not running at {url}. "
                        "Start it with: ollama serve")
        result = client.generate(model=model, prompt=prompt)
        if result.error:
            return _err("ask_ollama", result.error)
        return _ok("ask_ollama", result.response,
                   data={"model": model, "response": result.response})

    reg.register(Tool(
        name="ask_ollama",
        description="Send a prompt to a local Ollama LLM model and return its response.",
        params=[
            ToolParam("prompt", "str", "The question or instruction."),
            ToolParam("model", "str", "Ollama model name (default: llama3).",
                      required=False, default="llama3"),
            ToolParam("url", "str", "Ollama base URL.",
                      required=False, default="http://localhost:11434"),
        ],
        handler=_ask_ollama,
        category="ai",
    ))

    def _list_ollama_models(url: str = "http://localhost:11434") -> ToolResult:
        from .ai_integrations import OllamaClient  # noqa: PLC0415
        client = OllamaClient(base_url=url)
        models = client.list_models()
        if not models:
            return _ok("list_ollama_models", "No models found (or Ollama is not running).", data=[])
        lines = [f"  {m}" for m in models]
        return _ok("list_ollama_models", "Ollama models:\n" + "\n".join(lines),
                   data=[m.name for m in models])

    reg.register(Tool(
        name="list_ollama_models",
        description="List all models currently available in the local Ollama installation.",
        params=[ToolParam("url", "str", "Ollama base URL.",
                          required=False, default="http://localhost:11434")],
        handler=_list_ollama_models,
        category="ai",
    ))
