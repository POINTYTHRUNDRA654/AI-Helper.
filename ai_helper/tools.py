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

    # ------------------------------------------------------------------ #
    # MESH TOOLS                                                           #
    # ------------------------------------------------------------------ #

    def _mesh_ask(question: str) -> ToolResult:
        from .mesh_engine import MeshEngine  # noqa: PLC0415
        engine = MeshEngine()
        answer = engine.ask_knowledge(question)
        return _ok("mesh_ask", answer, data={"question": question, "answer": answer})

    reg.register(Tool(
        name="mesh_ask",
        description=(
            "Ask Mossy a Fallout 4 mesh / modding question. "
            "Covers polygon budgets, NIF format, textures, LOD, collision, workflow and free tools."
        ),
        params=[ToolParam("question", "str", "The modding or mesh question.")],
        handler=_mesh_ask,
        category="mesh",
    ))

    def _mesh_workflow() -> ToolResult:
        from .mesh_engine import MeshEngine  # noqa: PLC0415
        engine = MeshEngine()
        text = engine.get_workflow()
        return _ok("mesh_workflow", text)

    reg.register(Tool(
        name="mesh_workflow",
        description="Show the recommended image-to-Fallout-4-mesh workflow (free tools only).",
        params=[],
        handler=_mesh_workflow,
        category="mesh",
    ))

    def _mesh_list_free_tools() -> ToolResult:
        from .mesh_engine import FreeImage3DClient  # noqa: PLC0415
        text = FreeImage3DClient.list_backends()
        return _ok("mesh_list_free_tools", text)

    reg.register(Tool(
        name="mesh_list_free_tools",
        description=(
            "List all free open-source image-to-3D tools Mossy supports "
            "(TripoSG, TRELLIS, TripoSR, Shap-E — all from GitHub/HuggingFace)."
        ),
        params=[],
        handler=_mesh_list_free_tools,
        category="mesh",
    ))

    def _mesh_install_instructions(backend: str) -> ToolResult:
        from .mesh_engine import FreeImage3DClient  # noqa: PLC0415
        text = FreeImage3DClient.install_instructions(backend)
        return _ok("mesh_install_instructions", text, data={"backend": backend})

    reg.register(Tool(
        name="mesh_install_instructions",
        description=(
            "Get step-by-step install instructions for a free image-to-3D backend. "
            "Options: triposg, trellis, triposr, shap_e."
        ),
        params=[
            ToolParam("backend", "str",
                      "Backend name: triposg (recommended), trellis, triposr, or shap_e."),
        ],
        handler=_mesh_install_instructions,
        category="mesh",
    ))

    def _mesh_free_image_to_3d(
        image_path: str,
        output_dir: str = "",
        backend: str = "triposg",
        faces: int = 0,
    ) -> ToolResult:
        from .mesh_engine import MeshEngine  # noqa: PLC0415
        engine = MeshEngine()
        kwargs = {}
        if faces > 0:
            kwargs["faces"] = faces
        result = engine.free_image_to_3d(
            image_path=image_path,
            output_dir=output_dir or None,
            backend=backend,
            **kwargs,
        )
        return _ok("mesh_free_image_to_3d", result.summary, data={
            "backend": result.backend,
            "mesh_path": result.mesh_path,
            "exported_files": result.exported_files,
            "success": result.success,
        }) if result.success else _err("mesh_free_image_to_3d", result.error)

    reg.register(Tool(
        name="mesh_free_image_to_3d",
        description=(
            "Convert an image to a 3D mesh using a free open-source AI model "
            "(TripoSG by default — MIT license, 8 GB VRAM, best free quality). "
            "Outputs GLB/OBJ ready for Blender → NIF export."
        ),
        params=[
            ToolParam("image_path", "str", "Path to the input image (PNG or JPEG)."),
            ToolParam("output_dir", "str", "Where to save the output mesh.",
                      required=False, default=""),
            ToolParam("backend", "str",
                      "Free backend: triposg (default), trellis, triposr, shap_e.",
                      required=False, default="triposg"),
            ToolParam("faces", "int",
                      "Max triangle count (0 = model default). e.g. 5000 for FO4 weapon.",
                      required=False, default=0),
        ],
        handler=_mesh_free_image_to_3d,
        category="mesh",
    ))

    def _mesh_meshy_image_to_3d(
        image_path: str,
        output_dir: str = "",
        api_key: str = "",
        download_fmt: str = "obj",
        target_polycount: int = 10000,
    ) -> ToolResult:
        from .mesh_engine import MeshEngine  # noqa: PLC0415
        engine = MeshEngine()
        result = engine.meshy_image_to_3d(
            image_path=image_path,
            output_dir=output_dir or None,
            api_key=api_key or None,
            download_fmt=download_fmt,
            target_polycount=target_polycount,
        )
        if result.success:
            return _ok("mesh_meshy_image_to_3d", result.summary, data={
                "mesh_path": result.mesh_path,
                "exported_files": result.exported_files,
            })
        return _err("mesh_meshy_image_to_3d",
                    "; ".join(result.errors) or "Meshy conversion failed")

    reg.register(Tool(
        name="mesh_meshy_image_to_3d",
        description=(
            "Convert an image to a 3D mesh using the Meshy API (paid subscription). "
            "Requires a Meshy API key (set MESHY_API_KEY env var or pass api_key)."
        ),
        params=[
            ToolParam("image_path", "str", "Path to the input image (PNG or JPEG)."),
            ToolParam("output_dir", "str", "Where to save the downloaded mesh.",
                      required=False, default=""),
            ToolParam("api_key", "str", "Meshy API key (or use MESHY_API_KEY env var).",
                      required=False, default=""),
            ToolParam("download_fmt", "str", "Output format: obj, glb, fbx, usdz.",
                      required=False, default="obj"),
            ToolParam("target_polycount", "int",
                      "Polygon budget hint for Meshy (default 10000).",
                      required=False, default=10000),
        ],
        handler=_mesh_meshy_image_to_3d,
        category="mesh",
    ))

    def _mesh_validate(mesh_path: str, asset_type: str = "settlement_object_medium") -> ToolResult:
        from .mesh_engine import MeshEngine  # noqa: PLC0415
        engine = MeshEngine()
        validation = engine.validate_mesh(mesh_path, asset_type)
        return _ok("mesh_validate", str(validation), data={
            "passed": validation.passed,
            "poly_count": validation.poly_count,
            "issues": validation.issues,
        })

    reg.register(Tool(
        name="mesh_validate",
        description=(
            "Validate a mesh file against Fallout 4 requirements "
            "(polygon budget, UVs, normals, watertight check)."
        ),
        params=[
            ToolParam("mesh_path", "str", "Path to OBJ, PLY, GLB, or STL file."),
            ToolParam("asset_type", "str",
                      "FO4 asset type for budget check: weapon, armor_piece, "
                      "settlement_object_medium, character_head, etc.",
                      required=False, default="settlement_object_medium"),
        ],
        handler=_mesh_validate,
        category="mesh",
    ))

    def _mesh_check_deps() -> ToolResult:
        from .mesh_engine import MeshEngine  # noqa: PLC0415
        engine = MeshEngine()
        text = engine.check_dependencies()
        return _ok("mesh_check_deps", text)

    reg.register(Tool(
        name="mesh_check_deps",
        description=(
            "Check which mesh pipeline packages are installed and which free "
            "3D model repos have been cloned."
        ),
        params=[],
        handler=_mesh_check_deps,
        category="mesh",
    ))

    # ------------------------------------------------------------------ #
    # WEB TOOLS                                                            #
    # ------------------------------------------------------------------ #

    def _web_fetch(url: str, max_chars: int = 4000) -> ToolResult:
        """Fetch a URL and return the text content."""
        import urllib.request as _urllib  # noqa: PLC0415
        import urllib.error as _urllib_err  # noqa: PLC0415
        import urllib.parse as _urlparse  # noqa: PLC0415
        import html  # noqa: PLC0415
        import re as _re  # noqa: PLC0415
        # Validate URL scheme — only allow http/https to prevent SSRF
        parsed = _urlparse.urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return _err("web_fetch", f"Disallowed URL scheme {parsed.scheme!r}. Only http/https are allowed.")
        try:
            req = _urllib.Request(
                url,
                headers={"User-Agent": "AI-Helper/0.1 (+https://github.com/AI-Helper)"},
            )
            with _urllib.urlopen(req, timeout=15) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8", errors="replace")
            # Strip HTML tags for readability
            text = _re.sub(r"<[^>]+>", " ", raw)
            text = html.unescape(text)
            text = _re.sub(r"\s{2,}", " ", text).strip()
            return _ok("web_fetch", text[:max_chars], data={"url": url, "length": len(text)})
        except _urllib_err.HTTPError as exc:
            return _err("web_fetch", f"HTTP {exc.code} {exc.reason} — {url}")
        except Exception as exc:  # noqa: BLE001
            return _err("web_fetch", str(exc))

    reg.register(Tool(
        name="web_fetch",
        description="Fetch a URL and return its text content (HTML stripped).",
        params=[
            ToolParam("url", "str", "The URL to fetch."),
            ToolParam("max_chars", "int", "Maximum characters to return (default 4000).",
                      required=False, default=4000),
        ],
        handler=_web_fetch,
        category="web",
    ))

    def _open_url(url: str) -> ToolResult:
        """Open a URL in the system default browser."""
        import webbrowser  # noqa: PLC0415
        try:
            webbrowser.open(url)
            return _ok("open_url", f"Opened in browser: {url}", data={"url": url})
        except Exception as exc:  # noqa: BLE001
            return _err("open_url", str(exc))

    reg.register(Tool(
        name="open_url",
        description="Open a URL in the system default web browser.",
        params=[ToolParam("url", "str", "The URL to open.")],
        handler=_open_url,
        category="web",
    ))

    # ------------------------------------------------------------------ #
    # VOICE TOOLS                                                          #
    # ------------------------------------------------------------------ #

    def _speak(text: str, rate: int = 0, volume: float = 0.0) -> ToolResult:
        """Speak text aloud via the TTS engine."""
        from .voice import Speaker  # noqa: PLC0415
        speaker = Speaker()
        if rate > 0:
            speaker.set_rate(rate)
        if 0.0 < volume <= 1.0:
            speaker.set_volume(volume)
        speaker.speak(text)
        return _ok("speak", f"Speaking: {text[:80]}…" if len(text) > 80 else f"Speaking: {text}")

    reg.register(Tool(
        name="speak",
        description="Speak text aloud using the system TTS engine (pyttsx3 or OS fallback).",
        params=[
            ToolParam("text", "str", "Text to speak."),
            ToolParam("rate", "int", "Speech rate in words per minute (0 = default 175).",
                      required=False, default=0),
            ToolParam("volume", "float", "Volume 0.0–1.0 (0.0 = default).",
                      required=False, default=0.0),
        ],
        handler=_speak,
        category="voice",
    ))

    def _list_voices() -> ToolResult:
        """List available TTS voices."""
        from .voice import Speaker  # noqa: PLC0415
        speaker = Speaker(enabled=False)
        voices = speaker.list_voices()
        if not voices:
            return _ok("list_voices", "No pyttsx3 voices found (pyttsx3 may not be installed).", data=[])
        text = "Available TTS voices:\n" + "\n".join(f"  {v}" for v in voices)
        return _ok("list_voices", text, data=voices)

    reg.register(Tool(
        name="list_voices",
        description="List all available text-to-speech voices on this system.",
        params=[],
        handler=_list_voices,
        category="voice",
    ))

    def _set_voice(voice_id: str) -> ToolResult:
        """Select a TTS voice by name fragment."""
        from .voice import Speaker  # noqa: PLC0415
        speaker = Speaker()
        speaker.set_voice(voice_id)
        return _ok("set_voice", f"Voice set to: {voice_id!r}")

    reg.register(Tool(
        name="set_voice",
        description="Select a text-to-speech voice by name or id fragment (e.g. 'zira', 'david', 'daniel').",
        params=[ToolParam("voice_id", "str", "Name or id fragment of the desired voice.")],
        handler=_set_voice,
        category="voice",
    ))

    # ------------------------------------------------------------------ #
    # SCREENSHOT                                                           #
    # ------------------------------------------------------------------ #

    def _screenshot(output_path: str = "") -> ToolResult:
        """Take a screenshot and save it to disk."""
        import time as _time  # noqa: PLC0415
        from pathlib import Path as _Path  # noqa: PLC0415
        from .config import get_data_dir  # noqa: PLC0415
        if not output_path:
            ts = _time.strftime("%Y%m%d_%H%M%S")
            output_path = str(get_data_dir() / "screenshots" / f"screenshot_{ts}.png")
        dest = _Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            # Try PIL/Pillow ImageGrab first (Windows/macOS)
            from PIL import ImageGrab  # noqa: PLC0415
            img = ImageGrab.grab()
            img.save(str(dest))
            return _ok("screenshot", f"Screenshot saved: {dest}",
                       data={"path": str(dest), "size": list(img.size)})
        except ImportError:
            pass
        # Fallback: scrot (Linux)
        import shutil as _shutil  # noqa: PLC0415
        import subprocess as _subprocess  # noqa: PLC0415
        if _shutil.which("scrot"):
            proc = _subprocess.run(
                ["scrot", str(dest)], capture_output=True, text=True, timeout=10
            )
            if proc.returncode != 0:
                logger.warning("scrot failed (rc=%d): %s", proc.returncode, proc.stderr.strip())
            elif dest.exists():
                return _ok("screenshot", f"Screenshot saved: {dest}", data={"path": str(dest)})
        # Fallback: gnome-screenshot
        if _shutil.which("gnome-screenshot"):
            proc = _subprocess.run(
                ["gnome-screenshot", "-f", str(dest)], capture_output=True, text=True, timeout=10
            )
            if proc.returncode != 0:
                logger.warning("gnome-screenshot failed (rc=%d): %s",
                               proc.returncode, proc.stderr.strip())
            elif dest.exists():
                return _ok("screenshot", f"Screenshot saved: {dest}", data={"path": str(dest)})
        return _err("screenshot",
                    "No screenshot tool available. Install Pillow (pip install Pillow) or scrot.")

    reg.register(Tool(
        name="screenshot",
        description=(
            "Take a screenshot of the current screen and save it to disk. "
            "Requires Pillow (Windows/macOS) or scrot/gnome-screenshot (Linux)."
        ),
        params=[
            ToolParam("output_path", "str",
                      "Where to save the PNG. Defaults to AI-Helper/Data/screenshots/<timestamp>.png.",
                      required=False, default=""),
        ],
        handler=_screenshot,
        category="system",
    ))

    # ------------------------------------------------------------------ #
    # CLIPBOARD TOOLS                                                      #
    # ------------------------------------------------------------------ #

    def _clipboard_read() -> ToolResult:
        """Read current clipboard contents."""
        try:
            import pyperclip  # noqa: PLC0415
            text = pyperclip.paste() or ""
            return _ok("clipboard_read", text or "(clipboard is empty)", data={"content": text})
        except Exception as exc:  # noqa: BLE001
            return _err("clipboard_read", f"Could not read clipboard: {exc}")

    reg.register(Tool(
        name="clipboard_read",
        description="Read the current clipboard contents and return the text.",
        params=[],
        handler=_clipboard_read,
        category="system",
    ))

    def _clipboard_write(text: str) -> ToolResult:
        """Write text to the clipboard."""
        try:
            import pyperclip  # noqa: PLC0415
            pyperclip.copy(text)
            preview = text[:80] + "…" if len(text) > 80 else text
            return _ok("clipboard_write", f"Copied to clipboard: {preview}")
        except Exception as exc:  # noqa: BLE001
            return _err("clipboard_write", f"Could not write clipboard: {exc}")

    reg.register(Tool(
        name="clipboard_write",
        description="Write text to the system clipboard.",
        params=[ToolParam("text", "str", "Text to copy to the clipboard.")],
        handler=_clipboard_write,
        category="system",
    ))

    # ------------------------------------------------------------------ #
    # RESILIENCE / DIAGNOSTICS TOOLS                                       #
    # ------------------------------------------------------------------ #

    def _circuit_breaker_status() -> ToolResult:
        """Show status of all AI service circuit breakers."""
        try:
            from .ai_integrations import _BREAKERS  # noqa: PLC0415
            lines = ["Circuit breaker status for AI services:"]
            for name, cb in _BREAKERS.items():
                icon = "✓" if not cb.is_open else "✗ OPEN"
                lines.append(
                    f"  {name:<14} {icon}  (failures={cb._failure_count}/{cb.failure_threshold})"
                )
            return _ok("circuit_breaker_status", "\n".join(lines))
        except Exception as exc:  # noqa: BLE001
            return _err("circuit_breaker_status", str(exc))

    reg.register(Tool(
        name="circuit_breaker_status",
        description="Show whether each AI service circuit breaker is open (failing fast) or closed (healthy).",
        params=[],
        handler=_circuit_breaker_status,
        category="ai",
    ))

    def _reset_circuit_breaker(service: str) -> ToolResult:
        """Manually close a circuit breaker for a service."""
        try:
            from .ai_integrations import _BREAKERS  # noqa: PLC0415
            cb = _BREAKERS.get(service)
            if cb is None:
                valid = ", ".join(_BREAKERS)
                return _err("reset_circuit_breaker",
                            f"Unknown service {service!r}. Valid: {valid}")
            cb.reset()
            return _ok("reset_circuit_breaker", f"Circuit breaker for {service!r} reset to CLOSED.")
        except Exception as exc:  # noqa: BLE001
            return _err("reset_circuit_breaker", str(exc))

    reg.register(Tool(
        name="reset_circuit_breaker",
        description="Manually close (reset) a circuit breaker for an AI service after it has recovered.",
        params=[
            ToolParam("service", "str",
                      "Service name: ollama, lmstudio, comfyui, sdwebui, openwebui, "
                      "localai, textgen, oobabooga, jan, llamacpp."),
        ],
        handler=_reset_circuit_breaker,
        category="ai",
    ))

