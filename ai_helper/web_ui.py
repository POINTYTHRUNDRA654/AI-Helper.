"""Browser dashboard (web UI).

Serves a self-refreshing HTML dashboard on ``http://localhost:<port>``
using only Python's built-in ``http.server`` — no extra packages required.

The page shows the same data as the terminal dashboard:

* System stats (CPU, memory, disk)
* NVIDIA GPU stats
* Running AI programs
* Top processes
* Recent alerts

The page auto-refreshes every ``refresh_seconds`` seconds via a
``<meta http-equiv="refresh">`` tag and a small inline ``fetch()`` call to
``/api/status`` that updates the page without a full reload.

Usage
-----
::

    from ai_helper.web_ui import WebUI
    ui = WebUI(port=8765)
    ui.start()          # serves in background thread
    # … AI Helper runs …
    ui.stop()

Or from the CLI::

    python -m ai_helper --web-ui --web-port 8765
"""

from __future__ import annotations

import json
import logging
import platform
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 8765
_REFRESH_SECONDS = 5


def _collect_status() -> Dict[str, Any]:
    """Gather all system data into one JSON-serialisable dict."""
    data: Dict[str, Any] = {"ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    try:
        from .monitor import SystemMonitor  # noqa: PLC0415
        mon = SystemMonitor()
        snap = mon.snapshot()
        data["cpu"] = snap.cpu_percent
        data["memory"] = snap.memory_percent
        disks = []
        for d in snap.disk_partitions[:3]:
            disks.append({"mount": d.mountpoint, "pct": d.percent,
                          "used_gb": round(d.used_bytes / 1e9, 1),
                          "total_gb": round(d.total_bytes / 1e9, 1)})
        data["disks"] = disks
        data["alerts"] = mon.alerts(snap)
    except Exception as exc:  # noqa: BLE001
        data["alerts"] = [f"System monitor error: {exc}"]

    try:
        from .process_manager import ProcessManager  # noqa: PLC0415
        pm = ProcessManager()
        procs = pm.list_processes()
        top = sorted(procs, key=lambda p: p.cpu_percent, reverse=True)[:8]
        data["processes"] = [
            {"pid": p.pid, "name": p.name,
             "cpu": round(p.cpu_percent, 1), "mem_mb": round(p.memory_mb)}
            for p in top
        ]
    except Exception:  # noqa: BLE001
        data["processes"] = []

    try:
        from .gpu_monitor import GpuMonitor  # noqa: PLC0415
        gpu = GpuMonitor()
        snaps = gpu.snapshots()
        data["gpus"] = [
            {"index": s.index, "name": s.name,
             "vram_pct": round(s.vram_percent, 1),
             "vram_used_gb": s.vram_used_gb, "vram_total_gb": s.vram_total_gb,
             "temp_c": round(s.temperature_c), "util_pct": round(s.utilization_percent)}
            for s in snaps
        ]
    except Exception:  # noqa: BLE001
        data["gpus"] = []

    try:
        from .ai_integrations import AIAppRegistry  # noqa: PLC0415
        reg = AIAppRegistry(timeout=1.0)
        statuses = reg.discover()
        data["ai_apps"] = [
            {"name": s.name, "running": s.running, "url": s.url}
            for s in statuses
        ]
    except Exception:  # noqa: BLE001
        data["ai_apps"] = []

    return data


def _render_html(data: Dict[str, Any], port: int, refresh: int) -> str:
    def bar(pct: float) -> str:
        color = ("#4caf50" if pct < 60 else "#ff9800" if pct < 85 else "#f44336")
        return (f'<div class="bar-outer"><div class="bar-inner" '
                f'style="width:{min(pct,100):.0f}%;background:{color}"></div>'
                f'<span class="bar-label">{pct:.1f}%</span></div>')

    cpu_bar = bar(data.get("cpu", 0))
    mem_bar = bar(data.get("memory", 0))

    disk_rows = ""
    for d in data.get("disks", []):
        disk_rows += (
            f'<tr><td>{d["mount"]}</td>'
            f'<td>{bar(d["pct"])}</td>'
            f'<td>{d["used_gb"]}/{d["total_gb"]} GB</td></tr>'
        )

    gpu_rows = ""
    for g in data.get("gpus", []):
        gpu_rows += (
            f'<tr><td>GPU{g["index"]} {g["name"]}</td>'
            f'<td>{bar(g["vram_pct"])}</td>'
            f'<td>{g["temp_c"]}°C</td>'
            f'<td>{g["util_pct"]}%</td></tr>'
        )
    if not gpu_rows:
        gpu_rows = '<tr><td colspan="4">No NVIDIA GPU detected</td></tr>'

    proc_rows = ""
    for p in data.get("processes", []):
        cpu_color = ("#f44336" if p["cpu"] >= 85 else
                     "#ff9800" if p["cpu"] >= 60 else "#4caf50")
        proc_rows += (
            f'<tr><td>{p["pid"]}</td><td>{p["name"]}</td>'
            f'<td style="color:{cpu_color}">{p["cpu"]}%</td>'
            f'<td>{p["mem_mb"]} MB</td></tr>'
        )

    ai_rows = ""
    for a in data.get("ai_apps", []):
        dot = "🟢" if a["running"] else "🔴"
        link = f'<a href="{a["url"]}" target="_blank">{a["url"]}</a>' if a["running"] else a["url"]
        ai_rows += f'<tr><td>{dot}</td><td>{a["name"]}</td><td>{link}</td></tr>'

    alert_items = ""
    for al in data.get("alerts", []):
        alert_items += f'<li>⚠ {al}</li>'
    if not alert_items:
        alert_items = '<li style="color:#4caf50">✓ No active alerts</li>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="{refresh}">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI Helper Dashboard</title>
  <style>
    :root {{ --bg:#121212; --card:#1e1e1e; --border:#333; --text:#e0e0e0;
             --accent:#00bcd4; --muted:#888; }}
    * {{ box-sizing:border-box; margin:0; padding:0; }}
    body {{ background:var(--bg); color:var(--text); font-family:monospace;
             font-size:14px; padding:16px; }}
    h1 {{ color:var(--accent); font-size:1.4em; margin-bottom:4px; }}
    .ts {{ color:var(--muted); font-size:0.85em; margin-bottom:16px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
              gap:16px; }}
    .card {{ background:var(--card); border:1px solid var(--border);
              border-radius:8px; padding:16px; }}
    .card h2 {{ color:var(--accent); font-size:1em; margin-bottom:12px;
                 border-bottom:1px solid var(--border); padding-bottom:6px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th,td {{ text-align:left; padding:4px 8px; border-bottom:1px solid var(--border); }}
    th {{ color:var(--muted); font-size:0.85em; }}
    .bar-outer {{ width:100%; background:#333; border-radius:4px;
                   height:14px; position:relative; overflow:hidden; }}
    .bar-inner {{ height:100%; border-radius:4px; transition:width .3s; }}
    .bar-label {{ position:absolute; right:4px; top:0; font-size:11px;
                   line-height:14px; color:#fff; }}
    .stat-row {{ display:flex; gap:8px; align-items:center; margin-bottom:8px; }}
    .stat-label {{ width:70px; color:var(--muted); }}
    .stat-bar {{ flex:1; }}
    ul {{ padding-left:16px; }}
    li {{ margin:4px 0; }}
    a {{ color:var(--accent); }}
    footer {{ margin-top:24px; color:var(--muted); font-size:0.8em; text-align:center; }}
        .chat {{ display:flex; flex-direction:column; gap:8px; }}
        .chat textarea {{ width:100%; min-height:120px; background:#111; color:#e0e0e0;
                                            border:1px solid var(--border); border-radius:6px; padding:8px; }}
        .chat button {{ align-self:flex-start; background:var(--accent); color:#000; border:none;
                                        padding:8px 14px; border-radius:4px; cursor:pointer; font-weight:bold; }}
        .chat button:disabled {{ opacity:0.5; cursor:wait; }}
        .chat .result {{ white-space:pre-wrap; background:#0d0d0d; border:1px solid var(--border);
                                         border-radius:6px; padding:8px; min-height:60px; }}
        .muted {{ color:var(--muted); }}
  </style>
</head>
<body>
  <h1>🤖 AI Helper Dashboard</h1>
  <div class="ts">Last updated: {data['ts']} — auto-refreshes every {refresh}s
     &nbsp;|&nbsp; <a href="/api/status">JSON API</a></div>
  <div class="grid">

    <div class="card">
      <h2>System Resources</h2>
      <div class="stat-row"><span class="stat-label">CPU</span>
        <div class="stat-bar">{cpu_bar}</div></div>
      <div class="stat-row"><span class="stat-label">Memory</span>
        <div class="stat-bar">{mem_bar}</div></div>
      <table style="margin-top:8px">
        <tr><th>Mount</th><th>Usage</th><th>Capacity</th></tr>
        {disk_rows}
      </table>
    </div>

    <div class="card">
      <h2>GPU</h2>
      <table>
        <tr><th>GPU</th><th>VRAM</th><th>Temp</th><th>Util</th></tr>
        {gpu_rows}
      </table>
    </div>

    <div class="card">
      <h2>AI Programs</h2>
      <table>
        <tr><th></th><th>Name</th><th>URL</th></tr>
        {ai_rows}
      </table>
    </div>

    <div class="card">
      <h2>Top Processes (CPU)</h2>
      <table>
        <tr><th>PID</th><th>Name</th><th>CPU</th><th>Memory</th></tr>
        {proc_rows}
      </table>
    </div>

    <div class="card">
      <h2>Alerts</h2>
      <ul>{alert_items}</ul>
    </div>

        <div class="card">
            <h2>Ask AI Helper</h2>
            <div class="chat">
                <label for="prompt" class="muted">Type a request and click Ask. Runs with the built-in agent (no voice required).</label>
                <textarea id="prompt" placeholder="E.g. summarize current system status"></textarea>
                <div style="display:flex; gap:8px; align-items:center;">
                    <button id="ask-btn" onclick="ask()">Ask</button>
                    <span id="ask-status" class="muted"></span>
                </div>
                <div id="ask-result" class="result" aria-live="polite">Waiting for a question…</div>
            </div>
        </div>

  </div>
  <footer>AI Helper — serving on port {port} — {platform.node()}</footer>
    <script>
        async function ask() {{
            const btn = document.getElementById('ask-btn');
            const status = document.getElementById('ask-status');
            const result = document.getElementById('ask-result');
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {{ result.textContent = 'Please enter a prompt first.'; return; }}
            btn.disabled = true;
            status.textContent = 'Working…';
            result.textContent = '';
            try {{
                const resp = await fetch('/api/ask', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ prompt }})
                }});
                const data = await resp.json();
                if (!data.ok) {{
                    result.textContent = 'Error: ' + (data.error || 'Unknown error');
                }} else {{
                    const steps = (data.steps || []).join('\n\n');
                    result.textContent = (steps ? steps + '\n\n' : '') + (data.answer || '');
                }}
            }} catch (e) {{
                result.textContent = 'Request failed: ' + e;
            }} finally {{
                btn.disabled = false;
                status.textContent = '';
            }}
        }}
    </script>
</body>
</html>"""


class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP request handler for the dashboard."""

    def __init__(self, *args, port: int = _DEFAULT_PORT,
                 refresh: int = _REFRESH_SECONDS, **kwargs) -> None:
        self._port = port
        self._refresh = refresh
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            data = _collect_status()
            body = json.dumps(data, indent=2).encode()
            self._respond(200, "application/json", body)
        else:
            data = _collect_status()
            html = _render_html(data, self._port, self._refresh)
            self._respond(200, "text/html; charset=utf-8", html.encode())

    def _respond(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:  # noqa: ANN002
        logger.debug("WebUI: %s", args[1] if len(args) > 1 else args)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/ask":
            self._respond(404, "application/json", b"{}")
            return

        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
            prompt = (payload.get("prompt") or "").strip()
        except Exception:  # noqa: BLE001
            self._respond(400, "application/json", b'{"ok":false,"error":"Invalid JSON"}')
            return

        if not prompt:
            self._respond(400, "application/json", b'{"ok":false,"error":"Prompt is required"}')
            return

        try:
            from .agent import Agent  # noqa: PLC0415

            agent = Agent()
            result = agent.execute(prompt)
            body = json.dumps({
                "ok": True,
                "answer": getattr(result, "answer", ""),
                "steps": [str(s) for s in getattr(result, "steps", [])],
            }).encode()
            self._respond(200, "application/json", body)
        except Exception as exc:  # noqa: BLE001
            err = json.dumps({"ok": False, "error": str(exc)}).encode()
            self._respond(500, "application/json", err)


class WebUI:
    """Run the AI Helper browser dashboard.

    Parameters
    ----------
    port:
        TCP port to listen on (default 8765).
    host:
        Bind address (default ``"127.0.0.1"``).
    refresh_seconds:
        How often the browser auto-refreshes (default 5 s).
    """

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        host: str = "127.0.0.1",
        refresh_seconds: int = _REFRESH_SECONDS,
    ) -> None:
        self.port = port
        self.host = host
        self.refresh_seconds = refresh_seconds
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the web server in a background daemon thread."""
        def _handler_factory(*args, **kwargs):
            return _Handler(*args, port=self.port,
                            refresh=self.refresh_seconds, **kwargs)

        self._server = HTTPServer((self.host, self.port), _handler_factory)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="ai-helper-webui",
            daemon=True,
        )
        self._thread.start()
        logger.info("Web UI started at http://%s:%d", self.host, self.port)
        print(f"  Dashboard: http://{self.host}:{self.port}", flush=True)

    def stop(self) -> None:
        """Shut down the web server."""
        if self._server:
            self._server.shutdown()
        logger.info("Web UI stopped")

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def serve_forever(self) -> None:
        """Start the server and block until Ctrl-C."""
        self.start()
        print(f"AI Helper web dashboard running at http://{self.host}:{self.port}")
        print("Press Ctrl-C to stop.")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
