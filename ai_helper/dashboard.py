"""Live terminal dashboard.

Renders a continuously updating, full-screen terminal UI using Python's
built-in ``curses`` library (no extra packages required).

The dashboard shows:

* **Header** — AI Helper branding and current time
* **System** — CPU, memory, disk bar-charts with colour coding
* **GPU** — VRAM / temperature / utilisation (when NVIDIA GPU present)
* **AI Programs** — live status of all known AI apps
* **Processes** — top 5 by CPU, refreshed every cycle
* **Alerts** — last 6 notifications colour-coded by urgency
* **Footer** — keybindings help

Colour coding
-------------
* Green  < 60 %   (healthy)
* Yellow 60–84 %  (watch)
* Red    ≥ 85 %   (alert)

Usage
-----
::

    from ai_helper.dashboard import Dashboard
    dash = Dashboard(poll_interval=2.0)
    dash.run()   # blocks until Ctrl-C / 'q'

Or from the CLI::

    python -m ai_helper --dashboard
"""

from __future__ import annotations

import curses
import logging
import time
from typing import List, Optional

logger = logging.getLogger(__name__)


class Dashboard:
    """Curses-based live terminal dashboard.

    Parameters
    ----------
    poll_interval:
        Seconds between screen refreshes (default 2 s).
    """

    _BAR_WIDTH = 20

    def __init__(self, poll_interval: float = 2.0) -> None:
        self.poll_interval = poll_interval
        # Lazy imports so the rest of AI Helper works in headless environments
        from .monitor import SystemMonitor              # noqa: PLC0415
        from .process_manager import ProcessManager     # noqa: PLC0415
        from .gpu_monitor import GpuMonitor             # noqa: PLC0415
        from .ai_integrations import AIAppRegistry      # noqa: PLC0415
        from .notification_center import NotificationCenter  # noqa: PLC0415

        self._monitor = SystemMonitor()
        self._procs = ProcessManager()
        self._gpu = GpuMonitor()
        self._ai_reg = AIAppRegistry(timeout=1.0)
        self._nc = NotificationCenter()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the dashboard.  Blocks until the user presses 'q' or Ctrl-C."""
        try:
            curses.wrapper(self._main)
        except KeyboardInterrupt:
            pass

    # ------------------------------------------------------------------
    # Curses main loop
    # ------------------------------------------------------------------

    def _main(self, stdscr: "curses.window") -> None:
        curses.curs_set(0)
        stdscr.nodelay(True)
        curses.start_color()
        curses.use_default_colors()

        # Colour pairs: 1=green, 2=yellow, 3=red, 4=cyan, 5=white-bold, 6=magenta
        curses.init_pair(1, curses.COLOR_GREEN,   -1)
        curses.init_pair(2, curses.COLOR_YELLOW,  -1)
        curses.init_pair(3, curses.COLOR_RED,     -1)
        curses.init_pair(4, curses.COLOR_CYAN,    -1)
        curses.init_pair(5, curses.COLOR_WHITE,   -1)
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)

        while True:
            key = stdscr.getch()
            if key in (ord("q"), ord("Q"), 27):  # q / Q / Escape
                break

            try:
                self._draw(stdscr)
            except curses.error:
                pass   # Terminal too small — keep running

            time.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, stdscr: "curses.window") -> None:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        row = 0

        snap = self._monitor.snapshot()
        alerts = self._monitor.alerts(snap)

        # ---- Header --------------------------------------------------
        title = " AI Helper Dashboard "
        ts = time.strftime("%Y-%m-%d  %H:%M:%S")
        header = f"{'':=<{cols}}"
        self._addstr(stdscr, row, 0, header[:cols], curses.color_pair(4))
        row += 1
        self._addstr(stdscr, row, 2, title, curses.color_pair(4) | curses.A_BOLD)
        self._addstr(stdscr, row, cols - len(ts) - 2, ts, curses.color_pair(5))
        row += 1
        self._addstr(stdscr, row, 0, "─" * cols, curses.color_pair(5))
        row += 1

        # ---- System --------------------------------------------------
        row = self._draw_section(stdscr, row, "SYSTEM", cols)
        row = self._draw_bar(stdscr, row, "CPU",    snap.cpu_percent)
        row = self._draw_bar(stdscr, row, "Memory", snap.memory_percent)
        if snap.disk_partitions:
            first = snap.disk_partitions[0]
            row = self._draw_bar(stdscr, row, f"Disk {first.mountpoint}", first.percent)
        row += 1

        # ---- GPU (if available) --------------------------------------
        if row < rows - 4 and self._gpu.available:
            row = self._draw_section(stdscr, row, "GPU", cols)
            for gsnap in self._gpu.snapshots():
                row = self._draw_bar(stdscr, row, f"GPU{gsnap.index} VRAM", gsnap.vram_percent,
                                      suffix=f"  {gsnap.temperature_c:.0f}°C  {gsnap.name}")
            row += 1

        # ---- AI Programs ---------------------------------------------
        if row < rows - 4:
            row = self._draw_section(stdscr, row, "AI PROGRAMS", cols)
            statuses = self._ai_reg.discover()
            running = [s for s in statuses if s.running]
            offline = [s for s in statuses if not s.running]
            for s in running[:5]:
                self._addstr(stdscr, row, 2, f"  ✓  {s.name}", curses.color_pair(1))
                row += 1
                if row >= rows - 2:
                    break
            if offline and row < rows - 2:
                names = ", ".join(s.name for s in offline[:6])
                self._addstr(stdscr, row, 2,
                              f"  ✗  Offline: {names}"[:cols - 4],
                              curses.color_pair(2))
                row += 1
            row += 1

        # ---- Top Processes -------------------------------------------
        if row < rows - 4:
            row = self._draw_section(stdscr, row, "TOP PROCESSES (CPU)", cols)
            procs = self._procs.list_processes()
            top = sorted(procs, key=lambda p: p.cpu_percent, reverse=True)[:5]
            for p in top:
                if row >= rows - 2:
                    break
                line = (f"  PID {p.pid:6d}  CPU {p.cpu_percent:5.1f}%"
                        f"  MEM {p.memory_mb:5.0f}MB  {p.name}")[:cols - 2]
                color = self._pct_color(p.cpu_percent)
                self._addstr(stdscr, row, 0, line, curses.color_pair(color))
                row += 1
            row += 1

        # ---- Alerts --------------------------------------------------
        if row < rows - 2:
            row = self._draw_section(stdscr, row, "ALERTS", cols)
            if alerts:
                for a in alerts[:6]:
                    if row >= rows - 2:
                        break
                    self._addstr(stdscr, row, 2, f"  ⚠  {a}"[:cols - 4], curses.color_pair(3))
                    row += 1
            else:
                self._addstr(stdscr, row, 2, "  ✓  No active alerts", curses.color_pair(1))
                row += 1

        # ---- Footer --------------------------------------------------
        if rows > 2:
            footer = " [q] Quit  |  refreshes every {:.0f}s ".format(self.poll_interval)
            self._addstr(stdscr, rows - 1, 0, "─" * cols, curses.color_pair(5))
            self._addstr(stdscr, rows - 1, 2, footer[:cols - 4], curses.color_pair(5))

        stdscr.refresh()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_section(self, stdscr: "curses.window", row: int, title: str, cols: int) -> int:
        label = f" {title} "
        line = f"{'─' * 2}{label}{'─' * max(0, cols - len(label) - 2)}"
        self._addstr(stdscr, row, 0, line[:cols], curses.color_pair(4))
        return row + 1

    def _draw_bar(
        self,
        stdscr: "curses.window",
        row: int,
        label: str,
        pct: float,
        suffix: str = "",
    ) -> int:
        color = self._pct_color(pct)
        filled = int(self._BAR_WIDTH * pct / 100)
        bar = "█" * filled + "░" * (self._BAR_WIDTH - filled)
        line = f"  {label:<12} [{bar}] {pct:5.1f}%{suffix}"
        rows, cols = stdscr.getmaxyx()
        self._addstr(stdscr, row, 0, line[:cols - 1], curses.color_pair(color))
        return row + 1

    @staticmethod
    def _pct_color(pct: float) -> int:
        if pct >= 85:
            return 3   # red
        if pct >= 60:
            return 2   # yellow
        return 1       # green

    @staticmethod
    def _addstr(stdscr: "curses.window", row: int, col: int, text: str, attr: int = 0) -> None:
        rows, cols = stdscr.getmaxyx()
        if row < 0 or row >= rows:
            return
        if col >= cols:
            return
        text = text[:cols - col]
        try:
            stdscr.addstr(row, col, text, attr)
        except curses.error:
            pass
