"""Simple desktop HUD for AI Helper (offline-friendly).

Opens a small window showing CPU/memory/disk stats and lets you type a request
for the Agent. No Internet is needed unless your request itself triggers
networked tools.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Optional

from .agent import Agent
from .monitor import SystemMonitor


class HUDApp:
    def __init__(self) -> None:
        self.monitor = SystemMonitor()
        self.root = tk.Tk()
        self.root.title("AI Helper HUD")
        self.root.geometry("520x460")
        self.root.configure(bg="#111")
        self._build_ui()
        self._status_job: Optional[str] = None

    def _build_ui(self) -> None:
        # Styles
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", background="#111", foreground="#e0e0e0")
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Card.TLabelframe", background="#1b1b1b", foreground="#e0e0e0")
        style.configure(
            "Card.TLabelframe.Label",
            background="#1b1b1b",
            foreground="#8ad",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TButton", padding=6)

        main = ttk.Frame(self.root, padding=12, style="TFrame")
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="AI Helper – HUD", style="Title.TLabel")
        title.pack(anchor="w", pady=(0, 8))

        # Status card
        status = ttk.Labelframe(main, text="System status", style="Card.TLabelframe")
        status.pack(fill="x", pady=6)

        self.cpu_var = tk.StringVar(value="CPU: …")
        self.mem_var = tk.StringVar(value="Memory: …")
        self.disk_var = tk.StringVar(value="Disk: …")
        self.alerts_var = tk.StringVar(value="Alerts: none")

        ttk.Label(status, textvariable=self.cpu_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(status, textvariable=self.mem_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(status, textvariable=self.disk_var).pack(anchor="w", padx=8, pady=2)
        ttk.Label(status, textvariable=self.alerts_var, wraplength=480).pack(anchor="w", padx=8, pady=4)

        # Ask card
        ask = ttk.Labelframe(main, text="Ask AI Helper", style="Card.TLabelframe")
        ask.pack(fill="both", expand=True, pady=6)

        ttk.Label(ask, text="Type a request and press Ask (runs locally)").pack(anchor="w", padx=8, pady=(6, 4))
        self.prompt = tk.Text(ask, height=5, wrap="word", bg="#0f0f0f", fg="#f0f0f0", insertbackground="#f0f0f0")
        self.prompt.pack(fill="x", padx=8)

        controls = ttk.Frame(ask)
        controls.pack(anchor="w", padx=8, pady=6)
        self.ask_btn = ttk.Button(controls, text="Ask", command=self._on_ask)
        self.ask_btn.pack(side="left")
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=8)

        self.result = tk.Text(ask, height=10, wrap="word", bg="#0c0c0c", fg="#dcdcdc", insertbackground="#dcdcdc")
        self.result.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        self.result.insert("1.0", "Ready. No Internet needed unless your task requires it.")

    def _update_status(self) -> None:
        try:
            snap = self.monitor.snapshot()
            self.cpu_var.set(f"CPU: {snap.cpu_percent:.1f}%")
            self.mem_var.set(f"Memory: {snap.memory_percent:.1f}%")
            if snap.disk_partitions:
                d = snap.disk_partitions[0]
                self.disk_var.set(f"Disk {d.mountpoint}: {d.percent:.1f}% used")
            alerts = self.monitor.alerts(snap)
            self.alerts_var.set("Alerts: none" if not alerts else "Alerts: " + "; ".join(alerts))
        except Exception as exc:  # noqa: BLE001
            self.alerts_var.set(f"Status error: {exc}")

        # Schedule next update
        self._status_job = self.root.after(3000, self._update_status)

    def _on_ask(self) -> None:
        prompt = self.prompt.get("1.0", "end").strip()
        if not prompt:
            self.status_var.set("Enter a prompt first")
            return

        self.ask_btn.state(["disabled"])
        self.status_var.set("Working…")
        self.result.delete("1.0", "end")

        def worker() -> None:
            try:
                agent = Agent()
                res = agent.execute(prompt)
                answer = getattr(res, "answer", "")
                steps_raw = getattr(res, "steps", [])
                steps = [str(s) for s in steps_raw] if steps_raw else []
                text = "\n\n".join(steps + ([answer] if answer else [])) or "(no answer)"
            except Exception as exc:  # noqa: BLE001
                text = f"Error: {exc}"
            def finish() -> None:
                self.result.delete("1.0", "end")
                self.result.insert("1.0", text)
                self.status_var.set("Done")
                self.ask_btn.state(["!disabled"])
            self.root.after(0, finish)

        threading.Thread(target=worker, daemon=True).start()

    def run(self) -> None:
        self._update_status()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self) -> None:
        if self._status_job:
            self.root.after_cancel(self._status_job)
        self.root.destroy()


def run_hud() -> None:
    HUDApp().run()
