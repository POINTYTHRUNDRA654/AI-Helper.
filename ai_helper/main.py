"""AI Helper – command-line entry point.

Usage
-----
Run a one-shot status report::

    python -m ai_helper

Run the continuous monitoring daemon (Ctrl-C to stop)::

    python -m ai_helper --daemon

Options
-------
--daemon              Run as a background polling loop.
--interval SECONDS    Poll interval in seconds (default: 30).
--cpu-threshold PCT   CPU % to alert on (default: 85).
--mem-threshold PCT   Memory % to alert on (default: 85).
--disk-threshold PCT  Disk % to alert on (default: 90).
--log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from .communicator import Communicator, Message
from .monitor import SystemMonitor
from .orchestrator import Orchestrator
from .process_manager import ProcessManager


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_helper",
        description="AI Helper – keep your desktop running smooth, organised and communicating.",
    )
    parser.add_argument("--daemon", action="store_true", help="Run as a continuous monitoring daemon.")
    parser.add_argument("--interval", type=float, default=30.0, metavar="SECONDS", help="Poll interval (default: 30).")
    parser.add_argument("--cpu-threshold", type=float, default=85.0, metavar="PCT", help="CPU alert threshold %% (default: 85).")
    parser.add_argument("--mem-threshold", type=float, default=85.0, metavar="PCT", help="Memory alert threshold %% (default: 85).")
    parser.add_argument("--disk-threshold", type=float, default=90.0, metavar="PCT", help="Disk alert threshold %% (default: 90).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity (default: INFO).")
    return parser


def main(argv: list[str] | None = None) -> None:  # noqa: UP006
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    monitor = SystemMonitor(
        thresholds={
            "cpu": args.cpu_threshold,
            "memory": args.mem_threshold,
            "disk": args.disk_threshold,
        }
    )
    process_manager = ProcessManager()
    communicator = Communicator()

    # Echo every alert to stdout so the user sees it even without a desktop.
    def _print_alert(msg: Message) -> None:
        print(f"⚠  {msg}", flush=True)

    communicator.subscribe("alert", _print_alert)
    communicator.subscribe("process_alert", _print_alert)

    if not args.daemon:
        # One-shot mode: print a snapshot and exit.
        snap = monitor.snapshot()
        print(monitor.format_snapshot(snap))
        print()
        procs = process_manager.list_processes()
        print(process_manager.summary(procs))
        alerts = monitor.alerts(snap)
        if alerts:
            print("\nAlerts:")
            for a in alerts:
                print(f"  ⚠  {a}")
        return

    # Daemon mode
    orchestrator = Orchestrator(
        poll_interval=args.interval,
        monitor=monitor,
        process_manager=process_manager,
        communicator=communicator,
    )

    def _handle_signal(signum: int, _frame: object) -> None:
        print("\nShutting down AI Helper…", flush=True)
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print("AI Helper daemon started. Press Ctrl-C to stop.", flush=True)
    orchestrator.start()

    # Keep the main thread alive.
    while orchestrator.running:
        time.sleep(1)


if __name__ == "__main__":
    main()
