"""AI Helper – command-line entry point.

Usage
-----
Run a one-shot status report::

    python -m ai_helper

Run the continuous monitoring daemon (Ctrl-C to stop)::

    python -m ai_helper --daemon

Speak every alert aloud::

    python -m ai_helper --daemon --voice

Show GPU stats::

    python -m ai_helper --gpu-stats

List all discovered AI programs::

    python -m ai_helper --list-ai

Ask Ollama a question::

    python -m ai_helper --ollama-ask "What is the capital of France?" --ollama-model llama3

Install / uninstall the auto-start service::

    python -m ai_helper --install-service
    python -m ai_helper --uninstall-service

Options
-------
--daemon              Run as a background polling loop.
--interval SECONDS    Poll interval in seconds (default: 30).
--cpu-threshold PCT   CPU % to alert on (default: 85).
--mem-threshold PCT   Memory % to alert on (default: 85).
--disk-threshold PCT  Disk % to alert on (default: 90).
--voice               Enable text-to-speech for all alerts.
--voice-rate WPM      Speech rate in words per minute (default: 175).
--voice-volume VOL    Speech volume 0.0–1.0 (default: 1.0).
--list-voices         Print available TTS voices and exit.
--gpu-stats           Print NVIDIA GPU stats and exit.
--list-ai             Discover and print status of all known AI programs.
--ollama-ask TEXT     Send a one-shot prompt to Ollama and print the reply.
--ollama-model MODEL  Ollama model to use with --ollama-ask (default: llama3).
--ollama-url URL      Ollama base URL (default: http://localhost:11434).
--install-service     Install AI Helper as a boot-time auto-start service.
--uninstall-service   Remove the auto-start service.
--service-status      Show the current service installation status.
--log-level LEVEL     Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from .agent import Agent
from .ai_integrations import AIAppRegistry, OllamaClient
from .backup import BackupManager
from .communicator import Communicator, Message
from .gpu_monitor import GpuMonitor
from .memory import Memory
from .monitor import SystemMonitor
from .notification_center import NotificationCenter
from .orchestrator import Orchestrator
from .process_manager import ProcessManager
from .service import ServiceManager
from .updater import Updater
from .voice import Speaker, VoiceSettings
from .wake_word import WakeWordListener


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
    # Voice options
    parser.add_argument("--voice", action="store_true", help="Speak alerts aloud using text-to-speech.")
    parser.add_argument("--voice-rate", type=int, default=175, metavar="WPM", help="Speech rate in words-per-minute (default: 175).")
    parser.add_argument("--voice-volume", type=float, default=1.0, metavar="VOL", help="Speech volume 0.0–1.0 (default: 1.0).")
    parser.add_argument("--list-voices", action="store_true", help="List available TTS voices and exit.")
    # GPU options
    parser.add_argument("--gpu-stats", action="store_true", help="Print NVIDIA GPU stats and exit.")
    # AI program options
    parser.add_argument("--list-ai", action="store_true", help="Discover and list all known AI programs.")
    parser.add_argument("--ollama-ask", metavar="TEXT", help="Send a prompt to Ollama and print the reply.")
    parser.add_argument("--ollama-model", default="llama3", metavar="MODEL", help="Ollama model for --ollama-ask (default: llama3).")
    parser.add_argument("--ollama-url", default="http://localhost:11434", metavar="URL", help="Ollama base URL.")
    # Agent / tool use
    parser.add_argument("--ask", metavar="GOAL",
                        help="Give AI Helper a goal in plain English and let the agent "
                             "plan and execute it using all available tools.")
    parser.add_argument("--steps", type=int, default=10, metavar="N",
                        help="Maximum agent steps for --ask (default: 10).")
    # Service options
    parser.add_argument("--install-service", action="store_true", help="Install AI Helper as a boot-time auto-start service.")
    parser.add_argument("--uninstall-service", action="store_true", help="Remove the auto-start service.")
    parser.add_argument("--service-status", action="store_true", help="Show current service installation status.")
    # Dashboard
    parser.add_argument("--dashboard", action="store_true", help="Launch the live curses terminal dashboard.")
    parser.add_argument("--hud", action="store_true", help="Open the desktop HUD window (no voice, local only).")
    # Web UI
    parser.add_argument("--web-ui", action="store_true", help="Start the browser dashboard web server.")
    parser.add_argument("--web-port", type=int, default=8765, metavar="PORT", help="Web dashboard port (default: 8765).")
    # Wake word / voice activation (prototype)
    parser.add_argument("--wake-word", metavar="WORD", help="Enable wake-word listener (requires microphone + SpeechRecognition).")
    parser.add_argument(
        "--wake-phrase-seconds",
        type=float,
        default=6.0,
        metavar="SECONDS",
        help="Max seconds to capture a spoken command after the wake word (default: 6).",
    )
    parser.add_argument(
        "--wake-mic-index",
        type=int,
        metavar="IDX",
        help="Microphone device index for wake word (use --list-mics to see devices).",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List microphone devices and exit (requires SpeechRecognition).",
    )
    parser.add_argument(
        "--wake-stt",
        choices=["google", "whisper"],
        default="google",
        help="Speech-to-text backend for wake word commands (default: google).",
    )
    parser.add_argument(
        "--wake-whisper-model",
        default="small.en",
        help="Whisper model size/path when using --wake-stt whisper (default: small.en).",
    )
    # Memory
    parser.add_argument("--memory", action="store_true", help="Show the AI Helper persistent memory summary.")
    parser.add_argument("--memory-history", action="store_true", help="Show recent agent conversation history.")
    # Notification history
    parser.add_argument("--notify-history", action="store_true", help="Print the notification history and exit.")
    # Backup
    parser.add_argument("--backup", metavar="DIR", help="Immediately back up DIR to the D drive.")
    # Update check
    parser.add_argument("--check-update", action="store_true", help="Check for a new AI Helper release on GitHub.")
    # Diagnostics
    parser.add_argument("--diagnostics", action="store_true",
                        help="Run installation diagnostics: verify required packages, "
                             "core modules and basic functionality, then exit.")
    # Hotkey info
    parser.add_argument("--hotkeys", action="store_true", help="Print registered global hotkeys and exit.")
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

    # ------------------------------------------------------------------
    # Diagnostics (early exit)
    # ------------------------------------------------------------------
    if args.diagnostics:
        from .diagnostics import run_diagnostics  # noqa: PLC0415
        _, passed = run_diagnostics(verbose=True)
        sys.exit(0 if passed else 1)

    # ------------------------------------------------------------------
    # Microphone listing (early exit)
    # ------------------------------------------------------------------
    if args.list_mics:
        try:
            import speech_recognition as sr  # type: ignore

            names = sr.Microphone.list_microphone_names() or []
            if not names:
                print("No microphones found.")
            else:
                print("Available microphones:")
                for i, name in enumerate(names):
                    print(f"  [{i}] {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not list microphones: {exc}")
        return

    # ------------------------------------------------------------------
    # Voice / TTS setup
    # ------------------------------------------------------------------
    speaker = Speaker(
        settings=VoiceSettings(rate=args.voice_rate, volume=args.voice_volume),
        enabled=args.voice,
    )

    if args.list_voices:
        voices = speaker.list_voices()
        if voices:
            print("Available TTS voices:")
            for v in voices:
                print(f"  {v}")
        else:
            print("No pyttsx3 voices found (is pyttsx3 installed?).")
        return

    # ------------------------------------------------------------------
    # Service management (early exit)
    # ------------------------------------------------------------------
    if args.install_service or args.uninstall_service or args.service_status:
        svc = ServiceManager()
        if args.install_service:
            ok = svc.install()
            print("Service installed." if ok else "Service installation failed — check logs.")
        elif args.uninstall_service:
            ok = svc.uninstall()
            print("Service removed." if ok else "Service removal failed — check logs.")
        else:
            print(svc.status())
        return

    # ------------------------------------------------------------------
    # GPU stats (early exit)
    # ------------------------------------------------------------------
    if args.gpu_stats:
        gpu = GpuMonitor()
        print(gpu.format_snapshots())
        alerts = gpu.alerts()
        if alerts:
            print("\nGPU Alerts:")
            for a in alerts:
                print(f"  ⚠  {a}")
        return

    # ------------------------------------------------------------------
    # AI program discovery (early exit)
    # ------------------------------------------------------------------
    if args.list_ai:
        registry = AIAppRegistry()
        print(registry.format_status())
        return

    # ------------------------------------------------------------------
    # Agent / tool-use (early exit)
    # ------------------------------------------------------------------
    if args.ask:
        agent = Agent(
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            max_steps=args.steps,
        )
        print(f"AI Helper is working on: {args.ask!r}\n", flush=True)
        agent_result = agent.execute(args.ask)
        # Print each step
        for step in agent_result.steps:
            print(step)
            print()
        print("─" * 60)
        print(f"Answer:\n{agent_result.answer}")
        if args.voice:
            speaker.speak_now(agent_result.answer)
        return

    # ------------------------------------------------------------------
    # Terminal dashboard (blocks until quit)
    # ------------------------------------------------------------------
    if args.dashboard:
        from .dashboard import Dashboard  # noqa: PLC0415
        print("Starting AI Helper dashboard (press 'q' to quit)…", flush=True)
        Dashboard(poll_interval=2.0).run()
        return

    # ------------------------------------------------------------------
    # Desktop HUD (blocks until closed)
    # ------------------------------------------------------------------
    if args.hud:
        from .hud import run_hud  # noqa: PLC0415

        print("Starting AI Helper HUD…", flush=True)
        run_hud()
        return

    # ------------------------------------------------------------------
    # Web UI (early exit — blocks in foreground)
    # ------------------------------------------------------------------
    if args.web_ui:
        from .web_ui import WebUI  # noqa: PLC0415
        ui = WebUI(port=args.web_port)
        ui.serve_forever()
        return

    # ------------------------------------------------------------------
    # Memory summary
    # ------------------------------------------------------------------
    if args.memory:
        mem = Memory()
        print(mem.summary())
        return

    if args.memory_history:
        mem = Memory()
        convos = mem.recent_conversations(limit=20)
        if not convos:
            print("No conversations recorded yet.")
        for c in convos:
            print(c)
        return

    # ------------------------------------------------------------------
    # Notification history
    # ------------------------------------------------------------------
    if args.notify_history:
        nc = NotificationCenter()
        print(nc.format_history())
        return

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------
    if args.backup:
        from pathlib import Path  # noqa: PLC0415
        mgr = BackupManager()
        copied = mgr.backup_now(Path(args.backup))
        print(f"Backed up {copied} file(s) from {args.backup}")
        return

    # ------------------------------------------------------------------
    # Update check
    # ------------------------------------------------------------------
    if args.check_update:
        u = Updater()
        print("Checking for updates…", flush=True)
        info = u.check()
        print(info)
        if info.update_available:
            ans = input("Download update? [y/N] ").strip().lower()
            if ans == "y":
                dest = u.download(info)
                if dest:
                    print(f"Downloaded to {dest}")
        return

    # ------------------------------------------------------------------
    # Hotkey info
    # ------------------------------------------------------------------
    if args.hotkeys:
        from .hotkey import HotkeyManager  # noqa: PLC0415
        print(HotkeyManager().bindings_info())
        return

    # ------------------------------------------------------------------
    # Ollama one-shot query (early exit)
    # ------------------------------------------------------------------
    if args.ollama_ask:
        client = OllamaClient(base_url=args.ollama_url)
        if not client.is_running():
            print(f"Ollama is not running at {args.ollama_url}")
            print("Start it with: ollama serve")
            return
        print(f"Asking {args.ollama_model}: {args.ollama_ask!r} …", flush=True)
        result = client.generate(model=args.ollama_model, prompt=args.ollama_ask)
        print(result.response)
        if args.voice:
            speaker.speak_now(result.response)
        return

    # ------------------------------------------------------------------
    # Core modules
    # ------------------------------------------------------------------
    monitor = SystemMonitor(
        thresholds={
            "cpu": args.cpu_threshold,
            "memory": args.mem_threshold,
            "disk": args.disk_threshold,
        }
    )
    process_manager = ProcessManager()
    communicator = Communicator(speaker=speaker, speak_alerts=args.voice)

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
            if args.voice:
                for a in alerts:
                    speaker.speak_now(a)
        return

    # ------------------------------------------------------------------
    # Daemon mode
    # ------------------------------------------------------------------
    orchestrator = Orchestrator(
        poll_interval=args.interval,
        monitor=monitor,
        process_manager=process_manager,
        communicator=communicator,
    )

    # Optional wake-word listener (prototype)
    wake_listener = None
    if args.wake_word:
        try:
            wake_listener = WakeWordListener(
                wake_word=args.wake_word,
                phrase_time_limit=args.wake_phrase_seconds,
                device_index=args.wake_mic_index,
                backend=args.wake_stt,
                whisper_model=args.wake_whisper_model,
            )
            if wake_listener.available:
                agent_for_wake = Agent(max_steps=args.steps, ollama_url=args.ollama_url, ollama_model=args.ollama_model)

                def _on_wake_command(text: str) -> None:
                    print(f"[wake] Heard command: {text}", flush=True)
                    try:
                        result = agent_for_wake.execute(text)
                        print(result.answer, flush=True)
                        if args.voice:
                            speaker.speak(result.answer)
                    except Exception as exc:  # noqa: BLE001
                        print(f"Wake-word command error: {exc}", flush=True)

                wake_listener.start(_on_wake_command)
            else:
                print("Wake-word listener unavailable (install SpeechRecognition + PyAudio).", flush=True)
                wake_listener = None
        except Exception as exc:  # noqa: BLE001
            print(f"Wake-word listener failed to start: {exc}", flush=True)

    if args.voice:
        speaker.speak_now("AI Helper started.")

    def _handle_signal(signum: int, _frame: object) -> None:
        print("\nShutting down AI Helper…", flush=True)
        if args.voice:
            speaker.speak_now("AI Helper shutting down.")
        orchestrator.stop()
        if wake_listener:
            wake_listener.stop()
        speaker.shutdown()
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
