# AI Helper 🤖

**AI Helper** keeps your Windows desktop running smooth, organised and communicating.
It monitors your hardware, organises your files, talks to your AI programs (Ollama,
ComfyUI, LM Studio, Stable Diffusion and more), learns your habits over time, and can
autonomously use any file or program on your computer to help you get things done.

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 **System monitoring** | CPU, memory, disk, network — with ML-powered anomaly detection |
| 🖥️ **Live dashboard** | Full-screen terminal UI *and* browser dashboard |
| 🎙️ **Voice** | Speaks every alert and answer aloud (pyttsx3 + OS fallback) |
| 🧠 **AI agent** | Give any goal in plain English; the agent plans and executes it |
| 🦙 **Ollama** | List models, generate text, multi-turn chat |
| 🎨 **ComfyUI** | Queue workflows, check status, interrupt jobs |
| 🖼️ **Stable Diffusion** | txt2img, img2img, list models/samplers |
| 💬 **LM Studio** | Chat completions, list loaded models |
| 🧬 **Gemma Fine-Tuning** | Fine-tune & run Gemma 4 models locally with Unsloth (1.5x faster, 60% less VRAM) |
| 🎮 **NVIDIA GPU** | VRAM, temperature, utilisation, per-process memory |
| 📁 **File system** | Search, read, write, watch any file on disk |
| 💾 **Auto-backup** | Watch folders; version every change to D:\\AI-Helper\\Backups |
| 🔔 **Notification center** | Deduplication, throttling, escalation, history |
| 📋 **Clipboard monitor** | Detects paths/commands/errors/URLs; routes to agent |
| ⌨️ **Global hotkeys** | Trigger AI Helper actions from anywhere (Ctrl+Alt+A/S/O/N/G) |
| 🔄 **Self-updater** | Checks GitHub releases and downloads to D drive |
| 💾 **Persistent memory** | SQLite store: anomalies, conversations, preferences, file patterns |
| 🚀 **Auto-start** | Installs as systemd / launchd / Windows Task Scheduler service |
| 📦 **D-drive installs** | All downloads and packages go to D:\\AI-Helper |

---

## Quick Start

### Requirements

- Python 3.10 or later
- Windows 10/11 **or** Linux **or** macOS

### Install

```bash
# 1 – Clone to D drive (Windows) or home dir (Linux/macOS)
git clone https://github.com/POINTYTHRUNDRA654/AI-Helper. D:\AI-Helper\src

# 2a – Install as a package (recommended)
cd D:\AI-Helper\src
pip install .

# 2b – Or install dependencies only
pip install -r requirements.txt

# Optional – for GPU monitoring
pip install pynvml

# 3 – Verify the installation with diagnostics
python -m ai_helper --diagnostics

# 4 – One-shot system status check
python -m ai_helper
```

> **Linux / macOS users:** Replace `D:\AI-Helper\src` with any path, e.g. `~/AI-Helper`.

---

## CLI Reference

### Diagnostics

```bash
# Verify all required packages are installed and every module is working.
# Exits with code 0 if everything passes, 1 if any check fails.
python -m ai_helper --diagnostics
```

### Basic usage

```bash
# One-shot status report
python -m ai_helper

# Continuous daemon (Ctrl-C to stop)
python -m ai_helper --daemon

# Daemon with voice alerts
python -m ai_helper --daemon --voice
```

### Dashboards

```bash
# Live terminal dashboard (press q to quit)
python -m ai_helper --dashboard

# Browser dashboard at http://127.0.0.1:8765
python -m ai_helper --web-ui
python -m ai_helper --web-ui --web-port 9000
```

### AI Agent

```bash
# Ask AI Helper anything in plain English
python -m ai_helper --ask "What process is using the most memory?"
python -m ai_helper --ask "Find all Python files I edited today"
python -m ai_helper --ask "Show me my GPU temperature and VRAM usage"

# With voice readback
python -m ai_helper --ask "Summarise my system health" --voice
```

### Ollama

```bash
# Ask a model directly
python -m ai_helper --ollama-ask "Explain VRAM fragmentation" --ollama-model llama3

# Use a different Ollama server
python -m ai_helper --ollama-ask "Hello" --ollama-url http://192.168.1.100:11434
```

### GPU

```bash
python -m ai_helper --gpu-stats
```

### AI Programs

```bash
# Discover all known AI programs (Ollama, ComfyUI, LM Studio, etc.)
python -m ai_helper --list-ai
```

### File Backup

```bash
# Immediately back up a folder to D:\AI-Helper\Backups
python -m ai_helper --backup "C:\Users\YourName\Documents"
```

### Notifications & Memory

```bash
# View notification history
python -m ai_helper --notify-history

# View persistent memory (anomalies, conversations, preferences)
python -m ai_helper --memory

# View recent agent conversation history
python -m ai_helper --memory-history
```

### Auto-start Service

```bash
# Install – starts automatically on every login
python -m ai_helper --install-service

# Check status
python -m ai_helper --service-status

# Remove
python -m ai_helper --uninstall-service
```

### Updates

```bash
python -m ai_helper --check-update
```

### Voice

```bash
# List available TTS voices
python -m ai_helper --list-voices

# Custom speech rate and volume
python -m ai_helper --daemon --voice --voice-rate 160 --voice-volume 0.8
```

### Global Hotkeys

```bash
# Show registered hotkeys
python -m ai_helper --hotkeys
```

| Hotkey | Action |
|--------|--------|
| `Ctrl+Alt+A` | Ask the agent a question |
| `Ctrl+Alt+S` | Speak current system status |
| `Ctrl+Alt+O` | Organise the desktop now |
| `Ctrl+Alt+N` | Read the latest alert aloud |
| `Ctrl+Alt+G` | Speak GPU statistics |

> **Note:** Global hotkeys require `pynput` (`pip install pynput`).

---

## D Drive Setup

All packages and data are stored on the D drive to keep your C drive free.

Run the setup helper once:

```bash
python D:\AI-Helper\src\scripts\setup_d_drive.py
```

This creates the folder layout and writes a user-level `pip.ini` so that
`pip install` always targets `D:\AI-Helper\packages`.

Directory layout:

```
D:\AI-Helper\
    packages\       ← pip install target
    Backups\        ← auto-backup of watched folders
    Logs\           ← daemon and service logs
    Memory\         ← memory.db (SQLite)
    Organized\      ← desktop organiser output
    Updates\        ← downloaded release archives
```

---

## Python API

```python
from ai_helper.agent import Agent
from ai_helper.memory import Memory
from ai_helper.backup import BackupManager
from ai_helper.ai_integrations import OllamaClient, ComfyUIClient, SDWebUIClient, LMStudioClient
from ai_helper.notification_center import NotificationCenter

# Agent
agent = Agent()
result = agent.execute("What is using the most memory right now?")
print(result.answer)

# Memory
mem = Memory()
mem.set_preference("ollama_model", "mistral")
print(mem.summary())

# Backup
mgr = BackupManager()
mgr.add_watch(Path.home() / "Documents")
mgr.start()

# Ollama
client = OllamaClient()
for model in client.list_models():
    print(model)
reply = client.generate("llama3", "Write a haiku about Python")
print(reply.response)

# ComfyUI
comfy = ComfyUIClient()
if comfy.is_running():
    job_id = comfy.queue_prompt(my_workflow)
    print(f"Job queued: {job_id}")

# Stable Diffusion
sd = SDWebUIClient()
if sd.is_running():
    images = sd.txt2img("a beautiful sunset over mountains", steps=20)
    images[0].save("output.png")

# LM Studio
lms = LMStudioClient()
if lms.is_running():
    reply = lms.chat("What is 2+2?")
    print(reply.response)

# Notifications
nc = NotificationCenter()
nc.notify("CPU is high", source="monitor", urgency="critical")
print(nc.format_history())
```

---

## Module Overview

| Module | Purpose |
|--------|---------|
| `monitor.py` | CPU / memory / disk / network snapshots |
| `process_manager.py` | List, filter and signal processes |
| `gpu_monitor.py` | NVIDIA GPU monitoring (pynvml or nvidia-smi) |
| `ml_engine.py` | EWMA anomaly detection, trend prediction |
| `ai_integrations.py` | Ollama, LM Studio, ComfyUI, SD WebUI clients |
| `agent.py` | ReAct-style goal-directed agent |
| `tools.py` | Named tool registry for the agent |
| `file_system.py` | FileSearcher, FileReader, FileWriter, FileWatcher |
| `organizer.py` | Desktop file organiser → D drive |
| `backup.py` | Auto-backup watched folders with versioning |
| `memory.py` | SQLite persistent memory |
| `notification_center.py` | Alert dedup / throttle / escalation / history |
| `dashboard.py` | Live curses terminal dashboard |
| `web_ui.py` | Browser dashboard (stdlib http.server) |
| `clipboard_monitor.py` | Clipboard change detection + content routing |
| `hotkey.py` | Global keyboard shortcuts (pynput) |
| `communicator.py` | Pub/sub message bus + desktop notifications |
| `voice.py` | Text-to-speech (pyttsx3 + OS CLI fallback) |
| `scheduler.py` | Interval task scheduler |
| `service.py` | Auto-start service install (systemd/launchd/schtasks) |
| `updater.py` | GitHub release checker and downloader |
| `config.py` | Install-dir and path configuration |

---

## Optional Packages

| Package | Feature | Install |
|---------|---------|---------|
| `pynvml` | Better GPU monitoring | `pip install pynvml` |
| `pynput` | Global hotkeys | `pip install pynput` |
| `pyperclip` | Better clipboard support | `pip install pyperclip` |

All optional — AI Helper works without them, with graceful fallback.

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## License

MIT

