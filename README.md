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
| 🗿 **Mesh Engine** | Scan images → Fallout 4 mesh (TripoSG, TRELLIS, Meshy, NIF export) |

---

## 🗿 Mesh Engine (Fallout 4)

Mossy includes a full image-to-mesh pipeline purpose-built for Fallout 4 modding.
All tools are **free and open-source** (GitHub / HuggingFace) except Meshy, which
requires a paid subscription.

### Free-first philosophy

| Tool | Source | VRAM | License | Role |
|------|--------|------|---------|------|
| **TripoSG** ⭐ | [github.com/VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG) | 8 GB | MIT | Image → 3D (recommended) |
| **TRELLIS** | [github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | 16 GB | MIT | Image/text → 3D (highest quality) |
| **TripoSR** | [github.com/VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) | 6 GB | MIT | Image → 3D (lightest) |
| **Shap-E** | [github.com/openai/shap-e](https://github.com/openai/shap-e) | 8 GB | MIT | Image/text → 3D (CPU-capable) |
| **Depth-Anything v2** | [HuggingFace](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | — | Apache 2.0 | Free depth estimation |
| **Meshy** | [meshy.ai](https://www.meshy.ai) | Cloud | Paid ✱ | Image → 3D (subscription) |

✱ The user holds a paid Meshy annual subscription.

### Install mesh dependencies

```bash
# Core mesh packages (all free)
pip install "ai-helper[mesh]"

# Or install individually:
pip install opencv-python open3d trimesh numpy Pillow scipy scikit-image
pip install diffusers einops omegaconf huggingface_hub transformers torch
pip install pymeshlab moderngl tqdm requests
pip install pyffi          # NIF export for Fallout 4

# Clone free image-to-3D repos (pick whichever you have VRAM for)
git clone https://github.com/VAST-AI-Research/TripoSG.git  D:\AI-Helper\FreeModels\triposg
pip install -r D:\AI-Helper\FreeModels\triposg\requirements.txt

git clone https://github.com/VAST-AI-Research/TripoSR.git  D:\AI-Helper\FreeModels\triposr
pip install -r D:\AI-Helper\FreeModels\triposr\requirements.txt

git clone https://github.com/openai/shap-e.git  D:\AI-Helper\FreeModels\shap_e
pip install -e D:\AI-Helper\FreeModels\shap_e

# TRELLIS (16 GB VRAM, best quality)
git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git  D:\AI-Helper\FreeModels\trellis
cd D:\AI-Helper\FreeModels\trellis && . ./setup.sh --new-env --basic --xformers --flash-attn
```

### Quick start

```python
from ai_helper.mesh_engine import MeshEngine

engine = MeshEngine()

# Ask Mossy a Fallout 4 modding question
print(engine.ask_knowledge("How many polygons can a weapon mesh have?"))
print(engine.ask_knowledge("What NIF block type should I use for a static prop?"))

# See the full workflow
print(engine.get_workflow())

# List all free image-to-3D tools
print(engine.list_free_tools())

# Get install instructions for TripoSG
print(engine.install_instructions("triposg"))

# Convert image → 3D mesh (free, TripoSG — recommended)
result = engine.free_image_to_3d(
    image_path="my_object.jpg",
    output_dir="D:/AI-Helper/MeshOutput",
    backend="triposg",   # or: trellis, triposr, shap_e
    faces=5000,          # polygon budget for FO4 weapon
)
print(result.summary)   # path to .glb output

# Convert image → 3D mesh (Meshy subscription)
result = engine.meshy_image_to_3d(
    image_path="my_object.jpg",
    output_dir="D:/AI-Helper/MeshOutput",
    api_key="your-meshy-api-key",  # or set MESHY_API_KEY env var
    download_fmt="obj",
    target_polycount=5000,
)
print(result.summary)

# Validate a mesh against FO4 requirements
validation = engine.validate_mesh("my_mesh.obj", asset_type="weapon")
print(validation)

# Free depth estimation (HuggingFace Depth-Anything v2)
engine.estimate_depth("photo.jpg", output_dir="depth_output")

# Check which packages / repos are installed
print(engine.check_dependencies())
```

### CLI (via agent)

```bash
# Ask mesh questions
python -m ai_helper --ask "What polygon budget should I use for a weapon?"
python -m ai_helper --ask "Show me the workflow to convert images to Fallout 4 meshes"
python -m ai_helper --ask "List all free image to 3D tools"
python -m ai_helper --ask "Install instructions for triposg"
python -m ai_helper --ask "Convert my_object.jpg to a 3D mesh"
python -m ai_helper --ask "Validate my_mesh.obj as a weapon"
python -m ai_helper --ask "Check mesh dependencies"
```

### Fallout 4 mesh knowledge

Mossy has built-in domain knowledge covering:

| Topic | Details |
|-------|---------|
| **Polygon budgets** | weapon (5k), armor (3k), settlement (1–6k), character (4–6k), vehicle (15k) |
| **NIF format** | BSTriShape, BSFadeNode, BSLightingShaderProperty, BSXFlags, bhkCollisionObject |
| **NIF version** | 20.2.0.7 / user_version=12 / user_version_2=130 (required exact values) |
| **Texture channels** | `_d.dds` diffuse, `_n.dds` normal (BC5), `_s.dds` specular, `_g.dds` glow |
| **LOD tiers** | LOD0–LOD3 (full detail → 90% reduction); tools: xLODGen, DynDOLOD |
| **Collision** | bhkConvexVerticesShape, bhkMoppBvTreeShape; all statics need collision |
| **Scale** | 70 Bethesda units ≈ 1 metre |
| **Free tools** | NifSkope, NifTools Blender add-on, GIMP+DDS, Texconv, Creation Kit, xEdit |

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
| `mesh_engine.py` | Image → 3D mesh pipeline; Fallout 4 NIF export; TripoSG/TRELLIS/Meshy integration |
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

