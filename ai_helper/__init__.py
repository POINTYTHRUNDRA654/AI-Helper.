"""AI Helper – desktop monitoring, AI orchestration, and Fallout 4 mesh pipeline.

Quick start
-----------
::

    from ai_helper import Agent, MeshEngine, Speaker

    # Ask the agent anything
    agent = Agent()
    result = agent.execute("What is my CPU usage?")
    print(result.answer)

    # Convert an image to a Fallout 4 mesh (free — TripoSG)
    engine = MeshEngine()
    r = engine.free_image_to_3d("photo.jpg", backend="triposg")
    print(r.summary)

    # Give Mossy a voice
    speaker = Speaker()
    speaker.speak("AI Helper is running.")
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "AI Helper contributors"

# ---------------------------------------------------------------------------
# Core subsystems — always available (stdlib only)
# ---------------------------------------------------------------------------

from .agent import Agent, AgentResult, AgentStep
from .config import (
    get_install_dir,
    get_downloads_dir,
    get_data_dir,
    get_logs_dir,
    get_organized_dir,
    save_config,
    set_install_dir,
    ensure_dirs,
)
from .monitor import SystemMonitor, SystemSnapshot
from .process_manager import ProcessManager, ProcessInfo
from .notification_center import NotificationCenter, NotificationRecord
from .memory import Memory
from .scheduler import TaskScheduler, Task, TaskStatus
from .voice import Speaker, VoiceSettings
from .retry import CircuitBreaker, CircuitOpenError, with_retry
from .tools import ToolRegistry, Tool, ToolParam, ToolResult

# ---------------------------------------------------------------------------
# AI integrations — require running services but no extra Python packages
# ---------------------------------------------------------------------------

from .ai_integrations import (
    OllamaClient,
    LMStudioClient,
    ComfyUIClient,
    SDWebUIClient,
    KoboldCppClient,
    TabbyAPIClient,
    GPT4AllClient,
    AphroditeClient,
    AIAppRegistry,
    AIAppStatus,
    OllamaModel,
    GenerateResult,
    SDImage,
    ComfyUIJob,
)

# ---------------------------------------------------------------------------
# Mesh engine — gracefully absent when mesh deps not installed
# ---------------------------------------------------------------------------

try:
    from .mesh_engine import (
        MeshEngine,
        FreeImage3DClient,
        FreeImage3DResult,
        FREE_IMAGE_TO_3D_BACKENDS,
        HuggingFaceDepthEstimator,
        MeshyClient,
        MeshyTaskResult,
        Fallout4MeshKnowledge,
        MeshValidator,
        ValidationResult,
    )
    _MESH_AVAILABLE = True
except Exception:  # noqa: BLE001
    _MESH_AVAILABLE = False

# ---------------------------------------------------------------------------
# GPU monitoring — gracefully absent when pynvml not installed
# ---------------------------------------------------------------------------

try:
    from .gpu_monitor import GpuMonitor, GpuSnapshot
    _GPU_AVAILABLE = True
except Exception:  # noqa: BLE001
    _GPU_AVAILABLE = False

# ---------------------------------------------------------------------------
# Gemma fine-tuner — gracefully absent when torch/unsloth not installed
# ---------------------------------------------------------------------------

try:
    from .gemma_finetuner import (
        GemmaFineTuner,
        UnslothFineTuner,
        FineTuneConfig,
        GPUMemoryProfiler,
        GEMMA_MODELS,
        TrainingResult,
        InferenceResult,
        prepare_mossy_dataset,
        list_recommended_models,
    )
    _GEMMA_AVAILABLE = True
except Exception:  # noqa: BLE001
    _GEMMA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Agent
    "Agent", "AgentResult", "AgentStep",
    # Config
    "get_install_dir", "get_downloads_dir", "get_data_dir", "get_logs_dir",
    "get_organized_dir", "save_config", "set_install_dir", "ensure_dirs",
    # Monitor
    "SystemMonitor", "SystemSnapshot",
    # Processes
    "ProcessManager", "ProcessInfo",
    # Notifications
    "NotificationCenter", "NotificationRecord",
    # Memory
    "Memory",
    # Scheduler
    "TaskScheduler", "Task", "TaskStatus",
    # Voice
    "Speaker", "VoiceSettings",
    # Retry / resilience
    "CircuitBreaker", "CircuitOpenError", "with_retry",
    # Tools
    "ToolRegistry", "Tool", "ToolParam", "ToolResult",
    # AI integrations
    "OllamaClient", "LMStudioClient", "ComfyUIClient", "SDWebUIClient",
    "KoboldCppClient", "TabbyAPIClient", "GPT4AllClient", "AphroditeClient",
    "AIAppRegistry", "AIAppStatus", "OllamaModel", "GenerateResult",
    "SDImage", "ComfyUIJob",
    # Mesh (conditional)
    "MeshEngine", "FreeImage3DClient", "FreeImage3DResult",
    "FREE_IMAGE_TO_3D_BACKENDS", "HuggingFaceDepthEstimator",
    "MeshyClient", "MeshyTaskResult", "Fallout4MeshKnowledge",
    "MeshValidator", "ValidationResult",
    # GPU (conditional)
    "GpuMonitor", "GpuSnapshot",
    # Fine-tuning (conditional)
    "GemmaFineTuner", "UnslothFineTuner",
    "FineTuneConfig", "GPUMemoryProfiler",
    "GEMMA_MODELS", "TrainingResult", "InferenceResult",
    "prepare_mossy_dataset", "list_recommended_models",
    # Availability flags
    "_MESH_AVAILABLE", "_GPU_AVAILABLE", "_GEMMA_AVAILABLE",
]
