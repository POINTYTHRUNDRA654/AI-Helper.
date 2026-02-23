"""AI program integrations.

Discovers, monitors and interacts with the AI programs running on the
desktop — including local LLM runtimes, image generation servers and
GPU-accelerated AI tools.

Supported programs
------------------
+------------------------+-------------------------------+------------------+
| Program                | Default URL                   | Notes            |
+========================+===============================+==================+
| **Ollama**             | http://localhost:11434        | List/run models  |
| **LM Studio**          | http://localhost:1234         | OpenAI-compat    |
| **ComfyUI**            | http://localhost:8188         | Image generation |
| **SD WebUI (A1111)**   | http://localhost:7860         | Image generation |
| **Open WebUI**         | http://localhost:3000         | Chat front-end   |
| **LocalAI**            | http://localhost:8080         | OpenAI-compat    |
| **text-gen-webui**     | http://localhost:7861         | LLM server       |
| **Oobabooga (API)**    | http://localhost:5000         | LLM server       |
| **Jan**                | http://localhost:1337         | LLM desktop app  |
| **LLaMA.cpp server**   | http://localhost:8000         | Raw GGUF serving |
+------------------------+-------------------------------+------------------+

All HTTP calls use the standard-library ``urllib`` — no extra packages
needed.  Each integration gracefully returns ``running=False`` if the
server is not reachable.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 3.0   # seconds for HTTP health checks


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AIAppStatus:
    """Status snapshot for one AI program."""
    name: str
    url: str
    running: bool
    version: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        state = "✓ running" if self.running else "✗ not running"
        ver = f"  v{self.version}" if self.version else ""
        return f"[{self.name}]{ver}  {state}  ({self.url})"


@dataclass
class OllamaModel:
    name: str
    size_gb: float
    modified: str = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.size_gb:.1f} GB)"


@dataclass
class GenerateResult:
    model: str
    prompt: str
    response: str
    done: bool
    error: str = ""

    def __str__(self) -> str:
        if self.error:
            return f"[ERROR] {self.error}"
        return self.response


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------


def _get(url: str, timeout: float = _DEFAULT_TIMEOUT) -> Optional[Any]:
    """GET *url* and return parsed JSON, or None on any error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _post(url: str, payload: Dict[str, Any], timeout: float = _DEFAULT_TIMEOUT) -> Optional[Any]:
    """POST JSON *payload* to *url* and return parsed JSON response, or None."""
    try:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(  # noqa: S310
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except Exception:  # noqa: BLE001
        return None


def _reachable(url: str, timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Return True if *url* responds with any HTTP status."""
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
        return True
    except urllib.error.HTTPError:
        return True   # Any HTTP response (even 4xx) means server is up
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Interact with a running Ollama server.

    Parameters
    ----------
    base_url:
        Ollama API base URL (default ``http://localhost:11434``).
    """

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Status / discovery
    # ------------------------------------------------------------------

    def is_running(self) -> bool:
        """Return ``True`` if the Ollama server is reachable."""
        return _reachable(f"{self.base_url}/api/tags")

    def status(self) -> AIAppStatus:
        data = _get(f"{self.base_url}/api/tags")
        running = data is not None
        version = ""
        if running:
            ver_data = _get(f"{self.base_url}/api/version")
            if ver_data:
                version = ver_data.get("version", "")
        return AIAppStatus(
            name="Ollama", url=self.base_url, running=running, version=version,
        )

    def list_models(self) -> List[OllamaModel]:
        """Return all models available in Ollama."""
        data = _get(f"{self.base_url}/api/tags")
        if not data:
            return []
        models: List[OllamaModel] = []
        for m in data.get("models", []):
            size_gb = m.get("size", 0) / 1e9
            models.append(OllamaModel(
                name=m.get("name", ""),
                size_gb=size_gb,
                modified=m.get("modified_at", ""),
            ))
        return models

    def pull(self, model: str) -> bool:
        """Pull (download) a model by name.  Returns True on success."""
        result = _post(
            f"{self.base_url}/api/pull",
            {"name": model, "stream": False},
            timeout=300.0,
        )
        return result is not None and result.get("status") == "success"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        timeout: float = 120.0,
    ) -> GenerateResult:
        """Run inference with *model* on *prompt*.

        Parameters
        ----------
        model:
            Model name (e.g. ``"llama3"``, ``"mistral"``).
        prompt:
            The user prompt text.
        system:
            Optional system prompt to prepend.
        timeout:
            Seconds before giving up (default 120 s).
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
        }
        if system:
            payload["system"] = system

        result = _post(f"{self.base_url}/api/generate", payload, timeout=timeout)
        if result is None:
            return GenerateResult(
                model=model, prompt=prompt, response="", done=False,
                error="No response from Ollama (is the server running?)",
            )
        return GenerateResult(
            model=model,
            prompt=prompt,
            response=result.get("response", ""),
            done=result.get("done", False),
        )

    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        timeout: float = 120.0,
    ) -> GenerateResult:
        """Send a chat-format message list to *model*.

        Parameters
        ----------
        messages:
            List of ``{"role": "user"|"assistant"|"system", "content": "…"}``
            dicts, matching the OpenAI chat format.
        """
        payload = {"model": model, "messages": messages, "stream": False}
        result = _post(f"{self.base_url}/api/chat", payload, timeout=timeout)
        if result is None:
            return GenerateResult(
                model=model, prompt=str(messages), response="", done=False,
                error="No response from Ollama",
            )
        content = result.get("message", {}).get("content", "")
        return GenerateResult(
            model=model, prompt=str(messages),
            response=content, done=result.get("done", False),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Attempt to start Ollama using the system PATH.  Returns True if launched."""
        if shutil.which("ollama"):
            try:
                subprocess.Popen(  # noqa: S603
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                logger.info("Ollama serve launched")
                return True
            except OSError as exc:
                logger.error("Could not start Ollama: %s", exc)
        else:
            logger.warning("'ollama' not found on PATH")
        return False


# ---------------------------------------------------------------------------
# Generic AI app registry
# ---------------------------------------------------------------------------


# Known AI programs: (name, health-check URL, optional version path)
_KNOWN_APPS: List[tuple[str, str, Optional[str]]] = [
    ("Ollama",               "http://localhost:11434/api/tags",   "http://localhost:11434/api/version"),
    ("LM Studio",            "http://localhost:1234/v1/models",   None),
    ("ComfyUI",              "http://localhost:8188/system_stats", None),
    ("Stable Diffusion WebUI", "http://localhost:7860/sdapi/v1/options", None),
    ("Open WebUI",           "http://localhost:3000",             None),
    ("LocalAI",              "http://localhost:8080/v1/models",   None),
    ("text-generation-webui","http://localhost:7861/api/v1/model", None),
    ("Oobabooga API",        "http://localhost:5000/api/v1/model", None),
    ("Jan",                  "http://localhost:1337/v1/models",   None),
    ("LLaMA.cpp server",     "http://localhost:8000/health",      None),
]


class AIAppRegistry:
    """Discovers and monitors all known AI programs on the local machine.

    Performs fast parallel-ish health checks by iterating through the
    known registry and returning status for each.

    Parameters
    ----------
    timeout:
        Per-app HTTP timeout in seconds (default 2 s — kept short so the
        full scan completes quickly even when most apps are offline).
    extra_apps:
        Optional list of ``(name, health_url, version_url_or_None)`` tuples
        to extend the built-in registry.
    """

    def __init__(
        self,
        timeout: float = 2.0,
        extra_apps: Optional[List[tuple[str, str, Optional[str]]]] = None,
    ) -> None:
        self.timeout = timeout
        self._apps = list(_KNOWN_APPS) + (extra_apps or [])

    def discover(self) -> List[AIAppStatus]:
        """Return a :class:`AIAppStatus` for every known AI program."""
        results: List[AIAppStatus] = []
        for name, health_url, version_url in self._apps:
            running = _reachable(health_url, timeout=self.timeout)
            version = ""
            extra: Dict[str, Any] = {}
            if running and version_url:
                ver_data = _get(version_url, timeout=self.timeout)
                if ver_data and isinstance(ver_data, dict):
                    version = str(ver_data.get("version", ""))
            # Base URL = everything up to the path
            base_url = "/".join(health_url.split("/")[:3])
            results.append(AIAppStatus(
                name=name, url=base_url, running=running,
                version=version, extra=extra,
            ))
        return results

    def running(self) -> List[AIAppStatus]:
        """Return only the AI programs that are currently running."""
        return [s for s in self.discover() if s.running]

    def format_status(self, statuses: Optional[List[AIAppStatus]] = None) -> str:
        """Return a human-readable table of all AI program statuses."""
        if statuses is None:
            statuses = self.discover()
        lines = ["=== AI Programs ==="]
        for s in statuses:
            lines.append(f"  {s}")
        running_count = sum(1 for s in statuses if s.running)
        lines.append(f"\n  {running_count} of {len(statuses)} programs running.")
        return "\n".join(lines)

    def register(self, name: str, health_url: str, version_url: Optional[str] = None) -> None:
        """Add a custom AI program to the registry at runtime."""
        self._apps.append((name, health_url, version_url))
        logger.debug("Registered AI app %r at %s", name, health_url)


# ---------------------------------------------------------------------------
# LM Studio client (OpenAI-compatible)
# ---------------------------------------------------------------------------


class LMStudioClient:
    """Interact with a running LM Studio server (OpenAI-compatible API).

    Parameters
    ----------
    base_url:
        LM Studio server base URL (default ``http://localhost:1234``).
    """

    def __init__(self, base_url: str = "http://localhost:1234") -> None:
        self.base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        return _reachable(f"{self.base_url}/v1/models")

    def status(self) -> AIAppStatus:
        running = self.is_running()
        return AIAppStatus(name="LM Studio", url=self.base_url, running=running)

    def list_models(self) -> List[str]:
        """Return model IDs available in LM Studio."""
        data = _get(f"{self.base_url}/v1/models")
        if not data:
            return []
        return [m.get("id", "") for m in data.get("data", [])]

    def chat(
        self,
        prompt: str,
        model: str = "",
        system: str = "You are a helpful AI assistant.",
        temperature: float = 0.7,
        timeout: float = 120.0,
    ) -> GenerateResult:
        """Send a chat message and return the response.

        Parameters
        ----------
        prompt:
            User message text.
        model:
            Model ID to use.  If empty, LM Studio uses whichever is loaded.
        system:
            System prompt.
        temperature:
            Sampling temperature (0.0–2.0).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if model:
            payload["model"] = model

        result = _post(f"{self.base_url}/v1/chat/completions", payload, timeout=timeout)
        if result is None:
            return GenerateResult(model=model, prompt=prompt, response="", done=False,
                                  error="No response from LM Studio (is it running?)")
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        return GenerateResult(model=model, prompt=prompt, response=content, done=True)


# ---------------------------------------------------------------------------
# ComfyUI client
# ---------------------------------------------------------------------------


@dataclass
class ComfyUIJob:
    """A queued or completed ComfyUI workflow job."""
    prompt_id: str
    status: str   # "pending" | "running" | "complete" | "error"
    outputs: Dict[str, Any] = field(default_factory=dict)


class ComfyUIClient:
    """Interact with a running ComfyUI server.

    Parameters
    ----------
    base_url:
        ComfyUI server base URL (default ``http://localhost:8188``).
    """

    def __init__(self, base_url: str = "http://localhost:8188") -> None:
        self.base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        return _reachable(f"{self.base_url}/system_stats")

    def status(self) -> AIAppStatus:
        data = _get(f"{self.base_url}/system_stats")
        running = data is not None
        version = str(data.get("system", {}).get("comfyui_version", "")) if data else ""
        return AIAppStatus(name="ComfyUI", url=self.base_url, running=running, version=version)

    def queue_prompt(self, workflow: Dict[str, Any], client_id: str = "ai-helper") -> Optional[str]:
        """Queue a ComfyUI workflow for execution.

        Parameters
        ----------
        workflow:
            A ComfyUI API-format workflow dict (the ``"prompt"`` payload).
        client_id:
            Client identifier string (used to track job status).

        Returns
        -------
        str or None
            The ``prompt_id`` string if queued successfully, else ``None``.
        """
        payload = {"prompt": workflow, "client_id": client_id}
        result = _post(f"{self.base_url}/prompt", payload, timeout=30.0)
        if result:
            return result.get("prompt_id")
        return None

    def get_queue(self) -> Dict[str, Any]:
        """Return the current queue status (running + pending jobs)."""
        return _get(f"{self.base_url}/queue") or {}

    def get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Return the history entry for a completed job."""
        data = _get(f"{self.base_url}/history/{prompt_id}")
        if data and prompt_id in data:
            return data[prompt_id]
        return None

    def interrupt(self) -> bool:
        """Interrupt the currently running job."""
        result = _post(f"{self.base_url}/interrupt", {}, timeout=5.0)
        return result is not None

    def get_system_stats(self) -> Dict[str, Any]:
        """Return ComfyUI system stats (GPU memory, python version, etc.)."""
        return _get(f"{self.base_url}/system_stats") or {}


# ---------------------------------------------------------------------------
# Stable Diffusion WebUI (Automatic1111) client
# ---------------------------------------------------------------------------


@dataclass
class SDImage:
    """A generated image returned by the SD WebUI."""
    base64_data: str    # Base64-encoded PNG
    seed: int = -1
    prompt: str = ""
    width: int = 512
    height: int = 512

    def save(self, path: str) -> bool:
        """Save to disk.  Returns True on success."""
        import base64  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415
        try:
            Path(path).write_bytes(base64.b64decode(self.base64_data))
            return True
        except Exception:  # noqa: BLE001
            return False


class SDWebUIClient:
    """Interact with a running Automatic1111 Stable Diffusion WebUI server.

    Parameters
    ----------
    base_url:
        SD WebUI server base URL (default ``http://localhost:7860``).
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self.base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        return _reachable(f"{self.base_url}/sdapi/v1/options")

    def status(self) -> AIAppStatus:
        running = self.is_running()
        version = ""
        if running:
            data = _get(f"{self.base_url}/sdapi/v1/cmd-flags")
            if data:
                version = str(data.get("version", ""))
        return AIAppStatus(name="Stable Diffusion WebUI", url=self.base_url,
                           running=running, version=version)

    def txt2img(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        timeout: float = 180.0,
    ) -> List[SDImage]:
        """Generate images from a text prompt.

        Returns a list of :class:`SDImage` objects (usually one unless
        ``batch_size`` > 1).  Returns an empty list on failure.
        """
        payload: Dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }
        result = _post(f"{self.base_url}/sdapi/v1/txt2img", payload, timeout=timeout)
        if not result:
            return []
        images = result.get("images", [])
        info_raw = result.get("info", "{}")
        try:
            info = json.loads(info_raw) if isinstance(info_raw, str) else info_raw
        except (json.JSONDecodeError, TypeError):
            info = {}
        seeds = info.get("all_seeds", [-1] * len(images))
        return [
            SDImage(
                base64_data=img,
                seed=seeds[i] if i < len(seeds) else -1,
                prompt=prompt,
                width=width,
                height=height,
            )
            for i, img in enumerate(images)
        ]

    def img2img(
        self,
        init_image_b64: str,
        prompt: str,
        negative_prompt: str = "",
        denoising_strength: float = 0.75,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        timeout: float = 180.0,
    ) -> List[SDImage]:
        """Transform an existing image with a text prompt (img2img).

        Parameters
        ----------
        init_image_b64:
            Base64-encoded source image (PNG or JPEG).
        """
        payload: Dict[str, Any] = {
            "init_images": [init_image_b64],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "denoising_strength": denoising_strength,
            "width": width,
            "height": height,
            "steps": steps,
        }
        result = _post(f"{self.base_url}/sdapi/v1/img2img", payload, timeout=timeout)
        if not result:
            return []
        images = result.get("images", [])
        return [SDImage(base64_data=img, prompt=prompt, width=width, height=height)
                for img in images]

    def list_models(self) -> List[str]:
        """Return model checkpoint names available in the WebUI."""
        data = _get(f"{self.base_url}/sdapi/v1/sd-models") or []
        return [m.get("title", "") for m in data]

    def list_samplers(self) -> List[str]:
        """Return available sampler names."""
        data = _get(f"{self.base_url}/sdapi/v1/samplers") or []
        return [s.get("name", "") for s in data]
