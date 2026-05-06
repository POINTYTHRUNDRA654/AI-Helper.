"""Mesh Engine — image-to-mesh scanning and Fallout 4 conversion pipeline.

Mossy's specialised 3D-mesh subsystem.  It handles the full pipeline from
raw photographs to a game-ready mesh asset for Fallout 4 modding.

Free-first design
-----------------
Every tool and service used by this module is **free and open-source**,
downloaded from GitHub or Hugging Face — with one deliberate exception:

* **Meshy** (https://www.meshy.ai) — AI-powered image-to-3D conversion.
  The user holds a paid annual Meshy subscription, so the
  :class:`MeshyClient` integrates with the Meshy REST API.  All other
  services are free.

Free components
~~~~~~~~~~~~~~~
* ``opencv-python`` — image loading and feature detection (GitHub: opencv/opencv)
* ``open3d`` — point cloud + Poisson mesh reconstruction (GitHub: isl-org/Open3D)
* ``trimesh`` — mesh I/O and format conversion (GitHub: mikedh/trimesh)
* ``numpy`` / ``scipy`` — numerical computing (free)
* ``Pillow`` — image loading (GitHub: python-pillow/Pillow)
* ``pyffi`` — NIF file format I/O (GitHub: niftools/pyffi)
* HuggingFace ``transformers`` + ``Depth-Anything`` — free monocular depth
  estimation for single-image workflows (HF: LiheYoung/depth-anything)

Pipeline overview
-----------------
1. **Scan** — load photographs; extract 2-D image features with OpenCV.
2. **Depth** — estimate per-pixel depth via HuggingFace Depth-Anything (free)
   or Meshy's image-to-3D API (user subscription).
3. **Reconstruct** — build a 3-D point cloud with Open3D.
4. **Mesh** — Poisson surface reconstruction → triangle mesh.
5. **Export** — OBJ/PLY (universal), NIF (pyffi, for Fallout 4 direct import).

Usage (quick-start)
-------------------
::

    from ai_helper.mesh_engine import MeshEngine
    engine = MeshEngine()

    # Ask Mossy a modding question
    answer = engine.ask_knowledge("How many polygons can a weapon mesh use?")

    # Scan images using free pipeline
    result = engine.scan_images(
        image_paths=["front.jpg", "side.jpg", "back.jpg"],
        output_dir="my_mesh_output",
        export_formats=["obj"],
    )
    print(result.summary)

    # Use Meshy for high-quality AI image-to-3D (requires API key)
    result = engine.meshy_image_to_3d(
        image_path="photo.jpg",
        output_dir="meshy_output",
        api_key="your-meshy-api-key",
    )
    print(result.summary)

    # Validate a mesh file
    validation = engine.validate_mesh("my_mesh.obj")
    print(validation)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Meshy API client (user's paid subscription)
# ---------------------------------------------------------------------------

_MESHY_API_BASE = "https://api.meshy.ai/v2"


@dataclass
class MeshyTaskResult:
    """Result from a Meshy image-to-3D task."""
    task_id: str
    status: str               # "PENDING" | "IN_PROGRESS" | "SUCCEEDED" | "FAILED"
    model_urls: Dict[str, str] = field(default_factory=dict)  # format → download URL
    thumbnail_url: str = ""
    progress: int = 0          # 0-100
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"


class MeshyClient:
    """Client for the Meshy image-to-3D REST API.

    The user holds a paid Meshy annual subscription.  All other mesh
    services used by Mossy are free.

    Meshy API docs: https://docs.meshy.ai/api-image-to-3d

    Parameters
    ----------
    api_key:
        Meshy API key.  If not supplied here, read from the
        ``MESHY_API_KEY`` environment variable.
    timeout:
        HTTP request timeout in seconds.
    poll_interval:
        Seconds between status-poll requests.
    max_wait:
        Maximum seconds to wait for a task to complete.

    Example
    -------
    ::

        client = MeshyClient(api_key="your-key")
        task = client.image_to_3d("photo.jpg")
        if task.succeeded:
            client.download_result(task, output_dir="output", fmt="obj")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        poll_interval: float = 5.0,
        max_wait: float = 600.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("MESHY_API_KEY", "")
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.max_wait = max_wait

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def image_to_3d(
        self,
        image_path: str,
        enable_pbr: bool = True,
        ai_model: str = "meshy-4",
        topology: str = "quad",
        target_polycount: int = 10000,
        should_remesh: bool = True,
    ) -> MeshyTaskResult:
        """Submit an image-to-3D task and **wait** for completion.

        Parameters
        ----------
        image_path:
            Local image file (JPEG/PNG).  Uploaded as base-64 data URL.
        enable_pbr:
            When ``True``, Meshy generates PBR textures (diffuse, normal,
            metallic/roughness) suitable for Fallout 4 modding.
        ai_model:
            Meshy model version.  ``"meshy-4"`` is the current default.
        topology:
            Mesh topology: ``"quad"`` (cleaner for game assets) or ``"triangle"``.
        target_polycount:
            Polygon budget passed to Meshy; default 10 000 is a good starting
            point for Fallout 4 weapons/armour.
        should_remesh:
            Whether Meshy automatically remeshes to the target poly count.
        """
        if not self.api_key:
            raise ValueError(
                "Meshy API key is required.  Set it via the MESHY_API_KEY "
                "environment variable or pass api_key= to MeshyClient()."
            )

        # Encode image as data URL
        data_url = self._image_to_data_url(image_path)

        payload = {
            "image_url": data_url,
            "enable_pbr": enable_pbr,
            "ai_model": ai_model,
            "topology": topology,
            "target_polycount": target_polycount,
            "should_remesh": should_remesh,
        }

        # Submit task
        response = self._post("/image-to-3d", payload)
        task_id = response.get("result", "")
        if not task_id:
            return MeshyTaskResult(
                task_id="",
                status="FAILED",
                error=f"Unexpected Meshy response: {response}",
            )

        logger.info("Meshy task submitted: %s", task_id)
        return self._poll_until_done(task_id)

    def get_task(self, task_id: str) -> MeshyTaskResult:
        """Fetch the current status of a Meshy task."""
        data = self._get(f"/image-to-3d/{task_id}")
        return self._parse_task(data)

    def download_result(
        self,
        task: MeshyTaskResult,
        output_dir: str,
        fmt: str = "obj",
    ) -> Optional[str]:
        """Download the mesh from a completed task.

        Parameters
        ----------
        task:
            A :class:`MeshyTaskResult` with ``status == "SUCCEEDED"``.
        output_dir:
            Directory to save the downloaded file.
        fmt:
            Format to download: ``"obj"``, ``"glb"``, ``"fbx"``, ``"usdz"``.

        Returns
        -------
        str | None
            Local file path, or ``None`` if the format is unavailable.
        """
        url = task.model_urls.get(fmt)
        if not url:
            logger.warning("Meshy task %s has no %r download URL.", task.task_id, fmt)
            return None

        # Validate that the URL is from a Meshy domain to prevent SSRF
        parsed_url = urllib.parse.urlparse(url)
        if "meshy.ai" not in parsed_url.netloc and not parsed_url.netloc.endswith(".meshy.ai"):
            logger.error(
                "Meshy download URL has unexpected domain %r — refusing to fetch.",
                parsed_url.netloc,
            )
            return None

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"meshy_{task.task_id}.{fmt}"

        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            dest.write_bytes(resp.read())

        logger.info("Meshy download saved: %s", dest)
        return str(dest)

    def download_textures(
        self, task: MeshyTaskResult, output_dir: str
    ) -> List[str]:
        """Download all PBR texture maps from *task* into *output_dir*.

        Returns the list of saved file paths.
        """
        saved: List[str] = []
        texture_urls: Dict[str, str] = task.model_urls.get("textures", {})  # type: ignore[assignment]
        if not isinstance(texture_urls, dict):
            return saved
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for channel, url in texture_urls.items():
            ext = Path(urllib.parse.urlparse(url).path).suffix or ".png"
            dest = out_dir / f"meshy_{task.task_id}_{channel}{ext}"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    dest.write_bytes(resp.read())
                saved.append(str(dest))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to download texture %s: %s", channel, exc)
        return saved

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _image_to_data_url(self, image_path: str) -> str:
        import base64  # noqa: PLC0415
        import mimetypes  # noqa: PLC0415
        path = Path(image_path)
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/jpeg"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{data}"

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = _MESHY_API_BASE + endpoint
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Meshy API error {exc.code}: {body_text}") from exc

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = _MESHY_API_BASE + endpoint
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _poll_until_done(self, task_id: str) -> MeshyTaskResult:
        deadline = time.monotonic() + self.max_wait
        while time.monotonic() < deadline:
            task = self.get_task(task_id)
            logger.debug("Meshy task %s: %s (%d%%)", task_id, task.status, task.progress)
            if task.status in ("SUCCEEDED", "FAILED", "EXPIRED"):
                return task
            time.sleep(self.poll_interval)
        return MeshyTaskResult(
            task_id=task_id,
            status="FAILED",
            error=f"Timed out after {self.max_wait}s waiting for Meshy task.",
        )

    @staticmethod
    def _parse_task(data: Dict[str, Any]) -> MeshyTaskResult:
        model_urls = data.get("model_urls") or {}
        # Normalise nested textures dict if present
        textures = data.get("texture_urls") or {}
        if textures:
            model_urls["textures"] = textures
        return MeshyTaskResult(
            task_id=data.get("id", ""),
            status=data.get("status", "UNKNOWN"),
            model_urls=model_urls,
            thumbnail_url=data.get("thumbnail_url", ""),
            progress=data.get("progress", 0),
            error=data.get("message", "") if data.get("status") == "FAILED" else "",
        )


# ---------------------------------------------------------------------------
# HuggingFace depth estimator (free, no subscription needed)
# ---------------------------------------------------------------------------


class HuggingFaceDepthEstimator:
    """Free monocular depth estimation using models from Hugging Face.

    Uses ``Depth-Anything v2`` by default — a state-of-the-art depth model
    available for free at ``depth-anything/Depth-Anything-V2-Small-hf``.

    All models are downloaded from HuggingFace on first use and cached
    locally.  No API key or subscription is required.

    Parameters
    ----------
    model_id:
        HuggingFace model ID.  Defaults to the small Depth-Anything v2
        variant which runs on CPU in a few seconds per image.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected when not specified.
    cache_dir:
        Optional local directory to cache downloaded model weights.

    Example
    -------
    ::

        estimator = HuggingFaceDepthEstimator()
        depth = estimator.estimate("photo.jpg")       # numpy array
        estimator.save_depth_map(depth, "depth.png")  # visualisation
    """

    # Good free defaults from HuggingFace (ordered by quality/speed tradeoff)
    FREE_MODELS = {
        "depth_anything_v2_small": "depth-anything/Depth-Anything-V2-Small-hf",
        "depth_anything_v2_base": "depth-anything/Depth-Anything-V2-Base-hf",
        "depth_anything_v2_large": "depth-anything/Depth-Anything-V2-Large-hf",
        "midas_large": "Intel/dpt-large",
        "midas_hybrid": "Intel/dpt-hybrid-midas",
        "zoedepth": "isl-org/ZoeDepth",
    }

    DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL
        self._device = device
        self.cache_dir = cache_dir
        self._pipeline: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, image_path: str) -> Any:
        """Return a depth map as a NumPy array (H×W, float32, normalised 0-1).

        Requires ``transformers``, ``torch``, and ``Pillow``.
        """
        try:
            from transformers import pipeline as hf_pipeline  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "transformers, torch, and Pillow are required for depth estimation.\n"
                "Install: pip install transformers torch Pillow\n"
                "All packages are free and available on GitHub / PyPI."
            ) from exc

        if self._pipeline is None:
            device = self._device or self._auto_device()
            logger.info(
                "Loading depth model %r from HuggingFace (device=%s)…",
                self.model_id, device,
            )
            kwargs: Dict[str, Any] = {"device": device}
            if self.cache_dir:
                kwargs["model_kwargs"] = {"cache_dir": self.cache_dir}
            self._pipeline = hf_pipeline(
                task="depth-estimation",
                model=self.model_id,
                **kwargs,
            )

        img = PILImage.open(image_path).convert("RGB")
        output = self._pipeline(img)
        depth_img = output["depth"]  # PIL Image (grayscale)
        depth_arr = np.array(depth_img).astype(np.float32)
        # Normalise to 0-1
        dmin, dmax = depth_arr.min(), depth_arr.max()
        if dmax > dmin:
            depth_arr = (depth_arr - dmin) / (dmax - dmin)
        return depth_arr

    def save_depth_map(self, depth: Any, output_path: str) -> str:
        """Save a normalised depth array as a greyscale PNG for inspection."""
        try:
            import numpy as np  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("numpy and Pillow required.") from exc

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        img = PILImage.fromarray((depth * 255).astype("uint8"), mode="L")
        img.save(str(out))
        logger.info("Depth map saved: %s", out)
        return str(out)

    def estimate_and_save(self, image_path: str, output_dir: str) -> Tuple[Any, str]:
        """Estimate depth and save the visualisation.  Returns (depth_array, png_path)."""
        depth = self.estimate(image_path)
        stem = Path(image_path).stem
        png_path = self.save_depth_map(depth, str(Path(output_dir) / f"{stem}_depth.png"))
        return depth, png_path

    @staticmethod
    def list_free_models() -> str:
        """Return a formatted list of the free HuggingFace depth models."""
        lines = ["Free HuggingFace depth estimation models:"]
        for name, hf_id in HuggingFaceDepthEstimator.FREE_MODELS.items():
            lines.append(f"  {name:<28} → hf.co/{hf_id}")
        lines.append(
            "\nAll models are free and downloaded from HuggingFace on first use."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch  # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# Free open-source image-to-3D backends
# ---------------------------------------------------------------------------


# Registry of every free image-to-3D tool Mossy knows about.
# All are open-source (GitHub) and/or available on HuggingFace — 100% free.
FREE_IMAGE_TO_3D_BACKENDS: Dict[str, Dict[str, Any]] = {
    "triposg": {
        "name": "TripoSG (VAST-AI-Research)",
        "description": (
            "2025 successor to TripoSR — 1.5B rectified-flow transformer. "
            "Produces sharp, fine-detail GLB meshes from a single image. "
            "Best free quality/VRAM tradeoff."
        ),
        "github": "https://github.com/VAST-AI-Research/TripoSG",
        "hf_space": "https://huggingface.co/spaces/VAST-AI/TripoSG",
        "hf_model": "VAST-AI/TripoSG",
        "license": "MIT",
        "vram_gb": 8,
        "output_formats": ["glb"],
        "install": (
            "git clone https://github.com/VAST-AI-Research/TripoSG.git\n"
            "cd TripoSG && pip install -r requirements.txt"
        ),
        "run_cmd": "python -m scripts.inference_triposg --image-input IMAGE --output-path OUTPUT.glb",
    },
    "trellis": {
        "name": "Microsoft TRELLIS",
        "description": (
            "State-of-the-art unified 3D generation from image or text. "
            "Outputs textured GLB, PLY and radiance field."
        ),
        "github": "https://github.com/microsoft/TRELLIS",
        "hf_space": "https://huggingface.co/spaces/Microsoft/TRELLIS",
        "hf_model": "microsoft/TRELLIS-image-large",
        "license": "MIT",
        "vram_gb": 16,
        "output_formats": ["glb", "ply"],
        "install": (
            "git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git\n"
            "cd TRELLIS && . ./setup.sh --new-env --basic --xformers --flash-attn"
        ),
        "run_cmd": "python app.py  # Gradio UI",
    },
    "triposr": {
        "name": "TripoSR (Stability AI / VAST-AI)",
        "description": (
            "Original fast single-image 3D reconstruction. "
            "Lightest option at ~6 GB VRAM. Superseded by TripoSG."
        ),
        "github": "https://github.com/VAST-AI-Research/TripoSR",
        "hf_space": "https://huggingface.co/spaces/stabilityai/TripoSR",
        "hf_model": "stabilityai/TripoSR",
        "license": "MIT",
        "vram_gb": 6,
        "output_formats": ["obj", "glb"],
        "install": (
            "git clone https://github.com/VAST-AI-Research/TripoSR.git\n"
            "cd TripoSR && pip install -r requirements.txt"
        ),
        "run_cmd": "python run.py IMAGE --output-dir OUTPUT/",
    },
    "instantmesh": {
        "name": "InstantMesh (TencentARC)",
        "description": (
            "Image to high-quality 3D mesh with texture map in seconds. "
            "Apache 2.0 license."
        ),
        "github": "https://github.com/TencentARC/InstantMesh",
        "hf_space": "https://huggingface.co/spaces/TencentARC/InstantMesh",
        "hf_model": "TencentARC/InstantMesh",
        "license": "Apache 2.0",
        "vram_gb": 12,
        "output_formats": ["obj"],
        "install": (
            "git clone https://github.com/TencentARC/InstantMesh.git\n"
            "cd InstantMesh && pip install -r requirements.txt"
        ),
        "run_cmd": "python run.py configs/instant-mesh-large.yaml IMAGE --export_texmap",
    },
    "shap_e": {
        "name": "Shap-E (OpenAI)",
        "description": (
            "Text or image to 3D implicit function → OBJ/PLY via marching cubes. "
            "Pip-installable, CPU-capable."
        ),
        "github": "https://github.com/openai/shap-e",
        "hf_space": "https://huggingface.co/spaces/hysts/Shap-E",
        "hf_model": "openai/shap-e-img2img",
        "license": "MIT",
        "vram_gb": 8,
        "output_formats": ["obj", "ply"],
        "install": (
            "git clone https://github.com/openai/shap-e.git\n"
            "cd shap-e && pip install -e ."
        ),
        "run_cmd": "# See shap_e/examples/sample_image_to_3d.ipynb",
    },
}


@dataclass
class FreeImage3DResult:
    """Result from a free image-to-3D inference run."""
    backend: str
    input_image: str
    output_dir: str
    mesh_path: Optional[str] = None
    exported_files: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    success: bool = False
    error: str = ""

    @property
    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        files = ", ".join(Path(f).name for f in self.exported_files) or "none"
        info = FREE_IMAGE_TO_3D_BACKENDS.get(self.backend, {})
        lines = [
            f"{status} {info.get('name', self.backend)}",
            f"  Input  : {self.input_image}",
            f"  Output : {files}",
            f"  Time   : {self.elapsed_s:.1f}s",
        ]
        if self.error:
            lines.append(f"  Error  : {self.error}")
        return "\n".join(lines)


class FreeImage3DClient:
    """Run free open-source image-to-3D models from GitHub / HuggingFace.

    Meshy (meshy.ai) is a commercial SaaS with no open-source version.
    These are the best free alternatives — all MIT or Apache-2.0:

    +---------------+--------------------------------+------+----------+
    | Backend       | Source                         | VRAM | License  |
    +===============+================================+======+==========+
    | ``triposg``   | github.com/VAST-AI/TripoSG     |  8GB | MIT ★    |
    | ``trellis``   | github.com/microsoft/TRELLIS   | 16GB | MIT      |
    | ``triposr``   | github.com/VAST-AI/TripoSR     |  6GB | MIT      |
    | ``instantmesh``| github.com/TencentARC          | 12GB | Apache 2 |
    | ``shap_e``    | github.com/openai/shap-e       |  8GB | MIT      |
    +---------------+--------------------------------+------+----------+
    ★ TripoSG is the recommended default — best quality at 8 GB VRAM.

    Each backend is optional — only the one you actually use needs to be
    cloned and installed.  Call :meth:`list_backends` to see all options
    and :meth:`install_instructions` for setup steps.

    Parameters
    ----------
    repo_root:
        Directory containing the cloned GitHub repos, organised as
        sub-folders named after each backend key (``triposg/``,
        ``trellis/``, ``triposr/``, etc.).
        Defaults to ``~/AI-Helper/FreeModels``.
    device:
        ``"cuda"`` or ``"cpu"``.  Auto-detected when omitted.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self.repo_root = (
            Path(repo_root) if repo_root else Path.home() / "AI-Helper" / "FreeModels"
        )
        self._device = device

    # ------------------------------------------------------------------
    # Backend info
    # ------------------------------------------------------------------

    @staticmethod
    def list_backends() -> str:
        """Return a formatted table of all free image-to-3D backends."""
        lines = [
            "Free image-to-3D backends (open-source — GitHub / HuggingFace):",
            "",
            f"  {'Backend':<14} {'Name':<35} {'VRAM':>5}  {'License':<12}  HuggingFace Space",
            "  " + "-" * 105,
        ]
        for key, info in FREE_IMAGE_TO_3D_BACKENDS.items():
            star = " ★" if key == "triposg" else "  "
            lines.append(
                f"  {key:<14} {info['name']:<35} {info['vram_gb']:>4}GB  "
                f"{info['license']:<12}  {info.get('hf_space', '')}{star}"
            )
        lines += [
            "",
            "★  TripoSG is the recommended default (8 GB VRAM, MIT, best free quality).",
            "   Meshy (meshy.ai) is commercial — no open-source version exists on GitHub",
            "   or HuggingFace.  The Tripo team's open-source work is TripoSG / TripoSR.",
        ]
        return "\n".join(lines)

    @staticmethod
    def install_instructions(backend: str) -> str:
        """Return step-by-step installation instructions for *backend*."""
        info = FREE_IMAGE_TO_3D_BACKENDS.get(backend)
        if not info:
            valid = ", ".join(FREE_IMAGE_TO_3D_BACKENDS)
            return f"Unknown backend {backend!r}. Valid options: {valid}"
        return (
            f"{info['name']}  [{info['license']} license]\n"
            f"GitHub  : {info['github']}\n"
            f"HF Space: {info.get('hf_space', 'N/A')}\n"
            f"HF Model: {info.get('hf_model', 'N/A')}\n"
            f"GPU VRAM: {info['vram_gb']} GB minimum\n"
            f"Output  : {', '.join(info['output_formats'])}\n\n"
            f"Install:\n{info['install']}\n\n"
            f"Run:\n{info.get('run_cmd', 'See GitHub README')}"
        )

    # ------------------------------------------------------------------
    # TripoSG — recommended free default (8 GB VRAM, MIT, 2025)
    # ------------------------------------------------------------------

    def run_triposg(
        self,
        image_path: str,
        output_dir: str,
        faces: Optional[int] = None,
        seed: int = 42,
    ) -> FreeImage3DResult:
        """Run TripoSG to convert a single image to a textured GLB mesh.

        TripoSG is the 2025 successor to TripoSR by VAST-AI-Research.
        It uses a 1.5 B rectified-flow transformer and produces
        significantly sharper geometry than TripoSR, at the same
        8 GB VRAM cost.

        Clone it first:

        .. code-block:: bash

            git clone https://github.com/VAST-AI-Research/TripoSG.git ~/AI-Helper/FreeModels/triposg
            pip install -r ~/AI-Helper/FreeModels/triposg/requirements.txt

        Model weights are downloaded automatically from HuggingFace
        (``VAST-AI/TripoSG``) on the first run.

        Parameters
        ----------
        image_path:
            Input photo (PNG or JPEG).  Background is auto-removed.
        output_dir:
            Where to save the ``.glb`` output.
        faces:
            Optional triangle budget.  Pass e.g. ``5000`` to cap the
            output mesh at 5 000 triangles (good for Fallout 4 props).
            When ``None``, the model's default resolution is used.
        seed:
            Random seed for reproducible results.
        """
        t0 = time.monotonic()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = FreeImage3DResult(
            backend="triposg",
            input_image=str(image_path),
            output_dir=str(out),
        )

        repo_path = self.repo_root / "triposg"
        inference_module = repo_path / "scripts" / "inference_triposg.py"

        if not inference_module.exists():
            result.error = (
                f"TripoSG not found at {repo_path}.\n"
                "Clone it with:\n"
                f"  git clone https://github.com/VAST-AI-Research/TripoSG.git {repo_path}\n"
                f"  pip install -r {repo_path}/requirements.txt\n"
                "Weights will download automatically from HuggingFace on first run."
            )
            result.elapsed_s = time.monotonic() - t0
            return result

        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        stem = Path(image_path).stem
        output_glb = str(out / f"triposg_{stem}.glb")

        cmd = [
            sys.executable, "-m", "scripts.inference_triposg",
            "--image-input", str(image_path),
            "--output-path", output_glb,
            "--seed", str(seed),
        ]
        if faces is not None:
            cmd.extend(["--faces", str(faces)])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True,
                cwd=str(repo_path),
                timeout=600,
            )
            if proc.returncode != 0:
                result.error = (proc.stderr or proc.stdout).strip()
            else:
                if Path(output_glb).exists():
                    result.exported_files.append(output_glb)
                    result.mesh_path = output_glb
                    result.success = True
                else:
                    # Scan for any glb produced
                    glbs = list(out.rglob("*.glb"))
                    if glbs:
                        result.exported_files = [str(p) for p in glbs]
                        result.mesh_path = str(glbs[0])
                        result.success = True
                    else:
                        result.error = "TripoSG finished but no .glb output found."
        except subprocess.TimeoutExpired:
            result.error = "TripoSG timed out after 600 s."
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)

        result.elapsed_s = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # TripoSR — lightest free option (6 GB VRAM, original)
    # ------------------------------------------------------------------

    def run_triposr(
        self,
        image_path: str,
        output_dir: str,
        chunk_size: int = 8192,
        mc_resolution: int = 256,
        no_remove_bg: bool = False,
        bake_texture: bool = False,
        texture_resolution: int = 1024,
    ) -> FreeImage3DResult:
        """Run TripoSR (original, 6 GB VRAM) to convert an image to OBJ/GLB.

        Clone it first:

        .. code-block:: bash

            git clone https://github.com/VAST-AI-Research/TripoSR.git ~/AI-Helper/FreeModels/triposr
            pip install -r ~/AI-Helper/FreeModels/triposr/requirements.txt

        Parameters
        ----------
        image_path:
            Input photo (PNG or JPEG).
        output_dir:
            Where to save the OBJ/GLB output.
        chunk_size:
            Marching-cubes chunk size (lower = less VRAM).
        mc_resolution:
            Marching-cubes resolution (higher = more detail).
        no_remove_bg:
            Set ``True`` if the image already has a transparent background.
        bake_texture:
            When ``True``, bake a texture map instead of vertex colours.
        texture_resolution:
            Texture map size in pixels when ``bake_texture=True``.
        """
        t0 = time.monotonic()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = FreeImage3DResult(
            backend="triposr",
            input_image=str(image_path),
            output_dir=str(out),
        )

        repo_path = self.repo_root / "triposr"
        run_script = repo_path / "run.py"

        if not run_script.exists():
            result.error = (
                f"TripoSR not found at {repo_path}.\n"
                "Clone it with:\n"
                f"  git clone https://github.com/VAST-AI-Research/TripoSR.git {repo_path}\n"
                f"  pip install -r {repo_path}/requirements.txt"
            )
            result.elapsed_s = time.monotonic() - t0
            return result

        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415

        cmd = [
            sys.executable, str(run_script),
            str(image_path),
            "--output-dir", str(out),
            "--chunk-size", str(chunk_size),
            "--mc-resolution", str(mc_resolution),
        ]
        if no_remove_bg:
            cmd.append("--no-remove-bg")
        if bake_texture:
            cmd.extend(["--bake-texture", "--texture-resolution", str(texture_resolution)])

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=str(repo_path), timeout=600,
            )
            if proc.returncode != 0:
                result.error = (proc.stderr or proc.stdout).strip()
            else:
                for ext in ("obj", "glb", "mtl"):
                    result.exported_files.extend(str(p) for p in out.rglob(f"*.{ext}"))
                if result.exported_files:
                    result.mesh_path = result.exported_files[0]
                    result.success = True
                else:
                    result.error = "TripoSR completed but no output files found."
        except subprocess.TimeoutExpired:
            result.error = "TripoSR timed out after 600 s."
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)

        result.elapsed_s = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # TRELLIS — highest quality (16 GB VRAM)
    # ------------------------------------------------------------------

    def run_trellis(
        self,
        image_path: str,
        output_dir: str,
        simplify: float = 0.95,
        texture_size: int = 1024,
        seed: int = 1,
    ) -> FreeImage3DResult:
        """Run Microsoft TRELLIS to convert an image to a textured GLB.

        Clone and set up TRELLIS first:

        .. code-block:: bash

            git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git ~/AI-Helper/FreeModels/trellis
            cd ~/AI-Helper/FreeModels/trellis
            . ./setup.sh --new-env --basic --xformers --flash-attn

        Requires 16 GB+ GPU VRAM.

        Parameters
        ----------
        image_path:
            Input photo (PNG or JPEG).
        output_dir:
            Where to save GLB and PLY outputs.
        simplify:
            Mesh simplification ratio (0.95 = keep 95% of polygons).
        texture_size:
            Texture resolution in pixels.
        seed:
            Random seed for reproducibility.
        """
        t0 = time.monotonic()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = FreeImage3DResult(
            backend="trellis",
            input_image=str(image_path),
            output_dir=str(out),
        )

        repo_path = self.repo_root / "trellis"
        if not (repo_path / "trellis").exists():
            result.error = (
                f"TRELLIS not found at {repo_path}.\n"
                "Set it up with:\n"
                f"  git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git {repo_path}\n"
                f"  cd {repo_path} && . ./setup.sh --new-env --basic --xformers --flash-attn\n"
                "Requires 16 GB+ GPU VRAM."
            )
            result.elapsed_s = time.monotonic() - t0
            return result

        try:
            import sys  # noqa: PLC0415
            import os as _os  # noqa: PLC0415
            sys.path.insert(0, str(repo_path))
            _os.environ.setdefault("SPCONV_ALGO", "native")
            from PIL import Image as PILImage  # noqa: PLC0415
            from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: PLC0415
            from trellis.utils import postprocessing_utils  # noqa: PLC0415

            logger.info("Loading TRELLIS pipeline (microsoft/TRELLIS-image-large)…")
            pipeline = TrellisImageTo3DPipeline.from_pretrained(
                "microsoft/TRELLIS-image-large"
            )
            pipeline.cuda()

            image = PILImage.open(image_path).convert("RGB")
            outputs = pipeline.run(image, seed=seed)

            stem = Path(image_path).stem
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=simplify,
                texture_size=texture_size,
            )
            glb_path = str(out / f"trellis_{stem}.glb")
            glb.export(glb_path)
            result.exported_files.append(glb_path)

            ply_path = str(out / f"trellis_{stem}.ply")
            outputs["gaussian"][0].save_ply(ply_path)
            result.exported_files.append(ply_path)

            result.mesh_path = glb_path
            result.success = True

        except ImportError as exc:
            result.error = (
                f"TRELLIS import failed: {exc}\n"
                "Make sure you ran setup.sh and are in the TRELLIS conda environment."
            )
        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            logger.exception("TRELLIS inference failed")

        result.elapsed_s = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # Shap-E — pip-installable, CPU-capable fallback
    # ------------------------------------------------------------------

    def run_shap_e(
        self,
        image_path: str,
        output_dir: str,
        guidance_scale: float = 3.0,
        karras_steps: int = 64,
    ) -> FreeImage3DResult:
        """Run OpenAI Shap-E to generate a 3D mesh from an image.

        Shap-E is pip-installable and can run on CPU (slower):

        .. code-block:: bash

            git clone https://github.com/openai/shap-e.git ~/AI-Helper/FreeModels/shap_e
            pip install -e ~/AI-Helper/FreeModels/shap_e

        Parameters
        ----------
        image_path:
            Input photo.
        output_dir:
            Where to write the OBJ output.
        guidance_scale:
            How closely to match the input image (higher = closer).
        karras_steps:
            Diffusion noise schedule steps (higher = better quality, slower).
        """
        t0 = time.monotonic()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        result = FreeImage3DResult(
            backend="shap_e",
            input_image=str(image_path),
            output_dir=str(out),
        )

        repo_path = self.repo_root / "shap_e"
        if (repo_path / "shap_e").exists():
            import sys  # noqa: PLC0415
            sys.path.insert(0, str(repo_path))

        try:
            import torch  # noqa: PLC0415
            from PIL import Image as PILImage  # noqa: PLC0415
            from shap_e.diffusion.sample import sample_latents  # noqa: PLC0415
            from shap_e.diffusion.gaussian_diffusion import diffusion_from_config  # noqa: PLC0415
            from shap_e.models.download import load_model, load_config  # noqa: PLC0415
            from shap_e.util.notebooks import decode_latent_mesh  # noqa: PLC0415
        except ImportError as exc:
            result.error = (
                f"Shap-E import failed: {exc}\n"
                "Clone and install with:\n"
                f"  git clone https://github.com/openai/shap-e.git {repo_path}\n"
                f"  pip install -e {repo_path}"
            )
            result.elapsed_s = time.monotonic() - t0
            return result

        try:
            device = torch.device(
                self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            xm = load_model("transmitter", device=device)
            model = load_model("image300M", device=device)
            diffusion = diffusion_from_config(load_config("diffusion"))

            img = PILImage.open(image_path).convert("RGB")
            latents = sample_latents(
                batch_size=1,
                model=model,
                diffusion=diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(images=[img]),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=karras_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )
            obj_path = str(out / f"shap_e_{Path(image_path).stem}.obj")
            t_mesh = decode_latent_mesh(xm, latents[0]).tri_mesh()
            try:
                with open(obj_path, "w", encoding="utf-8") as fh:
                    t_mesh.write_obj(fh)
            except Exception as write_exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to write Shap-E OBJ: {write_exc}") from write_exc
            result.exported_files.append(obj_path)
            result.mesh_path = obj_path
            result.success = True

        except Exception as exc:  # noqa: BLE001
            result.error = str(exc)
            logger.exception("Shap-E inference failed")

        result.elapsed_s = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch  # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


# ---------------------------------------------------------------------------
# Fallout 4 domain knowledge
# ---------------------------------------------------------------------------


_FO4_KNOWLEDGE: Dict[str, Any] = {
    "polygon_budgets": {
        "weapon": 5000,
        "weapon_high": 10000,
        "armor_piece": 3000,
        "full_armor": 8000,
        "settlement_object_small": 1000,
        "settlement_object_medium": 3000,
        "settlement_object_large": 6000,
        "character_head": 4000,
        "character_body": 6000,
        "vehicle": 15000,
        "building_exterior": 20000,
    },
    "texture_channels": {
        "diffuse": "_d.dds",
        "normal": "_n.dds",
        "specular": "_s.dds",
        "glow": "_g.dds",
        "environment_mask": "_e.dds",
    },
    "texture_formats": {
        "diffuse": "BC1 (DXT1) or BC3 (DXT5) for transparency",
        "normal": "BC5 (ATI2/3Dc) — red=tangent X, green=tangent Y",
        "specular": "BC1 or BC3 depending on alpha channel usage",
        "glow": "BC1",
    },
    "texture_resolutions": {
        "small_prop": "512×512",
        "medium_prop": "1024×1024",
        "large_prop_character": "2048×2048",
        "high_res_pack": "4096×4096",
        "note": "Always power-of-two; mipmaps required for all in-game textures",
    },
    "nif_block_types": {
        "BSTriShape": "Primary mesh block in Fallout 4 (replaces NiTriShape from older games)",
        "BSLightingShaderProperty": "Shader property attached to BSTriShape for PBR-like shading",
        "BSEffectShaderProperty": "For glowing/effect meshes",
        "BSXFlags": "Flags block — controls collision, ragdoll, animation",
        "bhkCollisionObject": "Collision mesh root",
        "bhkRigidBody": "Physics rigid body data",
        "BSSubIndexTriShape": "LOD mesh block supporting multiple material indices",
        "NiNode": "Skeleton/hierarchy node",
        "BSFadeNode": "Root node type for most Fallout 4 static objects",
    },
    "nif_version": {
        "version": "20.2.0.7",
        "user_version": 12,
        "user_version_2": 130,
        "note": "These exact version numbers are required; NifSkope and the CK enforce them",
    },
    "lod_tiers": {
        "LOD0": "Full-detail mesh, used within ~30 metres",
        "LOD1": "~50% poly reduction, used 30-60 m",
        "LOD2": "~75% poly reduction, used 60-120 m",
        "LOD3": "~90% poly reduction, used beyond 120 m",
        "tools": ["xLODGen", "LODGen", "DynDOLOD (for worldspace LOD)"],
    },
    "collision_types": {
        "bhkConvexVerticesShape": "Best for simple convex objects — fastest physics",
        "bhkMoppBvTreeShape": "For complex concave shapes; auto-generated by Havok",
        "bhkListShape": "Compound shape combining multiple primitives",
        "bhkBoxShape": "Axis-aligned box — cheapest possible collision",
        "note": "All static objects need collision; havok_filter_layer = OL_STATIC (1)",
    },
    "scale": {
        "game_unit": "1 Bethesda unit = ~1.4285 cm (approximately 70 units per metre)",
        "human_height_units": "128",
        "recommended_export_scale": "1 unit in Blender = 1 unit in game (no scale modifier)",
    },
    "coordinate_system": {
        "handedness": "Right-handed (same as Blender default)",
        "up_axis": "Z-up",
        "forward_axis": "Y-forward",
        "note": "Blender uses Z-up / Y-forward which matches FO4 natively",
    },
    "workflow_steps": [
        "1. Capture photos from multiple angles (min 20-30, overlap ≥60%)",
        "2. Option A — Free AI (recommended): TripoSG (github.com/VAST-AI-Research/TripoSG, MIT, 8GB VRAM) — single image → GLB",
        "2. Option B — Paid AI (subscription): Meshy (meshy.ai) — upload image → download OBJ/GLB",
        "2. Option C — Free photogrammetry: Meshroom (free, GitHub) or COLMAP (free, GitHub/CLI)",
        "2. Option D — Free depth AI: Depth-Anything v2 (free, HuggingFace) for single-image depth maps",
        "3. Clean mesh in Blender (free): remove floating geometry, fill holes, retopo if needed",
        "4. UV unwrap and bake high-poly → low-poly textures in Blender",
        "5. Export textures as DDS using GIMP (free) + DDS plugin OR Texconv (free, Microsoft CLI)",
        "6. Import mesh into Blender with NifTools add-on (free, GitHub), assign BSLightingShaderProperty",
        "7. Set BSXFlags for collision/animation requirements",
        "8. Add collision mesh (bhkConvexVerticesShape for simple objects)",
        "9. Export as NIF via NifTools Blender add-on (File → Export → NetImmerse/Gamebryo)",
        "10. Test in Creation Kit (free, Bethesda Launcher) — place in world, verify scale and collision",
    ],
    "recommended_tools": {
        "ai_image_to_3d": [
            "Meshy (meshy.ai — user has paid subscription; best quality, fast)",
            "TripoSR (free, HuggingFace: stabilityai/TripoSR — single image to 3D)",
            "Zero123++ (free, HuggingFace: sudo-ai/zero123plus — multi-view from single image)",
        ],
        "photogrammetry_free": [
            "Meshroom (free, GitHub: alicevision/Meshroom — GPU-accelerated, best quality)",
            "COLMAP (free, GitHub: colmap/colmap — CLI, highly configurable)",
            "OpenMVG + OpenMVS (free, GitHub — lightweight pipeline)",
        ],
        "depth_estimation_free": [
            "Depth-Anything v2 (free, HuggingFace: depth-anything/Depth-Anything-V2)",
            "MiDaS (free, HuggingFace: Intel/dpt-large)",
            "ZoeDepth (free, HuggingFace: isl-org/ZoeDepth)",
        ],
        "mesh_editing_free": [
            "Blender 3.x/4.x (free, blender.org) with NifTools add-on",
            "MeshLab (free, GitHub: cnr-isti-vclab/meshlab — mesh cleaning/decimation)",
            "Instant Meshes (free, GitHub: wjakob/instant-meshes — auto retopo)",
        ],
        "nif_tools_free": [
            "NifSkope 2.0 (free, GitHub: niftools/nifskope — NIF viewer and editor)",
            "NifTools Blender add-on (free, GitHub: niftools/blender_niftools_addon)",
            "pyffi (free, GitHub: niftools/pyffi — Python NIF I/O library)",
        ],
        "texture_tools_free": [
            "GIMP (free) + DDS plugin (GitHub: FrancescoR/gimp-dds)",
            "Texconv (free, GitHub: Microsoft/DirectXTex — CLI batch DDS conversion)",
            "Paint.NET (free, getpaint.net) + DDS plugin",
            "Krita (free, krita.org) + DDS export",
        ],
        "testing_free": [
            "Creation Kit (free, Bethesda Launcher)",
            "xEdit / FO4Edit (free, GitHub: TES5Edit/TES5Edit)",
            "LODGen / xLODGen (free, GitHub)",
        ],
    },
    "common_errors": {
        "black_mesh": "Missing or wrong diffuse texture path; check NIF texture slot",
        "no_collision": "Object falls through floors — add bhkCollisionObject",
        "wrong_scale": "Object too large/small — verify export scale, check BSXFlags",
        "invisible_mesh": "Alpha flag wrong on BSLightingShaderProperty; check SLSF flags",
        "purple_mesh": "Missing normal map — ensure _n.dds is BC5 compressed",
        "t_pose": "Skeleton not properly bound; skinning weights need re-bake",
        "nif_version_mismatch": "Must be version 20.2.0.7 / uv2=12 / uv2_2=130",
    },
}

_KNOWLEDGE_KEYWORDS: List[Tuple[List[str], str]] = [
    (["polygon", "poly", "polycount", "triangle", "tris", "budget"],
     "polygon_budgets"),
    (["texture", "diffuse", "normal map", "specular", "glow", "dds", "bc1", "bc3", "bc5"],
     "texture_channels"),
    (["texture format", "dxt", "compression", "bc1", "bc3", "bc5", "ati2"],
     "texture_formats"),
    (["resolution", "texture size", "1024", "2048", "4096", "power of two", "mipmap"],
     "texture_resolutions"),
    (["nif block", "bstriShape", "bslighting", "ninode", "bsfadenode", "block type"],
     "nif_block_types"),
    (["nif version", "version number", "user_version", "20.2.0.7"],
     "nif_version"),
    (["lod", "level of detail", "distance", "xlodgen", "lodgen"],
     "lod_tiers"),
    (["collision", "havok", "bhk", "physics", "rigid body", "convex"],
     "collision_types"),
    (["scale", "unit", "size", "metre", "meter", "bethesda unit"],
     "scale"),
    (["axis", "coordinate", "handedness", "z-up", "y-forward", "blender"],
     "coordinate_system"),
    (["workflow", "steps", "how to", "process", "pipeline", "convert", "export"],
     "workflow_steps"),
    (["tool", "software", "meshroom", "colmap", "blender", "nifskope", "creation kit",
      "texconv", "triposg", "triposr", "trellis", "instantmesh", "shap-e", "free",
      "open source", "github", "huggingface"],
     "recommended_tools"),
    (["error", "bug", "problem", "black", "purple", "invisible", "wrong scale", "t-pose",
      "no collision", "falling"],
     "common_errors"),
]


class Fallout4MeshKnowledge:
    """Mossy's built-in knowledge base for Fallout 4 mesh and modding conventions.

    Covers polygon budgets, texture channels/formats, NIF block types, LOD
    tiers, collision shapes, scale, recommended tools and common error fixes.

    Example
    -------
    ::

        kb = Fallout4MeshKnowledge()
        print(kb.answer("How many polygons can a weapon have?"))
        print(kb.answer("What NIF block type should I use for a static prop?"))
    """

    def __init__(self) -> None:
        self._data = _FO4_KNOWLEDGE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, question: str) -> str:
        """Return a human-readable answer drawn from the knowledge base.

        The query is matched against keyword groups; if multiple sections
        match, all relevant sections are combined in the response.
        """
        question_lower = question.lower()
        matched_keys: List[str] = []

        for keywords, section_key in _KNOWLEDGE_KEYWORDS:
            if any(kw in question_lower for kw in keywords):
                if section_key not in matched_keys:
                    matched_keys.append(section_key)

        if not matched_keys:
            # Generic overview
            return self._overview()

        sections: List[str] = []
        for key in matched_keys:
            sections.append(self._format_section(key))
        return "\n\n".join(sections)

    def validate_mesh_metadata(self, poly_count: int, asset_type: str) -> List[str]:
        """Return a list of validation warnings for the given mesh.

        Parameters
        ----------
        poly_count:
            Triangle count of the mesh.
        asset_type:
            Category key such as ``"weapon"``, ``"armor_piece"``, etc.
            Case-insensitive; spaces are converted to underscores.
        """
        warnings: List[str] = []
        key = asset_type.lower().replace(" ", "_")
        budget = self._data["polygon_budgets"].get(key)

        if budget is None:
            warnings.append(
                f"Unknown asset type {asset_type!r}. "
                f"Known types: {', '.join(self._data['polygon_budgets'].keys())}"
            )
        elif poly_count > budget:
            warnings.append(
                f"Polygon count {poly_count:,} exceeds the FO4 budget of "
                f"{budget:,} for asset type {asset_type!r}. "
                "Consider retopology or poly reduction."
            )
        return warnings

    def get_workflow(self) -> List[str]:
        """Return the recommended image-to-FO4-mesh workflow steps."""
        return list(self._data["workflow_steps"])

    def get_nif_export_settings(self) -> Dict[str, Any]:
        """Return the NIF version settings required for Fallout 4."""
        return dict(self._data["nif_version"])

    def get_texture_requirements(self) -> Dict[str, str]:
        """Return texture channel → suffix mapping."""
        return dict(self._data["texture_channels"])

    def get_polygon_budget(self, asset_type: str) -> Optional[int]:
        """Return polygon budget for *asset_type*, or ``None`` if unknown."""
        return self._data["polygon_budgets"].get(asset_type.lower().replace(" ", "_"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _overview(self) -> str:
        budgets = ", ".join(
            f"{k} ({v:,} tris)"
            for k, v in list(self._data["polygon_budgets"].items())[:4]
        )
        return (
            "Fallout 4 Mesh Overview\n"
            "=======================\n"
            f"Polygon budgets (examples): {budgets} ...\n"
            f"NIF version: {self._data['nif_version']['version']} "
            f"(uv={self._data['nif_version']['user_version']}, "
            f"uv2={self._data['nif_version']['user_version_2']})\n"
            f"Scale: {self._data['scale']['game_unit']}\n"
            f"Texture channels: "
            + ", ".join(
                f"{ch}={suf}"
                for ch, suf in self._data["texture_channels"].items()
            )
            + "\n\nAsk me about: polygons, textures, NIF blocks, LOD, collision, "
            "scale, workflow, tools, or common errors."
        )

    def _format_section(self, key: str) -> str:
        data = self._data.get(key, {})
        title = key.replace("_", " ").title()
        if isinstance(data, dict):
            lines = [f"[{title}]"]
            for k, v in data.items():
                if isinstance(v, list):
                    lines.append(f"  {k}:")
                    for item in v:
                        lines.append(f"    - {item}")
                else:
                    lines.append(f"  {k}: {v}")
            return "\n".join(lines)
        if isinstance(data, list):
            lines = [f"[{title}]"] + [f"  {item}" for item in data]
            return "\n".join(lines)
        return f"[{title}]\n  {data}"


# ---------------------------------------------------------------------------
# Image loader / feature extractor
# ---------------------------------------------------------------------------


@dataclass
class ImageFeatures:
    """Keypoints and descriptors extracted from a single image."""
    path: str
    width: int
    height: int
    keypoint_count: int
    descriptors_shape: Tuple[int, int]  # (n_keypoints, descriptor_dim)


@dataclass
class ScanResult:
    """Outcome of a full image-to-mesh scan."""
    input_images: List[str]
    output_dir: str
    mesh_path: Optional[str]
    point_cloud_path: Optional[str]
    exported_files: List[str] = field(default_factory=list)
    poly_count: int = 0
    point_count: int = 0
    elapsed_s: float = 0.0
    success: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        lines = [
            f"{status} Mesh scan completed in {self.elapsed_s:.1f}s",
            f"  Input images  : {len(self.input_images)}",
            f"  Point cloud   : {self.point_count:,} points",
            f"  Mesh polygons : {self.poly_count:,}",
            f"  Output files  : {', '.join(Path(f).name for f in self.exported_files) or 'none'}",
        ]
        if self.warnings:
            lines += [f"  ⚠ {w}" for w in self.warnings]
        if self.errors:
            lines += [f"  ✗ {e}" for e in self.errors]
        return "\n".join(lines)


class ImageMeshScanner:
    """Converts a set of photographs into a 3-D mesh.

    The pipeline uses OpenCV for feature detection, Open3D for point cloud
    and mesh reconstruction, and trimesh for final mesh I/O.

    When heavy dependencies are not installed the scanner still works in
    *analysis mode* — it loads images, reports their dimensions and feature
    counts, and generates a stub OBJ that documents the scan metadata.

    Parameters
    ----------
    feature_detector:
        OpenCV feature detector type.  ``"sift"`` (default) or ``"orb"``.
    max_features:
        Maximum keypoints per image.
    depth_scale:
        Scaling factor applied to depth images (Open3D RGBD default 1000).
    voxel_size:
        Voxel size for point-cloud downsampling (metres).
    mesh_depth:
        Poisson reconstruction depth.  Higher = more detail but slower.
    """

    def __init__(
        self,
        feature_detector: str = "sift",
        max_features: int = 8000,
        depth_scale: float = 1000.0,
        voxel_size: float = 0.005,
        mesh_depth: int = 9,
    ) -> None:
        self.feature_detector = feature_detector.lower()
        self.max_features = max_features
        self.depth_scale = depth_scale
        self.voxel_size = voxel_size
        self.mesh_depth = mesh_depth

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_dependencies(self) -> Dict[str, bool]:
        """Return a dict of ``{package: available}`` for all required libs."""
        results = {}
        for pkg, import_name in [
            ("opencv-python", "cv2"),
            ("open3d", "open3d"),
            ("trimesh", "trimesh"),
            ("numpy", "numpy"),
            ("Pillow", "PIL"),
            ("scipy", "scipy"),
        ]:
            try:
                __import__(import_name)
                results[pkg] = True
            except ImportError:
                results[pkg] = False
        return results

    def extract_features(self, image_path: str) -> ImageFeatures:
        """Load *image_path* and extract keypoints / descriptors.

        Requires ``opencv-python``.  Falls back to metadata-only when
        the library is absent.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for feature extraction. "
                "Install it with: pip install opencv-python"
            ) from exc

        try:
            from PIL import Image as PILImage  # noqa: PLC0415
        except ImportError:
            PILImage = None  # type: ignore[assignment]

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if self.feature_detector == "sift":
            detector = cv2.SIFT_create(nfeatures=self.max_features)
        else:
            detector = cv2.ORB_create(nfeatures=self.max_features)

        keypoints, descriptors = detector.detectAndCompute(gray, None)
        desc_shape = descriptors.shape if descriptors is not None else (0, 0)

        return ImageFeatures(
            path=str(image_path),
            width=w,
            height=h,
            keypoint_count=len(keypoints),
            descriptors_shape=(int(desc_shape[0]), int(desc_shape[1])),
        )

    def images_to_point_cloud(
        self,
        image_paths: List[str],
        output_dir: str,
    ) -> Tuple[Optional[Any], int]:
        """Build a point cloud from *image_paths*.

        Uses Open3D's RGBD integration when depth maps are available,
        otherwise performs a colour-histogram-based placeholder cloud so
        the pipeline can complete even with basic input.

        Returns
        -------
        pcd:
            An ``open3d.geometry.PointCloud`` or ``None`` on failure.
        point_count:
            Number of points in the cloud.
        """
        try:
            import open3d as o3d  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "open3d, numpy and opencv-python are required for point cloud "
                "reconstruction.  Install with:\n"
                "  pip install open3d numpy opencv-python"
            ) from exc

        points_all: List[Any] = []
        colors_all: List[Any] = []

        for img_path in image_paths:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                logger.warning("Skipping unreadable image: %s", img_path)
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # ---- Attempt depth estimation via Laplacian sharpness map ----
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # Normalise to 0-1 as a rough depth proxy (sharp edges = closer)
            depth = np.abs(laplacian)
            dmax = depth.max()
            if dmax > 0:
                depth = depth / dmax
            depth = (depth * 2.0 + 0.5)  # push into a 0.5-2.5 m range

            # Sub-sample to avoid millions of points per image
            step = max(1, min(h, w) // 80)
            ys, xs = np.mgrid[0:h:step, 0:w:step]
            zs = depth[ys, xs]

            # Focal length heuristic
            focal = max(h, w) * 0.75
            cx, cy = w / 2.0, h / 2.0
            x3d = (xs - cx) * zs / focal
            y3d = (ys - cy) * zs / focal
            z3d = zs

            pts = np.stack([x3d.ravel(), y3d.ravel(), z3d.ravel()], axis=1)
            cols = img_rgb[ys, xs].reshape(-1, 3).astype(np.float64) / 255.0
            points_all.append(pts)
            colors_all.append(cols)

        if not points_all:
            return None, 0

        all_pts = np.vstack(points_all)
        all_cols = np.vstack(colors_all)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_pts)
        pcd.colors = o3d.utility.Vector3dVector(all_cols)

        # Downsample
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(self.voxel_size)

        # Estimate normals for Poisson
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 10, max_nn=30
            )
        )
        pcd.orient_normals_consistent_tangent_plane(100)

        ply_path = Path(output_dir) / "point_cloud.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        logger.info("Point cloud saved: %s (%d points)", ply_path, len(pcd.points))

        return pcd, len(pcd.points)

    def point_cloud_to_mesh(
        self,
        pcd: Any,
        output_dir: str,
    ) -> Tuple[Optional[Any], int]:
        """Run Poisson surface reconstruction on *pcd*.

        Returns
        -------
        mesh:
            An ``open3d.geometry.TriangleMesh`` or ``None`` on failure.
        poly_count:
            Triangle count after cleaning.
        """
        try:
            import open3d as o3d  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "open3d and numpy are required for mesh reconstruction."
            ) from exc

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.mesh_depth
        )

        # Remove low-density vertices (artefacts at the boundary)
        density_threshold = np.quantile(np.asarray(densities), 0.05)
        vertices_to_remove = np.asarray(densities) < density_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        ply_path = Path(output_dir) / "mesh.ply"
        o3d.io.write_triangle_mesh(str(ply_path), mesh)
        poly_count = len(mesh.triangles)
        logger.info("Mesh saved: %s (%d triangles)", ply_path, poly_count)
        return mesh, poly_count


# ---------------------------------------------------------------------------
# Fallout 4 mesh exporter
# ---------------------------------------------------------------------------


class FalloutMeshExporter:
    """Exports a 3-D mesh in Fallout 4 compatible formats.

    Supported export targets
    ~~~~~~~~~~~~~~~~~~~~~~~~
    * **OBJ** — universally supported; import into Blender then export NIF
      using the NifTools Blender add-on.
    * **PLY** — useful for further processing in MeshLab / Blender.
    * **NIF** — direct NIF export via pyffi (optional dependency).  Produces
      a minimal static-object NIF with ``BSFadeNode`` root,
      ``BSTriShape`` geometry, ``BSLightingShaderProperty``, and a stub
      ``bhkCollisionObject``.

    Parameters
    ----------
    game_scale:
        Multiplier applied to mesh coordinates on export.  Default converts
        from 1 m = 1 unit (photogrammetry) to Fallout 4's unit system where
        70 units ≈ 1 metre.
    """

    GAME_UNITS_PER_METRE = 70.0

    def __init__(self, game_scale: float = GAME_UNITS_PER_METRE) -> None:
        self.game_scale = game_scale

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_obj(
        self,
        mesh: Any,
        output_path: str,
        mesh_name: str = "fo4_mesh",
    ) -> str:
        """Export *mesh* as a Wavefront OBJ file.

        Works with any object that has ``.vertices`` and ``.triangles``
        attributes (Open3D ``TriangleMesh``), or falls through to trimesh
        when the input is a trimesh object.

        Returns
        -------
        str
            Path of the written file.
        """
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("numpy is required for OBJ export.") from exc

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Try Open3D mesh first, then trimesh, then duck-typing
        vertices = None
        faces = None
        vertex_colors = None

        if hasattr(mesh, "vertices") and hasattr(mesh, "triangles"):
            import numpy as np  # noqa: PLC0415, F811
            vertices = np.asarray(mesh.vertices) * self.game_scale
            faces = np.asarray(mesh.triangles)
            if hasattr(mesh, "vertex_colors") and len(mesh.vertex_colors) > 0:
                vertex_colors = np.asarray(mesh.vertex_colors)
        elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
            import numpy as np  # noqa: PLC0415, F811
            vertices = np.asarray(mesh.vertices) * self.game_scale
            faces = np.asarray(mesh.faces)

        if vertices is None:
            raise TypeError(f"Cannot export mesh of type {type(mesh)!r} to OBJ.")

        with open(out, "w", encoding="utf-8") as fh:
            fh.write(f"# Fallout 4 mesh exported by AI Helper / Mossy\n")
            fh.write(f"# Mesh name: {mesh_name}\n")
            fh.write(f"# Vertices: {len(vertices)}, Triangles: {len(faces)}\n")
            fh.write(f"# Scale applied: {self.game_scale} (FO4 units)\n\n")
            fh.write(f"o {mesh_name}\n\n")
            for v in vertices:
                fh.write(f"v {v[0]:.6f} {v[2]:.6f} {-v[1]:.6f}\n")  # Blender Z-up/Y-forward → OBJ Y-up convention (X, Z, -Y)
            fh.write("\n")
            for tri in faces:
                fh.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        logger.info("OBJ exported: %s", out)
        return str(out)

    def export_trimesh(
        self,
        mesh: Any,
        output_path: str,
        file_type: Optional[str] = None,
    ) -> str:
        """Export using trimesh — supports OBJ, GLB, STL, PLY, etc.

        Requires ``trimesh``.
        """
        try:
            import trimesh  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "trimesh and numpy are required for trimesh export.  "
                "Install: pip install trimesh numpy"
            ) from exc

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if hasattr(mesh, "vertices") and hasattr(mesh, "triangles"):
            import numpy as np  # noqa: PLC0415, F811
            verts = np.asarray(mesh.vertices) * self.game_scale
            faces = np.asarray(mesh.triangles)
            tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        elif isinstance(mesh, trimesh.Trimesh):
            tm = mesh
        else:
            raise TypeError(f"Cannot convert mesh of type {type(mesh)!r} to trimesh.")

        ext = file_type or out.suffix.lstrip(".")
        tm.export(str(out), file_type=ext)
        logger.info("Trimesh export (%s): %s", ext, out)
        return str(out)

    def export_nif_stub(
        self,
        obj_path: str,
        output_path: str,
        mesh_name: str = "FO4Mesh",
    ) -> str:
        """Export a minimal Fallout 4 NIF using pyffi.

        This writes the skeleton NIF structure — BSFadeNode root,
        BSTriShape, BSLightingShaderProperty, BSXFlags.  The caller
        still needs to open the result in NifSkope to assign texture
        paths and fine-tune shader flags.

        Requires ``pyffi``.  Install: ``pip install pyffi``

        Parameters
        ----------
        obj_path:
            Path to a Wavefront OBJ file to embed.  The OBJ is parsed
            manually so trimesh is not required here.
        output_path:
            Destination ``.nif`` path.
        mesh_name:
            Name embedded in the NIF block names.
        """
        try:
            from pyffi.formats.nif import NifFormat  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pyffi is required for NIF export.  "
                "Install: pip install pyffi"
            ) from exc

        vertices, faces = self._load_obj(obj_path)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # Build NIF in-memory
        nif = NifFormat.Data()
        nif.version = 0x14020007       # 20.2.0.7
        nif.user_version = 12
        nif.user_version_2 = 130

        # Root node
        root = NifFormat.BSFadeNode()
        root.name = mesh_name.encode()
        root.flags = 14  # standard static flags

        # BSXFlags
        bsx = NifFormat.BSXFlags()
        bsx.name = b"BSX"
        bsx.integer_data = 2  # BSX_EDITOR_MARKER | BSX_COLLISION
        root.add_extra_data(bsx)

        # BSTriShape
        shape = NifFormat.BSTriShape()
        shape.name = f"{mesh_name}:0".encode()
        shape.flags = 14

        # Vertices
        shape.num_vertices = len(vertices)
        shape.vertices.update_size()
        for i, (x, y, z) in enumerate(vertices):
            v = shape.vertices[i]
            v.x, v.y, v.z = float(x), float(y), float(z)

        # Triangles
        shape.num_triangles = len(faces)
        shape.triangles.update_size()
        for i, (a, b, c) in enumerate(faces):
            t = shape.triangles[i]
            t.v_1, t.v_2, t.v_3 = int(a), int(b), int(c)

        # BSLightingShaderProperty
        shader = NifFormat.BSLightingShaderProperty()
        shader.name = b""
        shader.shader_type = NifFormat.BSLightingShaderPropertyShaderType.default
        shape.shader_property = shader

        root.add_child(shape)
        nif.roots = [root]

        with open(out, "wb") as fh:
            nif.write(fh)

        logger.info("NIF exported: %s", out)
        return str(out)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_obj(obj_path: str) -> Tuple[List[Tuple[float, float, float]],
                                          List[Tuple[int, int, int]]]:
        """Parse a minimal Wavefront OBJ file (v/f lines only)."""
        vertices: List[Tuple[float, float, float]] = []
        faces: List[Tuple[int, int, int]] = []
        with open(obj_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("v "):
                    parts = line.split()
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("f "):
                    parts = line.split()
                    def _vi(tok: str) -> int:
                        return int(tok.split("/")[0]) - 1
                    idxs = [_vi(p) for p in parts[1:]]
                    # Fan-triangulate
                    for i in range(1, len(idxs) - 1):
                        faces.append((idxs[0], idxs[i], idxs[i + 1]))
        return vertices, faces


# ---------------------------------------------------------------------------
# Mesh validator
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Results from mesh validation against FO4 requirements."""
    path: str
    poly_count: int
    vertex_count: int
    has_normals: bool
    has_uvs: bool
    is_watertight: bool
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0

    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else f"✗ FAIL ({len(self.issues)} issues)"
        lines = [
            f"Mesh validation: {status}",
            f"  File     : {self.path}",
            f"  Polygons : {self.poly_count:,}",
            f"  Vertices : {self.vertex_count:,}",
            f"  Normals  : {'yes' if self.has_normals else 'NO — required for NIF'}",
            f"  UVs      : {'yes' if self.has_uvs else 'NO — required for texturing'}",
            f"  Watertight: {'yes' if self.is_watertight else 'no (acceptable for statics)'}",
        ]
        if self.issues:
            lines += [f"  ✗ {i}" for i in self.issues]
        if self.suggestions:
            lines += [f"  💡 {s}" for s in self.suggestions]
        return "\n".join(lines)


class MeshValidator:
    """Validates a mesh file against Fallout 4 modding requirements.

    Uses trimesh for loading and analysis.  Without trimesh, performs a
    basic polygon-count check only from OBJ/PLY file parsing.
    """

    def __init__(self, knowledge: Optional[Fallout4MeshKnowledge] = None) -> None:
        self.knowledge = knowledge or Fallout4MeshKnowledge()

    def validate(self, mesh_path: str, asset_type: str = "settlement_object_medium") -> ValidationResult:
        """Load and validate *mesh_path*.

        Parameters
        ----------
        mesh_path:
            Path to an OBJ, PLY, STL, or NIF file.
        asset_type:
            FO4 asset category for polygon budget check.
        """
        path = Path(mesh_path)
        if not path.exists():
            return ValidationResult(
                path=str(path),
                poly_count=0,
                vertex_count=0,
                has_normals=False,
                has_uvs=False,
                is_watertight=False,
                issues=[f"File not found: {path}"],
            )

        # Try trimesh first
        try:
            import trimesh  # noqa: PLC0415
            mesh = trimesh.load(str(path), force="mesh")
            poly_count = len(mesh.faces)
            vertex_count = len(mesh.vertices)
            has_normals = (
                hasattr(mesh, "vertex_normals") and mesh.vertex_normals is not None
                and len(mesh.vertex_normals) > 0
            )
            has_uvs = (
                hasattr(mesh, "visual") and hasattr(mesh.visual, "uv")
                and mesh.visual.uv is not None
            )
            is_watertight = mesh.is_watertight
        except ImportError:
            # Fallback: count vertices/faces from OBJ
            poly_count, vertex_count, has_normals, has_uvs = self._parse_obj_stats(path)
            is_watertight = False
        except Exception as exc:  # noqa: BLE001
            return ValidationResult(
                path=str(path),
                poly_count=0,
                vertex_count=0,
                has_normals=False,
                has_uvs=False,
                is_watertight=False,
                issues=[f"Failed to load mesh: {exc}"],
            )

        issues: List[str] = []
        suggestions: List[str] = []

        # Polygon budget
        budget_issues = self.knowledge.validate_mesh_metadata(poly_count, asset_type)
        issues.extend(budget_issues)
        if budget_issues:
            suggestions.append(
                "Use Blender's Decimate modifier or Instant Meshes to reduce polygon count."
            )

        # UV check
        if not has_uvs:
            issues.append("No UV coordinates — mesh cannot be textured in Fallout 4.")
            suggestions.append("UV-unwrap the mesh in Blender (Smart UV Project for a quick start).")

        # Normals
        if not has_normals:
            suggestions.append(
                "No vertex normals found.  Recalculate normals in Blender (Ctrl+N in Edit Mode) "
                "or they will be auto-computed on NIF export."
            )

        return ValidationResult(
            path=str(path),
            poly_count=poly_count,
            vertex_count=vertex_count,
            has_normals=has_normals,
            has_uvs=has_uvs,
            is_watertight=is_watertight,
            issues=issues,
            suggestions=suggestions,
        )

    @staticmethod
    def _parse_obj_stats(path: Path) -> Tuple[int, int, bool, bool]:
        """Minimal OBJ parser that counts v/f/vn/vt lines."""
        v = vn = vt = f = 0
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("v "):
                        v += 1
                    elif line.startswith("vn "):
                        vn += 1
                    elif line.startswith("vt "):
                        vt += 1
                    elif line.startswith("f "):
                        # Count triangles in fan (each face with N verts = N-2 tris)
                        parts = line.split()
                        f += max(1, len(parts) - 3)
        except OSError:
            pass
        return f, v, vn > 0, vt > 0


# ---------------------------------------------------------------------------
# MeshEngine façade
# ---------------------------------------------------------------------------


class MeshEngine:
    """Unified façade for Mossy's 3-D mesh capabilities.

    Combines image scanning, free AI image-to-3D models (TripoSG, TRELLIS,
    TripoSR, Shap-E), paid Meshy API (user subscription), HuggingFace depth
    estimation, mesh conversion, Fallout 4 domain knowledge and validation.

    Free-first philosophy
    ~~~~~~~~~~~~~~~~~~~~~
    All backends except Meshy are free and open-source (GitHub / HuggingFace).
    TripoSG is the **recommended default** for image-to-3D: MIT license,
    8 GB VRAM, best free quality.

    Parameters
    ----------
    feature_detector:
        OpenCV feature detector: ``"sift"`` (default) or ``"orb"``.
    mesh_depth:
        Poisson reconstruction depth (higher = more detail, slower).
    game_scale:
        Scale factor applied on export (default: 70 units/metre for FO4).
    output_root:
        Default directory for scan outputs.
    free_models_root:
        Directory containing cloned free model repos (TripoSG, TRELLIS, etc.).
        Defaults to ``~/AI-Helper/FreeModels``.
    meshy_api_key:
        Meshy API key for the paid image-to-3D service.  Falls back to the
        ``MESHY_API_KEY`` environment variable when not supplied.
    """

    def __init__(
        self,
        feature_detector: str = "sift",
        mesh_depth: int = 9,
        game_scale: float = FalloutMeshExporter.GAME_UNITS_PER_METRE,
        output_root: Optional[str] = None,
        free_models_root: Optional[str] = None,
        meshy_api_key: Optional[str] = None,
    ) -> None:
        self.knowledge = Fallout4MeshKnowledge()
        self.scanner = ImageMeshScanner(
            feature_detector=feature_detector,
            mesh_depth=mesh_depth,
        )
        self.exporter = FalloutMeshExporter(game_scale=game_scale)
        self.validator = MeshValidator(self.knowledge)
        self.free_client = FreeImage3DClient(repo_root=free_models_root)
        self.meshy = MeshyClient(api_key=meshy_api_key)
        self.depth_estimator = HuggingFaceDepthEstimator()
        self.output_root = Path(output_root) if output_root else Path.cwd() / "mesh_output"

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    def ask_knowledge(self, question: str) -> str:
        """Ask Mossy a Fallout 4 mesh / modding question."""
        return self.knowledge.answer(question)

    def get_workflow(self) -> str:
        """Return the recommended image-to-FO4-mesh workflow."""
        steps = self.knowledge.get_workflow()
        return "Recommended Image → Fallout 4 Mesh Workflow:\n" + "\n".join(steps)

    def list_free_tools(self) -> str:
        """Return a table of all free image-to-3D backends Mossy knows about."""
        return FreeImage3DClient.list_backends()

    def install_instructions(self, backend: str) -> str:
        """Return step-by-step install instructions for a free backend.

        Valid backend names: ``triposg``, ``trellis``, ``triposr``,
        ``instantmesh``, ``shap_e``.
        """
        return FreeImage3DClient.install_instructions(backend)

    # ------------------------------------------------------------------
    # Free AI image-to-3D (TripoSG recommended)
    # ------------------------------------------------------------------

    def free_image_to_3d(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        backend: str = "triposg",
        **kwargs: Any,
    ) -> FreeImage3DResult:
        """Convert a single image to a 3D mesh using a free open-source model.

        Parameters
        ----------
        image_path:
            Input photograph (PNG or JPEG).
        output_dir:
            Where to save the output mesh.  Defaults to
            ``output_root/free_<backend>_<timestamp>``.
        backend:
            Which free model to use.  Options:

            * ``"triposg"`` *(default)* — TripoSG, 8 GB VRAM, MIT,
              best free quality (github.com/VAST-AI-Research/TripoSG)
            * ``"trellis"`` — Microsoft TRELLIS, 16 GB VRAM, MIT
              (github.com/microsoft/TRELLIS)
            * ``"triposr"`` — TripoSR original, 6 GB VRAM, MIT
              (github.com/VAST-AI-Research/TripoSR)
            * ``"shap_e"`` — OpenAI Shap-E, CPU-capable, MIT
              (github.com/openai/shap-e)
        **kwargs:
            Additional arguments forwarded to the backend's ``run_*`` method.
            See :class:`FreeImage3DClient` for per-backend options.
        """
        out_dir = output_dir or str(
            self.output_root / f"free_{backend}_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        dispatch = {
            "triposg": self.free_client.run_triposg,
            "trellis": self.free_client.run_trellis,
            "triposr": self.free_client.run_triposr,
            "shap_e": self.free_client.run_shap_e,
        }
        runner = dispatch.get(backend)
        if runner is None:
            valid = ", ".join(dispatch)
            return FreeImage3DResult(
                backend=backend,
                input_image=str(image_path),
                output_dir=str(out_dir),
                error=f"Unknown backend {backend!r}. Valid: {valid}",
            )
        return runner(image_path, out_dir, **kwargs)

    # ------------------------------------------------------------------
    # Meshy (paid subscription)
    # ------------------------------------------------------------------

    def meshy_image_to_3d(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        download_fmt: str = "obj",
        target_polycount: int = 10000,
        enable_pbr: bool = True,
    ) -> ScanResult:
        """Convert an image to a 3D mesh using the Meshy API (paid subscription).

        Parameters
        ----------
        image_path:
            Input photograph (PNG or JPEG).
        output_dir:
            Where to save the downloaded mesh.
        api_key:
            Meshy API key.  Falls back to ``MESHY_API_KEY`` env var.
        download_fmt:
            Mesh format to download: ``"obj"``, ``"glb"``, ``"fbx"``, ``"usdz"``.
        target_polycount:
            Polygon budget hint passed to Meshy (default 10 000).
        enable_pbr:
            When ``True``, Meshy generates PBR texture maps.
        """
        t0 = time.monotonic()
        out_dir = Path(output_dir) if output_dir else (
            self.output_root / f"meshy_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        result = ScanResult(
            input_images=[str(image_path)],
            output_dir=str(out_dir),
            mesh_path=None,
            point_cloud_path=None,
        )

        if api_key:
            self.meshy.api_key = api_key

        try:
            task = self.meshy.image_to_3d(
                image_path=str(image_path),
                enable_pbr=enable_pbr,
                target_polycount=target_polycount,
            )
        except (ValueError, RuntimeError) as exc:
            result.errors.append(str(exc))
            result.elapsed_s = time.monotonic() - t0
            return result

        if not task.succeeded:
            result.errors.append(
                f"Meshy task {task.task_id} failed: {task.error or task.status}"
            )
            result.elapsed_s = time.monotonic() - t0
            return result

        # Download mesh
        mesh_path = self.meshy.download_result(task, str(out_dir), fmt=download_fmt)
        if mesh_path:
            result.exported_files.append(mesh_path)
            result.mesh_path = mesh_path

        # Download textures
        textures = self.meshy.download_textures(task, str(out_dir))
        result.exported_files.extend(textures)

        result.success = bool(mesh_path)
        result.elapsed_s = time.monotonic() - t0
        return result

    # ------------------------------------------------------------------
    # HuggingFace depth estimation (free)
    # ------------------------------------------------------------------

    def estimate_depth(
        self,
        image_path: str,
        output_dir: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """Estimate monocular depth for *image_path* using a free HF model.

        Downloads ``depth-anything/Depth-Anything-V2-Small-hf`` on first use
        (free, HuggingFace).  Returns a text summary and saves a depth PNG.
        """
        out_dir = output_dir or str(
            self.output_root / f"depth_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        if model_id:
            self.depth_estimator.model_id = model_id
        try:
            depth, png_path = self.depth_estimator.estimate_and_save(image_path, out_dir)
            return (
                f"Depth estimation complete.\n"
                f"  Model  : {self.depth_estimator.model_id}\n"
                f"  Input  : {image_path}\n"
                f"  Output : {png_path}\n"
                f"  Range  : 0.0 – 1.0 (normalised)"
            )
        except ImportError as exc:
            return f"Depth estimation unavailable: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Depth estimation failed: {exc}"

    # ------------------------------------------------------------------
    # Classic multi-image scan (open3d/opencv pipeline)
    # ------------------------------------------------------------------

    def scan_images(
        self,
        image_paths: List[str],
        output_dir: Optional[str] = None,
        export_formats: Optional[List[str]] = None,
        asset_type: str = "settlement_object_medium",
        mesh_name: str = "fo4_mesh",
    ) -> ScanResult:
        """Multi-image scan pipeline: images → point cloud → mesh → export.

        Uses OpenCV for feature extraction and Open3D for Poisson mesh
        reconstruction.  All dependencies are free and open-source.

        For better quality from a single image, use :meth:`free_image_to_3d`
        (TripoSG) or :meth:`meshy_image_to_3d` (Meshy subscription) instead.

        Parameters
        ----------
        image_paths:
            List of photograph paths (JPEG, PNG, etc.).
        output_dir:
            Where to write output files.
        export_formats:
            Formats to export: ``"obj"``, ``"ply"``, ``"nif"``.
            Defaults to ``["obj", "ply"]``.
        asset_type:
            FO4 asset category for polygon budget validation.
        mesh_name:
            Name embedded in exported files.
        """
        t0 = time.monotonic()
        if export_formats is None:
            export_formats = ["obj", "ply"]

        out_dir = Path(output_dir) if output_dir else (
            self.output_root / time.strftime("%Y%m%d_%H%M%S")
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        result = ScanResult(
            input_images=[str(p) for p in image_paths],
            output_dir=str(out_dir),
            mesh_path=None,
            point_cloud_path=str(out_dir / "point_cloud.ply"),
        )

        # Dependency check
        deps = self.scanner.check_dependencies()
        missing = [pkg for pkg, ok in deps.items() if not ok]
        if missing:
            result.warnings.append(
                f"Optional packages not installed: {', '.join(missing)}. "
                f"Install: pip install {' '.join(missing)}"
            )

        # Validate image files
        valid_images = []
        for p in image_paths:
            if Path(p).exists():
                valid_images.append(p)
            else:
                result.errors.append(f"Image not found: {p}")

        if not valid_images:
            result.errors.append("No valid input images found.")
            result.elapsed_s = time.monotonic() - t0
            self._write_scan_metadata(result, out_dir, asset_type, t0)
            return result

        # Point cloud
        pcd = None
        try:
            pcd, result.point_count = self.scanner.images_to_point_cloud(
                valid_images, str(out_dir)
            )
            if pcd is not None:
                result.point_cloud_path = str(out_dir / "point_cloud.ply")
        except ImportError as exc:
            result.warnings.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"Point cloud failed: {exc}")
            logger.exception("Point cloud reconstruction failed")

        # Mesh generation
        mesh = None
        if pcd is not None:
            try:
                mesh, result.poly_count = self.scanner.point_cloud_to_mesh(
                    pcd, str(out_dir)
                )
                if mesh is not None:
                    result.mesh_path = str(out_dir / "mesh.ply")
            except ImportError as exc:
                result.warnings.append(str(exc))
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"Mesh generation failed: {exc}")
                logger.exception("Mesh generation failed")

        # Export
        if mesh is not None:
            for fmt in export_formats:
                fmt_lower = fmt.lower()
                try:
                    if fmt_lower == "obj":
                        obj_path = str(out_dir / f"{mesh_name}.obj")
                        self.exporter.export_obj(mesh, obj_path, mesh_name=mesh_name)
                        result.exported_files.append(obj_path)
                    elif fmt_lower == "ply":
                        ply_path = str(out_dir / f"{mesh_name}.ply")
                        self.exporter.export_trimesh(mesh, ply_path, file_type="ply")
                        result.exported_files.append(ply_path)
                    elif fmt_lower == "nif":
                        obj_path = str(out_dir / f"{mesh_name}.obj")
                        if not Path(obj_path).exists():
                            self.exporter.export_obj(mesh, obj_path, mesh_name=mesh_name)
                        nif_path = str(out_dir / f"{mesh_name}.nif")
                        self.exporter.export_nif_stub(obj_path, nif_path, mesh_name=mesh_name)
                        result.exported_files.append(nif_path)
                    else:
                        result.warnings.append(f"Unknown export format: {fmt!r}")
                except ImportError as exc:
                    result.warnings.append(f"Cannot export {fmt}: {exc}")
                except Exception as exc:  # noqa: BLE001
                    result.errors.append(f"Export {fmt} failed: {exc}")
                    logger.exception("Export failed for format %s", fmt)

            budget_issues = self.knowledge.validate_mesh_metadata(
                result.poly_count, asset_type
            )
            result.warnings.extend(budget_issues)

        self._write_scan_metadata(result, out_dir, asset_type, t0)
        result.success = not result.errors and (mesh is not None or bool(result.warnings))
        result.elapsed_s = time.monotonic() - t0
        return result

    @staticmethod
    def _write_scan_metadata(
        result: ScanResult, out_dir: Path, asset_type: str, t0: float
    ) -> None:
        """Write scan_metadata.json into *out_dir*."""
        try:
            meta_path = out_dir / "scan_metadata.json"
            meta = {
                "input_images": result.input_images,
                "point_count": result.point_count,
                "poly_count": result.poly_count,
                "exported_files": result.exported_files,
                "asset_type": asset_type,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "warnings": result.warnings,
                "errors": result.errors,
            }
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write scan metadata: %s", exc)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_mesh(
        self, mesh_path: str, asset_type: str = "settlement_object_medium"
    ) -> ValidationResult:
        """Validate a mesh file against Fallout 4 requirements."""
        return self.validator.validate(mesh_path, asset_type)

    # ------------------------------------------------------------------
    # Feature extraction (single image)
    # ------------------------------------------------------------------

    def extract_image_features(self, image_path: str) -> str:
        """Return a text report of keypoint features from *image_path*."""
        try:
            features = self.scanner.extract_features(image_path)
            return (
                f"Image features for {Path(image_path).name}:\n"
                f"  Dimensions   : {features.width}×{features.height}\n"
                f"  Keypoints    : {features.keypoint_count:,}\n"
                f"  Descriptor   : {features.descriptors_shape[1]}-dim × "
                f"{features.descriptors_shape[0]} kp"
            )
        except ImportError as exc:
            return f"Feature extraction unavailable: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"Feature extraction failed: {exc}"

    def check_dependencies(self) -> str:
        """Return a formatted dependency status report for all mesh packages."""
        deps = self.scanner.check_dependencies()
        lines = ["Mesh Engine dependency status:"]
        for pkg, ok in deps.items():
            status = "✓ installed" if ok else f"✗ missing  →  pip install {pkg}"
            lines.append(f"  {pkg:<20} {status}")
        lines += [
            "",
            "Free image-to-3D models (clone from GitHub, no pip install needed):",
        ]
        for key, info in FREE_IMAGE_TO_3D_BACKENDS.items():
            repo_dir = self.free_client.repo_root / key
            installed = "✓ found" if repo_dir.exists() else f"✗ not found  →  git clone {info['github']}"
            lines.append(f"  {key:<14} {installed}")
        return "\n".join(lines)

