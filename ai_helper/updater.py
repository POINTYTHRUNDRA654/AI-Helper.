"""Self-updater.

Checks the GitHub Releases API for a newer version of AI Helper, downloads
the release archive to the D drive, and notifies the user.

No extra packages are required — uses only the standard-library
``urllib`` and ``zipfile``/``tarfile``.

Usage
-----
::

    from ai_helper.updater import Updater

    u = Updater()
    info = u.check()
    if info.update_available:
        print(f"New version: {info.latest_version}")
        u.download(info)
"""

from __future__ import annotations

import logging
import platform
import re
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GITHUB_API_URL = "https://api.github.com/repos/POINTYTHRUNDRA654/AI-Helper./releases/latest"
_TIMEOUT = 10.0
_CURRENT_VERSION = "0.1.0"   # Updated by release process


@dataclass
class UpdateInfo:
    """Result of an update check."""
    current_version: str
    latest_version: str
    update_available: bool
    release_url: str = ""
    release_notes: str = ""
    asset_url: str = ""
    asset_name: str = ""
    published_at: str = ""
    error: str = ""

    def __str__(self) -> str:
        if self.error:
            return f"Update check failed: {self.error}"
        if not self.update_available:
            return f"AI Helper is up to date (v{self.current_version})"
        return (
            f"Update available: v{self.current_version} → v{self.latest_version}\n"
            f"  Released : {self.published_at}\n"
            f"  Download : {self.release_url}"
        )


class Updater:
    """Check for and download AI Helper updates.

    Parameters
    ----------
    current_version:
        The currently running version string (default: ``_CURRENT_VERSION``).
    download_dir:
        Where to save downloaded archives (default: ``<INSTALL_DIR>/Updates``).
    api_url:
        GitHub API URL for the latest release.
    """

    def __init__(
        self,
        current_version: str = _CURRENT_VERSION,
        download_dir: Optional[Path] = None,
        api_url: str = _GITHUB_API_URL,
    ) -> None:
        self.current_version = current_version
        self.api_url = api_url
        if download_dir is None:
            from . import config as _cfg  # noqa: PLC0415
            download_dir = _cfg.get_install_dir() / "Updates"
        self.download_dir = Path(download_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> UpdateInfo:
        """Query the GitHub releases API and return an :class:`UpdateInfo`.

        Never raises — errors are captured in ``UpdateInfo.error``.
        """
        import json  # noqa: PLC0415
        try:
            req = urllib.request.Request(  # noqa: S310
                self.api_url,
                headers={"Accept": "application/vnd.github+json",
                         "User-Agent": "AI-Helper-Updater/1.0"},
            )
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:  # noqa: S310
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            return UpdateInfo(
                current_version=self.current_version,
                latest_version="unknown",
                update_available=False,
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            return UpdateInfo(
                current_version=self.current_version,
                latest_version="unknown",
                update_available=False,
                error=str(exc),
            )

        latest = data.get("tag_name", "").lstrip("v")
        release_url = data.get("html_url", "")
        notes = data.get("body", "")[:500]
        published_at = data.get("published_at", "")

        # Pick best asset for this platform
        asset_url, asset_name = self._pick_asset(data.get("assets", []))

        available = self._version_gt(latest, self.current_version)
        return UpdateInfo(
            current_version=self.current_version,
            latest_version=latest,
            update_available=available,
            release_url=release_url,
            release_notes=notes,
            asset_url=asset_url,
            asset_name=asset_name,
            published_at=published_at,
        )

    def download(self, info: UpdateInfo) -> Optional[Path]:
        """Download the release asset to the download directory.

        Returns the path of the downloaded file, or ``None`` on failure.
        """
        if not info.asset_url:
            logger.warning("No download asset available for this platform.")
            return None

        self.download_dir.mkdir(parents=True, exist_ok=True)
        dest = self.download_dir / info.asset_name

        logger.info("Downloading %s → %s", info.asset_url, dest)
        try:
            urllib.request.urlretrieve(info.asset_url, dest)  # noqa: S310
        except Exception as exc:  # noqa: BLE001
            logger.error("Download failed: %s", exc)
            return None

        logger.info("Downloaded %s  (%d bytes)", dest.name, dest.stat().st_size)
        return dest

    def extract(self, archive_path: Path, target_dir: Optional[Path] = None) -> Optional[Path]:
        """Extract a downloaded archive.

        Returns the extraction directory, or ``None`` on failure.
        """
        target_dir = target_dir or self.download_dir / archive_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        try:
            if archive_path.suffix == ".zip" or archive_path.name.endswith(".zip"):
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(target_dir)
            elif any(archive_path.name.endswith(ext)
                     for ext in (".tar.gz", ".tgz", ".tar.bz2")):
                with tarfile.open(archive_path) as tf:
                    tf.extractall(target_dir)  # noqa: S202
            else:
                logger.warning("Unknown archive format: %s", archive_path.suffix)
                return None
        except Exception as exc:  # noqa: BLE001
            logger.error("Extraction failed: %s", exc)
            return None

        logger.info("Extracted to %s", target_dir)
        return target_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _version_gt(a: str, b: str) -> bool:
        """Return True if version string *a* is greater than *b*."""
        def _parts(v: str):
            return tuple(int(x) for x in re.findall(r"\d+", v) or [0])
        return _parts(a) > _parts(b)

    @staticmethod
    def _pick_asset(assets: list) -> tuple[str, str]:
        """Choose the best release asset for the current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        prefer: list[str] = []
        if system == "windows":
            prefer = ["windows", "win64", "win32", ".zip"]
        elif system == "darwin":
            prefer = ["macos", "darwin", "osx", ".tar.gz", ".zip"]
        else:
            prefer = ["linux", "x86_64", "amd64", ".tar.gz", ".zip"]

        # Score each asset
        best_url, best_name, best_score = "", "", -1
        for asset in assets:
            name = asset.get("name", "").lower()
            url = asset.get("browser_download_url", "")
            score = sum(1 for p in prefer if p in name)
            if score > best_score:
                best_url, best_name, best_score = url, asset.get("name", ""), score

        return best_url, best_name
