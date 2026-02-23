"""Central configuration for AI Helper.

All paths used by AI Helper — including where packages are installed,
where downloaded resources are stored, and where organised files go —
flow from this single module so they can be controlled from one place.

D-drive policy (Windows)
------------------------
On Windows the install directory defaults to ``D:\\AI-Helper`` so that
packages, downloads and organised files never fill up the desktop (C drive).
On other platforms it defaults to ``~/AI-Helper``.

Override
--------
Set the environment variable ``AI_HELPER_INSTALL_DIR`` to any absolute
path to override the platform default at runtime::

    AI_HELPER_INSTALL_DIR=E:\\MyHelper python -m ai_helper

Or call :func:`set_install_dir` before constructing any AI Helper objects.

Persisting the setting
----------------------
Run ``python -m ai_helper --install-dir D:\\AI-Helper --save-config`` once
to write the path to ``~/.ai_helper.cfg`` so it survives reboots.
"""

from __future__ import annotations

import configparser
import os
import platform
from pathlib import Path

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SYSTEM = platform.system()
_CONFIG_FILE = Path.home() / ".ai_helper.cfg"
_ENV_VAR = "AI_HELPER_INSTALL_DIR"

# Sentinel so we can distinguish "not yet resolved" from an explicit None.
_UNSET = object()
_resolved: object = _UNSET


def _default_install_dir() -> Path:
    """Return the platform default install directory."""
    if _SYSTEM == "Windows":
        return Path("D:\\AI-Helper")
    return Path.home() / "AI-Helper"


def _load_from_config_file() -> Path | None:
    """Return the path saved in ``~/.ai_helper.cfg``, or *None*."""
    if not _CONFIG_FILE.exists():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(_CONFIG_FILE, encoding="utf-8")
        raw = parser.get("paths", "install_dir", fallback=None)
        return Path(raw) if raw else None
    except (configparser.Error, OSError):
        return None


def _resolve() -> Path:
    global _resolved  # noqa: PLW0603
    if _resolved is _UNSET:
        # Priority: env var > config file > platform default
        env = os.environ.get(_ENV_VAR)
        if env:
            _resolved = Path(env)
        else:
            from_file = _load_from_config_file()
            _resolved = from_file if from_file is not None else _default_install_dir()
    return _resolved  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_install_dir() -> Path:
    """Return the active install directory (resolved once, then cached)."""
    return _resolve()


def set_install_dir(path: Path | str) -> None:
    """Override the install directory for the current process.

    Does **not** persist the setting; call :func:`save_config` to do that.
    """
    global _resolved  # noqa: PLW0603
    _resolved = Path(path)


def save_config(install_dir: Path | str | None = None) -> Path:
    """Write the install directory to ``~/.ai_helper.cfg`` so it persists.

    Parameters
    ----------
    install_dir:
        Path to save.  If *None*, the current :func:`get_install_dir` value
        is used.

    Returns
    -------
    Path
        The config file path that was written.
    """
    target = Path(install_dir) if install_dir is not None else get_install_dir()
    parser = configparser.ConfigParser()
    if _CONFIG_FILE.exists():
        try:
            parser.read(_CONFIG_FILE, encoding="utf-8")
        except configparser.Error:
            pass
    if not parser.has_section("paths"):
        parser.add_section("paths")
    parser.set("paths", "install_dir", str(target))
    _CONFIG_FILE.write_text(
        "# AI Helper configuration – auto-generated\n" + _to_string(parser),
        encoding="utf-8",
    )
    return _CONFIG_FILE


def _to_string(parser: configparser.ConfigParser) -> str:
    import io
    buf = io.StringIO()
    parser.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Derived paths (all relative to the install directory)
# ---------------------------------------------------------------------------


def get_downloads_dir() -> Path:
    """Directory where downloaded resources are stored (inside install dir)."""
    return get_install_dir() / "Downloads"


def get_packages_dir() -> Path:
    """Directory where Python packages are installed (inside install dir)."""
    return get_install_dir() / "Lib" / "site-packages"


def get_organized_dir() -> Path:
    """Root directory where organised desktop files are moved to."""
    return get_install_dir() / "Organized"


def get_logs_dir() -> Path:
    """Directory for AI Helper log files."""
    return get_install_dir() / "Logs"


def get_data_dir() -> Path:
    """Directory for AI Helper runtime data (ML models, cache, etc.)."""
    return get_install_dir() / "Data"


def ensure_dirs() -> None:
    """Create all standard AI Helper directories under the install dir."""
    for getter in (
        get_install_dir,
        get_downloads_dir,
        get_packages_dir,
        get_organized_dir,
        get_logs_dir,
        get_data_dir,
    ):
        getter().mkdir(parents=True, exist_ok=True)


# Convenience read-only properties at module level for quick access.
INSTALL_DIR: Path = property(get_install_dir)  # type: ignore[assignment]
