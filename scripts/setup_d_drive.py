"""One-shot D-drive setup helper.

Run this script **once** after cloning the repository to:

1. Create the full AI Helper directory tree on the D drive
   (or the path given by ``--install-dir``).
2. Install all Python dependencies listed in ``requirements.txt``
   directly into the D-drive ``site-packages`` directory so they
   never touch the C drive / desktop.
3. Write the install path to ``~/.ai_helper.cfg`` so every subsequent
   ``python -m ai_helper`` run uses the D drive automatically.
4. Copy ``pip.ini`` (Windows) or ``pip.conf`` (Unix) to the appropriate
   user-level pip configuration location so that *future* ``pip install``
   commands also default to the D drive.

Usage
-----
::

    python scripts/setup_d_drive.py
    python scripts/setup_d_drive.py --install-dir D:\\MyHelper
    python scripts/setup_d_drive.py --install-dir /mnt/data/AI-Helper
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# Repo root is one level up from this script.
REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"

_SYSTEM = platform.system()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_install_dir() -> Path:
    return Path("D:\\AI-Helper") if _SYSTEM == "Windows" else Path.home() / "AI-Helper"


def _pip_config_dest() -> Path:
    """Return the recommended user-level pip config file path."""
    if _SYSTEM == "Windows":
        appdata = os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))
        return Path(appdata) / "pip" / "pip.ini"
    # Linux / macOS: XDG location
    xdg = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
    return Path(xdg) / "pip" / "pip.conf"


def _pip_config_src() -> Path:
    return REPO_ROOT / ("pip.ini" if _SYSTEM == "Windows" else "pip.conf")


def _create_dirs(install_dir: Path) -> None:
    subdirs = [
        install_dir,
        install_dir / "Downloads",
        install_dir / "Lib" / "site-packages",
        install_dir / "pip-cache",
        install_dir / "Organized",
        install_dir / "Logs",
        install_dir / "Data",
    ]
    for d in subdirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✓  {d}")


def _install_requirements(install_dir: Path) -> None:
    target = install_dir / "Lib" / "site-packages"
    cache = install_dir / "pip-cache"
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--requirement", str(REQUIREMENTS),
        "--target", str(target),
        "--cache-dir", str(cache),
        "--upgrade",
    ]
    print("\nInstalling requirements to", target)
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)  # noqa: S603
    if result.returncode != 0:
        print("  ⚠  pip install returned non-zero exit code — check output above.")
    else:
        print("  ✓  Requirements installed.")


def _write_ai_helper_config(install_dir: Path) -> None:
    # Import and use config module
    sys.path.insert(0, str(REPO_ROOT))
    from ai_helper import config as cfg  # noqa: PLC0415
    cfg.set_install_dir(install_dir)
    saved = cfg.save_config(install_dir)
    print(f"\n  ✓  AI Helper config saved → {saved}")


def _configure_pip(install_dir: Path) -> None:
    src = _pip_config_src()
    dest = _pip_config_dest()
    if not src.exists():
        print(f"\n  ⚠  pip config template not found at {src}; skipping.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Rewrite the target and cache-dir to the chosen install_dir.
    content = src.read_text(encoding="utf-8")
    target_line = f"target = {install_dir / 'Lib' / 'site-packages'}"
    cache_line = f"cache-dir = {install_dir / 'pip-cache'}"
    lines = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("target"):
            lines.append(target_line)
        elif stripped.startswith("cache-dir"):
            lines.append(cache_line)
        else:
            lines.append(line)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n  ✓  pip config written → {dest}")
    print("     Future `pip install` commands will default to the D drive.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up AI Helper on the D drive (or a custom location)."
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=_default_install_dir(),
        metavar="PATH",
        help=f"Root install directory (default: {_default_install_dir()}).",
    )
    parser.add_argument(
        "--skip-pip-config",
        action="store_true",
        help="Don't write the user-level pip.ini / pip.conf.",
    )
    parser.add_argument(
        "--skip-requirements",
        action="store_true",
        help="Don't run pip install for requirements.txt.",
    )
    args = parser.parse_args()
    install_dir: Path = args.install_dir.resolve()

    print(f"AI Helper D-drive setup")
    print(f"=======================")
    print(f"Install directory: {install_dir}\n")

    print("Creating directory structure:")
    _create_dirs(install_dir)

    if not args.skip_requirements:
        _install_requirements(install_dir)

    _write_ai_helper_config(install_dir)

    if not args.skip_pip_config:
        _configure_pip(install_dir)

    print(f"\nSetup complete.  Run AI Helper with:")
    print(f"  python -m ai_helper")
    print(f"  python -m ai_helper --daemon")


if __name__ == "__main__":
    main()
