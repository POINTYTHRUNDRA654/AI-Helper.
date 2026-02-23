"""Desktop / directory file organiser.

Scans any directory (defaults to the user's Desktop), categorises every
file by its extension, and moves files into tidy sub-directories inside
the configured install directory (``D:\\AI-Helper\\Organized`` on Windows
by default) — keeping the desktop *organised* and free of clutter.

All operations support a *dry_run* mode that reports what *would* happen
without touching any files.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config as _cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category definitions
# ---------------------------------------------------------------------------


class FileCategory(str, Enum):
    DOCUMENTS = "Documents"
    IMAGES = "Images"
    VIDEOS = "Videos"
    MUSIC = "Music"
    ARCHIVES = "Archives"
    CODE = "Code"
    DATA = "Data"
    EXECUTABLES = "Executables"
    FONTS = "Fonts"
    OTHER = "Other"


# Extension → category mapping (lower-case, with leading dot)
EXTENSION_MAP: Dict[str, FileCategory] = {
    # Documents
    ".pdf": FileCategory.DOCUMENTS,
    ".doc": FileCategory.DOCUMENTS,
    ".docx": FileCategory.DOCUMENTS,
    ".odt": FileCategory.DOCUMENTS,
    ".rtf": FileCategory.DOCUMENTS,
    ".txt": FileCategory.DOCUMENTS,
    ".md": FileCategory.DOCUMENTS,
    ".rst": FileCategory.DOCUMENTS,
    ".xls": FileCategory.DOCUMENTS,
    ".xlsx": FileCategory.DOCUMENTS,
    ".ods": FileCategory.DOCUMENTS,
    ".ppt": FileCategory.DOCUMENTS,
    ".pptx": FileCategory.DOCUMENTS,
    ".odp": FileCategory.DOCUMENTS,
    ".epub": FileCategory.DOCUMENTS,
    # Images
    ".jpg": FileCategory.IMAGES,
    ".jpeg": FileCategory.IMAGES,
    ".png": FileCategory.IMAGES,
    ".gif": FileCategory.IMAGES,
    ".bmp": FileCategory.IMAGES,
    ".svg": FileCategory.IMAGES,
    ".ico": FileCategory.IMAGES,
    ".tif": FileCategory.IMAGES,
    ".tiff": FileCategory.IMAGES,
    ".webp": FileCategory.IMAGES,
    ".heic": FileCategory.IMAGES,
    ".raw": FileCategory.IMAGES,
    # Videos
    ".mp4": FileCategory.VIDEOS,
    ".avi": FileCategory.VIDEOS,
    ".mkv": FileCategory.VIDEOS,
    ".mov": FileCategory.VIDEOS,
    ".wmv": FileCategory.VIDEOS,
    ".flv": FileCategory.VIDEOS,
    ".webm": FileCategory.VIDEOS,
    ".m4v": FileCategory.VIDEOS,
    ".mpeg": FileCategory.VIDEOS,
    ".mpg": FileCategory.VIDEOS,
    # Music
    ".mp3": FileCategory.MUSIC,
    ".wav": FileCategory.MUSIC,
    ".flac": FileCategory.MUSIC,
    ".aac": FileCategory.MUSIC,
    ".ogg": FileCategory.MUSIC,
    ".m4a": FileCategory.MUSIC,
    ".wma": FileCategory.MUSIC,
    ".opus": FileCategory.MUSIC,
    # Archives
    ".zip": FileCategory.ARCHIVES,
    ".tar": FileCategory.ARCHIVES,
    ".gz": FileCategory.ARCHIVES,
    ".bz2": FileCategory.ARCHIVES,
    ".xz": FileCategory.ARCHIVES,
    ".7z": FileCategory.ARCHIVES,
    ".rar": FileCategory.ARCHIVES,
    ".tgz": FileCategory.ARCHIVES,
    # Code
    ".py": FileCategory.CODE,
    ".js": FileCategory.CODE,
    ".ts": FileCategory.CODE,
    ".java": FileCategory.CODE,
    ".c": FileCategory.CODE,
    ".cpp": FileCategory.CODE,
    ".h": FileCategory.CODE,
    ".hpp": FileCategory.CODE,
    ".cs": FileCategory.CODE,
    ".go": FileCategory.CODE,
    ".rs": FileCategory.CODE,
    ".rb": FileCategory.CODE,
    ".php": FileCategory.CODE,
    ".swift": FileCategory.CODE,
    ".kt": FileCategory.CODE,
    ".sh": FileCategory.CODE,
    ".bash": FileCategory.CODE,
    ".zsh": FileCategory.CODE,
    ".ps1": FileCategory.CODE,
    ".lua": FileCategory.CODE,
    ".r": FileCategory.CODE,
    ".m": FileCategory.CODE,
    ".html": FileCategory.CODE,
    ".htm": FileCategory.CODE,
    ".css": FileCategory.CODE,
    ".scss": FileCategory.CODE,
    # Data
    ".json": FileCategory.DATA,
    ".xml": FileCategory.DATA,
    ".yaml": FileCategory.DATA,
    ".yml": FileCategory.DATA,
    ".toml": FileCategory.DATA,
    ".ini": FileCategory.DATA,
    ".cfg": FileCategory.DATA,
    ".conf": FileCategory.DATA,
    ".sql": FileCategory.DATA,
    ".db": FileCategory.DATA,
    ".sqlite": FileCategory.DATA,
    ".csv": FileCategory.DATA,
    # Executables
    ".exe": FileCategory.EXECUTABLES,
    ".msi": FileCategory.EXECUTABLES,
    ".dmg": FileCategory.EXECUTABLES,
    ".pkg": FileCategory.EXECUTABLES,
    ".deb": FileCategory.EXECUTABLES,
    ".rpm": FileCategory.EXECUTABLES,
    ".appimage": FileCategory.EXECUTABLES,
    # Fonts
    ".ttf": FileCategory.FONTS,
    ".otf": FileCategory.FONTS,
    ".woff": FileCategory.FONTS,
    ".woff2": FileCategory.FONTS,
}


def categorise_file(path: Path) -> FileCategory:
    """Return the :class:`FileCategory` for *path* based on its extension."""
    return EXTENSION_MAP.get(path.suffix.lower(), FileCategory.OTHER)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class MoveRecord:
    """Records one file move (real or simulated)."""

    source: Path
    destination: Path
    category: FileCategory
    dry_run: bool

    def __str__(self) -> str:
        prefix = "[DRY RUN] " if self.dry_run else ""
        return f"{prefix}{self.category.value}: {self.source.name} → {self.destination}"


@dataclass
class OrganiseResult:
    """Summary of an organise run."""

    target_dir: Path
    dry_run: bool
    moves: List[MoveRecord] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)
    errors: List[Tuple[Path, str]] = field(default_factory=list)

    @property
    def total_files(self) -> int:
        return len(self.moves) + len(self.skipped) + len(self.errors)

    def report(self) -> str:
        mode = "DRY RUN – " if self.dry_run else ""
        lines = [
            f"=== {mode}Organise Report: {self.target_dir} ===",
            f"  Files processed : {self.total_files}",
            f"  Moved           : {len(self.moves)}",
            f"  Skipped         : {len(self.skipped)}",
            f"  Errors          : {len(self.errors)}",
        ]
        if self.moves:
            lines.append("  Moves:")
            for m in self.moves:
                lines.append(f"    {m}")
        if self.skipped:
            lines.append("  Skipped (already in category folder or sub-dir):")
            for p in self.skipped:
                lines.append(f"    {p.name}")
        if self.errors:
            lines.append("  Errors:")
            for p, msg in self.errors:
                lines.append(f"    {p.name}: {msg}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Organiser
# ---------------------------------------------------------------------------


class FileOrganizer:
    """Organise files in a directory into categorised sub-folders.

    Files are moved *from* :attr:`target_dir` (the Desktop by default)
    *into* :attr:`downloads_dir` (``D:\\AI-Helper\\Organized`` on Windows
    by default) so that the Desktop stays clean and all organised content
    lives on the D drive where space is not constrained.

    Parameters
    ----------
    target_dir:
        Directory to scan for files to organise.  Defaults to the user's
        Desktop (``~/Desktop``).
    downloads_dir:
        Root destination directory where categorised sub-folders are
        created.  Defaults to :func:`ai_helper.config.get_organized_dir`
        (``D:\\AI-Helper\\Organized`` on Windows, ``~/AI-Helper/Organized``
        elsewhere).  Pass an explicit path to override.
    dry_run:
        When ``True``, compute and report what *would* happen without
        actually moving any files.
    recursive:
        When ``True``, scan sub-directories too.  Files already inside a
        category sub-directory are skipped to prevent re-organising.
    custom_map:
        Optional extra ``{extension: FileCategory}`` entries that extend
        (or override) the built-in :data:`EXTENSION_MAP`.
    """

    def __init__(
        self,
        target_dir: Optional[Path] = None,
        downloads_dir: Optional[Path] = None,
        dry_run: bool = False,
        recursive: bool = False,
        custom_map: Optional[Dict[str, FileCategory]] = None,
    ) -> None:
        self.target_dir = target_dir or (Path.home() / "Desktop")
        # All organised files land on the D drive (or configured install dir).
        self.downloads_dir: Path = downloads_dir or _cfg.get_organized_dir()
        self.dry_run = dry_run
        self.recursive = recursive
        self._ext_map: Dict[str, FileCategory] = {**EXTENSION_MAP, **(custom_map or {})}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> Dict[FileCategory, List[Path]]:
        """Return a dict mapping each category to its files, without moving anything."""
        result: Dict[FileCategory, List[Path]] = {cat: [] for cat in FileCategory}
        for path in self._iter_files():
            result[self._categorise(path)].append(path)
        return result

    def organise(self) -> OrganiseResult:
        """Move files into category sub-directories.

        Returns an :class:`OrganiseResult` summarising every action taken.
        """
        result = OrganiseResult(target_dir=self.target_dir, dry_run=self.dry_run)

        for path in self._iter_files():
            # Skip files already inside one of our category sub-directories
            if path.parent != self.target_dir:
                result.skipped.append(path)
                continue

            category = self._categorise(path)
            dest_dir = self.target_dir / category.value
            dest = self._unique_dest(dest_dir, path.name)

            record = MoveRecord(source=path, destination=dest, category=category, dry_run=self.dry_run)

            if not self.dry_run:
                try:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dest))
                    logger.info("Moved %s → %s", path.name, dest)
                except OSError as exc:
                    result.errors.append((path, str(exc)))
                    logger.error("Failed to move %s: %s", path, exc)
                    continue

            result.moves.append(record)

        return result

    def undo(self, result: OrganiseResult) -> int:
        """Reverse the moves recorded in *result*.  Returns number of files restored."""
        if result.dry_run:
            logger.warning("Cannot undo a dry-run result")
            return 0
        restored = 0
        for record in reversed(result.moves):
            try:
                if record.destination.exists():
                    shutil.move(str(record.destination), str(record.source))
                    restored += 1
                    logger.info("Restored %s", record.source.name)
            except OSError as exc:
                logger.error("Failed to restore %s: %s", record.source.name, exc)
        return restored

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _iter_files(self):
        if not self.target_dir.is_dir():
            return
        pattern = "**/*" if self.recursive else "*"
        for path in self.target_dir.glob(pattern):
            if path.is_file():
                yield path

    def _categorise(self, path: Path) -> FileCategory:
        return self._ext_map.get(path.suffix.lower(), FileCategory.OTHER)

    @staticmethod
    def _unique_dest(dest_dir: Path, name: str) -> Path:
        """Return a destination path that does not already exist."""
        candidate = dest_dir / name
        if not candidate.exists():
            return candidate
        stem = Path(name).stem
        suffix = Path(name).suffix
        counter = 1
        while True:
            candidate = dest_dir / f"{stem} ({counter}){suffix}"
            if not candidate.exists():
                return candidate
            counter += 1
