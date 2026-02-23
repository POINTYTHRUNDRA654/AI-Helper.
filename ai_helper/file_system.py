"""File system access module.

Gives AI Helper full access to the file system so it can read, write,
search and watch any file or directory on the computer — enabling it to
use files as part of performing its services.

Classes
-------
FileSearcher
    Find files anywhere on disk by name pattern, content keyword,
    extension, size or modification date.

FileReader
    Read text or binary files with automatic encoding detection and
    graceful fallback.

FileWriter
    Write or append to any file, with automatic backup creation.

FileWatcher
    Monitor a directory tree for new, modified or deleted files using
    a polling background thread.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class FileMatch:
    """One file returned by a search."""
    path: Path
    size_bytes: int
    modified: datetime
    snippet: str = ""          # matching line(s) for content searches

    @property
    def size_kb(self) -> float:
        return round(self.size_bytes / 1024, 1)

    def __str__(self) -> str:
        ts = self.modified.strftime("%Y-%m-%d %H:%M")
        snip = f"\n    …{self.snippet[:120]}…" if self.snippet else ""
        return f"{self.path}  ({self.size_kb} KB, {ts}){snip}"


@dataclass
class FileChangeEvent:
    """A change detected by :class:`FileWatcher`."""
    kind: str          # "created" | "modified" | "deleted"
    path: Path
    timestamp: float = field(default_factory=time.time)

    def __str__(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        return f"[{ts}] {self.kind.upper()} {self.path}"


# ---------------------------------------------------------------------------
# FileSearcher
# ---------------------------------------------------------------------------


class FileSearcher:
    """Search files anywhere on the file system.

    Parameters
    ----------
    default_root:
        Default directory tree to search when *root* is not given.
        Defaults to the user's home directory.
    max_results:
        Cap on the number of results returned (default 200).
    """

    def __init__(
        self,
        default_root: Optional[Path] = None,
        max_results: int = 200,
    ) -> None:
        self.default_root = default_root or Path.home()
        self.max_results = max_results

    def search(
        self,
        name_pattern: str = "*",
        root: Optional[Path] = None,
        content_keyword: str = "",
        extensions: Optional[List[str]] = None,
        min_size_bytes: int = 0,
        max_size_bytes: int = 0,
        modified_after: Optional[datetime] = None,
        recursive: bool = True,
    ) -> List[FileMatch]:
        """Search for files matching the given criteria.

        Parameters
        ----------
        name_pattern:
            Shell-style glob for the filename (e.g. ``"*.py"``, ``"report*"``).
            Defaults to ``"*"`` (all files).
        root:
            Directory tree to search.  Defaults to :attr:`default_root`.
        content_keyword:
            If non-empty, only return files whose text content contains
            this string (case-insensitive).
        extensions:
            List of extensions to include, e.g. ``[".py", ".txt"]``.
        min_size_bytes / max_size_bytes:
            Size filter (0 = no limit).
        modified_after:
            Only return files modified after this datetime.
        recursive:
            Whether to recurse into sub-directories (default ``True``).
        """
        root = Path(root) if root else self.default_root
        results: List[FileMatch] = []

        pattern = "**/" + name_pattern if recursive else name_pattern
        try:
            paths = root.glob(pattern) if recursive else root.glob(name_pattern)
        except (PermissionError, OSError):
            return results

        ext_set = {e.lower() if e.startswith(".") else f".{e.lower()}"
                   for e in (extensions or [])}

        for path in paths:
            if len(results) >= self.max_results:
                break
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue

            if ext_set and path.suffix.lower() not in ext_set:
                continue
            if min_size_bytes and stat.st_size < min_size_bytes:
                continue
            if max_size_bytes and stat.st_size > max_size_bytes:
                continue
            mtime = datetime.fromtimestamp(stat.st_mtime)
            if modified_after and mtime < modified_after:
                continue

            snippet = ""
            if content_keyword:
                snippet = self._find_snippet(path, content_keyword)
                if snippet is None:
                    continue   # keyword not found

            results.append(FileMatch(
                path=path,
                size_bytes=stat.st_size,
                modified=mtime,
                snippet=snippet or "",
            ))

        return sorted(results, key=lambda m: m.modified, reverse=True)

    def find_by_name(self, name: str, root: Optional[Path] = None) -> List[FileMatch]:
        """Convenience wrapper: exact filename match anywhere under *root*."""
        return self.search(name_pattern=name, root=root)

    def find_by_extension(self, extension: str, root: Optional[Path] = None) -> List[FileMatch]:
        """Return all files with a given extension under *root*."""
        ext = extension if extension.startswith(".") else f".{extension}"
        return self.search(name_pattern=f"*{ext}", root=root)

    def find_containing(self, keyword: str, root: Optional[Path] = None,
                         extensions: Optional[List[str]] = None) -> List[FileMatch]:
        """Return text files whose content contains *keyword*."""
        return self.search(
            root=root,
            content_keyword=keyword,
            extensions=extensions or [".txt", ".md", ".py", ".json", ".yaml",
                                       ".yml", ".toml", ".ini", ".cfg", ".log"],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_snippet(path: Path, keyword: str) -> Optional[str]:
        """Return the first matching line, or None if not found."""
        kw = keyword.lower()
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    if kw in line.lower():
                        return line.strip()
        except OSError:
            pass
        return None


# ---------------------------------------------------------------------------
# FileReader
# ---------------------------------------------------------------------------


class FileReader:
    """Read files in a robust, encoding-tolerant way.

    Parameters
    ----------
    max_bytes:
        Maximum bytes to read from a single file (default 10 MB).
        Protects against accidentally loading huge binary files.
    """

    _ENCODINGS = ["utf-8", "utf-16", "latin-1", "cp1252"]

    def __init__(self, max_bytes: int = 10 * 1024 * 1024) -> None:
        self.max_bytes = max_bytes

    def read(self, path: Path | str) -> str:
        """Return the text content of *path*.

        Tries UTF-8 first, then several fallback encodings.  Binary files
        return a placeholder message rather than raising.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"Path is a directory: {path}")

        size = path.stat().st_size
        if size > self.max_bytes:
            return (
                f"[File too large to read: {path.name}  "
                f"({size / 1e6:.1f} MB > {self.max_bytes / 1e6:.0f} MB limit)]"
            )

        for enc in self._ENCODINGS:
            try:
                return path.read_text(encoding=enc)
            except (UnicodeDecodeError, LookupError):
                continue

        # Binary fallback
        return f"[Binary file: {path.name}  ({size} bytes)]"

    def read_lines(self, path: Path | str, start: int = 1, end: Optional[int] = None) -> str:
        """Return a slice of lines *start*...*end* (1-based, inclusive)."""
        content = self.read(path)
        lines = content.splitlines()
        slice_ = lines[start - 1: end]
        return "\n".join(f"{start + i}: {ln}" for i, ln in enumerate(slice_))

    def checksum(self, path: Path | str, algorithm: str = "md5") -> str:
        """Return a hex checksum of the file contents."""
        path = Path(path)
        h = hashlib.new(algorithm)
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
        except OSError as exc:
            return f"[error: {exc}]"
        return h.hexdigest()


# ---------------------------------------------------------------------------
# FileWriter
# ---------------------------------------------------------------------------


class FileWriter:
    """Write or modify files with optional automatic backup.

    Parameters
    ----------
    backup:
        When ``True`` (default), a ``*.bak`` copy of the original file is
        created before overwriting it.
    create_parents:
        When ``True`` (default), missing parent directories are created.
    """

    def __init__(self, backup: bool = True, create_parents: bool = True) -> None:
        self.backup = backup
        self.create_parents = create_parents

    def write(self, path: Path | str, content: str, encoding: str = "utf-8") -> Path:
        """Write *content* to *path*, creating a backup of any existing file.

        Returns the path that was written.
        """
        path = Path(path)
        if self.create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        if self.backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
            logger.debug("Backup created: %s", backup_path)
        path.write_text(content, encoding=encoding)
        logger.info("Wrote %d bytes to %s", len(content.encode(encoding)), path)
        return path

    def append(self, path: Path | str, content: str, encoding: str = "utf-8") -> Path:
        """Append *content* to *path*.  Creates the file if it doesn't exist."""
        path = Path(path)
        if self.create_parents:
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding=encoding) as fh:
            fh.write(content)
        return path

    def delete(self, path: Path | str, backup: Optional[bool] = None) -> bool:
        """Delete *path*, optionally creating a backup first.

        Returns ``True`` if the file was deleted, ``False`` if not found.
        """
        path = Path(path)
        if not path.exists():
            return False
        use_backup = self.backup if backup is None else backup
        if use_backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        path.unlink()
        logger.info("Deleted %s", path)
        return True


# ---------------------------------------------------------------------------
# FileWatcher
# ---------------------------------------------------------------------------


class FileWatcher:
    """Poll a directory for file-system changes and fire callbacks.

    Uses a simple hash-of-mtimes approach (no OS-specific APIs needed)
    so it works on every platform.

    Parameters
    ----------
    path:
        Directory to watch.
    callback:
        Called with a :class:`FileChangeEvent` for each change detected.
    interval:
        Polling interval in seconds (default 5).
    recursive:
        Watch sub-directories too (default ``True``).
    """

    def __init__(
        self,
        path: Path | str,
        callback: Callable[[FileChangeEvent], None],
        interval: float = 5.0,
        recursive: bool = True,
    ) -> None:
        self.path = Path(path)
        self.callback = callback
        self.interval = interval
        self.recursive = recursive

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot: Dict[Path, float] = {}

    def start(self) -> None:
        """Start watching in a background daemon thread."""
        self._snapshot = self._take_snapshot()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="ai-helper-filewatcher", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background watcher thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 2)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(timeout=self.interval)
            try:
                self._check()
            except Exception:  # noqa: BLE001
                logger.exception("FileWatcher error")

    def _check(self) -> None:
        current = self._take_snapshot()
        prev = self._snapshot

        for path, mtime in current.items():
            if path not in prev:
                self._fire("created", path)
            elif prev[path] != mtime:
                self._fire("modified", path)

        for path in prev:
            if path not in current:
                self._fire("deleted", path)

        self._snapshot = current

    def _take_snapshot(self) -> Dict[Path, float]:
        snap: Dict[Path, float] = {}
        if not self.path.is_dir():
            return snap
        pattern = "**/*" if self.recursive else "*"
        try:
            for p in self.path.glob(pattern):
                if p.is_file():
                    try:
                        snap[p] = p.stat().st_mtime
                    except OSError:
                        pass
        except OSError:
            pass
        return snap

    def _fire(self, kind: str, path: Path) -> None:
        event = FileChangeEvent(kind=kind, path=path)
        try:
            self.callback(event)
        except Exception:  # noqa: BLE001
            logger.exception("FileWatcher callback error")
