"""Automatic file backup.

Watches one or more source directories and copies changed files to a
versioned backup tree on the D drive (or any configured install dir).

Directory layout::

    D:\\AI-Helper\\Backups\\
        my_docs\\               ← mirrors the watched source path name
            report.docx         ← latest copy
            .versions\\
                report.docx.20260223_143012  ← timestamped history

Features
--------
* **Continuous watching** — uses :class:`~ai_helper.file_system.FileWatcher`
  so changes are captured within seconds without hammering the disk.
* **Versioning** — up to ``keep_versions`` old copies are kept per file.
* **Dry-run mode** — pass ``dry_run=True`` to see what would be backed up
  without actually copying anything.
* **On-demand snapshot** — call ``backup_now(source_dir)`` to do a full
  sync immediately regardless of watchers.

Usage
-----
::

    from ai_helper.backup import BackupManager

    mgr = BackupManager()
    mgr.add_watch(Path.home() / "Documents")
    mgr.start()          # background watcher threads
    # … AI Helper runs …
    mgr.stop()

    # Or one-shot:
    mgr.backup_now(Path.home() / "Documents")
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .file_system import FileChangeEvent, FileWatcher

logger = logging.getLogger(__name__)

_VERSION_DIR = ".versions"
_TS_FMT = "%Y%m%d_%H%M%S"


class BackupManager:
    """Watch directories and automatically back up changed files.

    Parameters
    ----------
    backup_root:
        Root directory for all backups.
        Defaults to ``<INSTALL_DIR>/Backups``.
    keep_versions:
        Number of old versions to keep per file (default 10).
    poll_interval:
        Seconds between each FileWatcher poll (default 10).
    dry_run:
        If ``True``, log what would be copied but don't actually copy.
    """

    def __init__(
        self,
        backup_root: Optional[Path] = None,
        keep_versions: int = 10,
        poll_interval: float = 10.0,
        dry_run: bool = False,
    ) -> None:
        if backup_root is None:
            from . import config as _cfg  # noqa: PLC0415
            backup_root = _cfg.get_install_dir() / "Backups"
        self.backup_root = Path(backup_root)
        self.keep_versions = keep_versions
        self.poll_interval = poll_interval
        self.dry_run = dry_run

        self._watchers: Dict[Path, FileWatcher] = {}
        self._lock = threading.Lock()
        self._stats: Dict[str, int] = {"copied": 0, "skipped": 0, "errors": 0}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_watch(self, source_dir: Path | str) -> None:
        """Register *source_dir* to be watched and backed up continuously."""
        source_dir = Path(source_dir).resolve()
        with self._lock:
            if source_dir in self._watchers:
                logger.debug("Already watching %s", source_dir)
                return
            watcher = FileWatcher(
                path=source_dir,
                callback=lambda evt: self._on_change(evt, source_dir),
                interval=self.poll_interval,
                recursive=True,
            )
            self._watchers[source_dir] = watcher
            logger.info("Registered backup watch: %s", source_dir)

    def remove_watch(self, source_dir: Path | str) -> None:
        """Stop watching *source_dir*."""
        source_dir = Path(source_dir).resolve()
        with self._lock:
            watcher = self._watchers.pop(source_dir, None)
        if watcher:
            watcher.stop()
            logger.info("Removed backup watch: %s", source_dir)

    def start(self) -> None:
        """Start all registered watchers in background daemon threads."""
        with self._lock:
            for src, watcher in self._watchers.items():
                if not watcher.running:
                    watcher.start()
                    logger.info("Backup watcher started: %s", src)

    def stop(self) -> None:
        """Stop all watcher threads."""
        with self._lock:
            for watcher in self._watchers.values():
                watcher.stop()
            logger.info("All backup watchers stopped")

    def backup_now(self, source_dir: Path | str, recursive: bool = True) -> int:
        """Immediately back up all files in *source_dir*.

        Returns the number of files successfully copied.
        """
        source_dir = Path(source_dir).resolve()
        if not source_dir.is_dir():
            logger.error("backup_now: not a directory: %s", source_dir)
            return 0

        copied = 0
        pattern = "**/*" if recursive else "*"
        for src_file in source_dir.glob(pattern):
            if src_file.is_file():
                if self._copy_file(src_file, source_dir):
                    copied += 1
        logger.info("backup_now: copied %d files from %s", copied, source_dir)
        return copied

    @property
    def stats(self) -> Dict[str, int]:
        """Running count of files copied / skipped / errored."""
        return dict(self._stats)

    def format_stats(self) -> str:
        return (
            f"Backup stats — copied: {self._stats['copied']}, "
            f"skipped: {self._stats['skipped']}, errors: {self._stats['errors']}"
        )

    def list_versions(self, file_path: Path | str) -> List[Path]:
        """Return all backup versions of *file_path*, newest first."""
        file_path = Path(file_path).resolve()
        # Find which source_dir this file belongs to
        for src_dir in self._watchers:
            try:
                rel = file_path.relative_to(src_dir)
                dest_dir = self._dest_dir(src_dir) / rel.parent / _VERSION_DIR
                if dest_dir.is_dir():
                    prefix = file_path.name
                    versions = sorted(
                        dest_dir.glob(f"{prefix}.*"),
                        key=lambda p: p.stat().st_mtime,
                        reverse=True,
                    )
                    return versions
            except ValueError:
                continue
        return []

    def restore(self, version_path: Path | str, destination: Path | str) -> bool:
        """Restore *version_path* (a backup file) to *destination*.

        Returns ``True`` on success.
        """
        version_path = Path(version_path)
        destination = Path(destination)
        if not version_path.is_file():
            logger.error("restore: version not found: %s", version_path)
            return False
        if self.dry_run:
            logger.info("[DRY RUN] Would restore %s → %s", version_path, destination)
            return True
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(version_path, destination)
        logger.info("Restored %s → %s", version_path, destination)
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_change(self, event: FileChangeEvent, source_dir: Path) -> None:
        if event.kind == "deleted":
            return  # don't remove backups on deletion
        if not event.path.is_file():
            return
        self._copy_file(event.path, source_dir)

    def _dest_dir(self, source_dir: Path) -> Path:
        """Map a source directory to its backup root sub-folder."""
        return self.backup_root / source_dir.name

    def _copy_file(self, src_file: Path, source_dir: Path) -> bool:
        """Copy *src_file* into the backup tree.  Returns True on success."""
        try:
            rel = src_file.relative_to(source_dir)
        except ValueError:
            rel = Path(src_file.name)

        dest_dir = self._dest_dir(source_dir) / rel.parent
        dest_file = dest_dir / src_file.name
        version_dir = dest_dir / _VERSION_DIR

        if self.dry_run:
            logger.info("[DRY RUN] Would backup %s → %s", src_file, dest_file)
            self._stats["skipped"] += 1
            return False

        try:
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Archive existing copy before overwriting
            if dest_file.exists():
                version_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime(_TS_FMT)
                versioned = version_dir / f"{dest_file.name}.{ts}"
                shutil.copy2(dest_file, versioned)
                self._prune_versions(version_dir, dest_file.name)

            shutil.copy2(src_file, dest_file)
            self._stats["copied"] += 1
            logger.debug("Backed up: %s → %s", src_file, dest_file)
            return True

        except OSError as exc:
            logger.error("Backup failed for %s: %s", src_file, exc)
            self._stats["errors"] += 1
            return False

    def _prune_versions(self, version_dir: Path, filename: str) -> None:
        """Remove oldest versions beyond keep_versions limit."""
        versions = sorted(
            version_dir.glob(f"{filename}.*"),
            key=lambda p: p.stat().st_mtime,
        )
        excess = len(versions) - self.keep_versions
        for old in versions[:max(0, excess)]:
            try:
                old.unlink()
            except OSError:
                pass
