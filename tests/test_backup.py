"""Tests for ai_helper.backup."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_helper.backup import BackupManager


class TestBackupManager(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.src = self.root / "source"
        self.backup_root = self.root / "backups"
        self.src.mkdir()
        self.mgr = BackupManager(backup_root=self.backup_root, keep_versions=3)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, content: str) -> Path:
        p = self.src / name
        p.write_text(content)
        return p

    def test_backup_now_copies_files(self):
        self._write("notes.txt", "hello")
        self._write("data.csv", "a,b,c")
        copied = self.mgr.backup_now(self.src)
        self.assertEqual(copied, 2)

    def test_backup_now_creates_dest_file(self):
        self._write("report.txt", "contents")
        self.mgr.backup_now(self.src)
        dest = self.backup_root / "source" / "report.txt"
        self.assertTrue(dest.exists())
        self.assertEqual(dest.read_text(), "contents")

    def test_backup_now_versions_overwrite(self):
        p = self._write("file.txt", "v1")
        self.mgr.backup_now(self.src)
        p.write_text("v2")
        self.mgr.backup_now(self.src)
        version_dir = self.backup_root / "source" / ".versions"
        versions = list(version_dir.glob("file.txt.*"))
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].read_text(), "v1")

    def test_pruning_keeps_max_versions(self):
        p = self._write("prune_me.txt", "original")
        for i in range(5):
            self.mgr.backup_now(self.src)
            p.write_text(f"version {i}")
        version_dir = self.backup_root / "source" / ".versions"
        versions = list(version_dir.glob("prune_me.txt.*"))
        self.assertLessEqual(len(versions), self.mgr.keep_versions)

    def test_dry_run_does_not_copy(self):
        self._write("secret.txt", "private")
        mgr = BackupManager(backup_root=self.backup_root, dry_run=True)
        copied = mgr.backup_now(self.src)
        self.assertEqual(copied, 0)
        self.assertFalse((self.backup_root / "source" / "secret.txt").exists())

    def test_stats_updated(self):
        self._write("a.txt", "a")
        self._write("b.txt", "b")
        self.mgr.backup_now(self.src)
        self.assertEqual(self.mgr.stats["copied"], 2)
        self.assertEqual(self.mgr.stats["errors"], 0)

    def test_backup_nonexistent_dir_returns_zero(self):
        result = self.mgr.backup_now(self.root / "nonexistent")
        self.assertEqual(result, 0)

    def test_add_and_remove_watch(self):
        self.mgr.add_watch(self.src)
        self.assertIn(self.src, self.mgr._watchers)
        self.mgr.remove_watch(self.src)
        self.assertNotIn(self.src, self.mgr._watchers)

    def test_watcher_starts_and_stops(self):
        self.mgr.add_watch(self.src)
        self.mgr.start()
        watcher = self.mgr._watchers[self.src]
        self.assertTrue(watcher.running)
        self.mgr.stop()

    def test_restore(self):
        p = self._write("restore_me.txt", "original")
        self.mgr.backup_now(self.src)
        p.write_text("modified")
        self.mgr.backup_now(self.src)

        version_dir = self.backup_root / "source" / ".versions"
        versions = sorted(version_dir.glob("restore_me.txt.*"))
        self.assertTrue(len(versions) >= 1)

        restore_dest = self.root / "restored.txt"
        ok = self.mgr.restore(versions[0], restore_dest)
        self.assertTrue(ok)
        self.assertTrue(restore_dest.exists())

    def test_format_stats(self):
        self._write("x.txt", "x")
        self.mgr.backup_now(self.src)
        stats_str = self.mgr.format_stats()
        self.assertIn("copied", stats_str)


if __name__ == "__main__":
    unittest.main()
