"""Tests for ai_helper.file_system."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_helper.file_system import (
    FileChangeEvent,
    FileMatch,
    FileReader,
    FileSearcher,
    FileWatcher,
    FileWriter,
)


# ---------------------------------------------------------------------------
# FileSearcher
# ---------------------------------------------------------------------------

class TestFileSearcher(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Create test tree
        (self.root / "docs").mkdir()
        (self.root / "code").mkdir()
        (self.root / "docs" / "report.pdf.txt").write_text("quarterly report contents")
        (self.root / "docs" / "notes.txt").write_text("hello world notes")
        (self.root / "code" / "main.py").write_text("print('hello world')")
        (self.root / "code" / "utils.py").write_text("def helper(): pass")
        (self.root / "image.png").write_bytes(b"\x89PNG")

    def tearDown(self):
        self._tmp.cleanup()

    def test_search_all(self):
        s = FileSearcher(default_root=self.root)
        results = s.search(root=self.root)
        self.assertGreaterEqual(len(results), 5)

    def test_search_by_pattern(self):
        s = FileSearcher()
        results = s.search(name_pattern="*.py", root=self.root)
        self.assertEqual(len(results), 2)
        names = {r.path.name for r in results}
        self.assertIn("main.py", names)
        self.assertIn("utils.py", names)

    def test_search_by_extension(self):
        s = FileSearcher()
        results = s.find_by_extension(".txt", root=self.root)
        self.assertEqual(len(results), 2)

    def test_search_by_name(self):
        s = FileSearcher()
        results = s.find_by_name("notes.txt", root=self.root)
        self.assertEqual(len(results), 1)

    def test_search_by_content(self):
        s = FileSearcher()
        results = s.find_containing("hello world", root=self.root)
        names = {r.path.name for r in results}
        self.assertIn("notes.txt", names)
        self.assertIn("main.py", names)

    def test_content_not_found_excludes_file(self):
        s = FileSearcher()
        results = s.find_containing("XYZNOTFOUND", root=self.root)
        self.assertEqual(results, [])

    def test_snippet_included(self):
        s = FileSearcher()
        results = s.find_containing("quarterly", root=self.root)
        self.assertEqual(len(results), 1)
        self.assertIn("quarterly", results[0].snippet)

    def test_size_filter(self):
        s = FileSearcher()
        big = self.root / "big.txt"
        big.write_text("x" * 5000)
        results = s.search(root=self.root, min_size_bytes=1000)
        paths = {r.path for r in results}
        self.assertIn(big, paths)
        small_paths = {r.path for r in s.search(root=self.root, max_size_bytes=100)}
        self.assertNotIn(big, small_paths)

    def test_max_results_respected(self):
        s = FileSearcher(max_results=2)
        results = s.search(root=self.root)
        self.assertLessEqual(len(results), 2)

    def test_file_match_str(self):
        s = FileSearcher()
        results = s.search(name_pattern="*.py", root=self.root)
        self.assertTrue(str(results[0]))


# ---------------------------------------------------------------------------
# FileReader
# ---------------------------------------------------------------------------

class TestFileReader(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_text_file(self):
        p = self.root / "hello.txt"
        p.write_text("Hello, world!", encoding="utf-8")
        reader = FileReader()
        self.assertEqual(reader.read(p), "Hello, world!")

    def test_read_nonexistent_raises(self):
        reader = FileReader()
        with self.assertRaises(FileNotFoundError):
            reader.read(self.root / "does_not_exist.txt")

    def test_read_directory_raises(self):
        reader = FileReader()
        with self.assertRaises(IsADirectoryError):
            reader.read(self.root)

    def test_read_large_file_returns_message(self):
        p = self.root / "huge.txt"
        # Write 11 MB (more than default 10 MB limit)
        p.write_bytes(b"a" * (11 * 1024 * 1024))
        reader = FileReader(max_bytes=10 * 1024 * 1024)
        result = reader.read(p)
        self.assertIn("too large", result)

    def test_read_lines_slice(self):
        p = self.root / "multi.txt"
        p.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
        reader = FileReader()
        result = reader.read_lines(p, start=2, end=3)
        self.assertIn("line2", result)
        self.assertIn("line3", result)
        self.assertNotIn("line1", result)
        self.assertNotIn("line4", result)

    def test_checksum_consistent(self):
        p = self.root / "data.txt"
        p.write_text("checksum me", encoding="utf-8")
        reader = FileReader()
        a = reader.checksum(p)
        b = reader.checksum(p)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)   # MD5 hex length

    def test_checksum_different_files(self):
        p1 = self.root / "a.txt"
        p2 = self.root / "b.txt"
        p1.write_text("aaa")
        p2.write_text("bbb")
        reader = FileReader()
        self.assertNotEqual(reader.checksum(p1), reader.checksum(p2))

    def test_read_latin1_fallback(self):
        p = self.root / "latin.txt"
        p.write_bytes(b"\xe9\xe0\xfc")  # latin-1 encoded characters
        reader = FileReader()
        content = reader.read(p)
        self.assertIsInstance(content, str)


# ---------------------------------------------------------------------------
# FileWriter
# ---------------------------------------------------------------------------

class TestFileWriter(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_write_creates_file(self):
        p = self.root / "new.txt"
        writer = FileWriter()
        writer.write(p, "hello")
        self.assertTrue(p.exists())
        self.assertEqual(p.read_text(), "hello")

    def test_write_creates_parents(self):
        p = self.root / "a" / "b" / "c.txt"
        FileWriter(create_parents=True).write(p, "deep")
        self.assertTrue(p.exists())

    def test_write_creates_backup(self):
        p = self.root / "existing.txt"
        p.write_text("original")
        FileWriter(backup=True).write(p, "new content")
        backup = p.with_suffix(".txt.bak")
        self.assertTrue(backup.exists())
        self.assertEqual(backup.read_text(), "original")

    def test_write_no_backup_when_disabled(self):
        p = self.root / "nobackup.txt"
        p.write_text("old")
        FileWriter(backup=False).write(p, "new")
        backup = p.with_suffix(".txt.bak")
        self.assertFalse(backup.exists())

    def test_append_adds_content(self):
        p = self.root / "log.txt"
        writer = FileWriter()
        writer.append(p, "line1\n")
        writer.append(p, "line2\n")
        content = p.read_text()
        self.assertIn("line1", content)
        self.assertIn("line2", content)

    def test_delete_removes_file(self):
        p = self.root / "del.txt"
        p.write_text("bye")
        writer = FileWriter(backup=False)
        result = writer.delete(p)
        self.assertTrue(result)
        self.assertFalse(p.exists())

    def test_delete_nonexistent_returns_false(self):
        writer = FileWriter()
        result = writer.delete(self.root / "ghost.txt")
        self.assertFalse(result)

    def test_delete_with_backup(self):
        p = self.root / "precious.txt"
        p.write_text("keep me")
        FileWriter(backup=True).delete(p, backup=True)
        bak = p.with_suffix(".txt.bak")
        self.assertTrue(bak.exists())


# ---------------------------------------------------------------------------
# FileWatcher
# ---------------------------------------------------------------------------

class TestFileWatcher(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_detects_new_file(self):
        events = []
        watcher = FileWatcher(self.root, callback=events.append, interval=0.1)
        watcher.start()
        time.sleep(0.15)
        (self.root / "new.txt").write_text("created")
        time.sleep(0.4)
        watcher.stop()
        created = [e for e in events if e.kind == "created"]
        self.assertGreaterEqual(len(created), 1)

    def test_detects_modified_file(self):
        p = self.root / "existing.txt"
        p.write_text("original")
        events = []
        watcher = FileWatcher(self.root, callback=events.append, interval=0.1)
        watcher.start()
        time.sleep(0.15)
        p.write_text("modified")
        time.sleep(0.4)
        watcher.stop()
        modified = [e for e in events if e.kind == "modified"]
        self.assertGreaterEqual(len(modified), 1)

    def test_start_stop(self):
        watcher = FileWatcher(self.root, callback=lambda e: None, interval=0.1)
        watcher.start()
        self.assertTrue(watcher.running)
        watcher.stop()
        self.assertFalse(watcher.running)

    def test_file_change_event_str(self):
        e = FileChangeEvent(kind="created", path=Path("/tmp/test.txt"))
        self.assertIn("created", str(e).lower())
        self.assertIn("test.txt", str(e))


if __name__ == "__main__":
    unittest.main()
