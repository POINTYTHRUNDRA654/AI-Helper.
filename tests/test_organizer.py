"""Tests for ai_helper.organizer."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_helper.organizer import (
    FileCategory,
    FileOrganizer,
    OrganiseResult,
    categorise_file,
)


class TestCategoriseFile(unittest.TestCase):
    def test_pdf_is_document(self):
        self.assertEqual(categorise_file(Path("report.pdf")), FileCategory.DOCUMENTS)

    def test_jpg_is_image(self):
        self.assertEqual(categorise_file(Path("photo.jpg")), FileCategory.IMAGES)

    def test_mp3_is_music(self):
        self.assertEqual(categorise_file(Path("song.mp3")), FileCategory.MUSIC)

    def test_py_is_code(self):
        self.assertEqual(categorise_file(Path("script.py")), FileCategory.CODE)

    def test_zip_is_archive(self):
        self.assertEqual(categorise_file(Path("archive.zip")), FileCategory.ARCHIVES)

    def test_unknown_is_other(self):
        self.assertEqual(categorise_file(Path("mystery.xyz123")), FileCategory.OTHER)

    def test_case_insensitive(self):
        self.assertEqual(categorise_file(Path("IMAGE.PNG")), FileCategory.IMAGES)


class TestFileOrganizerDryRun(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self._src = TemporaryDirectory()
        self.src_path = Path(self._src.name)
        self.dst_path = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()
        self._src.cleanup()

    def _create_file(self, name: str) -> Path:
        p = self.src_path / name
        p.write_text("dummy")
        return p

    def test_dry_run_does_not_move_files(self):
        self._create_file("notes.txt")
        self._create_file("photo.png")
        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=True,
        )
        result = organizer.organise()
        self.assertTrue(result.dry_run)
        self.assertEqual(len(result.moves), 2)
        # Files must still be in source
        self.assertTrue((self.src_path / "notes.txt").exists())
        self.assertTrue((self.src_path / "photo.png").exists())

    def test_real_move_relocates_files(self):
        self._create_file("document.pdf")
        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=False,
        )
        result = organizer.organise()
        self.assertEqual(len(result.moves), 1)
        expected = self.dst_path / FileCategory.DOCUMENTS.value / "document.pdf"
        self.assertTrue(expected.exists(), f"Expected file at {expected}")

    def test_real_move_to_downloads_dir(self):
        """Organised files go to downloads_dir, not a subfolder of target_dir."""
        self._create_file("video.mp4")
        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=False,
        )
        organizer.organise()
        # File must NOT still be in the source directory
        self.assertFalse((self.src_path / "video.mp4").exists())
        # File must be in downloads_dir
        expected = self.dst_path / FileCategory.VIDEOS.value / "video.mp4"
        self.assertTrue(expected.exists())

    def test_scan_does_not_move(self):
        self._create_file("song.flac")
        organizer = FileOrganizer(target_dir=self.src_path, downloads_dir=self.dst_path)
        mapping = organizer.scan()
        self.assertTrue((self.src_path / "song.flac").exists())
        self.assertIn(Path(self.src_path / "song.flac"), mapping[FileCategory.MUSIC])

    def test_unique_dest_avoids_collision(self):
        # Create two files with same name
        self._create_file("readme.txt")
        # Pre-create a file at the destination to force collision handling
        dest_dir = self.dst_path / FileCategory.DOCUMENTS.value
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / "readme.txt").write_text("existing")

        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=False,
        )
        result = organizer.organise()
        self.assertEqual(len(result.errors), 0)
        # The moved file should have been renamed
        self.assertEqual(result.moves[0].destination.name, "readme (1).txt")

    def test_undo_restores_files(self):
        self._create_file("undo_me.json")
        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=False,
        )
        result = organizer.organise()
        self.assertFalse((self.src_path / "undo_me.json").exists())

        restored = organizer.undo(result)
        self.assertEqual(restored, 1)
        self.assertTrue((self.src_path / "undo_me.json").exists())

    def test_undo_dry_run_is_noop(self):
        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=True,
        )
        fake_result = OrganiseResult(target_dir=self.src_path, dry_run=True)
        restored = organizer.undo(fake_result)
        self.assertEqual(restored, 0)

    def test_report_contains_summary(self):
        self._create_file("img.jpeg")
        organizer = FileOrganizer(
            target_dir=self.src_path,
            downloads_dir=self.dst_path,
            dry_run=True,
        )
        result = organizer.organise()
        report = result.report()
        self.assertIn("Organise Report", report)
        self.assertIn("Moved", report)


if __name__ == "__main__":
    unittest.main()
