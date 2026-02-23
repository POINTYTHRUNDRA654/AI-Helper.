"""Tests for ai_helper.updater."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from ai_helper.updater import UpdateInfo, Updater


class TestVersionComparison(unittest.TestCase):
    def test_newer_is_available(self):
        self.assertTrue(Updater._version_gt("0.2.0", "0.1.0"))

    def test_same_version_not_newer(self):
        self.assertFalse(Updater._version_gt("0.1.0", "0.1.0"))

    def test_older_not_newer(self):
        self.assertFalse(Updater._version_gt("0.0.9", "0.1.0"))

    def test_major_bump(self):
        self.assertTrue(Updater._version_gt("2.0.0", "1.9.9"))


class TestAssetPicker(unittest.TestCase):
    def _assets(self, names):
        return [{"name": n, "browser_download_url": f"http://example.com/{n}"}
                for n in names]

    def test_windows_prefers_windows_zip(self):
        assets = self._assets(["ai-helper-linux.tar.gz",
                                "ai-helper-windows.zip",
                                "ai-helper-macos.tar.gz"])
        with patch("ai_helper.updater.platform.system", return_value="Windows"):
            url, name = Updater._pick_asset(assets)
        self.assertIn("windows", name.lower())

    def test_linux_prefers_linux_tarball(self):
        assets = self._assets(["ai-helper-linux.tar.gz",
                                "ai-helper-windows.zip"])
        with patch("ai_helper.updater.platform.system", return_value="Linux"):
            url, name = Updater._pick_asset(assets)
        self.assertIn("linux", name.lower())

    def test_empty_assets(self):
        url, name = Updater._pick_asset([])
        self.assertEqual(url, "")
        self.assertEqual(name, "")


class TestUpdaterCheck(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.download_dir = Path(self._tmp.name) / "updates"

    def tearDown(self):
        self._tmp.cleanup()

    def _make_release(self, tag="v0.2.0") -> dict:
        return {
            "tag_name": tag,
            "html_url": "https://github.com/test/releases/latest",
            "body": "Release notes",
            "published_at": "2026-01-01T00:00:00Z",
            "assets": [
                {"name": "ai-helper-windows.zip",
                 "browser_download_url": "http://example.com/ai-helper-windows.zip"},
            ],
        }

    def _mock_urlopen(self, release_data: dict):
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(release_data).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return patch("ai_helper.updater.urllib.request.urlopen", return_value=mock_resp)

    def test_update_available(self):
        u = Updater(current_version="0.1.0", download_dir=self.download_dir)
        with self._mock_urlopen(self._make_release("v0.2.0")):
            info = u.check()
        self.assertTrue(info.update_available)
        self.assertEqual(info.latest_version, "0.2.0")
        self.assertEqual(info.current_version, "0.1.0")

    def test_no_update_available(self):
        u = Updater(current_version="0.2.0", download_dir=self.download_dir)
        with self._mock_urlopen(self._make_release("v0.2.0")):
            info = u.check()
        self.assertFalse(info.update_available)

    def test_check_handles_network_error(self):
        import urllib.error  # noqa: PLC0415
        u = Updater(current_version="0.1.0", download_dir=self.download_dir)
        with patch("ai_helper.updater.urllib.request.urlopen",
                   side_effect=urllib.error.URLError("connection refused")):
            info = u.check()
        self.assertFalse(info.update_available)
        self.assertNotEqual(info.error, "")

    def test_update_info_str_available(self):
        info = UpdateInfo(
            current_version="0.1.0", latest_version="0.2.0",
            update_available=True, release_url="http://example.com",
            published_at="2026-01-01",
        )
        self.assertIn("0.2.0", str(info))

    def test_update_info_str_up_to_date(self):
        info = UpdateInfo(
            current_version="0.2.0", latest_version="0.2.0",
            update_available=False,
        )
        self.assertIn("up to date", str(info))

    def test_update_info_str_error(self):
        info = UpdateInfo(
            current_version="0.1.0", latest_version="unknown",
            update_available=False, error="network error",
        )
        self.assertIn("failed", str(info))


class TestUpdaterExtract(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_extract_zip(self):
        import zipfile  # noqa: PLC0415
        archive = self.root / "release.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("ai_helper/__init__.py", "# init")
        u = Updater(download_dir=self.root)
        target = u.extract(archive, self.root / "extracted")
        self.assertIsNotNone(target)
        self.assertTrue((target / "ai_helper" / "__init__.py").exists())

    def test_extract_unknown_format_returns_none(self):
        archive = self.root / "unknown.exe"
        archive.write_bytes(b"MZ")
        u = Updater(download_dir=self.root)
        result = u.extract(archive)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
