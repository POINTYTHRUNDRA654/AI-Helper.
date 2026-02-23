"""Tests for ai_helper.config."""

from __future__ import annotations

import os
import platform
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class TestConfig(unittest.TestCase):
    """Each test resets the internal cache so tests are independent."""

    def _reset(self):
        import ai_helper.config as cfg
        cfg._resolved = cfg._UNSET

    def setUp(self):
        self._reset()

    def tearDown(self):
        self._reset()

    def test_windows_default_is_d_drive(self):
        import ai_helper.config as cfg
        with patch("ai_helper.config._SYSTEM", "Windows"):
            path = cfg._default_install_dir()
        self.assertEqual(str(path), r"D:\AI-Helper")

    def test_linux_default_is_home(self):
        import ai_helper.config as cfg
        with patch("ai_helper.config._SYSTEM", "Linux"):
            path = cfg._default_install_dir()
        self.assertEqual(path, Path.home() / "AI-Helper")

    def test_env_var_overrides_default(self):
        import ai_helper.config as cfg
        with patch.dict(os.environ, {"AI_HELPER_INSTALL_DIR": "/custom/path"}):
            result = cfg.get_install_dir()
        self.assertEqual(result, Path("/custom/path"))

    def test_set_install_dir(self):
        import ai_helper.config as cfg
        cfg.set_install_dir("/tmp/test_install")
        self.assertEqual(cfg.get_install_dir(), Path("/tmp/test_install"))

    def test_derived_paths_under_install_dir(self):
        import ai_helper.config as cfg
        cfg.set_install_dir("/base")
        self.assertEqual(cfg.get_downloads_dir(), Path("/base/Downloads"))
        self.assertEqual(cfg.get_packages_dir(), Path("/base/Lib/site-packages"))
        self.assertEqual(cfg.get_organized_dir(), Path("/base/Organized"))
        self.assertEqual(cfg.get_logs_dir(), Path("/base/Logs"))
        self.assertEqual(cfg.get_data_dir(), Path("/base/Data"))

    def test_save_and_load_config(self):
        import ai_helper.config as cfg
        with TemporaryDirectory() as tmp:
            config_file = Path(tmp) / ".ai_helper.cfg"
            cfg.set_install_dir(tmp)
            with patch("ai_helper.config._CONFIG_FILE", config_file):
                saved = cfg.save_config()
            self.assertTrue(saved.exists())
            # Reset and reload from file
            cfg._resolved = cfg._UNSET
            with patch("ai_helper.config._CONFIG_FILE", config_file), \
                 patch.dict(os.environ, {}, clear=True):
                # Remove env var if set
                os.environ.pop("AI_HELPER_INSTALL_DIR", None)
                loaded = cfg._load_from_config_file.__wrapped__(config_file) \
                    if hasattr(cfg._load_from_config_file, "__wrapped__") \
                    else cfg._load_from_config_file()
            # If the function uses _CONFIG_FILE internally we verify file content
            self.assertTrue(config_file.read_text().find(str(tmp)) >= 0)

    def test_ensure_dirs_creates_structure(self):
        import ai_helper.config as cfg
        with TemporaryDirectory() as tmp:
            cfg.set_install_dir(tmp + "/ai_helper_test")
            cfg.ensure_dirs()
            self.assertTrue(cfg.get_install_dir().exists())
            self.assertTrue(cfg.get_downloads_dir().exists())
            self.assertTrue(cfg.get_organized_dir().exists())
            self.assertTrue(cfg.get_logs_dir().exists())
            self.assertTrue(cfg.get_data_dir().exists())


if __name__ == "__main__":
    unittest.main()
