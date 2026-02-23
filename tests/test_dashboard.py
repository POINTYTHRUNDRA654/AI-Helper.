"""Tests for ai_helper.dashboard."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestDashboard(unittest.TestCase):
    """Dashboard uses curses so we mock it completely."""

    def test_import(self):
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        d = Dashboard(poll_interval=1.0)
        self.assertEqual(d.poll_interval, 1.0)

    def test_pct_color_green(self):
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        self.assertEqual(Dashboard._pct_color(50), 1)

    def test_pct_color_yellow(self):
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        self.assertEqual(Dashboard._pct_color(70), 2)

    def test_pct_color_red(self):
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        self.assertEqual(Dashboard._pct_color(90), 3)

    def test_pct_color_boundary_85(self):
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        self.assertEqual(Dashboard._pct_color(85), 3)

    def test_pct_color_boundary_60(self):
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        self.assertEqual(Dashboard._pct_color(60), 2)

    def test_addstr_out_of_bounds_does_not_raise(self):
        import curses  # noqa: PLC0415
        from ai_helper.dashboard import Dashboard  # noqa: PLC0415
        mock_win = MagicMock()
        mock_win.getmaxyx.return_value = (10, 80)
        # row out of bounds
        Dashboard._addstr(mock_win, 100, 0, "text")
        mock_win.addstr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
