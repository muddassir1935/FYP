"""Unit tests for the rolling-window FPS counter (pure Python, no torch).

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.rolling_fps import RollingFps


class TestRollingFps(unittest.TestCase):
    def test_first_frame_reports_one(self) -> None:
        self.assertEqual(RollingFps().tick(100.0), 1.0)

    def test_steady_thirty_fps(self) -> None:
        counter = RollingFps()
        rate = 0.0
        for i in range(30):
            rate = counter.tick(100.0 + i / 30.0)
        self.assertAlmostEqual(rate, 30.0, places=6)

    def test_back_to_back_frames_never_spike(self) -> None:
        # The old 1/dt EMA reported 1/0.001 = 1000 FPS for this burst.
        counter = RollingFps()
        counter.tick(100.0)
        self.assertEqual(counter.tick(100.001), 2.0)

    def test_old_frames_expire(self) -> None:
        counter = RollingFps()
        for i in range(30):
            counter.tick(100.0 + i / 30.0)
        # More than a full window later, only the new frame counts.
        self.assertEqual(counter.tick(102.0), 1.0)

    def test_rate_decays_after_frames_stop(self) -> None:
        counter = RollingFps()
        for i in range(30):
            counter.tick(100.0 + i / 30.0)
        # Half the window later, half the frames have expired.
        self.assertAlmostEqual(counter.tick(101.5), 16.0, places=6)


if __name__ == "__main__":
    unittest.main()
