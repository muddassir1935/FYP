"""Unit tests for the demo web UI runner (no model, no camera).

Covers listings, config validation, and snapshot handling. Starting a
real run needs weights + a video and is covered by the headless worker
smoke test and the Playwright walkthrough instead.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backend.runner import DemoRunner, RunConfig


def _cfg(**overrides) -> RunConfig:
    base = {
        "source_type": "video",
        "source": "clip.mp4",
        "weights_id": "best.pt",
    }
    base.update(overrides)
    return RunConfig(**base)


class RunnerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.videos = root / "videos"
        self.weights = root / "weights"
        self.snaps = root / "snaps"
        self.videos.mkdir()
        self.weights.mkdir()
        self.runner = DemoRunner(
            videos_dir=self.videos,
            weights_dir=self.weights,
            snapshots_dir=self.snaps,
            save_video_dir=root / "rec",
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()


class TestListings(RunnerTestCase):
    def test_weights_lists_pt_and_openvino_only(self) -> None:
        (self.weights / "best.pt").write_bytes(b"x")
        (self.weights / "v3_openvino_model").mkdir()
        (self.weights / "notes.txt").write_text("not weights")
        (self.weights / "other_dir").mkdir()
        got = self.runner.list_weights()
        self.assertEqual(
            [(w["id"], w["kind"]) for w in got],
            [("best.pt", "pytorch"), ("v3_openvino_model", "openvino")],
        )

    def test_weights_empty_when_dir_missing(self) -> None:
        runner = DemoRunner(
            videos_dir=self.videos,
            weights_dir=self.weights / "nope",
            snapshots_dir=self.snaps,
        )
        self.assertEqual(runner.list_weights(), [])

    def test_videos_lists_extensions_and_subdirs(self) -> None:
        (self.videos / "a.mp4").write_bytes(b"x")
        (self.videos / "b.AVI").write_bytes(b"x")
        (self.videos / "c.txt").write_text("not a video")
        sub = self.videos / "uploads"
        sub.mkdir()
        (sub / "d.webm").write_bytes(b"x")
        got = self.runner.list_videos()
        self.assertEqual(
            [v["name"] for v in got], ["a.mp4", "b.AVI", "uploads/d.webm"]
        )
        self.assertIsNone(got[0]["duration_s"])  # bad file: no probe, still listed

    def test_snapshots_parsed_newest_first(self) -> None:
        self.snaps.mkdir(exist_ok=True)
        old = self.snaps / "threat_Gun_20261121_120000_t1.jpg"
        new = self.snaps / "threat_Knife_20261121_130000_t2.jpg"
        old.write_bytes(b"x")
        new.write_bytes(b"x")
        (self.snaps / "random.jpg").write_bytes(b"x")
        now = time.time()
        os.utime(old, (now - 100, now - 100))
        os.utime(new, (now, now))
        got = self.runner.list_snapshots()
        self.assertEqual([s["file"] for s in got], [new.name, old.name])
        self.assertEqual(got[0]["class"], "Knife")
        self.assertEqual(got[0]["track"], 2)
        self.assertEqual(got[0]["timestamp"], "20261121_130000")
        self.assertEqual(got[0]["url"], f"/snapshots/{new.name}")

    def test_clear_snapshots_removes_only_threat_files(self) -> None:
        self.snaps.mkdir(exist_ok=True)
        (self.snaps / "threat_Gun_20261121_120000_t1.jpg").write_bytes(b"x")
        (self.snaps / "keep.jpg").write_bytes(b"x")
        self.assertEqual(self.runner.clear_snapshots(), 1)
        self.assertTrue((self.snaps / "keep.jpg").is_file())
        self.assertEqual(self.runner.list_snapshots(), [])


class TestValidation(RunnerTestCase):
    def test_unknown_weights_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_weights("nope.pt")

    def test_weights_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_weights("../secret.pt")

    def test_video_outside_dir_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_source(_cfg(source="../escape.mp4"))

    def test_missing_video_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_source(_cfg(source="ghost.mp4"))

    def test_bad_webcam_index_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_source(
                _cfg(source_type="webcam", source="not-a-number")
            )

    def test_bad_source_type_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_source(_cfg(source_type="rtsp"))

    def test_webcam_index_resolves_to_int(self) -> None:
        self.assertEqual(
            self.runner._resolve_source(_cfg(source_type="webcam", source="0")), 0
        )

    def test_start_with_unknown_weights_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.runner.start(_cfg())
        self.assertEqual(self.runner.state, "idle")

    def test_stop_is_idempotent_when_idle(self) -> None:
        info = self.runner.stop()
        self.assertEqual(info.state, "idle")
        self.assertEqual(self.runner.stop().state, "idle")

    def test_status_idle_shape(self) -> None:
        info = self.runner.status()
        self.assertEqual(info.state, "idle")
        self.assertEqual(info.frames, 0)
        self.assertIsNone(info.error)


if __name__ == "__main__":
    unittest.main()
