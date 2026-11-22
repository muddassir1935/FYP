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

    def test_start_sets_starting_state_before_loading_detector(self) -> None:
        from unittest.mock import patch

        (self.weights / "best.pt").write_bytes(b"fake")
        (self.videos / "clip.mp4").write_bytes(b"fake")
        observed_states: list[str] = []

        def fake_detector(*_args, **_kwargs):
            observed_states.append(self.runner.state)
            # Concurrent start while starting must be rejected.
            with self.assertRaises(RuntimeError):
                self.runner.start(_cfg())
            raise RuntimeError("boom during load")

        with patch("src.backend.runner.WeaponDetector", side_effect=fake_detector):
            with self.assertRaises(RuntimeError):
                self.runner.start(_cfg())

        self.assertEqual(observed_states, ["starting"])
        self.assertEqual(self.runner.state, "error")

    def test_detector_warmup_and_cache_reused_across_runs(self) -> None:
        from unittest.mock import MagicMock, patch

        (self.weights / "best.pt").write_bytes(b"fake")
        (self.videos / "clip.mp4").write_bytes(b"fake")

        created: list[MagicMock] = []

        def fake_detector(*_args, **_kwargs):
            det = MagicMock()
            created.append(det)
            return det

        with patch("src.backend.runner.WeaponDetector", side_effect=fake_detector):
            det1 = self.runner._get_or_load_detector(
                self.weights / "best.pt", _cfg(conf_gun=0.44, conf_knife=0.43)
            )
            det2 = self.runner._get_or_load_detector(
                self.weights / "best.pt", _cfg(conf_gun=0.55, conf_knife=0.35)
            )
            # Changing imgsz must bust the cache and warm up a new instance.
            det3 = self.runner._get_or_load_detector(
                self.weights / "best.pt", _cfg(imgsz=416)
            )

        self.assertIs(det1, det2)
        self.assertIsNot(det1, det3)
        self.assertEqual(len(created), 2)
        det1.warmup.assert_called_once_with(tracker="bytetrack.yaml")
        det1.update_thresholds.assert_called_once_with(
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.55, "Knife": 0.35},
        )
        det1.reset_tracker.assert_called_once()
        det3.warmup.assert_called_once_with(tracker="bytetrack.yaml")

    def test_openvino_gpu_failure_falls_back_to_cpu(self) -> None:
        from unittest.mock import MagicMock, patch

        ov_dir = self.weights / "v3_openvino_model"
        ov_dir.mkdir()
        devices_tried: list[str | None] = []

        def fake_detector(_path, *, device=None, **_kwargs):
            devices_tried.append(device)
            det = MagicMock()
            if device == "intel:gpu":
                det.warmup.side_effect = RuntimeError("iGPU compile failed")
            return det

        with (
            patch.object(
                DemoRunner, "_detect_openvino_device", return_value="intel:gpu"
            ),
            patch("src.backend.runner.WeaponDetector", side_effect=fake_detector),
        ):
            det = self.runner._get_or_load_detector(
                ov_dir, _cfg(weights_id="v3_openvino_model")
            )

        self.assertEqual(devices_tried, ["intel:gpu", None])
        self.assertIsNotNone(det)
        self.assertEqual(self.runner._cached_detector_key[2], None)


if __name__ == "__main__":
    unittest.main()
