"""Unit tests for the demo web UI runner (no model, no camera).

Covers listings, config validation, and snapshot handling. Real runs
(start → frames → stop, pause → stop, worker-error propagation) are
covered by tests/test_demo_integration.py with a mocked detector; the
Playwright walkthrough still lives outside the repo at
~/workspace/sentinel-dataset-v2/walkthrough_demo_ui.py.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.backend.runner import DemoRunner, RunConfig

try:
    from helpers import CV2_AVAILABLE, make_tiny_video
except ImportError:  # pragma: no cover - helpers missing
    CV2_AVAILABLE = False
    make_tiny_video = None  # type: ignore[assignment]


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
        self.logs = root / "logs"
        self.runner = DemoRunner(
            videos_dir=self.videos,
            weights_dir=self.weights,
            snapshots_dir=self.snaps,
            save_video_dir=root / "rec",
            log_dir=self.logs,
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
            log_dir=self.logs,
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

    def test_clear_snapshots_resets_run_start_baseline(self) -> None:
        # status() reports len(snapshots) - _snapshots_at_start: without
        # the reset, clearing mid-run would silence gallery updates +
        # the siren for the next N threats (one per snapshot that
        # existed when the run started).
        self.snaps.mkdir(exist_ok=True)
        for i in range(3):
            (self.snaps / f"threat_Gun_20261121_12000{i}_t{i}.jpg").write_bytes(b"x")
        self.runner._snapshots_at_start = 3
        self.assertEqual(self.runner.clear_snapshots(), 3)
        self.assertEqual(self.runner._snapshots_at_start, 0)
        # A snapshot saved after the clear counts from zero again.
        (self.snaps / "threat_Knife_20261121_130000_t9.jpg").write_bytes(b"x")
        self.assertEqual(self.runner.status().snapshots, 1)


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

    def test_negative_webcam_index_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.runner._resolve_source(
                _cfg(source_type="webcam", source="-1")
            )

    def test_weights_dot_and_dotdot_rejected(self) -> None:
        for bad in (".", ".."):
            with self.assertRaises(ValueError):
                self.runner._resolve_weights(bad)

    @unittest.skipIf(
        sys.platform == "win32",
        "creating a symlink needs elevated privilege on Windows "
        "(WinError 1314)",
    )
    def test_weights_symlink_escape_rejected(self) -> None:
        outside = Path(self._tmp.name) / "outside.pt"
        outside.write_bytes(b"x")
        link = self.weights / "link.pt"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            self.runner._resolve_weights("link.pt")

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
        # The fallback must be recorded, not silent.
        warning = self.runner.status().warning
        self.assertIsNotNone(warning)
        self.assertIn("Intel GPU compile failed", warning)
        self.assertIn("running on CPU", warning)


class TestPauseResume(RunnerTestCase):
    def test_pause_and_resume_state_machine_and_timer(self) -> None:
        from unittest.mock import MagicMock

        # Idle cannot pause or resume.
        with self.assertRaises(RuntimeError):
            self.runner.pause()
        with self.assertRaises(RuntimeError):
            self.runner.resume()

        stream = MagicMock()
        worker = MagicMock()
        worker.latest.return_value = ([], [], 0.0)
        self.runner._state = "running"
        self.runner._started_at = time.time() - 5.0
        self.runner._stream = stream
        self.runner._worker = worker

        paused_info = self.runner.pause()
        self.assertEqual(paused_info.state, "paused")
        stream.pause.assert_called_once()
        worker.pause.assert_called_once()

        # Starting a new run while paused is rejected.
        (self.weights / "best.pt").write_bytes(b"x")
        (self.videos / "clip.mp4").write_bytes(b"x")
        with self.assertRaises(RuntimeError):
            self.runner.start(_cfg())

        # Elapsed timer stays frozen while paused.
        t_paused_1 = self.runner.status().elapsed_s
        time.sleep(0.08)
        t_paused_2 = self.runner.status().elapsed_s
        self.assertAlmostEqual(t_paused_1, t_paused_2, places=3)

        resumed_info = self.runner.resume()
        self.assertEqual(resumed_info.state, "running")
        stream.resume.assert_called_once()
        worker.resume.assert_called_once()

        # Stopping while paused works cleanly.
        self.runner.pause()
        stopped = self.runner.stop()
        self.assertEqual(stopped.state, "stopped")
        self.assertEqual(self.runner.state, "stopped")


class TestDeleteUploadedVideo(RunnerTestCase):
    def test_delete_uploaded_video_success(self) -> None:
        uploads = self.videos / "uploads"
        uploads.mkdir()
        target = uploads / "dup.mp4"
        target.write_bytes(b"fake")
        deleted = self.runner.delete_uploaded_video("uploads/dup.mp4")
        self.assertEqual(deleted, "uploads/dup.mp4")
        self.assertFalse(target.exists())

    def test_delete_rejects_non_uploads_and_traversal(self) -> None:
        (self.videos / "keep.mp4").write_bytes(b"fake")
        for bad in ("keep.mp4", "../escape.mp4", "uploads/../keep.mp4", "/uploads/x.mp4", ""):
            with self.assertRaises(ValueError):
                self.runner.delete_uploaded_video(bad)
        self.assertTrue((self.videos / "keep.mp4").is_file())

    def test_delete_missing_uploaded_video_raises_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.runner.delete_uploaded_video("uploads/ghost.mp4")

    def test_delete_conflict_when_run_uses_video(self) -> None:
        uploads = self.videos / "uploads"
        uploads.mkdir()
        target = uploads / "active.mp4"
        target.write_bytes(b"fake")
        self.runner._state = "paused"
        self.runner._config = _cfg(source="uploads/active.mp4")
        with self.assertRaises(RuntimeError):
            self.runner.delete_uploaded_video("uploads/active.mp4")
        self.assertTrue(target.is_file())


class TestStopRacingStart(RunnerTestCase):
    """Stop landing in the window between state="running" and the
    thread starts used to join a never-started thread (RuntimeError)
    and wedge the run in "stopping" forever."""

    def test_teardown_with_never_started_threads_does_not_raise(self) -> None:
        import threading
        from unittest.mock import MagicMock

        from src.pipeline import InferenceWorker, TrackClassVoter

        worker = InferenceWorker(
            MagicMock(),
            MagicMock(),
            TrackClassVoter(),
            cv2_module=MagicMock(),
            tracker="bytetrack.yaml",
            snapshots_dir=self.snaps,
            log_dir=self.logs,
        )
        self.runner._worker = worker
        self.runner._display_thread = threading.Thread(target=lambda: None)
        self.runner._state = "running"
        info = self.runner.stop()  # must not raise RuntimeError
        self.assertEqual(info.state, "stopped")
        self.assertEqual(self.runner.state, "stopped")

    @unittest.skipUnless(CV2_AVAILABLE, "cv2 not installed")
    def test_stop_called_during_start_does_not_wedge(self) -> None:
        from unittest.mock import MagicMock, patch

        from src.pipeline import InferenceWorker as RealWorker

        assert make_tiny_video is not None
        (self.weights / "best.pt").write_bytes(b"fake")
        make_tiny_video(self.videos / "clip.mp4")

        stop_results: list = []

        def worker_factory(*args, **kwargs):
            worker = RealWorker(*args, **kwargs)
            orig_start = worker.start

            def start_once():
                # Stop lands exactly in the race window: state is
                # "running", the worker thread is not started yet.
                stop_results.append(self.runner.stop())
                return orig_start()

            worker.start = start_once  # type: ignore[method-assign]
            return worker

        with (
            patch(
                "src.backend.runner.WeaponDetector",
                side_effect=lambda *a, **k: MagicMock(),
            ),
            patch("src.backend.runner.InferenceWorker", side_effect=worker_factory),
        ):
            info = self.runner.start(_cfg())

        self.assertEqual(len(stop_results), 1)
        self.assertEqual(stop_results[0].state, "stopped")
        # start() returned without raising and the run is not wedged
        # in "stopping".
        self.assertEqual(info.state, "stopped")
        self.assertEqual(self.runner.state, "stopped")
        worker_thread = self.runner._worker
        display_thread = self.runner._display_thread
        if worker_thread is not None:
            worker_thread.join(timeout=5)
            self.assertFalse(worker_thread.is_alive())
        if display_thread is not None:
            display_thread.join(timeout=5)
            self.assertFalse(display_thread.is_alive())


class TestRecordingFilenameCollision(RunnerTestCase):
    """Two runs planned in the same second must not overwrite each
    other's recording (same-second stamp collision). No video needed:
    the planning helper is deterministic given existing files."""

    def test_same_second_runs_get_distinct_recording_paths(self) -> None:
        rec = Path(self._tmp.name) / "rec"
        rec.mkdir()
        stamp = "20261126_120000"

        first = DemoRunner._unique_recording_path(rec, stamp, True)
        self.assertEqual(first.name, "demo_20261126_120000_raw.mp4")
        first.touch()  # first run's recording now exists

        second = DemoRunner._unique_recording_path(rec, stamp, True)
        self.assertEqual(second.name, "demo_20261126_120000_raw_1.mp4")
        self.assertNotEqual(first, second)
        second.touch()

        third = DemoRunner._unique_recording_path(rec, stamp, True)
        self.assertEqual(third.name, "demo_20261126_120000_raw_2.mp4")

    def test_annotated_recording_collides_independently(self) -> None:
        rec = Path(self._tmp.name) / "rec"
        rec.mkdir()
        stamp = "20261126_120000"
        (rec / "demo_20261126_120000.mp4").touch()
        got = DemoRunner._unique_recording_path(rec, stamp, False)
        self.assertEqual(got.name, "demo_20261126_120000_1.mp4")


if __name__ == "__main__":
    unittest.main()
