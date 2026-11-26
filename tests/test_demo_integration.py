"""Integration + unit tests for the demo pipeline's untested region.

* Real ``DemoRunner`` runs against tiny generated videos with a mocked
  detector (start -> frames -> stop, pause -> stop, worker-error
  propagation, clean EOF vs zero-frame failure, MJPEG stream reattach).
* ``FrameSource`` unit tests with a fake cv2 (EOF vs disconnect vs a
  reader thread that raises).
* ``InferenceWorker`` unit tests (inference error, snapshot save
  failure, snapshot exception).
* ``draw_annotations`` confirmed-match by track_id.
* CLI helpers (``resolve_thresholds``, ``--record-raw`` default,
  ``detect_openvino_device``) and the API schema defaults.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import CV2_AVAILABLE, make_tiny_video

from src.backend.runner import DemoRunner, RunConfig
from src.pipeline import InferenceWorker, TrackClassVoter
from src.pipeline.annotate import draw_annotations
from src.pipeline.capture import FrameSource
from src.pipeline.thresholds import CONF_GUN, CONF_KNIFE, DEFAULT_CONF
from src.pipeline.track_voting import TrackedDetection
from src.pipeline.weapon_detector import Detection, detect_openvino_device

try:
    from fastapi.testclient import TestClient

    from src.backend import create_app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    import httpx
    import uvicorn

    HTTP_AVAILABLE = True
except ImportError:
    HTTP_AVAILABLE = False


def _cfg(**overrides) -> RunConfig:
    base = {
        "source_type": "video",
        "source": "clip.mp4",
        "weights_id": "best.pt",
    }
    base.update(overrides)
    return RunConfig(**base)


class FakeDetector:
    """Stand-in for WeaponDetector: no model, instant, controllable."""

    def __init__(self, *, fail_after: int | None = None) -> None:
        self.fail_after = fail_after
        self.track_calls = 0
        self.warmup_called = False

    def track(self, frame, *, tracker="bytetrack.yaml"):
        self.track_calls += 1
        if self.fail_after is not None and self.track_calls >= self.fail_after:
            raise RuntimeError("boom in track")
        return []

    def warmup(self, *, tracker="bytetrack.yaml"):
        self.warmup_called = True

    def reset_tracker(self):
        pass

    def update_thresholds(self, **kwargs):
        pass


def _wait_for(predicate, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class RealRunTestCase(unittest.TestCase):
    """Real runner runs (mocked detector only) against a tiny video."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.videos = root / "videos"
        self.weights = root / "weights"
        self.snaps = root / "snaps"
        self.videos.mkdir()
        self.weights.mkdir()
        (self.weights / "best.pt").write_bytes(b"fake")
        self.runner = DemoRunner(
            videos_dir=self.videos,
            weights_dir=self.weights,
            snapshots_dir=self.snaps,
            save_video_dir=root / "rec",
        )

    def tearDown(self) -> None:
        try:
            self.runner.stop()
        except Exception:
            pass
        self._tmp.cleanup()

    def _start(self, detector: FakeDetector, **cfg_overrides):
        with patch(
            "src.backend.runner.WeaponDetector",
            side_effect=lambda *a, **k: detector,
        ):
            return self.runner.start(_cfg(**cfg_overrides))


@unittest.skipUnless(CV2_AVAILABLE, "cv2 not installed")
class TestRealRuns(RealRunTestCase):
    def test_start_frames_stop(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=30, fps=10)
        info = self._start(FakeDetector())
        self.assertEqual(info.state, "running")
        self.assertTrue(
            _wait_for(lambda: self.runner.status().frames > 5),
            f"frames never advanced: {self.runner.status()}",
        )
        # The MJPEG stream has a real JPEG to serve.
        _seq, jpeg = self.runner.latest_frame()
        self.assertIsNotNone(jpeg)
        assert jpeg is not None
        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        stopped = self.runner.stop()
        self.assertEqual(stopped.state, "stopped")
        self.assertGreater(stopped.frames, 5)
        self.assertIsNone(stopped.error)

    def test_pause_freezes_frames_then_resume_and_stop(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=60, fps=10)
        self._start(FakeDetector())
        self.assertTrue(_wait_for(lambda: self.runner.status().frames > 3))
        paused = self.runner.pause()
        self.assertEqual(paused.state, "paused")
        frames_1 = self.runner.status().frames
        time.sleep(0.5)
        frames_2 = self.runner.status().frames
        self.assertEqual(frames_1, frames_2)
        resumed = self.runner.resume()
        self.assertEqual(resumed.state, "running")
        self.assertTrue(
            _wait_for(lambda: self.runner.status().frames > frames_2)
        )
        stopped = self.runner.stop()
        self.assertEqual(stopped.state, "stopped")

    def test_worker_error_fails_the_run(self) -> None:
        # The worker dies on its 3rd frame while the video keeps
        # playing: the run must fail, not sit "Running" on frozen
        # verdicts (MUST-22).
        make_tiny_video(self.videos / "clip.mp4", n_frames=60, fps=10)
        self._start(FakeDetector(fail_after=3))
        self.assertTrue(
            _wait_for(lambda: self.runner.state == "error"),
            f"run never failed: {self.runner.status()}",
        )
        error = self.runner.status().error or ""
        self.assertIn("Inference failed", error)
        self.assertIn("boom in track", error)

    def test_short_video_ends_cleanly_on_eof(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=8, fps=10)
        self._start(FakeDetector())
        self.assertTrue(
            _wait_for(lambda: self.runner.state == "stopped"),
            f"run never stopped: {self.runner.status()}",
        )
        info = self.runner.status()
        self.assertEqual(info.frames, 8)
        self.assertIsNone(info.error)

    def test_zero_frame_video_fails_not_clean_stop(self) -> None:
        # A source OpenCV opens but never yields a frame (corrupt
        # file, dead camera): the run must fail with an error, not
        # end as a clean "Stopped" with 0 frames (MUST-23).
        (self.videos / "clip.mp4").write_bytes(b"not a video")

        class DeadCapture:
            def isOpened(self):
                return True

            def read(self):
                return False, None

            def get(self, prop):
                return 0.0

            def grab(self):
                return True

            def release(self):
                pass

        with patch("cv2.VideoCapture", return_value=DeadCapture()):
            self._start(FakeDetector())
            self.assertTrue(
                _wait_for(lambda: self.runner.state == "error"),
                f"run never failed: {self.runner.status()}",
            )
        error = self.runner.status().error or ""
        self.assertIn("Could not read video", error)
        self.assertIn("read_failures", error)

    def test_save_video_records_raw_by_default(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=20, fps=10)
        self._start(FakeDetector(), save_video=True)
        self.assertTrue(_wait_for(lambda: self.runner.status().frames > 5))
        stopped = self.runner.stop()
        self.assertIsNotNone(stopped.saved_video)
        assert stopped.saved_video is not None
        self.assertTrue(stopped.saved_video.endswith("_raw.mp4"))
        self.assertTrue(Path(stopped.saved_video).is_file())

    def test_save_video_annotated_on_request(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=20, fps=10)
        self._start(FakeDetector(), save_video=True, record_raw=False)
        self.assertTrue(_wait_for(lambda: self.runner.status().frames > 5))
        stopped = self.runner.stop()
        self.assertIsNotNone(stopped.saved_video)
        assert stopped.saved_video is not None
        self.assertFalse(stopped.saved_video.endswith("_raw.mp4"))
        self.assertTrue(Path(stopped.saved_video).is_file())

    def test_save_video_writes_timestamp_sidecar(self) -> None:
        # One sidecar line per recorded frame, frame_index 1..N in
        # order with strictly increasing monotonic timestamps — the
        # join key for the detection log (frame_index) and true timing.
        make_tiny_video(self.videos / "clip.mp4", n_frames=20, fps=10)
        self._start(FakeDetector(), save_video=True)
        self.assertTrue(
            _wait_for(lambda: self.runner.state == "stopped"),
            f"run never stopped: {self.runner.status()}",
        )
        stopped = self.runner.status()
        self.assertEqual(stopped.frames, 20)
        self.assertIsNotNone(stopped.saved_video)
        assert stopped.saved_video is not None
        sidecar_path = Path(f"{stopped.saved_video}.timestamps.jsonl")
        self.assertTrue(sidecar_path.is_file())
        lines = [
            json.loads(line)
            for line in sidecar_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 20)
        self.assertEqual(
            [line["frame_index"] for line in lines],
            list(range(1, 21)),
        )
        ts = [line["ts"] for line in lines]
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))


@unittest.skipUnless(
    FASTAPI_AVAILABLE and CV2_AVAILABLE and HTTP_AVAILABLE,
    "fastapi/cv2/httpx/uvicorn not installed",
)
class TestStreamReattach(unittest.TestCase):
    """The MJPEG stream serves frames, and re-attaches after a drop.

    Runs a real uvicorn server in a thread: TestClient buffers a
    streaming response until the generator ends (verified: a trivial
    3-chunk stream arrives all at once after 3 s), so it cannot test
    incremental MJPEG delivery at all.
    """

    def setUp(self) -> None:
        import socket
        import threading

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.videos = root / "videos"
        self.weights = root / "weights"
        self.snaps = root / "snaps"
        self.videos.mkdir()
        self.weights.mkdir()
        (self.weights / "best.pt").write_bytes(b"fake")
        make_tiny_video(self.videos / "clip.mp4", n_frames=120, fps=10)
        app = create_app(
            videos_dir=self.videos,
            weights_dir=self.weights,
            snapshots_dir=self.snaps,
            save_video_dir=root / "rec",
        )
        self.runner = app.state.runner
        self._patcher = patch(
            "src.backend.runner.WeaponDetector",
            side_effect=lambda *a, **k: FakeDetector(),
        )
        self._patcher.start()
        # Pick a free loopback port, then serve the app on it.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        config = uvicorn.Config(
            app, host="127.0.0.1", port=port, log_level="warning"
        )
        self.server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self.server.run, daemon=True
        )
        self._thread.start()
        self.client = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=30.0,
            trust_env=False,
        )
        healthy = _wait_for(
            lambda: self._is_healthy(), timeout=15.0, interval=0.2
        )
        if not healthy:
            self.tearDown()
            raise RuntimeError("test server did not come up")

    def _is_healthy(self) -> bool:
        try:
            return self.client.get("/api/health").status_code == 200
        except httpx.RequestError:
            return False

    def tearDown(self) -> None:
        try:
            self.client.post("/api/run/stop")
        except Exception:
            pass
        try:
            self.runner.stop()
        except Exception:
            pass
        self.server.should_exit = True
        self._thread.join(timeout=10)
        self.client.close()
        self._patcher.stop()
        self._tmp.cleanup()

    def _read_one_frame(self) -> bool:
        with self.client.stream("GET", "/api/stream") as stream:
            buf = b""
            for chunk in stream.iter_bytes():
                buf += chunk
                if b"\xff\xd8" in buf and b"\xff\xd9" in buf:
                    return True
                if len(buf) > 5_000_000:
                    return False
        return False

    def test_stream_attaches_twice_during_one_run(self) -> None:
        res = self.client.post(
            "/api/run/start",
            json={"source_type": "video", "source": "clip.mp4",
                  "weights_id": "best.pt"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertTrue(_wait_for(lambda: self.runner.status().frames > 2))
        self.assertTrue(self._read_one_frame())
        # Drop the connection and re-attach: the second stream must
        # serve frames too (the frontend retries this way on error).
        self.assertTrue(self._read_one_frame())
        stopped = self.client.post("/api/run/stop").json()
        self.assertEqual(stopped["state"], "stopped")


# -- FrameSource with a fake cv2 ------------------------------------------


class _FakeCapture:
    """VideoCapture stand-in: a fixed frame list, then EOF/failure."""

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, frames, *, total: int | None = None,
                 raise_on_read: Exception | None = None) -> None:
        self._frames = list(frames)
        self._total = len(self._frames) if total is None else total
        self._raise_on_read = raise_on_read
        self._pos = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._raise_on_read is not None:
            raise self._raise_on_read
        if self._pos < len(self._frames):
            frame = self._frames[self._pos]
            self._pos += 1
            return True, frame
        return False, None

    def get(self, prop):
        if prop == self.CAP_PROP_FPS:
            return 10.0
        if prop == self.CAP_PROP_FRAME_COUNT:
            return float(self._total)
        if prop == self.CAP_PROP_POS_FRAMES:
            return float(self._pos)
        return 0.0

    def set(self, prop, value):
        if prop == self.CAP_PROP_POS_FRAMES:
            self._pos = int(value)
            return True
        return False

    def grab(self):
        return True

    def release(self):
        self._opened = False


def _fake_cv2(capture: _FakeCapture) -> types.ModuleType:
    mod = types.ModuleType("cv2")
    mod.CAP_PROP_FPS = _FakeCapture.CAP_PROP_FPS
    mod.CAP_PROP_FRAME_COUNT = _FakeCapture.CAP_PROP_FRAME_COUNT
    mod.CAP_PROP_POS_FRAMES = _FakeCapture.CAP_PROP_POS_FRAMES
    mod.VideoCapture = lambda source: capture
    return mod


class TestFrameSourceFakeCv2(unittest.TestCase):
    def test_eof_reason_on_clean_file_end(self) -> None:
        cap = _FakeCapture([object(), object(), object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(
                "clip.mp4", pace_file=False, max_consecutive_failures=3
            )
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                self.assertEqual(src.end_reason, "eof")
                self.assertIsNone(src.end_error)
                self.assertFalse(src.is_live)
            finally:
                src.release()

    def test_read_failures_when_file_dies_midstream(self) -> None:
        # The file claims 10 frames but only delivers 1: not EOF.
        cap = _FakeCapture([object()], total=10)
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(
                "clip.mp4", pace_file=False, max_consecutive_failures=3
            )
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                self.assertEqual(src.end_reason, "read_failures")
            finally:
                src.release()

    def test_live_source_never_reports_eof(self) -> None:
        cap = _FakeCapture([object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(0, pace_file=False, max_consecutive_failures=3)
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                self.assertEqual(src.end_reason, "read_failures")
                self.assertTrue(src.is_live)
            finally:
                src.release()

    def test_reader_exception_sets_ended_and_error(self) -> None:
        cap = _FakeCapture(
            [], raise_on_read=RuntimeError("capture exploded")
        )
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(
                "clip.mp4", pace_file=False, max_consecutive_failures=3
            )
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                self.assertEqual(src.end_reason, "read_failures")
                self.assertEqual(src.end_error, "capture exploded")
            finally:
                src.release()

    def test_loop_rewinds(self) -> None:
        cap = _FakeCapture([object(), object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(
                "clip.mp4", loop_file=True, pace_file=False,
                max_consecutive_failures=3,
            )
            try:
                src.start()
                self.assertTrue(
                    _wait_for(
                        lambda: (src.read() or (0,))[0] >= 4, timeout=5
                    )
                )
                self.assertFalse(src.ended)
            finally:
                src.release()


# -- InferenceWorker --------------------------------------------------------


class _FakeStream:
    """FrameSource stand-in yielding a fixed number of packets."""

    def __init__(self, n_frames: int, frame: object) -> None:
        self._n = n_frames
        self._frame = frame
        self._i = 0
        self.ended = False

    def read(self):
        if self._i < self._n:
            self._i += 1
            return (self._i, time.monotonic(), self._frame)
        self.ended = True
        return None


class _FakeCv2Draw:
    """Minimal cv2 for draw_annotations + snapshot saving."""

    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self, *, imwrite_result=True, imwrite_raises=None) -> None:
        self.imwrite_result = imwrite_result
        self.imwrite_raises = imwrite_raises
        self.imwrite_calls: list[str] = []
        self.rectangles: list[tuple] = []

    def rectangle(self, img, pt1, pt2, color, thickness):
        self.rectangles.append((pt1, pt2, color, thickness))

    def getTextSize(self, text, font, scale, thickness):
        return ((10, 10), 2)

    def putText(self, *args, **kwargs):
        pass

    def imwrite(self, path, img):
        self.imwrite_calls.append(path)
        if self.imwrite_raises is not None:
            raise self.imwrite_raises
        return self.imwrite_result


class _TrackingDetector(FakeDetector):
    """Always sees one Gun on track 1 (for snapshot tests)."""

    def track(self, frame, *, tracker="bytetrack.yaml"):
        self.track_calls += 1
        return [
            (
                1,
                Detection(
                    class_id=0, class_name="Gun", confidence=0.9,
                    bbox=(10.0, 10.0, 40.0, 40.0),
                ),
            )
        ]


class TestInferenceWorker(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.snaps = Path(self._tmp.name) / "snaps"
        self.snaps.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_worker(self, detector, *, n_frames=3, cv2=None, voter=None):
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        stream = _FakeStream(n_frames, frame)
        worker = InferenceWorker(
            stream,
            detector,
            # min_span_s=0: these tests exercise snapshot saving, not
            # the span rule — confirmation must happen on the first
            # sighting, and the fake frames are milliseconds apart.
            voter or TrackClassVoter(min_hits=1, min_span_s=0),
            cv2_module=cv2 or _FakeCv2Draw(),
            tracker="bytetrack.yaml",
            snapshots_dir=self.snaps,
        )
        worker.start()
        worker.join(timeout=10)
        return worker

    def test_inference_error_sets_error_and_exits(self) -> None:
        worker = self._run_worker(FakeDetector(fail_after=2), n_frames=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.error, "boom in track")

    def test_snapshot_write_failure_warns_and_retries(self) -> None:
        # imwrite() returning False must NOT consume the cooldown:
        # the next confirmed frame retries the save.
        cv2 = _FakeCv2Draw(imwrite_result=False)
        worker = self._run_worker(
            _TrackingDetector(), n_frames=3, cv2=cv2
        )
        self.assertIsNone(worker.error)
        self.assertGreaterEqual(len(cv2.imwrite_calls), 2)

    def test_snapshot_write_success_consumes_cooldown(self) -> None:
        cv2 = _FakeCv2Draw(imwrite_result=True)
        worker = self._run_worker(
            _TrackingDetector(), n_frames=3, cv2=cv2
        )
        self.assertIsNone(worker.error)
        self.assertEqual(len(cv2.imwrite_calls), 1)

    def test_snapshot_exception_sets_worker_error(self) -> None:
        cv2 = _FakeCv2Draw(
            imwrite_raises=RuntimeError("disk gone")
        )
        worker = self._run_worker(
            _TrackingDetector(), n_frames=3, cv2=cv2
        )
        self.assertFalse(worker.is_alive())
        assert worker.error is not None
        self.assertIn("Snapshot save failed", worker.error)


class TestDrawAnnotationsTrackId(unittest.TestCase):
    def test_rebuilt_detection_still_draws_confirmed(self) -> None:
        import numpy as np

        cv2 = _FakeCv2Draw()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        original = Detection(
            class_id=0, class_name="Gun", confidence=0.9,
            bbox=(10.0, 10.0, 40.0, 40.0), track_id=7,
        )
        confirmed = [
            TrackedDetection(
                detection=original, track_id=7, hits=3, vote_share=1.0,
                raw_class_id=0, confirmed=True,
            )
        ]
        # A rebuilt Detection: same track, different object identity.
        rebuilt = Detection(
            class_id=0, class_name="Gun", confidence=0.9,
            bbox=(10.0, 10.0, 40.0, 40.0), track_id=7,
        )
        self.assertIsNot(rebuilt, original)
        draw_annotations(cv2, frame, [rebuilt], confirmed)
        boxes = [
            (color, thickness)
            for (_p1, _p2, color, thickness) in cv2.rectangles
        ]
        # Confirmed = thick red box; id()-based matching would have
        # drawn thin grey.
        self.assertIn(((0, 0, 255), 3), boxes)

    def test_detection_without_track_id_draws_unconfirmed(self) -> None:
        import numpy as np

        cv2 = _FakeCv2Draw()
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        det = Detection(
            class_id=0, class_name="Gun", confidence=0.9,
            bbox=(10.0, 10.0, 40.0, 40.0),
        )
        self.assertIsNone(det.track_id)
        draw_annotations(cv2, frame, [det], [])
        boxes = [
            (color, thickness)
            for (_p1, _p2, color, thickness) in cv2.rectangles
        ]
        self.assertIn(((170, 170, 170), 1), boxes)


# -- CLI helpers + schemas --------------------------------------------------


def _load_cli_module():
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "inference"
        / "run_weapon_demo.py"
    )
    spec = importlib.util.spec_from_file_location("run_weapon_demo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCliHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = _load_cli_module()

    def test_resolve_thresholds_defaults(self) -> None:
        import argparse

        args = argparse.Namespace(conf=None, conf_gun=None, conf_knife=None)
        self.assertEqual(
            self.cli.resolve_thresholds(args),
            (CONF_GUN, CONF_KNIFE, DEFAULT_CONF),
        )

    def test_resolve_thresholds_single_conf(self) -> None:
        import argparse

        args = argparse.Namespace(conf=0.5, conf_gun=None, conf_knife=None)
        self.assertEqual(
            self.cli.resolve_thresholds(args), (0.5, 0.5, 0.5)
        )

    def test_resolve_thresholds_per_class_wins(self) -> None:
        import argparse

        args = argparse.Namespace(conf=0.5, conf_gun=0.6, conf_knife=None)
        self.assertEqual(
            self.cli.resolve_thresholds(args), (0.6, 0.5, 0.5)
        )

    def test_record_raw_defaults_true(self) -> None:
        with patch.object(sys, "argv", ["run_weapon_demo.py"]):
            self.assertTrue(self.cli.parse_args().record_raw)
        with patch.object(
            sys, "argv", ["run_weapon_demo.py", "--no-record-raw"]
        ):
            self.assertFalse(self.cli.parse_args().record_raw)


@unittest.skipUnless(CV2_AVAILABLE, "cv2 not installed")
class TestCliMain(unittest.TestCase):
    """main() end to end with a mocked detector and stubbed GUI calls
    (no display in this environment, so imshow/waitKey are stubbed)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cli = _load_cli_module()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.videos = root / "videos"
        self.videos.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_cli(self, argv: list[str], detector: FakeDetector) -> int:
        import cv2

        with (
            patch.object(sys, "argv", argv),
            patch.object(self.cli, "WeaponDetector", return_value=detector),
            patch.object(cv2, "imshow"),
            patch.object(cv2, "waitKey", return_value=-1),
            patch.object(cv2, "getWindowProperty", return_value=1),
            patch.object(cv2, "destroyAllWindows"),
        ):
            return self.cli.main()

    def _argv(self, **extra) -> list[str]:
        argv = [
            "run_weapon_demo.py",
            "--source", str(self.videos / "clip.mp4"),
            "--weights", str(self.videos / "best.pt"),
            "--no-snapshots",
        ]
        for key, value in extra.items():
            argv += [f"--{key.replace('_', '-')}", str(value)]
        return argv

    def test_clean_run_returns_zero_and_warms_up(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=8, fps=10)
        (self.videos / "best.pt").write_bytes(b"fake")
        detector = FakeDetector()
        rc = self._run_cli(self._argv(), detector)
        self.assertEqual(rc, 0)
        self.assertTrue(detector.warmup_called)

    def test_worker_error_returns_nonzero(self) -> None:
        make_tiny_video(self.videos / "clip.mp4", n_frames=60, fps=10)
        (self.videos / "best.pt").write_bytes(b"fake")
        detector = FakeDetector(fail_after=2)
        rc = self._run_cli(self._argv(), detector)
        self.assertEqual(rc, 1)

    def test_zero_frame_video_returns_nonzero(self) -> None:
        (self.videos / "clip.mp4").write_bytes(b"not a video")
        (self.videos / "best.pt").write_bytes(b"fake")

        class DeadCapture:
            def isOpened(self):
                return True

            def read(self):
                return False, None

            def get(self, prop):
                return 0.0

            def grab(self):
                return True

            def release(self):
                pass

        import cv2

        with patch.object(cv2, "VideoCapture", return_value=DeadCapture()):
            rc = self._run_cli(self._argv(), FakeDetector())
        self.assertEqual(rc, 1)

    def test_missing_weights_returns_nonzero_without_traceback(self) -> None:
        # No WeaponDetector patch here: the real constructor must fail
        # cleanly (FileNotFoundError -> clean error + exit 1).
        import io
        from contextlib import redirect_stderr

        err = io.StringIO()
        with (
            patch.object(
                sys, "argv",
                ["run_weapon_demo.py", "--weights", "/tmp/nope/best.pt",
                 "--source", "0", "--no-snapshots"],
            ),
            redirect_stderr(err),
        ):
            rc = self.cli.main()
        self.assertEqual(rc, 1)
        self.assertIn("ERROR", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())


class TestDetectOpenvinoDevice(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.weights = root / "weights"
        self.weights.mkdir()
        (self.weights / "best.pt").write_bytes(b"x")
        self.ov_dir = self.weights / "v3_openvino_model"
        self.ov_dir.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    @staticmethod
    def _fake_ov(devices: list[str]) -> types.ModuleType:
        mod = types.ModuleType("openvino")

        class FakeCore:
            def __init__(self) -> None:
                self.available_devices = devices

        mod.Core = FakeCore  # type: ignore[attr-defined]
        return mod

    def test_pt_weights_never_select_gpu(self) -> None:
        self.assertIsNone(
            detect_openvino_device(self.weights / "best.pt")
        )

    def test_openvino_folder_with_gpu_selects_xpu(self) -> None:
        with patch.dict(
            sys.modules, {"openvino": self._fake_ov(["CPU", "GPU.0"])}
        ):
            self.assertEqual(detect_openvino_device(self.ov_dir), "xpu")

    def test_openvino_folder_without_gpu_selects_none(self) -> None:
        with patch.dict(
            sys.modules, {"openvino": self._fake_ov(["CPU"])}
        ):
            self.assertIsNone(detect_openvino_device(self.ov_dir))


class TestSchemaDefaults(unittest.TestCase):
    def test_run_start_defaults_match_thresholds(self) -> None:
        from src.backend.schemas import RunStartRequest

        req = RunStartRequest(
            source_type="video", source="x.mp4", weights_id="best.pt"
        )
        self.assertEqual(req.conf_gun, CONF_GUN)
        self.assertEqual(req.conf_knife, CONF_KNIFE)
        self.assertEqual(req.tracker, "bytetrack.yaml")
        self.assertTrue(req.record_raw)

    def test_run_status_has_warning_and_no_started_at(self) -> None:
        from src.backend.schemas import RunStatus

        self.assertIn("warning", RunStatus.model_fields)
        self.assertNotIn("started_at", RunStatus.model_fields)

    def test_run_config_defaults_match_thresholds(self) -> None:
        cfg = _cfg()
        self.assertEqual(cfg.conf_gun, CONF_GUN)
        self.assertEqual(cfg.conf_knife, CONF_KNIFE)
        self.assertTrue(cfg.record_raw)


if __name__ == "__main__":
    unittest.main()
