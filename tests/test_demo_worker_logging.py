"""Unit tests for per-detection JSONL logging in the inference worker.

No model, no camera: a fake detector + fake stream drive the real
InferenceWorker, and the JSONL file it writes is read back from disk.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.demo_worker import InferenceWorker, _DetectionLog
from src.pipeline.track_voting import TrackClassVoter
from src.pipeline.weapon_detector import Detection


def _det(class_id: int, track_id: int, conf: float = 0.9) -> tuple[int, Detection]:
    names = {0: "Gun", 1: "Knife"}
    return (
        track_id,
        Detection(
            class_id=class_id,
            class_name=names[class_id],
            confidence=conf,
            bbox=(10.0, 10.0, 40.0, 40.0),
        ),
    )


class FakeDetector:
    """Stand-in for WeaponDetector with real weights-path attributes."""

    def __init__(self, weights: Path, script: list[list[tuple]]) -> None:
        self.weights = weights
        self.conf_threshold = 0.40
        self.conf_thresholds = {0: 0.50, 1: 0.35}
        self.class_names = {0: "Gun", 1: "Knife"}
        self._script = script
        self._calls = 0

    def track(self, frame, *, tracker="bytetrack.yaml"):
        self._calls += 1
        index = min(self._calls - 1, len(self._script) - 1)
        return self._script[index]

    def reset_tracker(self) -> None:
        pass


class FakeDetectorNoAttrs:
    """Detector without weights/threshold attributes (header must cope)."""

    def __init__(self, script: list[list[tuple]]) -> None:
        self._script = script
        self._calls = 0

    def track(self, frame, *, tracker="bytetrack.yaml"):
        self._calls += 1
        return self._script[min(self._calls - 1, len(self._script) - 1)]

    def reset_tracker(self) -> None:
        pass


class FakeStream:
    """FrameSource stand-in yielding a fixed number of packets."""

    def __init__(self, n_frames: int, frame: object) -> None:
        self._n = n_frames
        self._frame = frame
        self._i = 0
        self.ended = False
        # Mirrors FrameSource.frames_seen (counted per successful read).
        self.frames_seen = 0

    def read(self):
        if self._i < self._n:
            self._i += 1
            self.frames_seen += 1
            return (self._i, time.monotonic() + 0.05 * self._i, self._frame)
        self.ended = True
        return None


_UNSET = object()  # _run_worker default: "use the tmp dir", distinct from None


class LoggingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        import numpy as np

        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.logs = root / "logs"
        self.snaps = root / "snaps"
        self.snaps.mkdir()
        self.frame = np.zeros((48, 64, 3), dtype=np.uint8)
        # A fake .pt file with known bytes, so the header hash is exact.
        self.weights = root / "best.pt"
        self.weights.write_bytes(b"fake weights bytes")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run_worker(
        self,
        detector,
        *,
        n_frames: int = 2,
        voter: TrackClassVoter | None = None,
        log_dir: Path | None = _UNSET,  # type: ignore[assignment]
        snapshots_dir: Path | None = _UNSET,  # type: ignore[assignment]
        source: str | None = None,
        recording: str | None = None,
    ) -> InferenceWorker:
        stream = FakeStream(n_frames, self.frame)
        worker = InferenceWorker(
            stream,
            detector,
            voter or TrackClassVoter(min_hits=3),
            cv2_module=MagicMock(),
            tracker="bytetrack.yaml",
            snapshots_dir=self.snaps if snapshots_dir is _UNSET else snapshots_dir,
            log_dir=self.logs if log_dir is _UNSET else log_dir,
            source=source,
            recording=recording,
        )
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        return worker

    def _read_lines(self, worker: InferenceWorker) -> list[dict]:
        assert worker.log_path is not None
        text = worker.log_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def test_header_carries_weights_hash_and_thresholds(self) -> None:
        detector = FakeDetector(
            self.weights, script=[[_det(0, 1)], [_det(0, 1)]]
        )
        worker = self._run_worker(detector)
        self.assertIsNone(worker.error)
        lines = self._read_lines(worker)
        header = lines[0]
        self.assertEqual(header["type"], "header")
        self.assertEqual(header["weights"], str(self.weights))
        self.assertEqual(
            header["weights_sha256"],
            hashlib.sha256(b"fake weights bytes").hexdigest(),
        )
        self.assertIsNone(header["weights_note"])
        self.assertEqual(header["model_label"], "best")
        self.assertEqual(header["conf_threshold"], 0.40)
        self.assertEqual(
            header["conf_thresholds"], {"Gun": 0.50, "Knife": 0.35}
        )
        self.assertEqual(header["min_hits"], 3)
        self.assertEqual(header["confirm_window_s"], 1.0)
        self.assertEqual(header["min_span_s"], 0.4)
        self.assertEqual(header["tracker"], "bytetrack.yaml")
        self.assertIsNone(header["source"])
        self.assertIsNone(header["recording"])
        self.assertIn("started_at", header)

    def test_header_carries_source_and_recording(self) -> None:
        detector = FakeDetector(self.weights, script=[[_det(0, 1)]])
        worker = self._run_worker(
            detector,
            n_frames=1,
            source="clip.mp4",
            recording="/tmp/rec/demo_20261126_160000_raw.mp4",
        )
        self.assertIsNone(worker.error)
        lines = self._read_lines(worker)
        header = lines[0]
        self.assertEqual(header["source"], "clip.mp4")
        self.assertEqual(
            header["recording"], "/tmp/rec/demo_20261126_160000_raw.mp4"
        )

    def test_footer_written_with_correct_counts(self) -> None:
        detector = FakeDetector(
            self.weights,
            script=[[_det(0, 1)], [_det(0, 1)], [_det(0, 1)]],
        )
        worker = self._run_worker(detector, n_frames=3)
        self.assertIsNone(worker.error)
        lines = self._read_lines(worker)
        # Header + 3 detections + closing run summary.
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0]["type"], "header")
        footer = lines[-1]
        self.assertEqual(footer["type"], "run_summary")
        self.assertEqual(footer["frames_processed"], 3)
        self.assertEqual(footer["frames_seen_by_capture"], 3)
        self.assertEqual(footer["started_at"], lines[0]["started_at"])
        self.assertIn("ended_at", footer)
        self.assertGreaterEqual(footer["duration_s"], 0)

    def test_footer_duration_excludes_paused_time(self) -> None:
        # The footer must agree with RunInfo.elapsed_s: paused time is
        # not run time, or false-alarms/hour comes out too low after
        # any pause during a Session A run.
        log = _DetectionLog(
            self.logs,
            detector=FakeDetector(self.weights, script=[]),
            voter=TrackClassVoter(min_hits=3),
            tracker="bytetrack.yaml",
        )
        time.sleep(0.02)
        log.pause()
        time.sleep(0.3)
        log.resume()
        log.close(frames_processed=0, frames_seen_by_capture=0)
        assert log.path is not None
        lines = [
            json.loads(line)
            for line in log.path.read_text().splitlines()
            if line.strip()
        ]
        footer = lines[-1]
        self.assertEqual(footer["type"], "run_summary")
        # ~0.02 s of live time; the 0.3 s pause must not be in it.
        self.assertLess(footer["duration_s"], 0.2)

    def test_footer_duration_excludes_in_progress_pause(self) -> None:
        # Closing while still paused (e.g. stop() on a paused run)
        # must count the open pause interval too.
        log = _DetectionLog(
            self.logs,
            detector=FakeDetector(self.weights, script=[]),
            voter=TrackClassVoter(min_hits=3),
            tracker="bytetrack.yaml",
        )
        time.sleep(0.02)
        log.pause()
        time.sleep(0.3)
        log.close(frames_processed=0, frames_seen_by_capture=0)
        assert log.path is not None
        lines = [
            json.loads(line)
            for line in log.path.read_text().splitlines()
            if line.strip()
        ]
        self.assertLess(lines[-1]["duration_s"], 0.2)

    def test_worker_pause_forwards_to_log(self) -> None:
        # InferenceWorker.pause()/resume() must reach the log, or the
        # footer fix above never sees a pause.
        stream = FakeStream(0, self.frame)
        worker = InferenceWorker(
            stream,
            FakeDetector(self.weights, script=[]),
            TrackClassVoter(min_hits=3),
            cv2_module=MagicMock(),
            tracker="bytetrack.yaml",
            snapshots_dir=self.snaps,
            log_dir=self.logs,
        )
        assert worker._log is not None
        worker.pause()
        time.sleep(0.05)
        worker.resume()
        self.assertGreater(worker._log._paused_total_s, 0.03)
        worker.stop()

    def test_footer_zero_counts_when_worker_never_started(self) -> None:
        # stop() on a never-started worker closes the log: the footer
        # is still written (a run that never started processed 0 frames).
        stream = FakeStream(0, self.frame)
        worker = InferenceWorker(
            stream,
            FakeDetector(self.weights, script=[]),
            TrackClassVoter(min_hits=3),
            cv2_module=MagicMock(),
            tracker="bytetrack.yaml",
            snapshots_dir=self.snaps,
            log_dir=self.logs,
        )
        worker.stop()
        assert worker.log_path is not None
        lines = [
            json.loads(line)
            for line in worker.log_path.read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(len(lines), 2)
        footer = lines[-1]
        self.assertEqual(footer["type"], "run_summary")
        self.assertEqual(footer["frames_processed"], 0)
        self.assertEqual(footer["frames_seen_by_capture"], 0)

    def test_one_line_per_detection_with_fields(self) -> None:
        detector = FakeDetector(
            self.weights,
            script=[[_det(0, 1)], [_det(0, 1), _det(1, 2, conf=0.8)]],
        )
        # min_span_s=0: this test checks the detection lines, not the
        # span rule — with the default 0.4 s span the 0.05 s-apart
        # fake frames would never confirm.
        worker = self._run_worker(
            detector,
            voter=TrackClassVoter(min_hits=1, min_span_s=0),
            snapshots_dir=None,
        )
        self.assertIsNone(worker.error)
        lines = self._read_lines(worker)
        self.assertEqual(lines[0]["type"], "header")
        self.assertEqual(lines[-1]["type"], "run_summary")
        dets = lines[1:-1]
        self.assertEqual(len(dets), 3)  # 1 on frame 1, 2 on frame 2
        self.assertEqual([d["frame_index"] for d in dets], [1, 2, 2])
        self.assertEqual([d["track_id"] for d in dets], [1, 1, 2])
        for d in dets:
            self.assertEqual(d["type"], "detection")
            self.assertIsInstance(d["timestamp"], str)
            self.assertIsInstance(d["monotonic_s"], float)
            self.assertIn(d["raw_class"], ("Gun", "Knife"))
            self.assertIn(d["voted_class"], ("Gun", "Knife"))
            self.assertIsInstance(d["confidence"], float)
            self.assertEqual(len(d["bbox"]), 4)
            self.assertTrue(d["confirmed"])  # min_hits=1: every sighting
        self.assertEqual(dets[2]["raw_class"], "Knife")
        self.assertEqual(dets[2]["voted_class"], "Knife")
        self.assertEqual(dets[2]["confidence"], 0.8)
        self.assertEqual(dets[2]["bbox"], [10.0, 10.0, 40.0, 40.0])

    def test_unconfirmed_detections_logged_with_flag_false(self) -> None:
        detector = FakeDetector(
            self.weights, script=[[_det(0, 1)], [_det(0, 1)]]
        )
        worker = self._run_worker(detector)  # min_hits=3, only 2 sightings
        self.assertIsNone(worker.error)
        lines = self._read_lines(worker)
        self.assertEqual(len(lines), 4)  # header + 2 detections + footer
        self.assertEqual(lines[-1]["type"], "run_summary")
        self.assertTrue(all(d["confirmed"] is False for d in lines[1:-1]))

    def test_openvino_folder_header_hashes_bin_and_xml(self) -> None:
        # The .bin (weight VALUES) identifies the model; the .xml only
        # describes structure, so weights_sha256 is now the .bin hash.
        ov_dir = Path(self._tmp.name) / "v3_openvino_model"
        ov_dir.mkdir()
        xml = ov_dir / "v3.xml"
        xml.write_bytes(b"<net/>")
        (ov_dir / "v3.bin").write_bytes(b"binary weights")
        detector = FakeDetector(ov_dir, script=[])
        log = _DetectionLog(
            self.logs,
            detector=detector,
            voter=TrackClassVoter(),
            tracker="bytetrack.yaml",
        )
        try:
            assert log.path is not None
            header = json.loads(log.path.read_text().splitlines()[0])
        finally:
            log.close()
        self.assertEqual(
            header["weights_sha256"],
            hashlib.sha256(b"binary weights").hexdigest(),
        )
        self.assertEqual(
            header["weights_xml_sha256"], hashlib.sha256(b"<net/>").hexdigest()
        )
        assert header["weights_note"] is not None
        self.assertIn("hashed v3.bin", header["weights_note"])
        self.assertEqual(header["model_label"], "v3")

    def test_openvino_folder_without_xml_hashes_bin(self) -> None:
        ov_dir = Path(self._tmp.name) / "empty_openvino_model"
        ov_dir.mkdir()
        (ov_dir / "model.bin").write_bytes(b"binary weights")
        detector = FakeDetector(ov_dir, script=[])
        log = _DetectionLog(
            self.logs,
            detector=detector,
            voter=TrackClassVoter(),
            tracker="bytetrack.yaml",
        )
        try:
            assert log.path is not None
            header = json.loads(log.path.read_text().splitlines()[0])
        finally:
            log.close()
        # The .bin alone still identifies the model (no .xml needed).
        self.assertEqual(
            header["weights_sha256"],
            hashlib.sha256(b"binary weights").hexdigest(),
        )
        self.assertIsNone(header["weights_xml_sha256"])
        assert header["weights_note"] is not None
        self.assertIn("no .xml", header["weights_note"])

    def test_openvino_folder_without_bin_hashes_xml_only(self) -> None:
        ov_dir = Path(self._tmp.name) / "xml_only_openvino_model"
        ov_dir.mkdir()
        (ov_dir / "v3.xml").write_bytes(b"<net/>")
        detector = FakeDetector(ov_dir, script=[])
        log = _DetectionLog(
            self.logs,
            detector=detector,
            voter=TrackClassVoter(),
            tracker="bytetrack.yaml",
        )
        try:
            assert log.path is not None
            header = json.loads(log.path.read_text().splitlines()[0])
        finally:
            log.close()
        # No .bin: the model is NOT identified (weights_sha256 null),
        # but the .xml hash is still recorded.
        self.assertIsNone(header["weights_sha256"])
        self.assertEqual(
            header["weights_xml_sha256"], hashlib.sha256(b"<net/>").hexdigest()
        )
        assert header["weights_note"] is not None
        self.assertIn("no .bin", header["weights_note"])

    def test_detector_without_weights_attrs_records_nulls(self) -> None:
        detector = FakeDetectorNoAttrs(script=[[_det(0, 1)]])
        worker = self._run_worker(detector, n_frames=1)
        self.assertIsNone(worker.error)
        lines = self._read_lines(worker)
        header = lines[0]
        self.assertIsNone(header["weights"])
        self.assertIsNone(header["weights_sha256"])
        assert header["weights_note"] is not None
        self.assertIn("unavailable", header["weights_note"])
        self.assertIsNone(header["model_label"])
        self.assertIsNone(header["conf_threshold"])
        self.assertIsNone(header["conf_thresholds"])
        # header + 1 detection + footer
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[-1]["type"], "run_summary")

    def test_log_dir_none_disables_logging(self) -> None:
        detector = FakeDetector(
            self.weights, script=[[_det(0, 1)], [_det(0, 1)]]
        )
        worker = self._run_worker(detector, log_dir=None)
        self.assertIsNone(worker.error)
        self.assertIsNone(worker.log_path)
        self.assertFalse(self.logs.exists())

    def test_unwritable_log_dir_does_not_stop_detection(self) -> None:
        # log_dir is an existing FILE, so mkdir fails: logging must
        # disable itself with a warning while detection runs on.
        blocker = Path(self._tmp.name) / "not-a-dir"
        blocker.write_text("x")
        detector = FakeDetector(
            self.weights, script=[[_det(0, 1)], [_det(0, 1)]]
        )
        worker = self._run_worker(detector, log_dir=blocker)
        self.assertIsNone(worker.error)
        self.assertIsNone(worker.log_path)
        dets, _confirmed, _fps = worker.latest()
        self.assertEqual(len(dets), 1)

    def test_log_file_created_under_given_dir_only(self) -> None:
        detector = FakeDetector(
            self.weights, script=[[_det(0, 1)], [_det(0, 1)]]
        )
        worker = self._run_worker(detector)
        self.assertIsNone(worker.error)
        assert worker.log_path is not None
        self.assertEqual(worker.log_path.parent, self.logs)
        self.assertTrue(worker.log_path.name.startswith("detections_"))
        self.assertTrue(worker.log_path.name.endswith(".jsonl"))
        self.assertEqual(list(self.logs.glob("*.jsonl")), [worker.log_path])


if __name__ == "__main__":
    unittest.main()
