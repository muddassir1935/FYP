"""Unit tests for FrameSource EOF / loop behaviour (fake cv2) and for
the InferenceWorker's loop-generation reset.

* EOF: a video file at EOF must end promptly (``"eof"``), not after
  ``max_consecutive_failures`` x 0.02 s of retries; a mid-stream
  failure must still retry and report ``"read_failures"``.
* Loop: every successful rewind increments ``FrameSource.loop_generation``.
* Worker: a generation change resets the detector tracker and the
  class voter (a generation that never changes resets nothing).

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import InferenceWorker, TrackClassVoter
from src.pipeline.capture import FrameSource
from src.pipeline.weapon_detector import Detection


def _wait_for(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class _FakeCapture:
    """VideoCapture stand-in: a fixed frame list, then EOF/failure."""

    CAP_PROP_FPS = 5
    CAP_PROP_FRAME_COUNT = 7
    CAP_PROP_POS_FRAMES = 1

    def __init__(self, frames, *, total: int | None = None) -> None:
        self._frames = list(frames)
        self._total = len(self._frames) if total is None else total
        self._pos = 0
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
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


class TestEofEndsPromptly(unittest.TestCase):
    def test_eof_does_not_wait_out_the_retry_budget(self) -> None:
        # The old loop retried 50 x 0.02 s = 1.0 s at every file EOF
        # before checking _is_at_eof; with the default budget the
        # source must now end well inside that second.
        cap = _FakeCapture([object(), object(), object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource("clip.mp4", pace_file=False)
            assert src.max_consecutive_failures == 50
            started = time.monotonic()
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                elapsed = time.monotonic() - started
                self.assertEqual(src.end_reason, "eof")
                self.assertIsNone(src.end_error)
                self.assertLess(elapsed, 0.5, f"EOF stall: {elapsed:.2f}s")
            finally:
                src.release()

    def test_midstream_failure_still_retries_then_reports_read_failures(
        self,
    ) -> None:
        # The file claims 10 frames but only delivers 1: not EOF, so
        # the retry budget still applies and the reason stays
        # "read_failures" (the short-circuit must not fire here).
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

    def test_live_source_never_short_circuits_to_eof(self) -> None:
        cap = _FakeCapture([object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(0, pace_file=False, max_consecutive_failures=3)
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                self.assertEqual(src.end_reason, "read_failures")
            finally:
                src.release()


class TestLoopGeneration(unittest.TestCase):
    def test_rewind_increments_loop_generation(self) -> None:
        cap = _FakeCapture([object(), object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource(
                "clip.mp4", loop_file=True, pace_file=False,
                max_consecutive_failures=3,
            )
            try:
                self.assertEqual(src.loop_generation, 0)
                src.start()
                self.assertTrue(
                    _wait_for(lambda: src.loop_generation >= 2, timeout=5),
                    f"generation never advanced: {src.loop_generation}",
                )
                self.assertFalse(src.ended)
                self.assertGreaterEqual(src.frames_seen, 4)
            finally:
                src.release()

    def test_no_loop_means_no_generation(self) -> None:
        cap = _FakeCapture([object(), object()])
        with patch.dict(sys.modules, {"cv2": _fake_cv2(cap)}):
            src = FrameSource("clip.mp4", pace_file=False)
            try:
                src.start()
                self.assertTrue(_wait_for(lambda: src.ended, timeout=5))
                self.assertEqual(src.loop_generation, 0)
                self.assertEqual(src.end_reason, "eof")
            finally:
                src.release()


# -- InferenceWorker loop-generation reset ---------------------------------


class _LoopingFakeStream:
    """FrameSource stand-in whose loop_generation flips mid-run."""

    def __init__(
        self, n_frames: int, frame: object, *, flip_after: int | None
    ) -> None:
        self._n = n_frames
        self._frame = frame
        self._flip_after = flip_after
        self._i = 0
        self.ended = False
        self.loop_generation = 0

    def read(self):
        if self._i < self._n:
            self._i += 1
            if self._flip_after is not None and self._i == self._flip_after:
                # Rewind happened between the previous packet and this
                # one: the next packet is frame 1 of the new loop.
                self.loop_generation += 1
            return (self._i, time.monotonic(), self._frame)
        self.ended = True
        return None


class _CountingDetector:
    """Always sees one Gun on track 1; counts tracker resets."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.track_calls = 0

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

    def reset_tracker(self) -> None:
        self.reset_calls += 1


class TestWorkerLoopReset(unittest.TestCase):
    def _run_worker(self, detector, voter, stream):
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        worker = InferenceWorker(
            stream,
            detector,
            voter,
            cv2_module=object(),
            tracker="bytetrack.yaml",
            snapshots_dir=None,
            log_dir=None,
        )
        worker.start()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())
        self.assertIsNone(worker.error)
        return worker

    def test_generation_change_resets_tracker_and_voter(self) -> None:
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        detector = _CountingDetector()
        voter = TrackClassVoter(min_hits=1, min_span_s=0)
        # 5 packets; the generation flips before packet 3, so the
        # voter must be cleared mid-run: only packets 3-5 count
        # towards its post-reset frame counter.
        stream = _LoopingFakeStream(5, frame, flip_after=3)
        self._run_worker(detector, voter, stream)
        self.assertEqual(detector.reset_calls, 1)
        self.assertEqual(voter._frame, 3)
        self.assertEqual(voter.active_tracks, 1)

    def test_no_generation_change_resets_nothing(self) -> None:
        import numpy as np

        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        detector = _CountingDetector()
        voter = TrackClassVoter(min_hits=1, min_span_s=0)
        stream = _LoopingFakeStream(5, frame, flip_after=None)
        self._run_worker(detector, voter, stream)
        self.assertEqual(detector.reset_calls, 0)
        self.assertEqual(voter._frame, 5)
        self.assertEqual(voter.active_tracks, 1)

    def test_voter_reset_clears_state(self) -> None:
        # Direct unit check of the reset the worker relies on: votes
        # and hit history from the old clip must not survive.
        voter = TrackClassVoter(min_hits=1, min_span_s=0)
        det = Detection(
            class_id=0, class_name="Gun", confidence=0.9,
            bbox=(10.0, 10.0, 40.0, 40.0),
        )
        voter.update([(1, det)], now=100.0)
        self.assertEqual(voter.active_tracks, 1)
        voter.reset()
        self.assertEqual(voter.active_tracks, 0)
        self.assertEqual(voter._frame, 0)


if __name__ == "__main__":
    unittest.main()
