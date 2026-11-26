"""Unit tests for WeaponDetector's predict confidence (no real model).

Covers the ``_predict_conf`` leak: with per-class thresholds Gun 0.65 /
Knife 0.60 and a 0.40 fallback, the model used to run at 0.40, feeding
0.40-0.64 boxes into ByteTrack, which activated/kept tracks alive across
sub-threshold gaps even though the per-class filter dropped them before
the voter. Now, when every class has an explicit per-class threshold,
the model runs at the lowest per-class threshold (0.60) and the
fallback is left out of the minimum.

The real ``__init__`` is exercised with a stubbed ``ultralytics.YOLO``
and a fake ``.pt`` file — no torch, no weights, no cv2 needed.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.weapon_detector import WeaponDetector

GUN = 0
KNIFE = 1
BOX = (100.0, 100.0, 200.0, 260.0)


class _Arr(list):
    """A list with a .tolist(), standing in for a 1-row tensor."""

    def tolist(self) -> list:
        return list(self)


class _FakeBox:
    """Mimics one ultralytics box: .cls[0], .conf[0], .xyxy[0].tolist()."""

    def __init__(self, class_id: int, conf: float) -> None:
        self.cls = [class_id]
        self.conf = [conf]
        self.xyxy = [_Arr([100.0, 100.0, 200.0, 260.0])]


class _FakeBoxes(list):
    """Mimics ultralytics Boxes: iterable boxes plus .id.tolist()."""

    def __init__(self, boxes, ids) -> None:
        super().__init__(boxes)
        self.id = _Arr(ids)


class _FakeYOLO:
    """Mimics ultralytics YOLO; records the conf it was run at."""

    def __init__(self, weights) -> None:
        self.weights = weights
        self.names = {GUN: "Gun", KNIFE: "Knife"}
        self.predict_calls: list[dict] = []
        self.track_calls: list[dict] = []
        self.predict_boxes = []
        self.track_boxes = None

    def predict(self, *, source, conf, iou, imgsz, device, verbose):
        self.predict_calls.append({"conf": conf})
        return [SimpleNamespace(boxes=self.predict_boxes)]

    def track(
        self, *, source, persist, tracker, conf, iou, imgsz, device, verbose
    ):
        self.track_calls.append({"conf": conf, "persist": persist})
        return [SimpleNamespace(boxes=self.track_boxes)]


class PredictConfTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.weights = Path(self._tmp.name) / "best.pt"
        self.weights.write_bytes(b"fake weights bytes")
        # __init__ imports YOLO lazily, so stubbing sys.modules here
        # is enough — the real ultralytics is never touched.
        self._patch = patch.dict(
            sys.modules,
            {"ultralytics": SimpleNamespace(YOLO=_FakeYOLO)},
        )
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._tmp.cleanup()

    def _make_detector(
        self, *, conf_threshold: float = 0.40, conf_thresholds=None
    ) -> WeaponDetector:
        return WeaponDetector(
            self.weights,
            conf_threshold=conf_threshold,
            conf_thresholds=conf_thresholds,
        )

    def test_all_classes_have_thresholds_excludes_fallback(self) -> None:
        det = self._make_detector(
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.65, "Knife": 0.60},
        )
        self.assertEqual(det._predict_conf, 0.60)

    def test_no_per_class_thresholds_preserves_fallback(self) -> None:
        det = self._make_detector(conf_threshold=0.40)
        self.assertEqual(det._predict_conf, 0.40)

    def test_partial_per_class_thresholds_keep_fallback_in_min(self) -> None:
        # Knife has no explicit threshold, so the 0.40 fallback is
        # genuinely needed for it and stays in the minimum.
        det = self._make_detector(
            conf_threshold=0.40, conf_thresholds={"Gun": 0.65}
        )
        self.assertEqual(det._predict_conf, 0.40)
        self.assertEqual(det.threshold_for(GUN), 0.65)
        self.assertEqual(det.threshold_for(KNIFE), 0.40)

    def test_update_thresholds_recomputes_predict_conf(self) -> None:
        det = self._make_detector(conf_threshold=0.40)
        self.assertEqual(det._predict_conf, 0.40)
        det.update_thresholds(
            conf_thresholds={"Gun": 0.65, "Knife": 0.60}
        )
        self.assertEqual(det._predict_conf, 0.60)

    def test_update_thresholds_partial_keeps_fallback(self) -> None:
        det = self._make_detector(
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.65, "Knife": 0.60},
        )
        self.assertEqual(det._predict_conf, 0.60)
        # Dropping Knife's explicit threshold brings the fallback
        # back into the minimum.
        det.update_thresholds(conf_thresholds={"Gun": 0.65})
        self.assertEqual(det._predict_conf, 0.40)

    def test_sub_threshold_box_never_reaches_tracker(self) -> None:
        # The leak, closed: with per-class 0.65/0.60, a 0.50 box must
        # never even be offered to ByteTrack, so track() must be
        # called with conf=0.60 — a real model would filter the 0.50
        # box at predict time. The stub ignores conf (so the test can
        # observe the call), which also proves the post-track
        # per-class filter still drops the box before the voter.
        det = self._make_detector(
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.65, "Knife": 0.60},
        )
        det._model.track_boxes = _FakeBoxes([_FakeBox(GUN, 0.50)], [1])
        tracked = det.track(object())
        self.assertEqual(len(det._model.track_calls), 1)
        self.assertEqual(det._model.track_calls[0]["conf"], 0.60)
        self.assertEqual(tracked, [])

    def test_detect_runs_at_predict_conf(self) -> None:
        # detect() uses _predict_conf, not the per-class values: with
        # all classes covered, no keepable box (>= its per-class
        # threshold) is pre-filtered, because each is >= 0.60.
        det = self._make_detector(
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.65, "Knife": 0.60},
        )
        det._model.predict_boxes = [
            _FakeBox(GUN, 0.50),    # below Gun 0.65 -> dropped
            _FakeBox(GUN, 0.70),    # kept
            _FakeBox(KNIFE, 0.62),  # kept
        ]
        kept = det.detect(object())
        self.assertEqual(det._model.predict_calls[0]["conf"], 0.60)
        self.assertEqual(
            [(d.class_id, d.confidence) for d in kept],
            [(GUN, 0.70), (KNIFE, 0.62)],
        )


if __name__ == "__main__":
    unittest.main()
