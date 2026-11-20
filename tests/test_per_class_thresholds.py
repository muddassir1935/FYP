"""Unit tests for per-class confidence thresholds (pure Python).

Covers WeaponDetector's threshold resolution and per-class filtering
with a fake YOLO model — no torch, no weights, no cv2 needed.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

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


class _FakeModel:
    """Mimics ultralytics YOLO.predict; records the conf it was given."""

    def __init__(self, boxes) -> None:
        self._boxes = boxes
        self.last_conf = None

    def predict(self, *, source, conf, iou, imgsz, device, verbose):
        self.last_conf = conf
        return [SimpleNamespace(boxes=self._boxes)]


def make_detector(
    boxes,
    *,
    conf_threshold: float = 0.40,
    conf_thresholds=None,
) -> WeaponDetector:
    """Build a WeaponDetector without __init__ (no weights, no torch)."""
    detector = WeaponDetector.__new__(WeaponDetector)
    detector.conf_threshold = conf_threshold
    detector.iou_threshold = 0.45
    detector.imgsz = 640
    detector.device = None
    detector.class_names = {GUN: "Gun", KNIFE: "Knife"}
    detector.conf_thresholds = detector._resolve_thresholds(
        conf_thresholds or {}
    )
    detector._predict_conf = min(
        [conf_threshold] + list(detector.conf_thresholds.values())
    )
    detector._model = _FakeModel(boxes)
    return detector


class TestResolveThresholds(unittest.TestCase):
    def test_name_keys_resolve_to_ids(self) -> None:
        det = make_detector([], conf_thresholds={"Gun": 0.55, "Knife": 0.25})
        self.assertEqual(det.conf_thresholds, {GUN: 0.55, KNIFE: 0.25})

    def test_int_keys_pass_through(self) -> None:
        det = make_detector([], conf_thresholds={GUN: 0.60})
        self.assertEqual(det.conf_thresholds, {GUN: 0.60})

    def test_unknown_name_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_detector([], conf_thresholds={"Rifle": 0.5})

    def test_unknown_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            make_detector([], conf_thresholds={7: 0.5})

    def test_out_of_range_raises(self) -> None:
        for bad in (0.0, 1.5, -0.1):
            with self.assertRaises(ValueError, msg=f"conf={bad}"):
                make_detector([], conf_thresholds={"Gun": bad})


class TestThresholdFor(unittest.TestCase):
    def test_per_class_wins_over_fallback(self) -> None:
        det = make_detector(
            [], conf_threshold=0.40, conf_thresholds={"Gun": 0.55}
        )
        self.assertEqual(det.threshold_for(GUN), 0.55)

    def test_fallback_for_unlisted_class(self) -> None:
        det = make_detector(
            [], conf_threshold=0.40, conf_thresholds={"Gun": 0.55}
        )
        self.assertEqual(det.threshold_for(KNIFE), 0.40)


class TestDetectPerClassFiltering(unittest.TestCase):
    def test_keeps_only_boxes_above_their_class_threshold(self) -> None:
        boxes = [
            _FakeBox(GUN, 0.50),    # below Gun 0.55 -> dropped
            _FakeBox(GUN, 0.60),    # above Gun 0.55 -> kept
            _FakeBox(KNIFE, 0.20),  # below Knife 0.25 -> dropped
            _FakeBox(KNIFE, 0.30),  # above Knife 0.25 -> kept
        ]
        det = make_detector(
            boxes,
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.55, "Knife": 0.25},
        )
        kept = det.detect(object())
        self.assertEqual(
            [(d.class_id, d.confidence) for d in kept],
            [(GUN, 0.60), (KNIFE, 0.30)],
        )

    def test_predict_runs_at_the_lowest_threshold(self) -> None:
        det = make_detector(
            [],
            conf_threshold=0.40,
            conf_thresholds={"Gun": 0.55, "Knife": 0.25},
        )
        det.detect(object())
        # The model must see everything above the lowest per-class
        # threshold, or Knife candidates would be pre-filtered away.
        self.assertEqual(det._model.last_conf, 0.25)

    def test_no_boxes_returns_empty(self) -> None:
        det = make_detector(None, conf_thresholds={"Gun": 0.55})
        self.assertEqual(det.detect(object()), [])


if __name__ == "__main__":
    unittest.main()
