"""Unit tests for the FR-05 temporal consistency buffer (pure Python).

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.temporal_buffer import TemporalConsistencyBuffer, bbox_iou
from src.pipeline.weapon_detector import Detection

GUN = 0
KNIFE = 1
BOX = (100.0, 100.0, 200.0, 260.0)
FAR_BOX = (800.0, 600.0, 900.0, 760.0)


def det(class_id: int = GUN, bbox=BOX, conf: float = 0.9) -> Detection:
    return Detection(
        class_id=class_id,
        class_name={GUN: "Gun", KNIFE: "Knife"}[class_id],
        confidence=conf,
        bbox=bbox,
    )


class TestBBoxIoU(unittest.TestCase):
    def test_identical_boxes(self) -> None:
        self.assertAlmostEqual(bbox_iou(BOX, BOX), 1.0)

    def test_disjoint_boxes(self) -> None:
        self.assertEqual(bbox_iou(BOX, FAR_BOX), 0.0)

    def test_half_overlap(self) -> None:
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 0.0, 15.0, 10.0)
        self.assertAlmostEqual(bbox_iou(a, b), 50.0 / 150.0)


class TestConfirmation(unittest.TestCase):
    def setUp(self) -> None:
        self.buffer = TemporalConsistencyBuffer(window=10, required_hits=6)

    def test_confirms_on_sixth_sighting_not_before(self) -> None:
        for frame in range(5):
            self.assertEqual(self.buffer.update([det()]), [], f"frame {frame + 1}")
        confirmed = self.buffer.update([det()])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].hits, 6)
        self.assertEqual(confirmed[0].detection.class_name, "Gun")

    def test_flicker_never_confirms(self) -> None:
        for _ in range(3):
            self.assertEqual(self.buffer.update([det()]), [])
        for _ in range(7):
            self.assertEqual(self.buffer.update([]), [])

    def test_classes_tracked_independently(self) -> None:
        for _ in range(3):
            self.buffer.update([det(GUN), det(KNIFE, FAR_BOX)])
        for _ in range(2):
            self.buffer.update([det(GUN)])
        confirmed = self.buffer.update([det(GUN), det(KNIFE, FAR_BOX)])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].detection.class_id, GUN)

    def test_moving_box_keeps_same_track(self) -> None:
        track_ids = set()
        confirmed_count = 0
        for step in range(8):  # drifts 4 px/frame: IoU stays well above 0.3
            moved = (BOX[0] + 4 * step, BOX[1], BOX[2] + 4 * step, BOX[3])
            for cd in self.buffer.update([det(bbox=moved)]):
                track_ids.add(cd.track_id)
                confirmed_count += 1
        self.assertEqual(track_ids, {1})
        self.assertEqual(confirmed_count, 3)  # frames 6, 7, 8

    def test_distant_object_is_a_separate_track(self) -> None:
        for _ in range(6):
            self.buffer.update([det()])
        # A new object far away must earn its own 6 sightings.
        for _ in range(5):
            self.assertEqual(self.buffer.update([det(bbox=FAR_BOX)]), [])
        confirmed = self.buffer.update([det(bbox=FAR_BOX)])
        self.assertEqual(len(confirmed), 1)
        self.assertEqual(confirmed[0].track_id, 2)

    def test_invalid_configuration_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TemporalConsistencyBuffer(window=5, required_hits=6)


if __name__ == "__main__":
    unittest.main()
