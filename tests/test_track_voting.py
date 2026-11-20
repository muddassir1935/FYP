"""Unit tests for track-level class voting (pure Python, no torch).

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.track_voting import TrackClassVoter
from src.pipeline.weapon_detector import Detection

GUN = 0
KNIFE = 1
BOX = (100.0, 100.0, 200.0, 260.0)
NAMES = {GUN: "Gun", KNIFE: "Knife"}


def det(class_id: int = GUN, conf: float = 0.9, bbox=BOX) -> Detection:
    return Detection(
        class_id=class_id,
        class_name=NAMES[class_id],
        confidence=conf,
        bbox=bbox,
    )


class TestConfirmation(unittest.TestCase):
    def test_confirms_on_third_sighting_not_before(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        for frame in range(2):
            out = voter.update([(1, det())])
            self.assertEqual(len(out), 1)
            self.assertFalse(out[0].confirmed, f"frame {frame + 1}")
            self.assertEqual(out[0].hits, frame + 1)
        out = voter.update([(1, det())])
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)
        self.assertEqual(out[0].track_id, 1)

    def test_two_frame_flicker_never_confirms(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det())])
        out = voter.update([(1, det())])
        self.assertFalse(out[0].confirmed)

    def test_three_frame_glimpse_confirms(self) -> None:
        # A brief true detection (3 frames) confirms under tracking,
        # where the 6-of-10 buffer would have dropped it.
        voter = TrackClassVoter(min_hits=3)
        for _ in range(2):
            voter.update([(7, det(KNIFE, 0.8))])
        out = voter.update([(7, det(KNIFE, 0.8))])
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].detection.class_name, "Knife")


class TestVoting(unittest.TestCase):
    def test_single_flip_flop_does_not_change_label(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det(GUN, 0.9))])
        voter.update([(1, det(GUN, 0.9))])
        # One Knife frame at lower confidence must not flip the label.
        out = voter.update([(1, det(KNIFE, 0.5))])
        self.assertEqual(out[0].detection.class_id, GUN)
        self.assertEqual(out[0].raw_class_id, KNIFE)
        self.assertAlmostEqual(out[0].vote_share, 1.8 / 2.3)

    def test_sustained_opposite_labels_eventually_flip(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det(GUN, 0.9))])
        voter.update([(1, det(GUN, 0.9))])
        out = voter.update([(1, det(KNIFE, 0.9))])
        self.assertEqual(out[0].detection.class_id, GUN)
        out = voter.update([(1, det(KNIFE, 0.9))])
        # 1.8 Gun vs 1.8 Knife: tie breaks to the most recent class.
        self.assertEqual(out[0].detection.class_id, KNIFE)
        out = voter.update([(1, det(KNIFE, 0.9))])
        self.assertEqual(out[0].detection.class_id, KNIFE)

    def test_confidence_weighted_not_plain_count(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det(GUN, 0.95))])
        voter.update([(1, det(KNIFE, 0.45))])
        # Two weak Knife votes (0.90 total) must not beat one strong Gun (0.95).
        out = voter.update([(1, det(KNIFE, 0.45))])
        self.assertEqual(out[0].detection.class_id, GUN)

    def test_tracks_vote_independently(self) -> None:
        voter = TrackClassVoter(min_hits=2)
        voter.update([(1, det(GUN, 0.9)), (2, det(KNIFE, 0.9))])
        out = voter.update([(1, det(GUN, 0.9)), (2, det(KNIFE, 0.9))])
        by_id = {t.track_id: t for t in out}
        self.assertEqual(by_id[1].detection.class_id, GUN)
        self.assertEqual(by_id[2].detection.class_id, KNIFE)
        self.assertTrue(all(t.confirmed for t in out))

    def test_voted_label_keeps_current_frame_bbox_and_conf(self) -> None:
        voter = TrackClassVoter(min_hits=2)
        voter.update([(1, det(GUN, 0.9))])
        moved = (120.0, 110.0, 220.0, 270.0)
        out = voter.update([(1, det(KNIFE, 0.55, bbox=moved))])
        self.assertEqual(out[0].detection.class_id, GUN)  # voted
        self.assertEqual(out[0].detection.bbox, moved)  # current frame
        self.assertAlmostEqual(out[0].detection.confidence, 0.55)


class TestLifecycle(unittest.TestCase):
    def test_stale_tracks_are_pruned(self) -> None:
        voter = TrackClassVoter(min_hits=2, max_age=3)
        voter.update([(1, det())])
        self.assertEqual(voter.active_tracks, 1)
        # Three frames with no sighting of track 1 (other track keeps
        # the frame counter moving); age 4 > max_age 3 prunes it.
        for _ in range(4):
            voter.update([(2, det())])
        self.assertEqual(voter.active_tracks, 1)

    def test_reset_clears_votes(self) -> None:
        voter = TrackClassVoter(min_hits=2)
        voter.update([(1, det(GUN, 0.9))])
        voter.reset()
        self.assertEqual(voter.active_tracks, 0)
        out = voter.update([(1, det(KNIFE, 0.9))])
        self.assertEqual(out[0].detection.class_id, KNIFE)
        self.assertEqual(out[0].hits, 1)
        self.assertFalse(out[0].confirmed)

    def test_invalid_params_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrackClassVoter(min_hits=0)
        with self.assertRaises(ValueError):
            TrackClassVoter(max_age=0)


if __name__ == "__main__":
    unittest.main()
