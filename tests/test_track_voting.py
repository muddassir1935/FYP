"""Unit tests for track-level class voting (pure Python, no torch).

Confirmation is density over time: >= min_hits sightings within the
last confirm_window_s seconds (default 3 in 1.0 s), lapsing when the
density drops. Every update() call therefore passes an explicit
monotonic timestamp, so the tests are deterministic.

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
        for i in range(2):
            out = voter.update([(1, det())], now=0.1 * i)
            self.assertEqual(len(out), 1)
            self.assertFalse(out[0].confirmed, f"frame {i + 1}")
            self.assertEqual(out[0].hits, i + 1)
        out = voter.update([(1, det())], now=0.2)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)
        self.assertEqual(out[0].track_id, 1)

    def test_two_frame_flicker_never_confirms(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det())], now=0.0)
        out = voter.update([(1, det())], now=0.1)
        self.assertFalse(out[0].confirmed)

    def test_three_frame_glimpse_confirms(self) -> None:
        # A brief true detection (3 frames, dense in time) confirms
        # under tracking, where the 6-of-10 buffer would have dropped it.
        voter = TrackClassVoter(min_hits=3)
        for i in range(2):
            voter.update([(7, det(KNIFE, 0.8))], now=0.1 * i)
        out = voter.update([(7, det(KNIFE, 0.8))], now=0.2)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].detection.class_name, "Knife")


class TestDensityConfirmation(unittest.TestCase):
    """The density rule: sparse sightings must never confirm."""

    def test_sparse_hits_never_confirm(self) -> None:
        # Two hits, a gap longer than the window, then one more hit:
        # the three sightings were never dense, so no confirmation —
        # the lifetime counter this replaces WOULD have confirmed here.
        voter = TrackClassVoter(min_hits=3, confirm_window_s=1.0)
        voter.update([(1, det())], now=0.0)
        voter.update([(1, det())], now=0.1)
        out = voter.update([(1, det())], now=1.5)
        self.assertFalse(out[0].confirmed)
        self.assertEqual(out[0].hits, 1)  # only the 1.5 s sighting is in-window

    def test_three_hits_within_window_confirm(self) -> None:
        voter = TrackClassVoter(min_hits=3, confirm_window_s=1.0)
        voter.update([(1, det())], now=0.0)
        voter.update([(1, det())], now=0.4)
        out = voter.update([(1, det())], now=0.8)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)

    def test_confirmation_lapses_after_silence(self) -> None:
        voter = TrackClassVoter(min_hits=3, confirm_window_s=1.0)
        for i in range(3):
            out = voter.update([(1, det())], now=0.1 * i)
        self.assertTrue(out[0].confirmed)
        # At t=1.1 only the 0.1 s and 0.2 s sightings are still
        # in-window (the 0.0 s one aged out): density dropped to 2.
        out = voter.update([(1, det())], now=1.1)
        self.assertFalse(out[0].confirmed)
        self.assertEqual(out[0].hits, 2)
        # After a longer silence only the fresh sighting counts.
        out = voter.update([(1, det())], now=3.0)
        self.assertFalse(out[0].confirmed)
        self.assertEqual(out[0].hits, 1)

    def test_tracks_confirm_independently(self) -> None:
        # Track 1 is seen densely (confirms); track 2 sparsely (never).
        voter = TrackClassVoter(min_hits=3, confirm_window_s=1.0)
        voter.update([(1, det(GUN)), (2, det(KNIFE))], now=0.0)
        voter.update([(1, det(GUN))], now=0.1)
        out = voter.update([(1, det(GUN)), (2, det(KNIFE))], now=0.2)
        by_id = {t.track_id: t for t in out}
        self.assertTrue(by_id[1].confirmed)
        self.assertEqual(by_id[1].hits, 3)
        self.assertFalse(by_id[2].confirmed)
        self.assertEqual(by_id[2].hits, 2)


class TestVoting(unittest.TestCase):
    def test_single_flip_flop_does_not_change_label(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det(GUN, 0.9))], now=0.0)
        voter.update([(1, det(GUN, 0.9))], now=0.1)
        # One Knife frame at lower confidence must not flip the label.
        out = voter.update([(1, det(KNIFE, 0.5))], now=0.2)
        self.assertEqual(out[0].detection.class_id, GUN)
        self.assertEqual(out[0].raw_class_id, KNIFE)
        self.assertAlmostEqual(out[0].vote_share, 1.8 / 2.3)

    def test_sustained_opposite_labels_eventually_flip(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det(GUN, 0.9))], now=0.0)
        voter.update([(1, det(GUN, 0.9))], now=0.1)
        out = voter.update([(1, det(KNIFE, 0.9))], now=0.2)
        self.assertEqual(out[0].detection.class_id, GUN)
        out = voter.update([(1, det(KNIFE, 0.9))], now=0.3)
        # 1.8 Gun vs 1.8 Knife: tie breaks to the most recent class.
        self.assertEqual(out[0].detection.class_id, KNIFE)
        out = voter.update([(1, det(KNIFE, 0.9))], now=0.4)
        self.assertEqual(out[0].detection.class_id, KNIFE)

    def test_confidence_weighted_not_plain_count(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det(GUN, 0.95))], now=0.0)
        voter.update([(1, det(KNIFE, 0.45))], now=0.1)
        # Two weak Knife votes (0.90 total) must not beat one strong Gun (0.95).
        out = voter.update([(1, det(KNIFE, 0.45))], now=0.2)
        self.assertEqual(out[0].detection.class_id, GUN)

    def test_tracks_vote_independently(self) -> None:
        voter = TrackClassVoter(min_hits=2)
        voter.update([(1, det(GUN, 0.9)), (2, det(KNIFE, 0.9))], now=0.0)
        out = voter.update([(1, det(GUN, 0.9)), (2, det(KNIFE, 0.9))], now=0.1)
        by_id = {t.track_id: t for t in out}
        self.assertEqual(by_id[1].detection.class_id, GUN)
        self.assertEqual(by_id[2].detection.class_id, KNIFE)
        self.assertTrue(all(t.confirmed for t in out))

    def test_voted_label_keeps_current_frame_bbox_and_conf(self) -> None:
        voter = TrackClassVoter(min_hits=2)
        voter.update([(1, det(GUN, 0.9))], now=0.0)
        moved = (120.0, 110.0, 220.0, 270.0)
        out = voter.update([(1, det(KNIFE, 0.55, bbox=moved))], now=0.1)
        self.assertEqual(out[0].detection.class_id, GUN)  # voted
        self.assertEqual(out[0].detection.bbox, moved)  # current frame
        self.assertAlmostEqual(out[0].detection.confidence, 0.55)


class TestLifecycle(unittest.TestCase):
    def test_stale_tracks_are_pruned(self) -> None:
        voter = TrackClassVoter(min_hits=2, max_age=3)
        voter.update([(1, det())], now=0.0)
        self.assertEqual(voter.active_tracks, 1)
        # Three frames with no sighting of track 1 (other track keeps
        # the frame counter moving); age 4 > max_age 3 prunes it.
        for i in range(4):
            voter.update([(2, det())], now=0.1 * (i + 1))
        self.assertEqual(voter.active_tracks, 1)

    def test_reset_clears_votes(self) -> None:
        voter = TrackClassVoter(min_hits=2)
        voter.update([(1, det(GUN, 0.9))], now=0.0)
        voter.reset()
        self.assertEqual(voter.active_tracks, 0)
        out = voter.update([(1, det(KNIFE, 0.9))], now=0.1)
        self.assertEqual(out[0].detection.class_id, KNIFE)
        self.assertEqual(out[0].hits, 1)
        self.assertFalse(out[0].confirmed)

    def test_invalid_params_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrackClassVoter(min_hits=0)
        with self.assertRaises(ValueError):
            TrackClassVoter(max_age=0)
        with self.assertRaises(ValueError):
            TrackClassVoter(confirm_window_s=0)
        with self.assertRaises(ValueError):
            TrackClassVoter(confirm_window_s=-1.0)


if __name__ == "__main__":
    unittest.main()
