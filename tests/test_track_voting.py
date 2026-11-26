"""Unit tests for track-level class voting (pure Python, no torch).

Confirmation is density over time plus a minimum span: >= min_hits
sightings within the last confirm_window_s seconds (default 3 in
1.0 s) AND the oldest in-window sighting at least min_span_s old
(default 0.4 s), lapsing when the density drops. Every update() call
therefore passes an explicit monotonic timestamp, so the tests are
deterministic.

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
        # Sightings at 0.0 s and 0.2 s: dense, but the span (0.2 s) is
        # under min_span_s (0.4 s), so neither confirms.
        for i, t in enumerate((0.0, 0.2)):
            out = voter.update([(1, det())], now=t)
            self.assertEqual(len(out), 1)
            self.assertFalse(out[0].confirmed, f"sighting {i + 1}")
            self.assertEqual(out[0].hits, i + 1)
        # Third sighting at 0.4 s: 3 hits spanning exactly 0.4 s.
        out = voter.update([(1, det())], now=0.4)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)
        self.assertEqual(out[0].track_id, 1)

    def test_two_frame_flicker_never_confirms(self) -> None:
        voter = TrackClassVoter(min_hits=3)
        voter.update([(1, det())], now=0.0)
        out = voter.update([(1, det())], now=0.1)
        self.assertFalse(out[0].confirmed)

    def test_three_frame_glimpse_confirms(self) -> None:
        # A brief true detection (3 sightings spanning 0.4 s) confirms
        # under tracking, where the 6-of-10 buffer would have dropped it.
        # A denser burst inside 0.4 s would NOT confirm — see TestMinSpan.
        voter = TrackClassVoter(min_hits=3)
        for t in (0.0, 0.2):
            voter.update([(7, det(KNIFE, 0.8))], now=t)
        out = voter.update([(7, det(KNIFE, 0.8))], now=0.4)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].detection.class_name, "Knife")


class TestDensityConfirmation(unittest.TestCase):
    """The density rule: sightings must be dense within the window.

    Only in-window sightings count, so sightings more than a window
    apart never combine — but gaps inside the window are tolerated:
    three sightings at 0.0, 0.5 and 0.9 s DO confirm (span is
    first-to-last, not continuity).
    """

    def test_sparse_sightings_within_window_do_confirm(self) -> None:
        # Claude round-6 correction: the old docstring claimed sparse
        # sightings spread over a second never confirm — but 0.0, 0.5
        # and 0.9 s are 3 in-window hits spanning 0.9 s, so they do.
        voter = TrackClassVoter(min_hits=3, confirm_window_s=1.0)
        voter.update([(1, det())], now=0.0)
        voter.update([(1, det())], now=0.5)
        out = voter.update([(1, det())], now=0.9)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)

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
        for t in (0.0, 0.25, 0.5):
            out = voter.update([(1, det())], now=t)
        self.assertTrue(out[0].confirmed)
        # At t=1.3 the 0.0 s and 0.25 s sightings aged out of the
        # 1.0 s window: only 0.5 s and 1.3 s remain, density dropped to 2.
        out = voter.update([(1, det())], now=1.3)
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
        voter.update([(1, det(GUN))], now=0.2)
        out = voter.update([(1, det(GUN)), (2, det(KNIFE))], now=0.4)
        by_id = {t.track_id: t for t in out}
        self.assertTrue(by_id[1].confirmed)
        self.assertEqual(by_id[1].hits, 3)
        self.assertFalse(by_id[2].confirmed)
        self.assertEqual(by_id[2].hits, 2)


class TestMinSpan(unittest.TestCase):
    """The span rule: confirmation needs observed presence, not just density."""

    def test_three_sightings_inside_span_do_not_confirm(self) -> None:
        # 3 sightings in 0.2 s: dense, but the object was only visibly
        # present for 0.2 s — a phone sweeping through frame.
        voter = TrackClassVoter(min_hits=3, min_span_s=0.4)
        for t in (0.0, 0.1, 0.2):
            out = voter.update([(1, det())], now=t)
        self.assertEqual(out[0].hits, 3)
        self.assertFalse(out[0].confirmed)

    def test_three_sightings_spanning_min_span_confirm(self) -> None:
        voter = TrackClassVoter(min_hits=3, min_span_s=0.4)
        for t in (0.0, 0.2, 0.4):
            out = voter.update([(1, det())], now=t)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 3)

    def test_gap_inside_span_is_tolerated(self) -> None:
        # Sightings at 0.0, 0.1, then 0.5: a 0.4 s gap in the middle,
        # but the span (0.5 s) clears the floor — near-threshold
        # flicker must not drop a real detection.
        voter = TrackClassVoter(min_hits=3, min_span_s=0.4)
        for t in (0.0, 0.1, 0.5):
            out = voter.update([(1, det())], now=t)
        self.assertTrue(out[0].confirmed)

    def test_span_measured_from_oldest_in_window_sighting(self) -> None:
        # 0.0, 0.1, 0.2 do not confirm; one more sighting at 0.45
        # makes the span 0.45 s — the oldest sighting counts, not the
        # most recent three.
        voter = TrackClassVoter(min_hits=3, min_span_s=0.4)
        for t in (0.0, 0.1, 0.2):
            out = voter.update([(1, det())], now=t)
        self.assertFalse(out[0].confirmed)
        out = voter.update([(1, det())], now=0.45)
        self.assertTrue(out[0].confirmed)
        self.assertEqual(out[0].hits, 4)

    def test_min_span_zero_disables_span_check(self) -> None:
        # The pre-span rule: density alone confirms.
        voter = TrackClassVoter(min_hits=3, min_span_s=0)
        for t in (0.0, 0.1, 0.2):
            out = voter.update([(1, det())], now=t)
        self.assertTrue(out[0].confirmed)

    def test_negative_min_span_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrackClassVoter(min_span_s=-0.1)


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
        # The second sighting is 0.5 s later, so the 0.4 s span passes
        # and both tracks confirm (at 0.1 s they would not).
        out = voter.update([(1, det(GUN, 0.9)), (2, det(KNIFE, 0.9))], now=0.5)
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
        with self.assertRaises(ValueError):
            TrackClassVoter(min_span_s=-0.1)


if __name__ == "__main__":
    unittest.main()
