"""Unit tests for the offline replay benchmark (pure Python, no model).

Covers the gate-critical logic — sidecar timing assertion, alarm-event
merging, timing-source selection, and CLI defaults — by importing the
harness module directly. No real weights or video are needed.

The one end-to-end path (real weights on videos/Test4.mp4) is NOT in
this suite: on this VM a full 1897-frame CPU replay takes far longer
than 60 s (see the timing probe in the task report), so a Test4 smoke
test would make the suite unusable. Run it manually instead:
    python scripts/evaluation/replay_benchmark.py --video videos/Test4.mp4

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluation.replay_benchmark import (
    DEFAULT_COOLDOWN_S,
    DEFAULT_CONFIRM_MIN_SPAN_S,
    DEFAULT_CONFIRM_WINDOW_S,
    DEFAULT_CONF_GUN,
    DEFAULT_CONF_KNIFE,
    DEFAULT_MIN_HITS,
    AlarmEvent,
    ConfirmedFrame,
    build_parser,
    find_clip_entry,
    frame_timestamp,
    load_intervals_doc,
    load_sidecar_timestamps,
    merge_confirmed_frames,
    score_against_intervals,
    score_intervals,
    score_labeled_frames,
    score_negative_clip,
    write_events_csv,
)


def rec(frame: int, ts: float, cls: str = "Gun", conf: float = 0.9,
        tracks=(1,)) -> ConfirmedFrame:
    return ConfirmedFrame(
        frame_index=frame,
        ts=ts,
        class_name=cls,
        peak_confidence=conf,
        track_ids=frozenset(tracks),
    )


class TestSidecarAssertion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.sidecar = Path(self._tmp.name) / "clip.mp4.timestamps.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write(self, entries) -> None:
        self.sidecar.write_text(
            "".join(
                json.dumps({"frame_index": i, "ts": ts}) + "\n"
                for i, ts in entries
            ),
            encoding="utf-8",
        )

    def test_exact_line_count_loads(self) -> None:
        self._write([(0, 100.0), (1, 100.1), (2, 100.2)])
        self.assertEqual(
            load_sidecar_timestamps(self.sidecar, 3), [100.0, 100.1, 100.2]
        )

    def test_fewer_lines_than_frames_raises(self) -> None:
        # The gate-critical case: a short sidecar must fail loudly,
        # never silently replay at the wrong speed.
        self._write([(0, 100.0), (1, 100.1)])
        with self.assertRaisesRegex(ValueError, "2 lines but the video has 3"):
            load_sidecar_timestamps(self.sidecar, 3)

    def test_more_lines_than_frames_raises(self) -> None:
        self._write([(0, 100.0), (1, 100.1), (2, 100.2), (3, 100.3)])
        with self.assertRaisesRegex(ValueError, "4 lines but the video has 3"):
            load_sidecar_timestamps(self.sidecar, 3)

    def test_frame_index_out_of_order_raises(self) -> None:
        self._write([(0, 100.0), (5, 100.1), (2, 100.2)])
        with self.assertRaisesRegex(ValueError, "frame_index 5, expected 1"):
            load_sidecar_timestamps(self.sidecar, 3)

    def test_malformed_line_raises(self) -> None:
        self.sidecar.write_text(
            '{"frame_index": 0, "ts": 100.0}\nnot json\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "line 2 is not a"):
            load_sidecar_timestamps(self.sidecar, 2)


class TestAlarmEventMerging(unittest.TestCase):
    def test_within_cooldown_merges_into_one_event(self) -> None:
        events = merge_confirmed_frames(
            [rec(10, 1.0), rec(20, 1.5, conf=0.95, tracks=(2,)), rec(30, 2.0)],
            cooldown_s=5.0,
        )
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.class_name, "Gun")
        self.assertEqual((event.first_frame, event.last_frame), (10, 30))
        self.assertEqual((event.first_ts, event.last_ts), (1.0, 2.0))
        self.assertEqual(event.confirmed_frames, 3)
        self.assertAlmostEqual(event.peak_confidence, 0.95)
        self.assertEqual(event.track_ids, {1, 2})

    def test_gap_over_cooldown_splits_events(self) -> None:
        events = merge_confirmed_frames(
            [rec(10, 1.0), rec(20, 1.5), rec(30, 7.0), rec(40, 7.5)],
            cooldown_s=5.0,
        )
        self.assertEqual(len(events), 2)
        self.assertEqual(
            [(e.first_frame, e.last_frame) for e in events],
            [(10, 20), (30, 40)],
        )
        self.assertEqual([e.confirmed_frames for e in events], [2, 2])

    def test_gap_exactly_at_cooldown_still_merges(self) -> None:
        events = merge_confirmed_frames(
            [rec(10, 1.0), rec(20, 6.0)], cooldown_s=5.0
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].confirmed_frames, 2)

    def test_different_classes_never_merge(self) -> None:
        events = merge_confirmed_frames(
            [
                rec(10, 1.0, "Gun"),
                rec(11, 1.05, "Knife"),
                rec(12, 1.1, "Gun"),
                rec(13, 1.15, "Knife"),
            ],
            cooldown_s=5.0,
        )
        self.assertEqual(len(events), 2)
        by_class = {e.class_name: e for e in events}
        self.assertEqual(by_class["Gun"].confirmed_frames, 2)
        self.assertEqual(by_class["Knife"].confirmed_frames, 2)
        self.assertEqual(by_class["Gun"].track_ids, {1})

    def test_interleaved_classes_keep_separate_gaps(self) -> None:
        # Gun goes quiet for > cooldown while Knife keeps firing: only
        # the Gun event splits.
        events = merge_confirmed_frames(
            [
                rec(10, 1.0, "Gun"),
                rec(11, 1.1, "Knife"),
                rec(12, 3.0, "Knife"),
                rec(13, 5.0, "Knife"),
                rec(14, 7.0, "Gun"),
            ],
            cooldown_s=5.0,
        )
        self.assertEqual(len(events), 3)
        guns = [e for e in events if e.class_name == "Gun"]
        knives = [e for e in events if e.class_name == "Knife"]
        self.assertEqual(len(guns), 2)
        self.assertEqual(len(knives), 1)
        self.assertEqual(knives[0].confirmed_frames, 3)

    def test_empty_input_no_events(self) -> None:
        self.assertEqual(merge_confirmed_frames([], 5.0), [])

    def test_event_dict_shape(self) -> None:
        events = merge_confirmed_frames([rec(10, 1.0, conf=0.9)], 5.0)
        self.assertEqual(
            events[0].to_dict(),
            {
                "class": "Gun",
                "first_frame": 10,
                "last_frame": 10,
                "first_ts": 1.0,
                "last_ts": 1.0,
                "peak_confidence": 0.9,
                "confirmed_frames": 1,
                "track_ids": [1],
            },
        )


class TestTimingSource(unittest.TestCase):
    def test_sidecar_present_uses_sidecar_ts(self) -> None:
        sidecar = [100.0, 100.033, 100.066]
        self.assertEqual(frame_timestamp(2, sidecar, 30.0), 100.066)

    def test_sidecar_absent_uses_frame_index_over_fps(self) -> None:
        self.assertAlmostEqual(frame_timestamp(90, None, 30.0), 3.0)
        self.assertAlmostEqual(frame_timestamp(51, None, 51.52307322497284), 0.99, places=2)

    def test_no_sidecar_and_no_fps_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "no FPS"):
            frame_timestamp(0, None, 0.0)


class TestArgDefaults(unittest.TestCase):
    def test_defaults_match_shipped_operating_point(self) -> None:
        args = build_parser().parse_args(["--video", "clip.mp4"])
        self.assertEqual(args.conf_gun, 0.44)
        self.assertEqual(args.conf_knife, 0.43)
        self.assertEqual(args.min_hits, 3)
        self.assertEqual(args.confirm_window, 1.0)
        self.assertEqual(args.confirm_min_span, 0.4)
        self.assertEqual(args.cooldown_s, 5.0)
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.device, "auto")
        self.assertEqual(args.scene_seen_in_training, "unknown")
        self.assertIsNone(args.output)
        self.assertIsNone(args.csv)
        self.assertIsNone(args.intervals)
        # Defaults agree with the module constants.
        self.assertEqual(args.conf_gun, DEFAULT_CONF_GUN)
        self.assertEqual(args.conf_knife, DEFAULT_CONF_KNIFE)
        self.assertEqual(args.min_hits, DEFAULT_MIN_HITS)
        self.assertEqual(args.confirm_window, DEFAULT_CONFIRM_WINDOW_S)
        self.assertEqual(args.confirm_min_span, DEFAULT_CONFIRM_MIN_SPAN_S)
        self.assertEqual(args.cooldown_s, DEFAULT_COOLDOWN_S)

    def test_scene_flag_accepts_yes_no_unknown_only(self) -> None:
        for value in ("yes", "no", "unknown"):
            args = build_parser().parse_args(
                ["--video", "clip.mp4", "--scene-seen-in-training", value]
            )
            self.assertEqual(args.scene_seen_in_training, value)
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                ["--video", "clip.mp4", "--scene-seen-in-training", "maybe"]
            )


class TestEventsCsv(unittest.TestCase):
    def test_one_row_per_event(self) -> None:
        events = merge_confirmed_frames(
            [rec(10, 1.0, "Gun"), rec(30, 7.0, "Gun"), rec(11, 1.1, "Knife")],
            cooldown_s=5.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            write_events_csv(path, events)
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 4)  # header + 3 events
        self.assertTrue(rows[0].startswith("event_id,class,first_frame"))
        self.assertIn("Gun,10,10", rows[1])
        self.assertIn("Knife,11,11", rows[2])
        self.assertIn("Gun,30,30", rows[3])


class TestIntervalScoring(unittest.TestCase):
    def _event(self, cls, first_ts, last_ts):
        return AlarmEvent(
            class_name=cls, first_frame=0, last_frame=0,
            first_ts=first_ts, last_ts=last_ts,
            peak_confidence=0.9, confirmed_frames=1, track_ids={1},
        )

    def test_interval_recall_and_inside_outside(self) -> None:
        events = [
            self._event("Knife", 5.0, 6.0),    # inside [2, 30]
            self._event("Knife", 60.0, 61.0),  # outside every interval
            self._event("Gun", 5.0, 6.0),      # wrong class: not inside
        ]
        scored = score_intervals(
            events, [[2.0, 30.0], [33.0, 43.0], [46.0, 55.0]], "Knife"
        )
        self.assertEqual(scored["gt_intervals"], 3)
        self.assertEqual(scored["recalled_intervals"], 1)
        self.assertAlmostEqual(scored["recall"], 1 / 3)
        self.assertEqual(scored["alarms_inside_intervals"], 1)
        self.assertEqual(scored["alarms_outside_intervals"], 1)
        self.assertEqual(scored["alarms_other_class"], 1)

    def test_full_recall(self) -> None:
        events = [self._event("Knife", 3.0, 4.0), self._event("Knife", 14.0, 15.0)]
        scored = score_intervals(events, [[3.0, 11.0], [13.0, 16.0]], "Knife")
        self.assertEqual(scored["recall"], 1.0)
        self.assertEqual(scored["alarms_outside_intervals"], 0)

    def test_negative_clip_every_alarm_is_fp(self) -> None:
        scored = score_negative_clip(
            [self._event("Knife", 5.0, 6.0), self._event("Gun", 9.0, 10.0)]
        )
        self.assertEqual(scored["mode"], "negative_clip")
        self.assertEqual(scored["false_positives"], 2)
        self.assertNotIn("recall", scored)
        self.assertNotIn("precision", scored)


class TestLabeledFrameScoring(unittest.TestCase):
    def test_recall_within_plus_minus_one_second(self) -> None:
        confirmed = [rec(300, 10.0, "Gun"), rec(600, 20.5, "Gun")]
        labeled = [
            {"t": 10.0, "decision": "positive"},   # exact hit
            {"t": 20.0, "decision": "positive"},   # 0.5 s away: recalled
            {"t": 40.0, "decision": "positive"},   # no detection near: missed
            {"t": 60.0, "decision": "negative"},   # no detection near: clean
        ]
        scored = score_labeled_frames(confirmed, labeled, "Gun")
        self.assertEqual(scored["mode"], "recall_only_labeled_frames")
        self.assertEqual(scored["positive_labeled_frames"], 3)
        self.assertEqual(scored["recalled_labeled_frames"], 2)
        self.assertAlmostEqual(scored["recall"], 2 / 3)
        self.assertEqual(scored["negative_frames_overlapped"], 0)
        self.assertNotIn("precision", scored)

    def test_no_positive_labeled_frames_recall_is_null_not_zero(self) -> None:
        # The shipped UCF clips: all 40 labeled frames are negatives.
        confirmed = [rec(300, 10.0, "Gun")]
        labeled = [
            {"t": 10.0, "decision": "negative"},  # overlapped: FP indicator
            {"t": 60.0, "decision": "negative"},
        ]
        scored = score_labeled_frames(confirmed, labeled, "Gun")
        self.assertIsNone(scored["recall"])
        self.assertIsNotNone(scored["recall_note"])
        self.assertEqual(scored["negative_frames_overlapped"], 1)
        self.assertEqual(scored["positive_labeled_frames"], 0)

    def test_wrong_class_does_not_recall(self) -> None:
        confirmed = [rec(300, 10.0, "Knife")]
        labeled = [{"t": 10.0, "decision": "positive"}]
        scored = score_labeled_frames(confirmed, labeled, "Gun")
        self.assertEqual(scored["recall"], 0.0)


class TestScoringDispatch(unittest.TestCase):
    def setUp(self) -> None:
        self.doc = {
            "clips": [
                {"clip": "knife", "video": "videos/v5-knife-handheld.mp4",
                 "class": "Knife", "intervals": [[2.0, 30.0]]},
                {"clip": "neg", "video": "videos/v5-scissors.mp4",
                 "intervals": []},
                {"clip": "ucf", "video": "ucf/Shooting011_x264.mp4",
                 "class": "Gun", "scoring": "recall_only_labeled_frames",
                 "labeled_frames_s": [{"t": 6.0, "decision": "negative"}]},
            ]
        }

    def test_intervals_clip(self) -> None:
        event = AlarmEvent(
            class_name="Knife", first_frame=0, last_frame=0,
            first_ts=5.0, last_ts=6.0, peak_confidence=0.9,
            confirmed_frames=1, track_ids={1},
        )
        scored = score_against_intervals(
            self.doc, Path("videos/v5-knife-handheld.mp4"), [event], []
        )
        self.assertEqual(scored["mode"], "intervals")
        self.assertEqual(scored["recall"], 1.0)

    def test_negative_clip(self) -> None:
        scored = score_against_intervals(
            self.doc, Path("videos/v5-scissors.mp4"),
            [AlarmEvent("Gun", 0, 0, 5.0, 6.0, 0.9, 1, {1})], [],
        )
        self.assertEqual(scored["mode"], "negative_clip")
        self.assertEqual(scored["false_positives"], 1)

    def test_ucf_clip_never_reports_precision(self) -> None:
        scored = score_against_intervals(
            self.doc, Path("ucf/Shooting011_x264.mp4"), [],
            [rec(180, 6.0, "Gun")],
        )
        self.assertEqual(scored["mode"], "recall_only_labeled_frames")
        self.assertIsNone(scored["recall"])
        self.assertEqual(scored["negative_frames_overlapped"], 1)
        self.assertNotIn("precision", scored)

    def test_unknown_video_reports_no_match(self) -> None:
        scored = score_against_intervals(
            self.doc, Path("videos/Test4.mp4"), [], []
        )
        self.assertEqual(scored["mode"], "no_match")

    def test_find_clip_entry_matches_by_basename(self) -> None:
        entry = find_clip_entry(
            self.doc, Path("/abs/dataset/videos/v5-knife-handheld.mp4")
        )
        self.assertEqual(entry["clip"], "knife")
        self.assertIsNone(find_clip_entry(self.doc, Path("other.mp4")))


class TestShippedIntervalsFile(unittest.TestCase):
    """The committed holdout_intervals.json itself: structure + sanity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = load_intervals_doc(
            Path(__file__).resolve().parents[1]
            / "scripts/evaluation/holdout_intervals.json"
        )

    def test_sixteen_clips(self) -> None:
        self.assertEqual(len(self.doc["clips"]), 16)

    def test_knife_intervals_inside_durations(self) -> None:
        by_clip = {c["clip"]: c for c in self.doc["clips"]}
        home = by_clip["v5-knife-handheld"]
        self.assertEqual(len(home["intervals"]), 8)
        self.assertEqual(home["gt_positive_frames"], 77)
        for start, end in home["intervals"]:
            self.assertGreaterEqual(start, 0.0)
            self.assertLessEqual(end, home["duration_s"])
        faiz = by_clip["knife2-handheld"]
        self.assertEqual(len(faiz["intervals"]), 5)
        self.assertEqual(faiz["gt_positive_frames"], 18)
        for start, end in faiz["intervals"]:
            self.assertLessEqual(end, faiz["duration_s"])

    def test_negative_clips_have_empty_intervals(self) -> None:
        by_clip = {c["clip"]: c for c in self.doc["clips"]}
        for name in (
            "v5-scissors", "v5-pliers-green", "v5-screwdriver",
            "lowlight-mobile", "lowlight-hammer2", "lowlight-screwdriver2",
        ):
            self.assertEqual(by_clip[name]["intervals"], [])

    def test_ucf_clips_recall_only_and_all_negative(self) -> None:
        by_clip = {c["clip"]: c for c in self.doc["clips"]}
        counts = {"Shooting011": 7, "Shooting002": 10, "Shooting051": 7,
                  "Shooting005": 5, "Shooting023": 5, "Shooting048": 3,
                  "Shooting043": 2, "Shooting044": 1}
        total = 0
        for name, expected in counts.items():
            entry = by_clip[name]
            self.assertEqual(entry["scoring"], "recall_only_labeled_frames")
            frames = entry["labeled_frames_s"]
            self.assertEqual(len(frames), expected)
            self.assertTrue(all(f["decision"] == "negative" for f in frames))
            for f in frames:
                self.assertLessEqual(f["t"], entry["duration_s"])
            total += len(frames)
        self.assertEqual(total, 40)


if __name__ == "__main__":
    unittest.main()
