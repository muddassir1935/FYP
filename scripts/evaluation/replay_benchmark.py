#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Offline replay benchmark: one video through the real detector + voter.

Replays a recorded clip through ``WeaponDetector.track()`` +
``TrackClassVoter`` and reports alarm events — the gate vehicle for the
v5 holdout, so it replays EVERY frame (no pacing, no dropping) and feeds
the voter VIDEO time, never wall-clock time.

Timing: if ``<video>.timestamps.jsonl`` exists next to the video it is
consumed (one ``{"frame_index", "ts"}`` line per frame, the format
written by :mod:`src.pipeline.timestamp_sidecar`). The sidecar is
1-indexed and may be non-contiguous (live captures drop frames), so
lines are matched to frames positionally and only checked for
strictly-increasing ``frame_index`` and non-decreasing ``ts``; the
timestamps are normalized to elapsed seconds (``t == 0`` at the first
line) because sidecars store raw ``time.monotonic()`` uptime. The
sidecar line count must equal the video frame count (checked upfront
when the video reports one, else after EOF) or the run FAILS LOUDLY
(non-zero exit) instead of silently evaluating at the wrong speed.
Otherwise the frame timestamp is ``frame_index / fps`` from the video.

Weights may be a ``.pt`` file or an OpenVINO model folder; ``--device
auto`` selects the OpenVINO Intel GPU (``"intel:gpu"``) for model folders
when one is available.

Confirmed detections merge into ALARM EVENTS: same class, gap <=
``--cooldown-s`` (default 5.0 s) between consecutive confirmed frames.
Each event records class, first/last frame + timestamp, peak
confidence, confirmed-frame count, and track IDs seen.

Scoring (``--intervals <holdout_intervals.json>``): the output gains a
``scoring`` block. For clips with GT intervals — interval recall (the
fraction of GT intervals overlapped by at least one alarm event of the
matching class) plus alarm counts inside/outside the intervals. For
``intervals: []`` (negative) clips — every alarm is a false positive
(FP count reported, no precision denominator). For UCF clips
(``scoring: "recall_only_labeled_frames"``) — ONLY recall on the
labeled frames (a labeled frame counts as recalled when a confirmed
detection of the matching class falls within ±1 s), NEVER a precision
number. That restriction is deliberate and absolute: no full-clip UCF
alarm-event precision/recall is ever to be reported from this data,
because the sampled labeled frames cannot support it — and in the
shipped file all 40 labeled UCF frames are negatives, so labeled-frame
recall there is vacuous (null) and the only honest UCF number is
whether alarms overlap labeled negative frames (an FP indicator).

The v5 holdout clips scored here are video-level. The frozen grown
test set is a separate image-level gate: 1,207 images survive the
freeze (the UNION of CLIP cos >= 0.90 (333) + pHash <= 10 (147) = 360
dropped, overlap 120; gun-only 864 / knife-only 139 / dual 5 /
negative 199; knife-only share 11.5%; new-only survivors 943, old-511
survivors 264). Report ALL and ``tg_*`` new-only separately.

Usage (from the repo root):
    python scripts/evaluation/replay_benchmark.py \\
        --video videos/Test4.mp4 \\
        --weights models/weights/weapon_yolov8s_v3_best.pt \\
        --output /tmp/test4.replay.json
    python scripts/evaluation/replay_benchmark.py \\
        --video /path/to/v5-knife-handheld.mp4 \\
        --intervals scripts/evaluation/holdout_intervals.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline.demo_worker import _hash_weights  # noqa: E402
from src.pipeline.thresholds import CONF_GUN, CONF_KNIFE  # noqa: E402
from src.pipeline.timestamp_sidecar import sidecar_path_for  # noqa: E402
from src.pipeline.track_voting import TrackClassVoter  # noqa: E402
from src.pipeline.weapon_detector import (  # noqa: E402
    WeaponDetector,
    detect_openvino_device,
)

DEFAULT_WEIGHTS = "models/weights/weapon_yolov8s_v3_best.pt"
# Operating-point defaults live in src/pipeline/thresholds (single
# source of truth) — never retype the numbers here.
DEFAULT_CONF_GUN = CONF_GUN
DEFAULT_CONF_KNIFE = CONF_KNIFE
DEFAULT_MIN_HITS = 3
DEFAULT_CONFIRM_WINDOW_S = 1.0
DEFAULT_CONFIRM_MIN_SPAN_S = 0.4
DEFAULT_COOLDOWN_S = 5.0
DEFAULT_IMGSZ = 640


@dataclass(frozen=True)
class ConfirmedFrame:
    """One frame with at least one confirmed detection of one class.

    One record per (frame, class): ``peak_confidence`` is the highest
    confidence among that class's confirmed detections in the frame
    and ``track_ids`` collects every track confirmed in it.
    """

    frame_index: int
    ts: float
    class_name: str
    peak_confidence: float
    track_ids: frozenset = frozenset()


@dataclass
class AlarmEvent:
    """Confirmed detections of one class merged across a short gap."""

    class_name: str
    first_frame: int
    last_frame: int
    first_ts: float
    last_ts: float
    peak_confidence: float
    confirmed_frames: int
    track_ids: set = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_ts": round(self.first_ts, 3),
            "last_ts": round(self.last_ts, 3),
            "peak_confidence": round(self.peak_confidence, 4),
            "confirmed_frames": self.confirmed_frames,
            "track_ids": sorted(self.track_ids),
        }


def build_parser() -> argparse.ArgumentParser:
    """CLI parser (kept separate so tests can check the defaults)."""
    parser = argparse.ArgumentParser(
        description="Replay one video offline through detector + voter."
    )
    parser.add_argument("--video", required=True, help="Video file to replay.")
    parser.add_argument(
        "--weights", default=DEFAULT_WEIGHTS, help="Path to best.pt."
    )
    parser.add_argument("--conf-gun", type=float, default=DEFAULT_CONF_GUN)
    parser.add_argument("--conf-knife", type=float, default=DEFAULT_CONF_KNIFE)
    parser.add_argument("--min-hits", type=int, default=DEFAULT_MIN_HITS)
    parser.add_argument(
        "--confirm-window", type=float, default=DEFAULT_CONFIRM_WINDOW_S,
        help="Seconds the --min-hits sightings must fall inside.",
    )
    parser.add_argument(
        "--confirm-min-span", type=float, default=DEFAULT_CONFIRM_MIN_SPAN_S,
        help="Minimum observed presence for confirmation (0 disables).",
    )
    parser.add_argument(
        "--cooldown-s", type=float, default=DEFAULT_COOLDOWN_S,
        help="Gap (s) between confirmed frames of the same class that "
        "still merges into one alarm event.",
    )
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument(
        "--device", default="auto",
        help="'cuda', 'cpu', 'intel:gpu', or 'auto' (default) for ultralytics' choice.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON path for the result (also printed to stdout).",
    )
    parser.add_argument(
        "--csv", default=None,
        help="Optional CSV path: one row per alarm event.",
    )
    parser.add_argument(
        "--intervals", default=None,
        help="Optional holdout_intervals.json path: adds a scoring block "
        "to the output (interval recall / FP counts / labeled-frame recall).",
    )
    parser.add_argument(
        "--scene-seen-in-training", default="unknown",
        choices=("yes", "no", "unknown"),
        help="Whether this scene appeared in training (recorded verbatim).",
    )
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """Return an error message for an invalid arg combination, else None."""
    for name in ("conf_gun", "conf_knife"):
        value = getattr(args, name)
        if not 0.0 < value <= 1.0:
            return f"--{name.replace('_', '-')} must be in (0, 1], got {value}"
    if args.min_hits < 1:
        return f"--min-hits must be >= 1, got {args.min_hits}"
    if args.confirm_window <= 0:
        return f"--confirm-window must be > 0, got {args.confirm_window}"
    if args.confirm_min_span < 0:
        return f"--confirm-min-span must be >= 0, got {args.confirm_min_span}"
    if args.confirm_min_span > args.confirm_window:
        return (
            "--confirm-min-span must be <= --confirm-window "
            f"(got {args.confirm_min_span} > {args.confirm_window}); "
            "a larger span makes confirmation impossible"
        )
    if args.cooldown_s < 0:
        return f"--cooldown-s must be >= 0, got {args.cooldown_s}"
    if args.imgsz < 32:
        return f"--imgsz must be at least 32, got {args.imgsz}"
    return None


def load_sidecar_timestamps(
    path: Path, frame_count: Optional[int]
) -> List[float]:
    """Read sidecar ``ts`` values, normalized to elapsed seconds.

    Returns one timestamp per sidecar line, in file order, with
    ``t == 0.0`` at the first line (``[t - raw[0] for t in raw]``) —
    sidecars store raw ``time.monotonic()`` uptime, but the GT
    intervals in ``--intervals`` are relative, so returning uptime
    would make every overlap check miss (0% interval recall).

    Raises ``ValueError`` with a clear message when the line count
    disagrees with ``frame_count`` (skipped when ``frame_count`` is
    None — the caller then verifies the count after EOF), when a
    ``frame_index`` is not strictly increasing, or when a ``ts`` goes
    backwards. Real sidecars are **1-indexed** (``capture.py`` starts
    its index at 0 and increments before storing the first packet) and
    **non-contiguous** (a live webcam drops frames), so the
    ``frame_index`` values are only ever checked for order — NEVER
    compared to the line number. The caller maps line ``i`` to replay
    frame ``i`` positionally.
    """
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if frame_count is not None and len(lines) != frame_count:
        raise ValueError(
            f"Timestamp sidecar {path} has {len(lines)} lines but the "
            f"video has {frame_count} frames — refusing to replay at "
            "the wrong speed. Re-record the sidecar or delete it to "
            "fall back to frame_index/fps timing."
        )
    raw: List[float] = []
    prev_index: Optional[int] = None
    prev_ts: Optional[float] = None
    for i, line in enumerate(lines):
        try:
            entry = json.loads(line)
            frame_index = int(entry["frame_index"])
            ts = float(entry["ts"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Timestamp sidecar {path} line {i + 1} is not a "
                f'{{"frame_index", "ts"}} record: {line!r} ({exc})'
            ) from exc
        if prev_index is not None and frame_index <= prev_index:
            raise ValueError(
                f"Timestamp sidecar {path} line {i + 1} has "
                f"frame_index {frame_index}, which is not strictly "
                f"greater than the previous {prev_index} — the sidecar "
                "does not line up with the video frames."
            )
        if prev_ts is not None and ts < prev_ts:
            raise ValueError(
                f"Timestamp sidecar {path} line {i + 1} has ts {ts}, "
                f"which goes backwards from the previous {prev_ts}."
            )
        prev_index = frame_index
        prev_ts = ts
        raw.append(ts)
    if not raw:
        return []
    first = raw[0]
    return [t - first for t in raw]


def frame_timestamp(
    frame_index: int,
    sidecar_ts: Optional[Sequence[float]],
    fps: float,
) -> float:
    """Video time for one frame: sidecar ``ts`` when present, else ``frame_index / fps``.

    ``sidecar_ts`` holds ELAPSED seconds (see
    :func:`load_sidecar_timestamps`): line ``i`` is replay frame ``i``
    positionally. A frame past the end of the sidecar is a clean
    ``ValueError`` (the caller exits 2) — never an ``IndexError``:
    some containers decode more frames than ``CAP_PROP_FRAME_COUNT``
    reports, and an uncaught traceback would hide which timing broke.
    """
    if sidecar_ts is not None:
        if frame_index >= len(sidecar_ts):
            raise ValueError(
                f"frame {frame_index} is past the end of the timestamp "
                f"sidecar ({len(sidecar_ts)} lines) — the video decoded "
                "more frames than the sidecar covers."
            )
        return float(sidecar_ts[frame_index])
    if not fps or fps <= 0:
        raise ValueError(
            "Video reports no FPS and no timestamp sidecar exists — "
            "cannot compute video time. Provide a sidecar or a video "
            "with a valid FPS."
        )
    return frame_index / fps


def merge_confirmed_frames(
    confirmed: Sequence[ConfirmedFrame], cooldown_s: float
) -> List[AlarmEvent]:
    """Merge confirmed frames into alarm events.

    Same class, gap <= ``cooldown_s`` between consecutive confirmed
    frames: one event. A longer gap splits the event; different classes
    never merge (each class keeps its own open event).
    """
    events: List[AlarmEvent] = []
    open_by_class: Dict[str, AlarmEvent] = {}
    for record in sorted(confirmed, key=lambda r: (r.frame_index, r.class_name)):
        event = open_by_class.get(record.class_name)
        if event is None or record.ts - event.last_ts > cooldown_s:
            event = AlarmEvent(
                class_name=record.class_name,
                first_frame=record.frame_index,
                last_frame=record.frame_index,
                first_ts=record.ts,
                last_ts=record.ts,
                peak_confidence=record.peak_confidence,
                confirmed_frames=1,
                track_ids=set(record.track_ids),
            )
            events.append(event)
            open_by_class[record.class_name] = event
        else:
            event.last_frame = record.frame_index
            event.last_ts = record.ts
            event.peak_confidence = max(
                event.peak_confidence, record.peak_confidence
            )
            event.confirmed_frames += 1
            event.track_ids.update(record.track_ids)
    events.sort(key=lambda e: (e.first_frame, e.class_name))
    return events


def weights_fingerprint(weights_path: Path) -> dict:
    """Hash info for a weights path — never raises.

    A ``.pt`` file hashes directly; an OpenVINO model folder has no
    single weights file, so :func:`src.pipeline.demo_worker._hash_weights`
    hashes the ``.bin`` (+ ``.xml``) instead. A path with nothing to
    hash records null hashes with a note saying so.
    """
    bin_hash, xml_hash, note = _hash_weights(weights_path)
    return {
        "weights_sha256": bin_hash,
        "weights_xml_sha256": xml_hash,
        "weights_note": note,
    }


def resolve_device(device_arg: str, weights_path: Path) -> Optional[str]:
    """Detector device for a replay run.

    An explicit ``--device`` wins. ``auto`` selects the OpenVINO Intel
    GPU (``"intel:gpu"``) for OpenVINO model folders when one is available
    and leaves everything else to ultralytics' choice (``None``) —
    the same selection the web UI runner and the CLI demo use.
    """
    if device_arg != "auto":
        return device_arg
    return detect_openvino_device(weights_path)


def load_intervals_doc(path: Path) -> dict:
    """Parse a holdout_intervals.json file (raises ValueError if malformed)."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read intervals file {path}: {exc}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("clips"), list):
        raise ValueError(
            f"intervals file {path} has no top-level 'clips' list"
        )
    return doc


def find_clip_entry(doc: dict, video_path: Path) -> Optional[dict]:
    """The intervals entry for a video, matched by file basename.

    The file stores dataset-relative video paths while a replay run
    points at the same file by absolute path, so the basename is the
    stable key (unique across the 16 holdout clips).
    """
    for entry in doc["clips"]:
        if Path(str(entry.get("video", ""))).name == video_path.name:
            return entry
    return None


def _event_overlaps(event: AlarmEvent, start: float, end: float) -> bool:
    return event.first_ts <= end and event.last_ts >= start


def score_intervals(
    events: Sequence[AlarmEvent],
    intervals: Sequence[Sequence[float]],
    class_name: str,
) -> dict:
    """Interval recall + inside/outside alarm counts for one GT class.

    An interval counts as recalled when at least one alarm event of
    ``class_name`` overlaps it. Alarms of the matching class that
    overlap no interval are outside (false positives); alarms of any
    other class are reported separately — a Gun alarm on a Knife clip
    is an FP too and must not vanish from the report.
    """
    matching = [e for e in events if e.class_name == class_name]
    other_class = [e for e in events if e.class_name != class_name]
    recalled = sum(
        1
        for start, end in intervals
        if any(_event_overlaps(e, float(start), float(end)) for e in matching)
    )
    inside = sum(
        1
        for e in matching
        if any(_event_overlaps(e, float(s), float(en)) for s, en in intervals)
    )
    return {
        "mode": "intervals",
        "class": class_name,
        "gt_intervals": len(intervals),
        "recalled_intervals": recalled,
        "recall": (recalled / len(intervals)) if intervals else None,
        "alarms_inside_intervals": inside,
        "alarms_outside_intervals": len(matching) - inside,
        "alarms_other_class": len(other_class),
    }


def score_negative_clip(events: Sequence[AlarmEvent]) -> dict:
    """Negative clip: every alarm is a false positive (no denominator)."""
    return {
        "mode": "negative_clip",
        "false_positives": len(events),
        "alarm_events": len(events),
    }


def score_labeled_frames(
    confirmed: Sequence[ConfirmedFrame],
    labeled_frames: Sequence[dict],
    class_name: str,
    window_s: float = 1.0,
) -> dict:
    """Recall on labeled frames only (UCF) — NEVER a precision number.

    A positive labeled frame counts as recalled when a confirmed
    detection of ``class_name`` falls within +/- ``window_s`` of its
    timestamp. With no positive labeled frames recall is vacuous
    (``None``, not 0): the file's UCF clips have only negative labeled
    frames, so the honest number there is how many labeled negative
    frames a confirmed detection of the class overlaps (an FP
    indicator), reported as ``negative_frames_overlapped``.
    """
    positives = [
        f for f in labeled_frames if f.get("decision") == "positive"
    ]
    negatives = [
        f for f in labeled_frames if f.get("decision") != "positive"
    ]
    matching = [c for c in confirmed if c.class_name == class_name]

    def overlapped(t: float) -> bool:
        return any(abs(c.ts - t) <= window_s for c in matching)

    # Negative frames: ANY confirmed class overlapping one is an FP
    # indicator, not just the clip's own class — on the UCF clips
    # ("class": "Gun") a confirmed Knife FP would otherwise be
    # silently ignored.
    def overlapped_any(t: float) -> bool:
        return any(abs(c.ts - t) <= window_s for c in confirmed)

    recalled = sum(1 for f in positives if overlapped(float(f["t"])))
    neg_hit = sum(1 for f in negatives if overlapped_any(float(f["t"])))
    return {
        "mode": "recall_only_labeled_frames",
        "class": class_name,
        "labeled_frames": len(labeled_frames),
        "positive_labeled_frames": len(positives),
        "recalled_labeled_frames": recalled,
        "recall": (recalled / len(positives)) if positives else None,
        "recall_note": (
            None
            if positives
            else "no positive labeled frames — recall is vacuous, not 0"
        ),
        "negative_labeled_frames": len(negatives),
        "negative_frames_overlapped": neg_hit,
    }


def score_against_intervals(
    doc: dict,
    video_path: Path,
    events: Sequence[AlarmEvent],
    confirmed: Sequence[ConfirmedFrame],
) -> dict:
    """Dispatch to the right scoring mode for one clip (pure logic)."""
    entry = find_clip_entry(doc, video_path)
    if entry is None:
        return {
            "mode": "no_match",
            "clip": None,
            "note": f"no intervals entry for video {video_path.name}",
        }
    if entry.get("scoring") == "recall_only_labeled_frames":
        result = score_labeled_frames(
            confirmed,
            entry.get("labeled_frames_s", []),
            str(entry.get("class", "Gun")),
        )
        result["clip"] = entry.get("clip")
        return result
    intervals = entry.get("intervals")
    if intervals is not None and len(intervals) == 0:
        result = score_negative_clip(events)
        result["clip"] = entry.get("clip")
        return result
    if intervals is not None:
        result = score_intervals(
            events, intervals, str(entry.get("class", "Gun"))
        )
        result["clip"] = entry.get("clip")
        return result
    return {
        "mode": "no_match",
        "clip": entry.get("clip"),
        "note": "intervals entry has neither intervals nor labeled frames",
    }


def load_timing_sidecar(
    video_path: Path, reported_frames: int
) -> tuple[Optional[List[float]], str]:
    """Load the sidecar next to a video, when one exists.

    Returns ``(timestamps, "sidecar")`` or ``(None, "fps")``. A present
    sidecar is ALWAYS loaded — even when the video reports no frame
    count (``reported_frames <= 0``), in which case the line count is
    verified against the frames actually read, after EOF. Raises
    ``ValueError`` when a present sidecar is malformed or its line
    count disagrees with a positive reported frame count, so the
    caller fails loudly before spending any time on inference.
    """
    sidecar_path = sidecar_path_for(video_path)
    if not sidecar_path.is_file():
        return None, "fps"
    return load_sidecar_timestamps(
        sidecar_path, reported_frames if reported_frames > 0 else None
    ), "sidecar"


def write_events_csv(path: Path, events: List[AlarmEvent]) -> None:
    """One row per alarm event."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "event_id",
                "class",
                "first_frame",
                "last_frame",
                "first_ts",
                "last_ts",
                "peak_confidence",
                "confirmed_frames",
                "track_ids",
            ]
        )
        for i, event in enumerate(events):
            writer.writerow(
                [
                    i,
                    event.class_name,
                    event.first_frame,
                    event.last_frame,
                    round(event.first_ts, 3),
                    round(event.last_ts, 3),
                    round(event.peak_confidence, 4),
                    event.confirmed_frames,
                    ";".join(str(t) for t in sorted(event.track_ids)),
                ]
            )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    error = validate_args(args)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    video_path = Path(args.video)
    if not video_path.is_file():
        print(f"error: video not found: {video_path}", file=sys.stderr)
        return 2
    weights_path = Path(args.weights)
    # A weights path may be a .pt file OR an OpenVINO model folder —
    # is_file() would reject every OpenVINO export.
    if not weights_path.exists():
        print(f"error: weights not found: {weights_path}", file=sys.stderr)
        return 2
    device = resolve_device(args.device, weights_path)

    try:
        import cv2
    except ImportError:
        print("error: opencv-python is not installed", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"error: cannot open video: {video_path}", file=sys.stderr)
        return 2
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Always load the sidecar when it exists — even when the video
    # reports no frame count (reported_frames <= 0): the line count is
    # then verified against the frames actually read, after EOF below.
    # timing_source only says "sidecar" when one was actually loaded.
    try:
        sidecar_ts, timing_source = load_timing_sidecar(
            video_path, reported_frames
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if timing_source == "fps" and (not fps or fps <= 0):
        print(
            f"error: video {video_path} reports no FPS and no timestamp "
            f"sidecar exists at {sidecar_path_for(video_path)} — cannot "
            "compute video time.",
            file=sys.stderr,
        )
        return 2

    try:
        detector = WeaponDetector(
            weights_path,
            conf_threshold=min(args.conf_gun, args.conf_knife),
            conf_thresholds={"Gun": args.conf_gun, "Knife": args.conf_knife},
            imgsz=args.imgsz,
            device=device,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: cannot load detector: {exc}", file=sys.stderr)
        return 2
    voter = TrackClassVoter(
        min_hits=args.min_hits,
        confirm_window_s=args.confirm_window,
        min_span_s=args.confirm_min_span,
    )

    # Pure offline loop: every frame, no pacing, no dropping. The voter
    # gets VIDEO time (sidecar ts or frame_index/fps), never wall clock.
    cap = cv2.VideoCapture(str(video_path))
    confirmed: List[ConfirmedFrame] = []
    total_confirmed_frames = 0
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                ts = frame_timestamp(frame_index, sidecar_ts, fps)
            except ValueError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            tracked = detector.track(frame)
            voted = voter.update(tracked, now=ts)
            by_class: Dict[str, ConfirmedFrame] = {}
            for item in voted:
                if not item.confirmed:
                    continue
                name = item.detection.class_name
                record = by_class.get(name)
                if record is None:
                    by_class[name] = ConfirmedFrame(
                        frame_index=frame_index,
                        ts=ts,
                        class_name=name,
                        peak_confidence=item.detection.confidence,
                        track_ids=frozenset({item.track_id}),
                    )
                else:
                    by_class[name] = ConfirmedFrame(
                        frame_index=frame_index,
                        ts=ts,
                        class_name=name,
                        peak_confidence=max(
                            record.peak_confidence,
                            item.detection.confidence,
                        ),
                        track_ids=record.track_ids | {item.track_id},
                    )
            if by_class:
                total_confirmed_frames += 1
                confirmed.extend(by_class.values())
            frame_index += 1
    finally:
        cap.release()

    if frame_index == 0:
        print(f"error: no frames read from video: {video_path}", file=sys.stderr)
        return 2
    # A sidecar loaded without an upfront check (video reported no frame
    # count) still has to match the frames actually read.
    if sidecar_ts is not None and len(sidecar_ts) != frame_index:
        print(
            f"error: timestamp sidecar {sidecar_path_for(video_path)} has "
            f"{len(sidecar_ts)} lines but {frame_index} frames were read "
            "— refusing to report results at the wrong speed.",
            file=sys.stderr,
        )
        return 2

    events = merge_confirmed_frames(confirmed, args.cooldown_s)
    scoring: Optional[dict] = None
    if args.intervals:
        intervals_path = Path(args.intervals)
        if not intervals_path.is_file():
            print(
                f"error: intervals file not found: {intervals_path}",
                file=sys.stderr,
            )
            return 2
        try:
            intervals_doc = load_intervals_doc(intervals_path)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        scoring = score_against_intervals(
            intervals_doc, video_path, events, confirmed
        )
    fingerprint = weights_fingerprint(weights_path)
    result = {
        "video": str(video_path),
        "weights": weights_path.name,
        "weights_sha256": fingerprint["weights_sha256"],
        "weights_xml_sha256": fingerprint["weights_xml_sha256"],
        "weights_note": fingerprint["weights_note"],
        "conf_gun": args.conf_gun,
        "conf_knife": args.conf_knife,
        "min_hits": args.min_hits,
        "confirm_window_s": args.confirm_window,
        "confirm_min_span_s": args.confirm_min_span,
        "cooldown_s": args.cooldown_s,
        "imgsz": args.imgsz,
        "device": args.device,
        "fps": fps,
        "timing_source": timing_source,
        "frame_count": frame_index,
        "total_confirmed_frames": total_confirmed_frames,
        "alarm_events": [event.to_dict() for event in events],
        "scene_seen_in_training": args.scene_seen_in_training,
    }
    if scoring is not None:
        result["scoring"] = scoring
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"Replay result written to {args.output}")
    if args.csv:
        write_events_csv(Path(args.csv), events)
        print(f"Alarm events written to {args.csv}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
