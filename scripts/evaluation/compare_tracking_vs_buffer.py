#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Compare ByteTrack + track voting against the 6-of-10 buffer on one clip.

Runs the SAME clip twice with the SAME model and confidence threshold
(two passes, so each method runs its real code path) and reports:

* flip-flops — frames where the same physical object switches Gun/Knife,
  counted on raw per-frame labels (baseline), on buffer-confirmed labels,
  and on track-voted labels;
* glimpses recovered — brief object appearances (4-5 consecutive frames)
  confirmed by each method (the 6/10 buffer can never confirm these;
  with min_hits=3 the tracker needs 4 consecutive frames, so 3-frame
  blips stay unconfirmed by design and are reported separately);
* flicker rejection — 1-2 frame appearances confirmed by each method
  (both should be zero);
* confirmation latency — frames from first sighting to first confirmation.

Object identity for the baseline/buffer side is IoU-linked chains
(class-agnostic, IoU >= 0.3, gaps of at most 1 frame — fast hand motion
at low FPS fragments one object under a stricter rule); for the
tracking side it is the ByteTrack track ID. The two passes' raw
detection totals are printed for reference: the track pass sees fewer
detections by design, because a new track's first sighting stays hidden
until it is matched in the next frame.

Usage (from the repo root):
    python scripts/evaluation/compare_tracking_vs_buffer.py \\
        --weights /path/to/best.pt --source clip.mp4 --max-frames 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import (  # noqa: E402
    ConfirmedDetection,
    Detection,
    TemporalConsistencyBuffer,
    TrackClassVoter,
    TrackedDetection,
    WeaponDetector,
    bbox_iou,
)

CHAIN_IOU = 0.3  # same physical object across frames (matches the buffer's IoU)
CHAIN_MAX_GAP = 1  # frames a chain may skip and still continue (fast motion
# at low FPS drops consecutive-frame IoU below 0.5 and fragments one
# object into many short chains; a 1-frame bridge keeps chains ~ objects)


@dataclass
class Chain:
    """One physical object followed across frames (raw labels)."""

    frames: List[int] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    boxes: List[Tuple[float, float, float, float]] = field(default_factory=list)

    @property
    def length(self) -> int:
        """Sightings of this object (not span — gaps do not count)."""
        return len(self.frames)

    @property
    def raw_switches(self) -> int:
        return sum(
            1 for a, b in zip(self.classes, self.classes[1:]) if a != b
        )


def build_chains(raw_by_frame: List[List[Detection]]) -> List[Chain]:
    """Link raw detections into class-agnostic IoU chains (gaps <= 1 frame)."""
    chains: List[Chain] = []
    active: List[Chain] = []  # chains seen within the last CHAIN_MAX_GAP frames
    for frame_index, detections in enumerate(raw_by_frame):
        candidates: List[Tuple[float, int, Chain, Detection]] = []
        for chain in active:
            gap = frame_index - chain.frames[-1] - 1
            if gap > CHAIN_MAX_GAP:
                continue
            for det in detections:
                iou = bbox_iou(chain.boxes[-1], det.bbox)
                if iou >= CHAIN_IOU:
                    # Prefer the smallest gap, then the best IoU.
                    candidates.append((iou, -gap, chain, det))
        candidates.sort(key=lambda c: (c[1], c[0]), reverse=True)
        used_chains: set[int] = set()
        used_dets: set[int] = set()
        next_active: List[Chain] = []
        for _iou, _gap, chain, det in candidates:
            if id(chain) in used_chains or id(det) in used_dets:
                continue
            used_chains.add(id(chain))
            used_dets.add(id(det))
            chain.frames.append(frame_index)
            chain.classes.append(det.class_id)
            chain.boxes.append(det.bbox)
            next_active.append(chain)
        for chain in active:
            if id(chain) not in used_chains and frame_index - chain.frames[-1] <= CHAIN_MAX_GAP:
                next_active.append(chain)  # stays open one more frame
        for det in detections:
            if id(det) in used_dets:
                continue
            chain = Chain(
                frames=[frame_index], classes=[det.class_id], boxes=[det.bbox]
            )
            chains.append(chain)
            next_active.append(chain)
        active = next_active
    return chains


def confirmed_labels_for_chain(
    chain: Chain,
    confirmed_by_frame: List[List[Detection]],
) -> List[Tuple[int, int]]:
    """(frame, confirmed class) for each frame this chain was confirmed in."""
    labels: List[Tuple[int, int]] = []
    for frame_index, box in zip(chain.frames, chain.boxes):
        for det in confirmed_by_frame[frame_index]:
            if bbox_iou(box, det.bbox) >= CHAIN_IOU:
                labels.append((frame_index, det.class_id))
                break
    return labels


def count_switches(labels: List[int]) -> int:
    deduped = [c for i, c in enumerate(labels) if i == 0 or c != labels[i - 1]]
    return max(0, len(deduped) - 1)


def iter_frames(
    cv2_module: object, source: str, start_frame: int, max_frames: int
):
    """Yield the clip's frames (streamed; each pass re-reads from disk)."""
    cap = cv2_module.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {source}")
    if start_frame:
        cap.set(cv2_module.CAP_PROP_POS_FRAMES, start_frame)
    count = 0
    try:
        while max_frames == 0 or count < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
            yield frame
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ByteTrack + voting vs 6-of-10 buffer, same clip."
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt (or OpenVINO folder).")
    parser.add_argument("--source", required=True, help="Video clip path.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole clip.")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-hits", type=int, default=3, help="Tracker confirmation threshold.")
    parser.add_argument("--buffer-window", type=int, default=10)
    parser.add_argument("--buffer-hits", type=int, default=6)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--out", default=None, help="Optional JSON path for the numbers.")
    parser.add_argument(
        "--dump",
        default=None,
        help="Optional JSON path for the per-frame detections/tracks "
        "(lets later analysis re-run without re-doing inference).",
    )
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed", file=sys.stderr)
        return 1

    detector = WeaponDetector(
        args.weights, conf_threshold=args.conf, imgsz=args.imgsz, device=args.device
    )

    # ---- Pass A: detect() + 6-of-10 buffer --------------------------------
    buffer = TemporalConsistencyBuffer(
        window=args.buffer_window, required_hits=args.buffer_hits
    )
    raw_by_frame: List[List[Detection]] = []
    buffer_conf_by_frame: List[List[Detection]] = []
    t0 = time.monotonic()
    for frame in iter_frames(cv2, args.source, args.start_frame, args.max_frames):
        detections = detector.detect(frame)
        raw_by_frame.append(detections)
        buffer_conf_by_frame.append(
            [cd.detection for cd in buffer.update(detections)]
        )
    buffer_seconds = time.monotonic() - t0
    n_frames = len(raw_by_frame)
    print(f"Clip: {args.source} — {n_frames} frames "
          f"(start={args.start_frame}, conf={args.conf}, imgsz={args.imgsz})")

    # ---- Pass B: track() + voter ------------------------------------------
    detector.reset_tracker()
    voter = TrackClassVoter(min_hits=args.min_hits)
    tracked_by_frame: List[List[Tuple[int, Detection]]] = []
    voted_by_frame: List[List[TrackedDetection]] = []
    voted_by_track: Dict[int, List[Tuple[int, int]]] = {}  # id -> [(frame, voted class)]
    raw_by_track: Dict[int, List[int]] = {}  # id -> [raw class per sighting]
    t0 = time.monotonic()
    for frame_index, frame in enumerate(
        iter_frames(cv2, args.source, args.start_frame, args.max_frames)
    ):
        tracked = detector.track(frame, tracker=args.tracker)
        tracked_by_frame.append(tracked)
        for track_id, det in tracked:
            raw_by_track.setdefault(track_id, []).append(det.class_id)
        voted = voter.update(tracked)
        voted_by_frame.append(voted)
        for t in voted:
            if t.confirmed:
                voted_by_track.setdefault(t.track_id, []).append(
                    (frame_index, t.detection.class_id)
                )
    track_seconds = time.monotonic() - t0
    tracked_raw_by_frame = [[d for _tid, d in t] for t in tracked_by_frame]
    voted_conf_by_frame = [
        [t.detection for t in voted if t.confirmed] for voted in voted_by_frame
    ]

    # ---- Raw detection totals (expected to differ: the track pass only ----
    # ---- sees VISIBLE tracks — a new track's first sighting is hidden ----
    # ---- until it is matched in the next frame, by ByteTrack's design) ----
    total_raw_a = sum(len(d) for d in raw_by_frame)
    total_raw_b = sum(len(d) for d in tracked_raw_by_frame)
    print(f"\nRaw detections — buffer pass: {total_raw_a}, "
          f"track pass (visible only): {total_raw_b}")
    print(f"  ({total_raw_a - total_raw_b} first-sightings hidden by "
          f"ByteTrack's 2-frame activation gate)")

    # ---- Chains + metrics --------------------------------------------------
    chains = build_chains(raw_by_frame)
    raw_switches = sum(c.raw_switches for c in chains)

    def bucket(length: int) -> str:
        if length <= 2:
            return "1-2"
        if length == 3:
            return "3"
        if length <= 5:
            return "4-5"
        return "6+"

    def chain_report(
        confirmed_by_frame: List[List[Detection]],
    ) -> Tuple[int, int, Dict[str, int], List[float]]:
        """(confirmed chains, confirmed-label switches, by-length counts, latencies)."""
        confirmed_chains = 0
        switches = 0
        by_length = {"1-2": 0, "3": 0, "4-5": 0, "6+": 0}
        latencies: List[float] = []
        for chain in chains:
            labels = confirmed_labels_for_chain(chain, confirmed_by_frame)
            if not labels:
                continue
            confirmed_chains += 1
            switches += count_switches([c for _f, c in labels])
            by_length[bucket(chain.length)] += 1
            latencies.append(labels[0][0] - chain.frames[0] + 1)
        return confirmed_chains, switches, by_length, latencies

    buf_chains, buf_switches, buf_by_len, buf_lat = chain_report(buffer_conf_by_frame)
    trk_chains, trk_switches, trk_by_len, trk_lat = chain_report(voted_conf_by_frame)

    # Per-track voted-label switches (the tracker's own flip-flop count).
    track_voted_switches = sum(
        count_switches([c for _f, c in seq]) for seq in voted_by_track.values()
    )
    track_raw_switches = sum(
        count_switches(seq) for seq in raw_by_track.values()
    )

    length_hist = {"1-2": 0, "3": 0, "4-5": 0, "6+": 0}
    for chain in chains:
        length_hist[bucket(chain.length)] += 1

    # Confirmed frames overall (earlier confirmation + longer bridging
    # should show up here even for objects whose detections are gappy).
    buf_conf_frames = sum(len(d) for d in buffer_conf_by_frame)
    trk_conf_frames = sum(len(d) for d in voted_conf_by_frame)

    def mean(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    print(f"\nObject chains (IoU-linked, consecutive frames): {len(chains)}")
    print(f"  by length: 1-2f: {length_hist['1-2']}, 3f: {length_hist['3']}, "
          f"4-5f: {length_hist['4-5']}, 6+f: {length_hist['6+']}")
    print(f"\nFlip-flops (same object switches Gun/Knife):")
    print(f"  raw per-frame labels:      {raw_switches}")
    print(f"  buffer-confirmed labels:   {buf_switches}")
    print(f"  track-voted labels:        {trk_switches} "
          f"(per-track count: {track_voted_switches}; "
          f"raw along tracks: {track_raw_switches})")
    print(f"\nConfirmed chains: buffer {buf_chains}, tracker {trk_chains}")
    print(f"  glimpses (4-5f) confirmed: buffer {buf_by_len['4-5']}, "
          f"tracker {trk_by_len['4-5']} of {length_hist['4-5']}")
    print(f"  blips (3f) confirmed:      buffer {buf_by_len['3']}, "
          f"tracker {trk_by_len['3']} of {length_hist['3']} "
          f"(tracker needs 4 consecutive frames at min_hits=3)")
    print(f"  flicker (1-2f) confirmed:  buffer {buf_by_len['1-2']}, "
          f"tracker {trk_by_len['1-2']} of {length_hist['1-2']} (want 0)")
    print(f"  long (6+f) confirmed:      buffer {buf_by_len['6+']}, "
          f"tracker {trk_by_len['6+']} of {length_hist['6+']}")
    print(f"  confirmed frames total:    buffer {buf_conf_frames}, "
          f"tracker {trk_conf_frames}")
    print(f"  mean frames to confirm:    buffer {mean(buf_lat)}, "
          f"tracker {mean(trk_lat)}")
    print(f"\nSpeed: buffer pass {buffer_seconds / n_frames:.2f}s/frame, "
          f"track pass {track_seconds / n_frames:.2f}s/frame "
          f"(CPU, {args.imgsz}px)")

    if args.dump:
        def det_dict(d: Detection) -> dict:
            return {
                "class_id": d.class_id,
                "class_name": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox": [round(v, 1) for v in d.bbox],
            }

        dump = {
            "source": args.source,
            "start_frame": args.start_frame,
            "frames": n_frames,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "min_hits": args.min_hits,
            "raw_by_frame": [[det_dict(d) for d in f] for f in raw_by_frame],
            "buffer_conf_by_frame": [
                [det_dict(d) for d in f] for f in buffer_conf_by_frame
            ],
            "tracked_by_frame": [
                [{"track_id": tid, **det_dict(d)} for tid, d in f]
                for f in tracked_by_frame
            ],
            "voted_by_frame": [
                [
                    {
                        "track_id": t.track_id,
                        "hits": t.hits,
                        "vote_share": round(t.vote_share, 4),
                        "raw_class_id": t.raw_class_id,
                        "confirmed": t.confirmed,
                        **det_dict(t.detection),
                    }
                    for t in f
                ]
                for f in voted_by_frame
            ],
        }
        Path(args.dump).write_text(json.dumps(dump))
        print(f"Per-frame dump written to {args.dump}")

    if args.out:
        result = {
            "source": args.source,
            "start_frame": args.start_frame,
            "frames": n_frames,
            "conf": args.conf,
            "imgsz": args.imgsz,
            "min_hits": args.min_hits,
            "buffer_window": args.buffer_window,
            "buffer_hits": args.buffer_hits,
            "raw_detections_buffer_pass": total_raw_a,
            "raw_detections_track_pass": total_raw_b,
            "chains": len(chains),
            "chain_length_hist": length_hist,
            "flip_flops_raw": raw_switches,
            "flip_flops_buffer_confirmed": buf_switches,
            "flip_flops_track_voted": trk_switches,
            "flip_flops_track_voted_per_track": track_voted_switches,
            "flip_flops_raw_along_tracks": track_raw_switches,
            "confirmed_chains_buffer": buf_chains,
            "confirmed_chains_tracker": trk_chains,
            "confirmed_by_length_buffer": buf_by_len,
            "confirmed_by_length_tracker": trk_by_len,
            "confirmed_frames_buffer": buf_conf_frames,
            "confirmed_frames_tracker": trk_conf_frames,
            "mean_frames_to_confirm_buffer": mean(buf_lat),
            "mean_frames_to_confirm_tracker": mean(trk_lat),
            "seconds_per_frame_buffer": buffer_seconds / n_frames,
            "seconds_per_frame_track": track_seconds / n_frames,
        }
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nNumbers written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
