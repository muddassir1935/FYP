#!/usr/bin/env python3
"""Standalone weapon-detection demo (FYP-I).

Runs the SENTINEL weapon pipeline on a webcam, a video file, or an RTSP
stream: YOLOv8 detection per frame (FR-02), temporal consistency
confirmation (FR-05, 6-of-10 by default), on-screen boxes, and a snapshot
saved whenever a threat is confirmed.

Model swap: ``--weights`` points at the model file. Upgrading the model
(e.g. after a 100-epoch tuning run) means dropping the new ``best.pt`` at
``models/weights/best.pt`` — no code changes.

Examples (from the repo root):
    python scripts/inference/run_weapon_demo.py                # webcam
    python scripts/inference/run_weapon_demo.py --source demo.mp4
    python scripts/inference/run_weapon_demo.py --source rtsp://cam/stream

Controls: ``q`` quit, ``r`` reset the temporal buffer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import FrameSource, TemporalConsistencyBuffer, WeaponDetector

SNAPSHOT_COOLDOWN_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SENTINEL weapon demo (FR-02 + FR-05)")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (default 0), video file path, or RTSP URL.",
    )
    parser.add_argument(
        "--weights",
        default=str(REPO_ROOT / "models" / "weights" / "best.pt"),
        help="Path to the YOLOv8 weights file (default: models/weights/best.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--window", type=int, default=10, help="Temporal buffer window (frames).")
    parser.add_argument("--hits", type=int, default=6, help="Hits inside the window needed to confirm.")
    parser.add_argument(
        "--snapshots-dir",
        default=str(REPO_ROOT / "snapshots"),
        help="Where confirmed-threat snapshots are saved.",
    )
    parser.add_argument("--no-snapshots", action="store_true", help="Disable snapshot saving.")
    parser.add_argument("--save-video", default=None, help="Optional path to record the annotated video.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop a video file when it ends instead of exiting (file sources only).",
    )
    parser.add_argument("--device", default=None, help="'cuda', 'cpu', or omit for auto.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed; run: pip install opencv-python", file=sys.stderr)
        return 1

    detector = WeaponDetector(args.weights, conf_threshold=args.conf, device=args.device)
    buffer = TemporalConsistencyBuffer(window=args.window, required_hits=args.hits)
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    snapshots_dir = Path(args.snapshots_dir)
    if not args.no_snapshots:
        snapshots_dir.mkdir(parents=True, exist_ok=True)
    last_snapshot_at: dict[int, float] = {}

    writer = None
    fps = 0.0
    last_frame_index = 0
    print(f"SENTINEL demo — source={args.source!r} weights={args.weights}")
    print("Press q to quit, r to reset the temporal buffer.")

    with FrameSource(source, loop_file=args.loop) as stream:
        while True:
            packet = stream.read()
            if packet is None:
                if stream.ended:
                    break
                time.sleep(0.005)
                continue
            frame_index, _ts, frame = packet
            if frame_index == last_frame_index:
                if stream.ended:
                    break
                time.sleep(0.002)
                continue
            tick = time.monotonic()
            if last_frame_index:
                instant = 1.0 / max(tick - last_tick, 1e-6)
                fps = instant if fps == 0.0 else 0.9 * fps + 0.1 * instant
            last_tick = tick
            last_frame_index = frame_index

            display = frame.copy()
            detections = detector.detect(frame)
            confirmed = buffer.update(detections)
            confirmed_by_id = {id(cd.detection): cd for cd in confirmed}

            for det in detections:
                x1, y1, x2, y2 = (int(v) for v in det.bbox)
                cd = confirmed_by_id.get(id(det))
                if cd is not None:
                    color, thickness = (0, 0, 255), 3
                    label = f"{det.class_name} {det.confidence:.2f} CONFIRMED {cd.hits}/{args.window}"
                else:
                    color, thickness = (0, 200, 0), 1
                    label = f"{det.class_name} {det.confidence:.2f}"
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(
                    display, label, (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
                )

            hud = f"FPS {fps:.1f} | frame {frame_index} | confirmed threats: {len(confirmed)}"
            cv2.putText(
                display, hud, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
            )

            now = time.monotonic()
            for cd in confirmed:
                if args.no_snapshots:
                    break
                if now - last_snapshot_at.get(cd.track_id, 0.0) < SNAPSHOT_COOLDOWN_SECONDS:
                    continue
                last_snapshot_at[cd.track_id] = now
                stamp = time.strftime("%Y%m%d_%H%M%S")
                path = snapshots_dir / f"threat_{cd.detection.class_name}_{stamp}_t{cd.track_id}.jpg"
                cv2.imwrite(str(path), display)
                print(f"[ALERT] {cd.detection.class_name} confirmed "
                      f"({cd.hits}/{args.window} frames) — snapshot: {path}")

            if args.save_video:
                if writer is None:
                    height, width = display.shape[:2]
                    writer = cv2.VideoWriter(
                        args.save_video,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        25.0,
                        (width, height),
                    )
                writer.write(display)

            cv2.imshow("SENTINEL — weapon detection demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                buffer.reset()
                print("Temporal buffer reset.")

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Demo ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
