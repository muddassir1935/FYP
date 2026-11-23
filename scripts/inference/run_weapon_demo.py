#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Standalone weapon-detection demo (FYP-I).

Runs the SENTINEL weapon pipeline on a webcam, a video file, or an RTSP
stream: YOLOv8 detection (FR-02), ByteTrack tracking + track-level class
voting for confirmation (FR-05), on-screen boxes, and a snapshot saved
whenever a threat is confirmed.

Confirmation works in two stages. ByteTrack (``model.track``) gives each
physical object a stable ID across frames — its two-stage association
keeps a track alive through weak detections, so brief true glimpses are
not lost. The track voter then assigns ONE stable label per track (a
confidence-weighted majority vote over the track's lifetime), so the
same object no longer flip-flops between Gun and Knife. A track
confirms after ``--min-hits`` visible sightings (default 3). ByteTrack
itself only shows a track after detections in 2 consecutive frames, so
in practice a threat confirms after 4 consecutive detections (against 6
for the old buffer) and 1-3 frame blips never confirm. The old 6-of-10
temporal buffer (``src/pipeline/temporal_buffer.py``) is kept intact
for RQ3 reproducibility — the demo simply no longer uses it.

Playback design: capture, display and inference run on separate threads.
The display shows every frame at the source's own pace (video files are
paced to their native FPS), while detection works through frames as fast
as the hardware allows and its newest boxes are overlaid on the live
picture. On a slow (CPU-only) machine the video therefore still plays
smoothly — only the box updates arrive at the inference rate.

Model swap: ``--weights`` points at the model file. Upgrading the model
(e.g. after a 100-epoch tuning run) means dropping the new ``best.pt`` at
``models/weights/best.pt`` — no code changes. An exported OpenVINO
folder works too, and tracking runs on its boxes the same way.

Examples (from the repo root):
    python scripts/inference/run_weapon_demo.py                # webcam
    python scripts/inference/run_weapon_demo.py --source demo.mp4 --loop
    python scripts/inference/run_weapon_demo.py --source rtsp://cam/stream
    python scripts/inference/run_weapon_demo.py --imgsz 416    # faster on CPU
    python scripts/inference/run_weapon_demo.py --conf-gun 0.60 --conf-knife 0.25

Confidence thresholds: Gun is the false-positive-prone class, so it runs
a higher threshold than Knife by default (val-tuned operating point, see
``sentinel-dataset-v2/thresholds/operating_point.md``). ``--conf`` still
works as a single threshold for all classes when given; ``--conf-gun`` /
``--conf-knife`` override it per class.

Controls: ``q`` quit, ``r`` reset the tracker and the class voter.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ASCII-only on purpose: a non-ASCII window title can be encoded
# inconsistently by Windows/OpenCV between calls, which made the OS show
# a second ghost window next to the real one.
WINDOW_NAME = "SENTINEL - Weapon Detection Demo"

from src.pipeline import (
    FALLBACK_FPS,
    Detection,
    FrameSource,
    InferenceWorker,
    RollingFps,
    TrackClassVoter,
    TrackedDetection,
    WeaponDetector,
    detect_openvino_device,
    draw_annotations,
)
from src.pipeline.thresholds import CONF_GUN, CONF_KNIFE, DEFAULT_CONF

# How long the display loop waits for the inference worker on exit.
WORKER_JOIN_TIMEOUT_S = 2.0

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
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Single confidence threshold for all classes. When omitted, "
        "the per-class operating point below is used instead (Gun "
        f"{CONF_GUN:.2f}, Knife {CONF_KNIFE:.2f}); when "
        "given, it is the fallback for any class without its own "
        "--conf-<class> override.",
    )
    parser.add_argument(
        "--conf-gun",
        type=float,
        default=None,
        help="Confidence threshold for Gun only (default: "
        f"{CONF_GUN:.2f}, the val-tuned operating point). "
        "Overrides --conf for this class.",
    )
    parser.add_argument(
        "--conf-knife",
        type=float,
        default=None,
        help="Confidence threshold for Knife only (default: "
        f"{CONF_KNIFE:.2f}, the val-tuned operating point). "
        "Overrides --conf for this class.",
    )
    parser.add_argument(
        "--hide-unconfirmed",
        action="store_true",
        help="Draw only FR-05-confirmed threats, hiding raw unconfirmed detections.",
    )
    parser.add_argument("--display-width", type=int, default=1280, help="Max on-screen width (default 1280).")
    parser.add_argument("--display-height", type=int, default=720, help="Max on-screen height (default 720).")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size. Smaller (e.g. 416) is much faster on CPU, slightly less accurate.",
    )
    parser.add_argument(
        "--min-hits",
        type=int,
        default=3,
        help="Visible sightings of a track needed to confirm it (default 3 — "
        "with ByteTrack's 2-frame activation gate this confirms after 4 "
        "consecutive detections and recovers 4-5 frame glimpses the old "
        "6-of-10 buffer dropped; 1-3 frame blips still never confirm).",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Ultralytics tracker config (default bytetrack.yaml; "
        "ocsort.yaml also works).",
    )
    parser.add_argument(
        "--snapshots-dir",
        default=str(REPO_ROOT / "snapshots"),
        help="Where confirmed-threat snapshots are saved.",
    )
    parser.add_argument("--no-snapshots", action="store_true", help="Disable snapshot saving.")
    parser.add_argument(
        "--save-video",
        default=None,
        help="Optional path to record the video. Records the RAW "
        "(un-annotated) frames by default — see --no-record-raw.",
    )
    parser.add_argument(
        "--no-record-raw",
        dest="record_raw",
        action="store_false",
        help="Record the ANNOTATED video instead of the raw frames. "
        "By default --save-video records raw frames, so a gate "
        "session yields minable training data.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop a video file when it ends instead of exiting (file sources only).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="'cuda', 'cpu', 'xpu', or omit for auto: OpenVINO folders "
        "auto-select the Intel GPU (xpu) when one is available.",
    )
    args = parser.parse_args()
    for name in ("conf", "conf_gun", "conf_knife"):
        value = getattr(args, name)
        if value is not None and not 0.0 < value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in (0, 1]")
    if args.min_hits < 1:
        parser.error("--min-hits must be >= 1")
    if args.display_width < 1 or args.display_height < 1:
        parser.error("--display-width and --display-height must be positive")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    return args


def resolve_thresholds(args: argparse.Namespace) -> tuple[float, float, float]:
    """Return (conf_gun, conf_knife, fallback_conf) for the demo run.

    Per class: an explicit ``--conf-<class>`` wins, then an explicit
    ``--conf`` (single-threshold mode, as before), then the val-tuned
    per-class operating point. The fallback is the generic threshold
    for any class without a per-class override.
    """
    conf_gun = (
        args.conf_gun
        if args.conf_gun is not None
        else args.conf if args.conf is not None else CONF_GUN
    )
    conf_knife = (
        args.conf_knife
        if args.conf_knife is not None
        else args.conf if args.conf is not None else CONF_KNIFE
    )
    fallback_conf = args.conf if args.conf is not None else DEFAULT_CONF
    return conf_gun, conf_knife, fallback_conf


def draw_hud(
    cv2_module: object,
    frame: object,
    display_fps: float,
    inference_fps: float,
    frame_index: int,
    confirmed_count: int,
) -> object:
    """Draw the HUD on a dark translucent bar at the top of ``frame``.

    Two short lines instead of one long line, so the text fits narrow
    portrait frames instead of running off the right edge and getting cut.
    """
    bar = frame.copy()
    cv2_module.rectangle(bar, (0, 0), (frame.shape[1], 68), (0, 0, 0), -1)
    cv2_module.addWeighted(bar, 0.55, frame, 0.45, 0, frame)
    font = cv2_module.FONT_HERSHEY_SIMPLEX
    cv2_module.putText(
        frame, f"FPS {display_fps:.1f} (detect {inference_fps:.1f})",
        (12, 27), font, 0.65, (255, 255, 255), 2, cv2_module.LINE_AA,
    )
    cv2_module.putText(
        frame, f"frame {frame_index} | confirmed: {confirmed_count}",
        (12, 55), font, 0.65, (255, 255, 255), 2, cv2_module.LINE_AA,
    )
    return frame


def fit_for_display(cv2_module: object, frame: object, max_width: int, max_height: int) -> object:
    """Scale ``frame`` to fit inside (max_width, max_height), keeping aspect."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale == 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    interpolation = cv2_module.INTER_AREA if scale < 1.0 else cv2_module.INTER_LINEAR
    return cv2_module.resize(frame, new_size, interpolation=interpolation)


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed; run: pip install opencv-python", file=sys.stderr)
        return 1

    conf_gun, conf_knife, fallback_conf = resolve_thresholds(args)
    device = args.device
    if device is None:
        # Auto-select the Intel GPU for OpenVINO folders, the same way
        # the web UI runner does (shared helper); .pt weights keep
        # ultralytics' auto-detect.
        device = detect_openvino_device(args.weights)
    try:
        detector = WeaponDetector(
            args.weights,
            conf_threshold=fallback_conf,
            conf_thresholds={"Gun": conf_gun, "Knife": conf_knife},
            imgsz=args.imgsz,
            device=device,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"ERROR: could not load weights {args.weights!r}: {exc}",
            file=sys.stderr,
        )
        return 1
    try:
        # Warm up BEFORE capture starts, so the first seconds do not
        # stream past unanalysed while the graph compiles.
        detector.warmup(tracker=args.tracker)
    except Exception as exc:
        print(f"ERROR: model warmup failed: {exc}", file=sys.stderr)
        return 1
    voter = TrackClassVoter(min_hits=args.min_hits)
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    snapshots_dir: Path | None = None
    if not args.no_snapshots:
        snapshots_dir = Path(args.snapshots_dir)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    print(f"SENTINEL demo - source={args.source!r} weights={args.weights} imgsz={args.imgsz}")
    print(f"Confidence thresholds: Gun={conf_gun:.2f} Knife={conf_knife:.2f}")
    print(f"Device: {device or 'auto'}")
    print(f"Confirmation: {args.tracker} + track voting (min_hits={args.min_hits})")
    print("Press q to quit, r to reset the tracker and class voter.")

    writer = None
    writer_failed = False
    display_fps = 0.0
    display_fps_counter = RollingFps()
    last_shown_index = 0
    user_quit = False
    stream = FrameSource(source, loop_file=args.loop)
    try:
        stream.start()
    except RuntimeError as exc:
        # Unopenable source (missing file, busy webcam): a clear error,
        # not a raw traceback.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    worker = InferenceWorker(
        stream,
        detector,
        voter,
        cv2_module=cv2,
        tracker=args.tracker,
        snapshots_dir=snapshots_dir,
    )
    worker.start()
    try:
        while True:
            # A dead worker freezes the verdicts while the video
            # plays on — check every iteration, not never.
            if worker.error is not None:
                break
            packet = stream.read()
            if packet is not None and packet[0] != last_shown_index:
                frame_index, _ts, frame = packet
                last_shown_index = frame_index
                tick = time.monotonic()
                display_fps = display_fps_counter.tick(tick)

                detections, confirmed, inference_fps = worker.latest()
                display = draw_annotations(
                    cv2, frame, detections, confirmed,
                    hide_unconfirmed=args.hide_unconfirmed,
                )
                draw_hud(cv2, display, display_fps, inference_fps, frame_index, len(confirmed))
                if args.save_video and not writer_failed:
                    if writer is None:
                        height, width = display.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save_video,
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            stream.source_fps or FALLBACK_FPS,
                            (width, height),
                        )
                        if not writer.isOpened():
                            print(
                                f"WARNING: cannot record to {args.save_video!r}; "
                                "recording disabled for this run."
                            )
                            writer.release()
                            writer = None
                            writer_failed = True
                    if writer is not None:
                        # Raw frames by default: the annotated
                        # recording is useless as training data.
                        writer.write(frame if args.record_raw else display)
                # Only the on-screen copy is scaled to fit the screen;
                # recordings and snapshots keep the original resolution.
                shown = fit_for_display(cv2, display, args.display_width, args.display_height)
                cv2.imshow(WINDOW_NAME, shown)
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        user_quit = True
                        break  # window closed with the X button
                except cv2.error:
                    user_quit = True
                    break
            elif stream.ended and not args.loop:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                user_quit = True
                break
            if key == ord("r"):
                worker.request_reset()
            if packet is None or packet[0] == last_shown_index:
                time.sleep(0.002)
    finally:
        worker.stop()
        worker.join(timeout=WORKER_JOIN_TIMEOUT_S)
        stream.release()

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    if worker.error is not None:
        print(f"ERROR: inference failed: {worker.error}", file=sys.stderr)
        return 1
    if stream.ended and not user_quit:
        # A live source only ends by dying, and a file that produced
        # zero frames never played: neither is a clean "Demo ended."
        if stream.is_live:
            print(
                "ERROR: camera disconnected (the live stream ended "
                f"unexpectedly: {stream.end_reason or 'unknown reason'}).",
                file=sys.stderr,
            )
            return 1
        if last_shown_index == 0:
            print(
                "ERROR: could not read video (no frames decoded).",
                file=sys.stderr,
            )
            return 1
    print("Demo ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
