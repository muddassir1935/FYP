#!/usr/bin/env python3
"""Standalone weapon-detection demo (FYP-I).

Runs the SENTINEL weapon pipeline on a webcam, a video file, or an RTSP
stream: YOLOv8 detection (FR-02), temporal consistency confirmation
(FR-05, 6-of-10 by default), on-screen boxes, and a snapshot saved whenever
a threat is confirmed.

Playback design: capture, display and inference run on separate threads.
The display shows every frame at the source's own pace (video files are
paced to their native FPS), while detection works through frames as fast
as the hardware allows and its newest boxes are overlaid on the live
picture. On a slow (CPU-only) machine the video therefore still plays
smoothly — only the box updates arrive at the inference rate.

Model swap: ``--weights`` points at the model file. Upgrading the model
(e.g. after a 100-epoch tuning run) means dropping the new ``best.pt`` at
``models/weights/best.pt`` — no code changes.

Examples (from the repo root):
    python scripts/inference/run_weapon_demo.py                # webcam
    python scripts/inference/run_weapon_demo.py --source demo.mp4 --loop
    python scripts/inference/run_weapon_demo.py --source rtsp://cam/stream
    python scripts/inference/run_weapon_demo.py --imgsz 416    # faster on CPU

Controls: ``q`` quit, ``r`` reset the temporal buffer.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import (
    ConfirmedDetection,
    Detection,
    FrameSource,
    TemporalConsistencyBuffer,
    WeaponDetector,
)

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
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Detection confidence threshold (default 0.40 — at 0.25 the model "
        "emits many background false alarms; see docs/training_run_log.md).",
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


def draw_annotations(
    cv2_module: object,
    frame: object,
    detections: list[Detection],
    confirmed: list[ConfirmedDetection],
    window: int,
    *,
    hide_unconfirmed: bool = False,
) -> object:
    """Return a copy of ``frame`` with detection/confirmation boxes drawn.

    Confirmed threats are thick red with a filled label chip; unconfirmed
    detections are thin muted gray (raw model output always contains some
    noise — the FR-05 confirmation is the system's actual verdict).
    """
    display = frame.copy()
    confirmed_by_id = {id(cd.detection): cd for cd in confirmed}
    font = cv2_module.FONT_HERSHEY_SIMPLEX
    for det in detections:
        cd = confirmed_by_id.get(id(det))
        if cd is None and hide_unconfirmed:
            continue
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        if cd is not None:
            color, thickness = (0, 0, 255), 3
            label = f"{det.class_name} {det.confidence:.2f} CONFIRMED {cd.hits}/{window}"
        else:
            color, thickness = (170, 170, 170), 1
            label = f"{det.class_name} {det.confidence:.2f}"
        cv2_module.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        (text_w, text_h), baseline = cv2_module.getTextSize(label, font, 0.55, 2)
        chip_h = text_h + baseline + 6
        chip_y1 = y1 - chip_h if y1 >= chip_h else y1
        chip_y2 = chip_y1 + chip_h
        cv2_module.rectangle(display, (x1, chip_y1), (x1 + text_w + 8, chip_y2), color, -1)
        cv2_module.putText(
            display, label, (x1 + 4, chip_y2 - baseline - 3),
            font, 0.55, (255, 255, 255), 2, cv2_module.LINE_AA,
        )
    return display


def fit_for_display(cv2_module: object, frame: object, max_width: int, max_height: int) -> object:
    """Scale ``frame`` to fit inside (max_width, max_height), keeping aspect."""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale == 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    interpolation = cv2_module.INTER_AREA if scale < 1.0 else cv2_module.INTER_LINEAR
    return cv2_module.resize(frame, new_size, interpolation=interpolation)


class InferenceWorker(threading.Thread):
    """Runs detection + temporal confirmation on the newest frames.

    Owns the detector and the buffer (single-threaded access, no locking
    around them) and publishes the latest result for the display loop.
    Snapshots are saved here, from the frame that was actually analysed.
    """

    def __init__(
        self,
        stream: FrameSource,
        detector: WeaponDetector,
        buffer: TemporalConsistencyBuffer,
        *,
        cv2_module: object,
        window: int,
        snapshots_dir: Path | None,
    ) -> None:
        super().__init__(name="sentinel-inference", daemon=True)
        self._stream = stream
        self._detector = detector
        self._buffer = buffer
        self._cv2 = cv2_module
        self._window = window
        self._snapshots_dir = snapshots_dir
        self._stop_event = threading.Event()
        self._reset_requested = threading.Event()
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._confirmed: list[ConfirmedDetection] = []
        self._fps = 0.0
        self._last_snapshot_at: dict[int, float] = {}

    def stop(self) -> None:
        self._stop_event.set()

    def request_reset(self) -> None:
        self._reset_requested.set()

    def latest(self) -> tuple[list[Detection], list[ConfirmedDetection], float]:
        with self._lock:
            return self._detections, self._confirmed, self._fps

    def run(self) -> None:
        last_index = 0
        last_tick = 0.0
        while not self._stop_event.is_set():
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self._buffer.reset()
                print("Temporal buffer reset.")
            packet = self._stream.read()
            if packet is None:
                if self._stream.ended:
                    break
                time.sleep(0.003)
                continue
            frame_index, _ts, frame = packet
            if frame_index == last_index:
                if self._stream.ended:
                    break
                time.sleep(0.002)
                continue
            tick = time.monotonic()
            if last_tick:
                instant = 1.0 / max(tick - last_tick, 1e-6)
                self._fps = instant if self._fps == 0.0 else 0.9 * self._fps + 0.1 * instant
            last_tick = tick
            last_index = frame_index

            detections = self._detector.detect(frame)
            confirmed = self._buffer.update(detections)
            with self._lock:
                self._detections = detections
                self._confirmed = confirmed
            self._save_snapshots(frame, detections, confirmed)

    def _save_snapshots(
        self,
        frame: object,
        detections: list[Detection],
        confirmed: list[ConfirmedDetection],
    ) -> None:
        if self._snapshots_dir is None or not confirmed:
            return
        now = time.monotonic()
        fresh = [
            cd
            for cd in confirmed
            if now - self._last_snapshot_at.get(cd.track_id, 0.0)
            >= SNAPSHOT_COOLDOWN_SECONDS
        ]
        if not fresh:
            return
        annotated = draw_annotations(self._cv2, frame, detections, confirmed, self._window)
        for cd in fresh:
            self._last_snapshot_at[cd.track_id] = now
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = (
                self._snapshots_dir
                / f"threat_{cd.detection.class_name}_{stamp}_t{cd.track_id}.jpg"
            )
            self._cv2.imwrite(str(path), annotated)
            print(
                f"[ALERT] {cd.detection.class_name} confirmed "
                f"({cd.hits}/{self._window} frames) — snapshot: {path}"
            )


def main() -> int:
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed; run: pip install opencv-python", file=sys.stderr)
        return 1

    detector = WeaponDetector(
        args.weights, conf_threshold=args.conf, imgsz=args.imgsz, device=args.device
    )
    buffer = TemporalConsistencyBuffer(window=args.window, required_hits=args.hits)
    source: int | str = int(args.source) if args.source.isdigit() else args.source

    snapshots_dir: Path | None = None
    if not args.no_snapshots:
        snapshots_dir = Path(args.snapshots_dir)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    print(f"SENTINEL demo — source={args.source!r} weights={args.weights} imgsz={args.imgsz}")
    print("Press q to quit, r to reset the temporal buffer.")

    writer = None
    display_fps = 0.0
    last_shown_index = 0
    last_show_tick = 0.0
    window_name = "SENTINEL — weapon detection demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    with FrameSource(source, loop_file=args.loop) as stream:
        worker = InferenceWorker(
            stream,
            detector,
            buffer,
            cv2_module=cv2,
            window=args.window,
            snapshots_dir=snapshots_dir,
        )
        worker.start()
        try:
            while True:
                packet = stream.read()
                if packet is not None and packet[0] != last_shown_index:
                    frame_index, _ts, frame = packet
                    last_shown_index = frame_index
                    tick = time.monotonic()
                    if last_show_tick:
                        instant = 1.0 / max(tick - last_show_tick, 1e-6)
                        display_fps = (
                            instant
                            if display_fps == 0.0
                            else 0.9 * display_fps + 0.1 * instant
                        )
                    last_show_tick = tick

                    detections, confirmed, inference_fps = worker.latest()
                    display = draw_annotations(
                        cv2, frame, detections, confirmed, args.window,
                        hide_unconfirmed=args.hide_unconfirmed,
                    )
                    hud = (
                        f"FPS {display_fps:.1f} (detect {inference_fps:.1f}) | "
                        f"frame {frame_index} | confirmed threats: {len(confirmed)}"
                    )
                    # Dark translucent bar keeps the white HUD readable on
                    # bright frames (it used to vanish against the sky etc.).
                    bar = display.copy()
                    cv2.rectangle(bar, (0, 0), (display.shape[1], 44), (0, 0, 0), -1)
                    cv2.addWeighted(bar, 0.55, display, 0.45, 0, display)
                    cv2.putText(
                        display, hud, (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
                    )
                    if args.save_video:
                        if writer is None:
                            height, width = display.shape[:2]
                            writer = cv2.VideoWriter(
                                args.save_video,
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                stream.source_fps or 25.0,
                                (width, height),
                            )
                        writer.write(display)
                    # Only the on-screen copy is scaled to fit the screen;
                    # recordings and snapshots keep the original resolution.
                    shown = fit_for_display(cv2, display, args.display_width, args.display_height)
                    cv2.imshow(window_name, shown)
                    cv2.resizeWindow(window_name, shown.shape[1], shown.shape[0])
                elif stream.ended and not args.loop:
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    worker.request_reset()
                if packet is None or packet[0] == last_shown_index:
                    time.sleep(0.002)
        finally:
            worker.stop()
            worker.join(timeout=2.0)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("Demo ended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
