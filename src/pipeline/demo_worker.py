# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Inference worker thread (shared by the CLI demo and the web UI).

Owns the detector and the class voter, runs them on the newest frames,
and publishes the latest result. Snapshots are saved here, from the
frame that was actually analysed.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .annotate import draw_annotations
from .capture import FrameSource
from .rolling_fps import RollingFps
from .track_voting import TrackClassVoter, TrackedDetection
from .weapon_detector import Detection, WeaponDetector

SNAPSHOT_COOLDOWN_SECONDS = 5.0


class InferenceWorker(threading.Thread):
    """Runs detection + ByteTrack tracking + class voting on the newest frames.

    Owns the detector and the voter (single-threaded access, no locking
    around them) and publishes the latest result for the display loop.
    Snapshots are saved here, from the frame that was actually analysed.
    """

    def __init__(
        self,
        stream: FrameSource,
        detector: WeaponDetector,
        voter: TrackClassVoter,
        *,
        cv2_module: object,
        tracker: str,
        snapshots_dir: Path | None,
    ) -> None:
        super().__init__(name="sentinel-inference", daemon=True)
        self._stream = stream
        self._detector = detector
        self._voter = voter
        self._cv2 = cv2_module
        self._tracker = tracker
        self._snapshots_dir = snapshots_dir
        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._reset_requested = threading.Event()
        self._lock = threading.Lock()
        self._detections: list[Detection] = []
        self._confirmed: list[TrackedDetection] = []
        self._fps_counter = RollingFps()
        self._fps = 0.0
        self.error: str | None = None
        self._last_snapshot_at: dict[int, float] = {}

    def stop(self) -> None:
        self._paused_event.clear()
        self._stop_event.set()

    def pause(self) -> None:
        self._paused_event.set()

    def resume(self) -> None:
        self._paused_event.clear()

    @property
    def paused(self) -> bool:
        return self._paused_event.is_set()

    def request_reset(self) -> None:
        self._reset_requested.set()

    def latest(self) -> tuple[list[Detection], list[TrackedDetection], float]:
        with self._lock:
            return self._detections, self._confirmed, self._fps

    def run(self) -> None:
        last_index = 0
        while not self._stop_event.is_set():
            if self._paused_event.is_set():
                time.sleep(0.02)
                continue
            if self._reset_requested.is_set():
                self._reset_requested.clear()
                self._detector.reset_tracker()
                self._voter.reset()
                print("Tracker and class voter reset.")
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
            self._fps = self._fps_counter.tick(tick)
            last_index = frame_index

            try:
                tracked = self._voter.update(
                    self._detector.track(frame, tracker=self._tracker)
                )
            except Exception as exc:  # fail loudly instead of freezing silently
                self.error = str(exc)
                print(f"Inference error, stopping detection: {exc}")
                break
            detections = [t.detection for t in tracked]
            confirmed = [t for t in tracked if t.confirmed]
            with self._lock:
                self._detections = detections
                self._confirmed = confirmed
            self._save_snapshots(frame, detections, confirmed)

    def _save_snapshots(
        self,
        frame: object,
        detections: list[Detection],
        confirmed: list[TrackedDetection],
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
        annotated = draw_annotations(self._cv2, frame, detections, confirmed)
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
                f"(track {cd.track_id}, {cd.hits} frames, "
                f"vote {cd.vote_share:.2f}) - snapshot: {path}"
            )
