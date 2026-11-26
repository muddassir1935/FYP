# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Inference worker thread (shared by the CLI demo and the web UI).

Owns the detector and the class voter, runs them on the newest frames,
and publishes the latest result. Snapshots are saved here, from the
frame that was actually analysed.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from .annotate import draw_annotations
from .capture import FrameSource
from .rolling_fps import RollingFps
from .track_voting import TrackClassVoter, TrackedDetection
from .weapon_detector import Detection, WeaponDetector

SNAPSHOT_COOLDOWN_SECONDS = 5.0
# Idle sleep while paused.
PAUSED_SLEEP_S = 0.02
# Repo-root logs/ dir (derived from this file, never hardcoded): the
# default home of per-detection JSONL logs. Verified gitignored
# (.gitignore section 6: `logs/`), so log files are never committed.
_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
# Buffered log writes are flushed every N lines and on stop.
LOG_FLUSH_EVERY_LINES = 50


def _sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed in 1 MiB chunks (one pass)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_weights(weights: object) -> tuple[str | None, str | None]:
    """(sha256 hex | None, note | None) identifying the model weights.

    A ``.pt`` file is hashed directly. An OpenVINO folder has no single
    weights file, so the model ``.xml`` inside it is hashed instead (the
    ``.bin`` holds the same weights the ``.xml`` describes); a folder
    with no ``.xml`` records a null hash with a note saying so.
    Anything else (detector without a weights path, missing file)
    likewise records null + a note — never raises.
    """
    if not isinstance(weights, (str, Path)):
        return None, "weights path unavailable from detector"
    path = Path(weights)
    if path.is_file():
        return _sha256_file(path), None
    if path.is_dir():
        xmls = sorted(path.glob("*.xml"))
        if xmls:
            return (
                _sha256_file(xmls[0]),
                f"openvino model folder; hashed {xmls[0].name}",
            )
        return None, "openvino model folder; no .xml found to hash"
    return None, f"weights path does not exist: {path}"


def _model_label(weights: object) -> str | None:
    """Short model label for the log header (``best`` / ``v3`` ...)."""
    if not isinstance(weights, (str, Path)):
        return None
    path = Path(weights)
    if path.is_file():
        return path.stem
    if path.is_dir():
        name = path.name
        return name[: -len("_openvino_model")] if name.endswith("_openvino_model") else name
    return None


class _DetectionLog:
    """Buffered per-detection JSONL log for one worker run.

    One file per run (``detections_<stamp>_<pid>.jsonl`` in the log
    dir). The first line is a header carrying the weights SHA-256, the
    confidence thresholds and the voter settings, so a log file is
    self-describing. A log failure (unwritable dir, full disk) disables
    logging with one WARNING and never stops detection.
    """

    def __init__(
        self,
        log_dir: Path,
        *,
        detector: object,
        voter: TrackClassVoter,
        tracker: str,
    ) -> None:
        self.path: Path | None = None
        self._handle: object | None = None
        self._broken = False
        self._lines_since_flush = 0
        try:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            base = f"detections_{stamp}_{os.getpid()}"
            path = log_dir / f"{base}.jsonl"
            n = 2
            while path.exists():
                path = log_dir / f"{base}_{n}.jsonl"
                n += 1
            self.path = path
            self._handle = open(path, "w", encoding="utf-8", buffering=8192)
            if not self._write_line(self._header(detector, voter, tracker)):
                raise OSError("could not write log header")
            # The header must be durable immediately: a run that is
            # killed (or crashes) before 50 lines still leaves a
            # self-describing file behind.
            self.flush()
        except Exception as exc:
            print(f"WARNING: detection logging disabled ({exc})")
            self._broken = True
            self._handle = None
            self.path = None

    @staticmethod
    def _header(
        detector: object, voter: TrackClassVoter, tracker: str
    ) -> dict:
        weights = getattr(detector, "weights", None)
        weights_hash, weights_note = _hash_weights(weights)
        thresholds: dict[str, float] | None = None
        raw = getattr(detector, "conf_thresholds", None)
        names = getattr(detector, "class_names", None)
        if isinstance(raw, dict):
            thresholds = {}
            for class_id, value in raw.items():
                if not isinstance(value, (int, float)):
                    continue
                label = (
                    names.get(class_id, str(class_id))
                    if isinstance(names, dict)
                    else str(class_id)
                )
                thresholds[label] = round(float(value), 4)
        fallback = getattr(detector, "conf_threshold", None)
        return {
            "type": "header",
            "started_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "weights": (
                str(weights) if isinstance(weights, (str, Path)) else None
            ),
            "weights_sha256": weights_hash,
            "weights_note": weights_note,
            "model_label": _model_label(weights),
            "conf_threshold": (
                float(fallback) if isinstance(fallback, (int, float)) else None
            ),
            "conf_thresholds": thresholds,
            "min_hits": getattr(voter, "min_hits", None),
            "confirm_window_s": getattr(voter, "confirm_window_s", None),
            "tracker": tracker,
        }

    def write_detection(
        self,
        *,
        timestamp: str,
        monotonic_s: float,
        frame_index: int,
        tracked: TrackedDetection,
        raw_class_name: str,
    ) -> None:
        """Append one JSONL line for a detection (no-op if disabled)."""
        if self._broken or self._handle is None:
            return
        det = tracked.detection
        line = {
            "type": "detection",
            "timestamp": timestamp,
            "monotonic_s": round(monotonic_s, 3),
            "frame_index": frame_index,
            "track_id": tracked.track_id,
            "raw_class": raw_class_name,
            "voted_class": det.class_name,
            "confidence": round(float(det.confidence), 4),
            "bbox": [round(float(v), 1) for v in det.bbox],
            "confirmed": bool(tracked.confirmed),
        }
        if self._write_line(line):
            self._lines_since_flush += 1
            if self._lines_since_flush >= LOG_FLUSH_EVERY_LINES:
                self.flush()

    def flush(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            self._lines_since_flush = 0
        except Exception as exc:
            self._fail(f"flush failed: {exc}")

    def close(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass

    def _write_line(self, record: dict) -> bool:
        try:
            self._handle.write(json.dumps(record) + "\n")
            return True
        except Exception as exc:
            self._fail(f"write failed: {exc}")
            return False

    def _fail(self, why: str) -> None:
        if not self._broken:
            print(
                f"WARNING: detection logging disabled ({why}) "
                "— detection continues."
            )
        self._broken = True
        self.close()



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
        log_dir: Path | None = _DEFAULT_LOG_DIR,
    ) -> None:
        super().__init__(name="sentinel-inference", daemon=True)
        self._stream = stream
        self._detector = detector
        self._voter = voter
        self._cv2 = cv2_module
        self._tracker = tracker
        self._snapshots_dir = snapshots_dir
        # Per-detection JSONL log (one file per run, repo-root logs/ by
        # default — both the CLI demo and the UI runner construct this
        # worker, so both get logging with no behaviour change to the
        # stream). Explicit None disables logging; a failure to open
        # the log only warns (see _DetectionLog), it never stops work.
        self._log = (
            _DetectionLog(
                log_dir, detector=detector, voter=voter, tracker=tracker
            )
            if log_dir is not None
            else None
        )
        self.log_path: Path | None = self._log.path if self._log else None
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
        # A worker that never started never runs its finally-close:
        # close the log here instead (idempotent; a no-op while the
        # thread is alive, when run()'s finally owns the close).
        if not self.is_alive() and self._log is not None:
            self._log.close()

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

    def _set_error(self, message: str) -> None:
        """Record a fatal worker error (under the lock, like _fps)."""
        with self._lock:
            self.error = message

    def run(self) -> None:
        try:
            self._run_loop()
        finally:
            # Flush + close the detection log when the thread exits,
            # whatever stopped it (end of stream, error, stop()).
            if self._log is not None:
                self._log.close()

    def _run_loop(self) -> None:
        last_index = 0
        while not self._stop_event.is_set():
            if self._paused_event.is_set():
                time.sleep(PAUSED_SLEEP_S)
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
            frame_index, ts, frame = packet
            if frame_index == last_index:
                if self._stream.ended:
                    break
                time.sleep(0.002)
                continue
            tick = time.monotonic()
            with self._lock:
                self._fps = self._fps_counter.tick(tick)
            last_index = frame_index

            try:
                # The packet timestamp (monotonic, stamped at capture),
                # not the processing time: a slow inference loop must
                # not stretch the voter's confirmation window.
                tracked = self._voter.update(
                    self._detector.track(frame, tracker=self._tracker), now=ts
                )
            except Exception as exc:  # fail loudly instead of freezing silently
                self._set_error(str(exc))
                print(f"Inference error, stopping detection: {exc}")
                break
            detections = [t.detection for t in tracked]
            confirmed = [t for t in tracked if t.confirmed]
            with self._lock:
                self._detections = detections
                self._confirmed = confirmed
            # One JSONL line per detection in this frame (raw + voted
            # class, confirmed flag). A log failure only disables
            # logging — _DetectionLog swallows it, detection continues.
            self._log_detections(frame_index, ts, tracked)
            # Snapshot saving is inside the worker's error handling too:
            # it used to sit after the try, so an exception here killed
            # the thread without setting self.error — a silent freeze.
            try:
                self._save_snapshots(frame, detections, confirmed)
            except Exception as exc:
                self._set_error(f"Snapshot save failed: {exc}")
                print(f"Snapshot error, stopping detection: {exc}")
                break

    def _log_detections(
        self,
        frame_index: int,
        monotonic_s: float,
        tracked: list[TrackedDetection],
    ) -> None:
        """One JSONL line per detection in this frame (no-op if logging
        is disabled). The wall-clock ``timestamp`` is for humans; the
        packet's monotonic ``monotonic_s`` is the voter's time base."""
        if self._log is None or not tracked:
            return
        names = getattr(self._detector, "class_names", None)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        for t in tracked:
            raw_name = (
                names.get(t.raw_class_id, str(t.raw_class_id))
                if isinstance(names, dict)
                else str(t.raw_class_id)
            )
            self._log.write_detection(
                timestamp=timestamp,
                monotonic_s=monotonic_s,
                frame_index=frame_index,
                tracked=t,
                raw_class_name=raw_name,
            )

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
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = (
                self._snapshots_dir
                / f"threat_{cd.detection.class_name}_{stamp}_t{cd.track_id}.jpg"
            )
            # Check the write: a failed save (full disk, deleted dir)
            # must not consume the cooldown or print a success line.
            if not self._cv2.imwrite(str(path), annotated):
                print(
                    f"WARNING: could not save snapshot {path} "
                    "(disk full or snapshots dir gone?) — will retry "
                    "on the next confirmed frame."
                )
                continue
            self._last_snapshot_at[cd.track_id] = now
            print(
                f"[ALERT] {cd.detection.class_name} confirmed "
                f"(track {cd.track_id}, {cd.hits} frames, "
                f"vote {cd.vote_share:.2f}) - snapshot: {path}"
            )
