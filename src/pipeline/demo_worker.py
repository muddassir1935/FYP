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


def _hash_weights(
    weights: object,
) -> tuple[str | None, str | None, str | None]:
    """(weights sha256 | None, xml sha256 | None, note | None).

    A ``.pt`` file is hashed directly. An OpenVINO folder has no single
    weights file: the ``.bin`` holds the weight VALUES, so it is the
    hash that identifies the model — the ``.xml`` only describes
    structure, and two models with the same architecture (v3 vs v4)
    can have near-identical ``.xml`` files. Both are therefore hashed:
    ``weights_sha256`` is the ``.bin`` (preferring the one matching the
    ``.xml`` stem), ``weights_xml_sha256`` is the ``.xml``. A folder
    with neither records null hashes with a note saying so. Anything
    else (detector without a weights path, missing file) likewise
    records nulls + a note — never raises.
    """
    if not isinstance(weights, (str, Path)):
        return None, None, "weights path unavailable from detector"
    path = Path(weights)
    if path.is_file():
        return _sha256_file(path), None, None
    if path.is_dir():
        xmls = sorted(path.glob("*.xml"))
        bins = sorted(path.glob("*.bin"))
        xml_hash = _sha256_file(xmls[0]) if xmls else None
        bin_hash: str | None = None
        bin_name: str | None = None
        if bins:
            if xmls:
                match = [b for b in bins if b.stem == xmls[0].stem]
                chosen = match[0] if match else bins[0]
            else:
                chosen = bins[0]
            bin_hash = _sha256_file(chosen)
            bin_name = chosen.name
        if bin_hash is not None and xml_hash is not None:
            note = (
                "openvino model folder; hashed "
                f"{bin_name} (+ {xmls[0].name})"
            )
        elif bin_hash is not None:
            note = (
                "openvino model folder; hashed "
                f"{bin_name} (no .xml found)"
            )
        elif xml_hash is not None:
            note = (
                "openvino model folder; hashed "
                f"{xmls[0].name} (no .bin found)"
            )
        else:
            note = "openvino model folder; no .xml or .bin found to hash"
        return bin_hash, xml_hash, note
    return None, None, f"weights path does not exist: {path}"


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
        source: str | None = None,
        recording: str | None = None,
    ) -> None:
        self.path: Path | None = None
        self._handle: object | None = None
        self._broken = False
        self._lines_since_flush = 0
        self._summary_written = False
        # Wall-clock start (for the header/footer) + monotonic start
        # (for the run duration — wall clock can jump, monotonic cannot).
        self._started_at = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        self._started_monotonic = time.monotonic()
        self._source = source
        self._recording = recording
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

    def _header(
        self, detector: object, voter: TrackClassVoter, tracker: str
    ) -> dict:
        weights = getattr(detector, "weights", None)
        weights_hash, weights_xml_hash, weights_note = _hash_weights(weights)
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
            "started_at": self._started_at,
            # Source + recording tie this log to its video and its raw
            # recording (the detection log joins the timestamp sidecar
            # on frame_index). Both are None when not applicable.
            "source": self._source,
            "recording": self._recording,
            "weights": (
                str(weights) if isinstance(weights, (str, Path)) else None
            ),
            # For OpenVINO this is the .bin hash (the weight values);
            # the .xml hash rides alongside it.
            "weights_sha256": weights_hash,
            "weights_xml_sha256": weights_xml_hash,
            "weights_note": weights_note,
            "model_label": _model_label(weights),
            "conf_threshold": (
                float(fallback) if isinstance(fallback, (int, float)) else None
            ),
            "conf_thresholds": thresholds,
            "min_hits": getattr(voter, "min_hits", None),
            "confirm_window_s": getattr(voter, "confirm_window_s", None),
            "min_span_s": getattr(voter, "min_span_s", None),
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

    def close(
        self,
        *,
        frames_processed: int | None = None,
        frames_seen_by_capture: int | None = None,
    ) -> None:
        """Flush, write the closing run-summary line, and close.

        The footer is what makes false alarms per hour computable:
        without it a log of detections alone cannot say how long the
        run lasted or how many frames it processed. Counts default to
        0 (a run that never started processed nothing). Written once —
        a second close() is a no-op — and never on a broken log.
        """
        handle, self._handle = self._handle, None
        if handle is None:
            return
        if not self._broken and not self._summary_written:
            self._summary_written = True
            ended_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            summary = {
                "type": "run_summary",
                "frames_processed": (
                    int(frames_processed) if frames_processed is not None else 0
                ),
                "frames_seen_by_capture": (
                    int(frames_seen_by_capture)
                    if frames_seen_by_capture is not None
                    else 0
                ),
                "started_at": self._started_at,
                "ended_at": ended_at,
                "duration_s": round(
                    time.monotonic() - self._started_monotonic, 3
                ),
            }
            try:
                handle.write(json.dumps(summary) + "\n")
            except Exception:
                pass  # best effort: the header + detections are already there
        try:
            handle.flush()
        except Exception:
            pass
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
        source: str | None = None,
        recording: str | None = None,
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
        # `source` (video file / webcam name) and `recording` (raw
        # recording filename, when recording is on) go in the header so
        # each log ties to its recording and its interval labels.
        self._log = (
            _DetectionLog(
                log_dir,
                detector=detector,
                voter=voter,
                tracker=tracker,
                source=source,
                recording=recording,
            )
            if log_dir is not None
            else None
        )
        # Frames this worker actually ran inference on (the footer's
        # frames_processed — distinct from frames the capture saw).
        self._frames_processed = 0
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
            # whatever stopped it (end of stream, error, stop()). The
            # footer carries the frame counts for false alarms per hour.
            if self._log is not None:
                self._log.close(
                    frames_processed=self._frames_processed,
                    frames_seen_by_capture=self._frames_seen_by_capture(),
                )

    def _frames_seen_by_capture(self) -> int:
        """Frames the capture source read (0 when the stream has no
        counter — e.g. a test fake). Guarded by type, not just
        presence: a Mock attribute must not leak into the footer."""
        seen = getattr(self._stream, "frames_seen", None)
        return seen if isinstance(seen, int) else 0

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
            self._frames_processed += 1
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
