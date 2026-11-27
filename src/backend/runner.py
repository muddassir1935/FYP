# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""One detection run for the demo web UI.

Owns the pipeline (``FrameSource`` -> ``WeaponDetector`` -> ``ByteTrack``
-> ``TrackClassVoter``) in two threads, exactly like the CLI demo:

* the inference worker (``src.pipeline.InferenceWorker``) runs detection
  as fast as the model allows;
* a display thread reads frames at video rate, draws the latest
  detections on them, JPEG-encodes the result, and (optionally) records
  the video — raw (un-annotated) frames by default, so a gate session
  yields minable training data, annotated on request.

The FastAPI layer only ever touches this class: it starts/stops runs,
reads the latest JPEG for the MJPEG stream, and polls ``status()``.
Inference never runs on the FastAPI event loop.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline import (
    FALLBACK_FPS,
    FrameSource,
    InferenceWorker,
    RollingFps,
    TimestampSidecar,
    TrackClassVoter,
    WeaponDetector,
    detect_openvino_device,
    draw_annotations,
)
from src.pipeline.thresholds import CONF_GUN, CONF_KNIFE, DEFAULT_CONF

logger = logging.getLogger("sentinel.demo_runner")

# How long teardown waits for the worker / display thread before giving
# up (and, for the worker, dropping the cached detector).
JOIN_TIMEOUT_S = 5.0
# Idle sleep in the display loop while paused.
PAUSED_POLL_SLEEP_S = 0.02

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
SNAPSHOT_RE = re.compile(
    r"^threat_(?P<class>Gun|Knife)_(?P<stamp>\d{8}_\d{6})_t(?P<track>\d+)\.jpg$"
)


@dataclass(frozen=True)
class RunConfig:
    """Validated settings for one run (built by the API layer)."""

    source_type: str  # "webcam" | "video"
    source: str  # webcam index ("0") or video filename inside videos_dir
    weights_id: str  # filename (.pt) or dirname (OpenVINO) inside weights_dir
    conf_gun: float = CONF_GUN
    conf_knife: float = CONF_KNIFE
    imgsz: int = 640
    min_hits: int = 3
    confirm_window_s: float = 1.0
    # Minimum observed presence for confirmation (see TrackClassVoter):
    # a burst of sightings inside this span never confirms.
    confirm_min_span_s: float = 0.4
    tracker: str = "bytetrack.yaml"
    loop: bool = False
    hide_unconfirmed: bool = False
    save_video: bool = False
    # With save_video on, record the RAW frames by default (annotated
    # video is useless as training data). False restores the old
    # annotated recording.
    record_raw: bool = True


@dataclass
class RunInfo:
    """What the API returns about a run (live or finished)."""

    state: str  # idle | starting | running | paused | stopping | stopped | error
    source_label: str = ""
    weights_label: str = ""
    display_fps: float = 0.0
    detect_fps: float = 0.0
    frames: int = 0
    detections: int = 0
    confirmed: int = 0
    snapshots: int = 0
    started_at: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    # Non-fatal problem for the status line (GPU fallback, failed
    # recording). Cleared on every start.
    warning: str | None = None
    saved_video: str | None = None


class DemoRunner:
    """Runs one detection at a time; thread-safe for the API layer."""

    def __init__(
        self,
        *,
        videos_dir: Path,
        weights_dir: Path,
        snapshots_dir: Path,
        save_video_dir: Path | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.videos_dir = Path(videos_dir)
        self.weights_dir = Path(weights_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.save_video_dir = Path(save_video_dir) if save_video_dir else None
        # Per-detection JSONL log dir, forwarded to InferenceWorker.
        # None means "let the worker use its own default" (repo-root
        # logs/) — passing None through would DISABLE logging, so the
        # worker kwarg is only set when a dir was given (see start()).
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._state = "idle"
        self._config: RunConfig | None = None
        self._error: str | None = None
        self._warning: str | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._paused_at: float | None = None
        self._paused_total_s: float = 0.0
        self._saved_video: str | None = None
        self._snapshots_at_start = 0
        # Set by stop() while state is "starting"; the loader checks it
        # after each slow step so a start can be cancelled.
        self._cancel_start = threading.Event()
        # worker.latest() failures are logged once per run, not on
        # every 500 ms status poll.
        self._latest_error_logged = False

        # Live frame + stats, written by the display thread.
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self._frames_shown = 0
        self._display_fps = 0.0

        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None
        self._worker: InferenceWorker | None = None
        self._stream: FrameSource | None = None
        # Recording filename fixed in start() (see below) so the
        # worker's log header can name the recording.
        self._planned_recording_path: str | None = None
        # Warmed-up detector cache keyed by (resolved_weights_path, imgsz, device).
        # Avoids recompiling OpenVINO/CUDA graphs on every consecutive run.
        self._cached_detector_key: tuple[Path, int, str | None] | None = None
        self._cached_detector: WeaponDetector | None = None

    # -- public API -----------------------------------------------------

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def start(self, config: RunConfig) -> RunInfo:
        """Start a run. Raises ValueError on a bad config, RuntimeError
        if a run is already active or the model cannot load."""
        with self._lock:
            if self._state in ("starting", "running", "paused"):
                raise RuntimeError("A run is already active — stop it first.")
            weights_path = self._resolve_weights(config.weights_id)
            source = self._resolve_source(config)
            self._state = "starting"
            self._config = config
            self._error = None
            self._warning = None
            self._latest_error_logged = False
            self._cancel_start.clear()
            self._saved_video = None
            self._planned_recording_path = None
            self._snapshots_at_start = len(self.list_snapshots())
            self._latest_jpeg = None
            self._frame_seq = 0
            self._frames_shown = 0
            self._display_fps = 0.0
            self._started_at = None
            self._stopped_at = None
            self._paused_at = None
            self._paused_total_s = 0.0
            self._stop_event.clear()

        # Model load + warmup happen outside the lock (while state is
        # "starting") and BEFORE FrameSource opens, so frame 1 of the
        # video is analyzed at full speed with zero missed opening seconds.
        try:
            detector = self._get_or_load_detector(weights_path, config)
        except FileNotFoundError as exc:
            self._fail(str(exc))
            raise ValueError(str(exc)) from exc
        except Exception as exc:  # bad weights file, missing openvino, ...
            self._fail(f"Could not load weights {config.weights_id!r}: {exc}")
            raise RuntimeError(
                f"Could not load weights {config.weights_id!r}: {exc}"
            ) from exc
        self._check_cancelled()

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment issue
            self._fail("opencv-python is not installed.")
            raise RuntimeError(
                "opencv-python is not installed; run: pip install opencv-python"
            ) from exc

        voter = TrackClassVoter(
            min_hits=config.min_hits,
            confirm_window_s=config.confirm_window_s,
            min_span_s=config.confirm_min_span_s,
        )
        # The recording filename is fixed up front — not when the first
        # frame is recorded — so the worker's log header can name it.
        # A stamp collision across runs in the same second gets a
        # numeric suffix (demo_<stamp>_raw.mp4, demo_<stamp>_raw_1.mp4,
        # ...) instead of overwriting the earlier recording.
        planned_recording: str | None = None
        if config.save_video and self.save_video_dir is not None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            planned_recording = str(
                self._unique_recording_path(
                    self.save_video_dir, stamp, config.record_raw
                )
            )
        self._planned_recording_path = planned_recording
        stream = FrameSource(source, loop_file=config.loop)
        try:
            stream.start()
        except Exception as exc:
            try:
                stream.release()
            except Exception:
                pass
            self._fail(f"Could not open source {config.source!r}: {exc}")
            # A bad source is the caller's fault (bad file, busy
            # webcam), not a server error — ValueError maps to 400.
            raise ValueError(
                f"Could not open source {config.source!r}: {exc}"
            ) from exc
        if self._cancel_start.is_set():
            stream.release()
            self._fail("Run cancelled while starting.")
            raise RuntimeError("Run cancelled while starting.")

        worker_kwargs: dict = {}
        if self.log_dir is not None:
            # Explicit log dir only: omitting the kwarg keeps the
            # worker's own default (repo-root logs/).
            worker_kwargs["log_dir"] = self.log_dir
        worker = InferenceWorker(
            stream,
            detector,
            voter,
            cv2_module=cv2,
            tracker=config.tracker,
            snapshots_dir=self.snapshots_dir,
            # Header provenance: the video/webcam name + the planned
            # recording filename. When the writer cannot be opened the
            # header still names the planned file (the display loop
            # warns loudly in that case) — the file simply never appears.
            source=self._source_label(config),
            recording=planned_recording,
            **worker_kwargs,
        )
        with self._lock:
            self._stream = stream
            self._worker = worker
            self._state = "running"
            self._started_at = time.time()
            self._stopped_at = None
        if self._cancel_start.is_set():
            # Stop landed between the last check and the thread starts:
            # tear down (the joins are guarded for never-started
            # threads) instead of launching a cancelled run.
            self._teardown()
            self._fail("Run cancelled while starting.")
            raise RuntimeError("Run cancelled while starting.")
        try:
            worker.start()
            with self._lock:
                torn_down = self._state != "running"
            if torn_down:
                # stop() raced start() and tore the run down while the
                # worker was starting: do not launch a display thread
                # for a run that is already finished.
                return self.status()
            self._display_thread = threading.Thread(
                target=self._display_loop, args=(cv2,), name="sentinel-display",
                daemon=True,
            )
            self._display_thread.start()
        except Exception as exc:
            self._teardown()
            self._fail(f"Could not start the run threads: {exc}")
            raise RuntimeError(f"Could not start the run threads: {exc}") from exc
        logger.info(
            "run started: source=%s (%s) weights=%s conf Gun=%.2f Knife=%.2f "
            "imgsz=%d min_hits=%d confirm_window=%.1fs confirm_min_span=%.1fs "
            "save_video=%s record_raw=%s",
            config.source, config.source_type, config.weights_id,
            config.conf_gun, config.conf_knife, config.imgsz,
            config.min_hits, config.confirm_window_s,
            config.confirm_min_span_s, config.save_video, config.record_raw,
        )
        return self.status()

    def _check_cancelled(self) -> None:
        """Raise if stop() cancelled this start while it was loading."""
        if self._cancel_start.is_set():
            self._fail("Run cancelled while starting.")
            raise RuntimeError("Run cancelled while starting.")

    def pause(self) -> RunInfo:
        """Pause an active run, holding the current frame and timer."""
        with self._lock:
            if self._state != "running":
                raise RuntimeError("No active run to pause.")
            self._state = "paused"
            self._paused_at = time.time()
            stream, worker = self._stream, self._worker
        if stream is not None:
            stream.pause()
        if worker is not None:
            worker.pause()
        return self.status()

    def resume(self) -> RunInfo:
        """Resume a paused run from the exact paused position."""
        with self._lock:
            if self._state != "paused":
                raise RuntimeError("No paused run to resume.")
            if self._paused_at is not None:
                self._paused_total_s += max(0.0, time.time() - self._paused_at)
                self._paused_at = None
            self._state = "running"
            stream, worker = self._stream, self._worker
        if stream is not None:
            stream.resume()
        if worker is not None:
            worker.resume()
        return self.status()

    def stop(self) -> RunInfo:
        """Stop the active or paused run (idempotent) and return final stats.

        While a run is "starting", Stop cancels the start instead: it
        sets the flag the loader checks, and the start fails with
        "Run cancelled while starting." once the current slow step
        (model load / warmup / source open) returns.

        ``status()`` takes ``self._lock`` itself, so it is never called
        while this method holds the lock (``threading.Lock`` is not
        re-entrant — that deadlocked the first version).
        """
        with self._lock:
            cancelling = self._state == "starting"
            if cancelling:
                self._cancel_start.set()
            idle = self._state not in ("running", "paused")
            if not idle and not cancelling:
                if self._paused_at is not None:
                    self._paused_total_s += max(
                        0.0, time.time() - self._paused_at
                    )
                    self._paused_at = None
                self._state = "stopping"
        if cancelling or idle:
            return self.status()
        # Confirmed count must be read before teardown drops the worker.
        confirmed = 0
        if self._worker is not None:
            try:
                _dets, conf, _fps = self._worker.latest()
                confirmed = len(conf)
            except Exception:
                pass
        self._teardown()
        with self._lock:
            self._state = "stopped"
            self._stopped_at = time.time()
        info = self.status()
        logger.info(
            "run stopped: frames=%d confirmed=%d snapshots=%d elapsed=%.1fs",
            info.frames, confirmed, info.snapshots, info.elapsed_s,
        )
        return info

    def _teardown(self) -> None:
        """Release the worker, display thread, and stream without
        changing the run state. Safe to call from the display thread
        itself (it never joins the current thread)."""
        self._stop_event.set()
        worker, stream = self._worker, self._stream
        display_thread = self._display_thread
        if worker is not None:
            worker.stop()
            # Guard the join: Stop racing start() can reach here while
            # the worker thread was created but never started, and
            # join() on a never-started thread raises RuntimeError.
            if worker.is_alive():
                worker.join(timeout=JOIN_TIMEOUT_S)
                if worker.is_alive():
                    # The old worker may still be inside
                    # detector.track(); drop the cached detector so the
                    # next run loads a fresh one instead of sharing a
                    # single YOLO object (and its tracker state)
                    # between two threads.
                    logger.warning(
                        "worker thread did not exit in %.1fs; dropping "
                        "the cached detector",
                        JOIN_TIMEOUT_S,
                    )
                    self._cached_detector = None
                    self._cached_detector_key = None
        if (
            display_thread is not None
            and display_thread is not threading.current_thread()
            and display_thread.is_alive()
        ):
            display_thread.join(timeout=JOIN_TIMEOUT_S)
        if stream is not None:
            stream.release()
        with self._lock:
            self._worker = None
            self._stream = None
            self._display_thread = None

    def status(self) -> RunInfo:
        """Snapshot of the run state for the UI (cheap, polled at 2 Hz)."""
        with self._lock:
            state = self._state
            config = self._config
            started = self._started_at
            stopped = self._stopped_at
            paused_at = self._paused_at
            paused_total_s = self._paused_total_s
            error = self._error
            warning = self._warning
            saved_video = self._saved_video
            frames = self._frames_shown
            display_fps = self._display_fps
            worker = self._worker
        return self._build_status(
            state, config, started, stopped, paused_at, paused_total_s,
            error, warning, saved_video, frames, display_fps, worker,
        )

    def _build_status(
        self,
        state: str,
        config: RunConfig | None,
        started: float | None,
        stopped: float | None,
        paused_at: float | None,
        paused_total_s: float,
        error: str | None,
        warning: str | None,
        saved_video: str | None,
        frames: int,
        display_fps: float,
        worker: InferenceWorker | None,
    ) -> RunInfo:
        """Assemble a RunInfo from a snapshot taken under the lock.

        ``worker.latest()`` and ``list_snapshots()`` run here, outside
        the lock's critical section, because they take other locks / do
        filesystem I/O.
        """
        detections, confirmed, detect_fps = 0, 0, 0.0
        if worker is not None:
            try:
                dets, conf, detect_fps = worker.latest()
                detections, confirmed = len(dets), len(conf)
            except Exception:
                # Log once per run, not on every 500 ms poll.
                if not self._latest_error_logged:
                    self._latest_error_logged = True
                    logger.warning(
                        "status: worker.latest() failed", exc_info=True
                    )
        if started is not None:
            if paused_at is not None:
                end = paused_at
            elif stopped is not None:
                end = stopped
            else:
                end = time.time()
            elapsed = max(0.0, end - started - paused_total_s)
        else:
            elapsed = 0.0
        return RunInfo(
            state="running" if state == "stopping" else state,
            source_label=self._source_label(config) if config else "",
            weights_label=config.weights_id if config else "",
            display_fps=display_fps,
            detect_fps=detect_fps,
            frames=frames,
            detections=detections,
            confirmed=confirmed,
            snapshots=max(0, len(self.list_snapshots()) - self._snapshots_at_start),
            started_at=started,
            elapsed_s=elapsed,
            error=error,
            warning=warning,
            saved_video=saved_video,
        )

    def latest_frame(self) -> tuple[int, bytes | None]:
        """(sequence, JPEG bytes) of the newest annotated frame."""
        with self._lock:
            return self._frame_seq, self._latest_jpeg

    # -- listings --------------------------------------------------------

    def list_videos(self) -> list[dict]:
        """Video files in videos_dir (sorted by name). Duration is probed
        with OpenCV; a file that cannot be probed still lists (no duration)."""
        if not self.videos_dir.is_dir():
            return []
        videos = []
        for path in sorted(self.videos_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            try:
                size_mb = round(path.stat().st_size / 1_000_000, 1)
            except FileNotFoundError:
                continue  # deleted between the scan and the stat
            videos.append(
                {
                    "name": path.relative_to(self.videos_dir).as_posix(),
                    "size_mb": size_mb,
                    "duration_s": self._probe_duration(path),
                }
            )
        return videos

    def delete_uploaded_video(self, name: str) -> str:
        """Delete a video inside ``videos/uploads/``.

        Raises ``ValueError`` for invalid paths or non-uploaded files,
        ``RuntimeError`` if the video is in use by an active/paused run,
        ``FileNotFoundError`` if the file does not exist, and
        ``OSError`` if the delete itself fails (e.g. the file is open
        on Windows — the API maps that to 409).

        The in-use check, the existence check, and the unlink all run
        under the lock, so a start() racing the delete cannot open the
        file first.
        """
        if not name or "\\" in name or name.startswith("/"):
            raise ValueError(f"Invalid video name: {name!r}")
        parts = Path(name).parts
        if any(p in ("", ".", "..") for p in parts) or parts[0] != "uploads":
            raise ValueError(
                "Only uploaded videos inside uploads/ can be deleted."
            )
        uploads_root = (self.videos_dir / "uploads").resolve()
        target = (self.videos_dir / name).resolve()
        if uploads_root not in target.parents:
            raise ValueError(
                "Only uploaded videos inside uploads/ can be deleted."
            )
        with self._lock:
            if self._state in ("starting", "running", "paused"):
                in_use = self._config is None or (
                    self._config.source_type == "video"
                    and (self.videos_dir / self._config.source).resolve()
                    == target
                )
                if in_use:
                    raise RuntimeError(
                        "Cannot delete a video while it is used by an active run."
                    )
            if not target.is_file():
                raise FileNotFoundError(f"Uploaded video not found: {name!r}")
            target.unlink()
        return target.relative_to(self.videos_dir.resolve()).as_posix()

    def list_weights(self) -> list[dict]:
        """Available weights: .pt files and *_openvino_model folders."""
        if not self.weights_dir.is_dir():
            return []
        out = []
        for path in sorted(self.weights_dir.iterdir()):
            if path.is_file() and path.suffix.lower() == ".pt":
                out.append(
                    {"id": path.name, "kind": "pytorch", "label": path.stem}
                )
            elif path.is_dir() and path.name.endswith("_openvino_model"):
                out.append(
                    {
                        "id": path.name,
                        "kind": "openvino",
                        "label": path.name[: -len("_openvino_model")],
                    }
                )
        return out

    def list_snapshots(self) -> list[dict]:
        """Snapshots, newest first (parsed from the worker's filenames)."""
        if not self.snapshots_dir.is_dir():
            return []
        # Stat each file defensively: a concurrent clear_snapshots()
        # can delete one between the glob and the stat, and that must
        # not 500 the 500 ms status poll.
        entries: list[tuple[float, Path]] = []
        for path in self.snapshots_dir.glob("threat_*.jpg"):
            try:
                entries.append((path.stat().st_mtime, path))
            except FileNotFoundError:
                continue
        out = []
        for _mtime, path in sorted(entries, reverse=True):
            m = SNAPSHOT_RE.match(path.name)
            if not m:
                continue
            out.append(
                {
                    "file": path.name,
                    "class": m.group("class"),
                    "track": int(m.group("track")),
                    "timestamp": m.group("stamp"),
                    "url": f"/snapshots/{path.name}",
                }
            )
        return out

    def clear_snapshots(self) -> int:
        """Delete all snapshots; returns how many were removed."""
        removed = 0
        for path in self.snapshots_dir.glob("threat_*.jpg"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        # Reset the run-start baseline too: status() reports
        # len(snapshots) - _snapshots_at_start, so without this a
        # mid-run clear would silence gallery updates + the siren
        # for the next N threats (one per snapshot that existed
        # when the run started).
        with self._lock:
            self._snapshots_at_start = 0
        return removed

    # -- internals -------------------------------------------------------

    @staticmethod
    def _detect_openvino_device(weights_path: Path) -> str | None:
        """Select ``"intel:gpu"`` for OpenVINO folders when an Intel GPU is
        available, leaving PyTorch (.pt) and CPU-only machines untouched.

        Delegates to :func:`src.pipeline.detect_openvino_device`, the
        helper shared with the CLI demo. (``"xpu"`` must never be
        returned: ``select_device("xpu")`` raises on the pinned
        ``torch==2.15.0+cpu`` — see that helper's docstring.)
        """
        return detect_openvino_device(weights_path)

    def _get_or_load_detector(
        self, weights_path: Path, config: RunConfig
    ) -> WeaponDetector:
        """Return a warmed-up ``WeaponDetector``, reusing the cached
        instance when weights, image size, and device are unchanged."""
        import os

        device = self._detect_openvino_device(weights_path)
        key = (weights_path.resolve(), config.imgsz, device)
        if (
            self._cached_detector is not None
            and self._cached_detector_key == key
        ):
            self._cached_detector.update_thresholds(
                conf_threshold=DEFAULT_CONF,
                conf_thresholds={
                    "Gun": config.conf_gun,
                    "Knife": config.conf_knife,
                },
            )
            self._cached_detector.reset_tracker()
            return self._cached_detector

        if weights_path.is_dir():
            cache_dir = self.weights_dir / ".ov_cache"
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault("OPENVINO_CACHE_DIR", str(cache_dir))
            except OSError:
                pass

        try:
            detector = WeaponDetector(
                weights_path,
                conf_threshold=DEFAULT_CONF,
                conf_thresholds={
                    "Gun": config.conf_gun,
                    "Knife": config.conf_knife,
                },
                imgsz=config.imgsz,
                device=device,
            )
            detector.warmup(tracker=config.tracker)
        except Exception as exc:
            if device != "intel:gpu":
                raise
            # Fall back to auto/CPU if Intel GPU compilation fails —
            # but say so loudly: a silent fallback runs the jury demo
            # at CPU speed with no indication anywhere.
            warning = f"Intel GPU compile failed ({exc}); running on CPU"
            with self._lock:
                self._warning = warning
            print(f"WARNING: {warning}")
            logger.warning(warning)
            device = None
            key = (weights_path.resolve(), config.imgsz, device)
            detector = WeaponDetector(
                weights_path,
                conf_threshold=DEFAULT_CONF,
                conf_thresholds={
                    "Gun": config.conf_gun,
                    "Knife": config.conf_knife,
                },
                imgsz=config.imgsz,
                device=None,
            )
            detector.warmup(tracker=config.tracker)

        self._cached_detector = detector
        self._cached_detector_key = key
        return detector

    def _resolve_weights(self, weights_id: str) -> Path:
        if (
            not weights_id
            or weights_id in (".", "..")
            or "/" in weights_id
            or "\\" in weights_id
        ):
            raise ValueError(f"Unknown weights: {weights_id!r}")
        # The resolved path must sit directly inside weights_dir:
        # that rejects "." / ".." (already filtered above) and any
        # symlink inside the dir that points elsewhere.
        weights_root = self.weights_dir.resolve()
        path = (self.weights_dir / weights_id).resolve()
        if path.parent != weights_root:
            raise ValueError(f"Unknown weights: {weights_id!r}")
        if not path.exists():
            known = [w["id"] for w in self.list_weights()]
            raise ValueError(
                f"Weights {weights_id!r} not found in {self.weights_dir}. "
                f"Available: {', '.join(known) or 'none — place best.pt there'}."
            )
        return path

    def _resolve_source(self, config: RunConfig) -> int | str:
        if config.source_type == "webcam":
            try:
                index = int(config.source)
            except ValueError as exc:
                raise ValueError(
                    f"Webcam source must be a camera index, got {config.source!r}"
                ) from exc
            if index < 0:
                raise ValueError(
                    "Webcam source must be a camera index >= 0, "
                    f"got {config.source!r}"
                )
            return index
        if config.source_type != "video":
            raise ValueError(
                f"source_type must be 'webcam' or 'video', got {config.source_type!r}"
            )
        # The name must stay inside videos_dir (no ../ escapes).
        candidate = (self.videos_dir / config.source).resolve()
        base = self.videos_dir.resolve()
        if base not in candidate.parents and candidate != base:
            raise ValueError(f"Video {config.source!r} is outside the videos folder.")
        if not candidate.is_file():
            raise ValueError(f"Video not found: {config.source!r}")
        return str(candidate)

    @staticmethod
    def _unique_recording_path(
        save_video_dir: Path, stamp: str, record_raw: bool
    ) -> Path:
        """Recording path for a stamp, with a numeric suffix when the
        planned file already exists (two runs in the same second):
        ``demo_<stamp>[_raw].mp4``, then ``..._1.mp4``, ``..._2.mp4``.
        Same idea as ``_DetectionLog``'s collision suffix."""
        suffix = "_raw" if record_raw else ""
        candidate = save_video_dir / f"demo_{stamp}{suffix}.mp4"
        n = 1
        while candidate.exists():
            candidate = save_video_dir / f"demo_{stamp}{suffix}_{n}.mp4"
            n += 1
        return candidate

    @staticmethod
    def _source_label(config: RunConfig | None) -> str:
        if config is None:
            return ""
        return (
            f"Webcam {config.source}"
            if config.source_type == "webcam"
            else config.source
        )

    def _fail(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._error = message
            self._stopped_at = time.time()
            frames = self._frames_shown
        logger.error("run failed: %s (frames=%d)", message, frames)

    def _teardown_then_fail(self, message: str) -> None:
        """Release the worker/stream handles BEFORE reporting the
        error — mirroring stop(), which also sets "stopping" and tears
        down first. If _fail ran first, the UI would see state "error"
        and re-enable Start/Delete while the webcam/file handle was
        still locked (WinError 32 on Windows). Used by _display_loop
        only; the start() failure paths are unchanged."""
        with self._lock:
            self._state = "stopping"
        self._teardown()
        self._fail(message)

    @staticmethod
    def _probe_duration(path: Path) -> float | None:
        try:
            import cv2

            cap = cv2.VideoCapture(str(path))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                if fps and fps > 0 and frames and frames > 0:
                    return round(frames / fps, 1)
            finally:
                cap.release()
        except Exception:
            pass
        return None

    def _display_loop(self, cv2_module) -> None:
        """Mirror of the CLI demo's display loop: newest frame + latest
        detections -> annotated JPEG, at video rate."""
        config = self._config
        stream = self._stream
        worker = self._worker
        # Explicit failure instead of asserts before the try: an assert
        # firing here used to skip the finally below and wedge the run
        # in "running" forever. (If the run already finished — a stop()
        # raced start() and tore it down before this thread got going —
        # just exit quietly instead of failing a finished run.)
        if config is None or stream is None or worker is None:
            with self._lock:
                finished = self._state in ("stopped", "error", "idle")
            if not finished:
                self._teardown_then_fail(
                    "Display loop started without a run configured."
                )
            return
        me = threading.current_thread()
        fps_counter = RollingFps()
        last_index = 0
        writer = None
        sidecar: TimestampSidecar | None = None
        saved_video_path: str | None = None
        recording_failed = False
        try:
            while not self._stop_event.is_set():
                # Check FIRST, every iteration: the worker can die while
                # frames keep flowing, and the old check (in the
                # no-new-frame branch below) then never ran — verdicts
                # froze while the UI stayed "Running".
                if worker.error is not None:
                    break
                if self.state == "paused":
                    time.sleep(PAUSED_POLL_SLEEP_S)
                    continue
                packet = stream.read()
                if packet is None or packet[0] == last_index:
                    if stream.ended and not config.loop:
                        break
                    if worker.error is not None:
                        break
                    time.sleep(0.002)
                    continue
                frame_index, _ts, frame = packet
                last_index = frame_index
                tick = time.monotonic()
                display_fps = fps_counter.tick(tick)
                detections, confirmed, _detect_fps = worker.latest()
                display = draw_annotations(
                    cv2_module, frame, detections, confirmed,
                    hide_unconfirmed=config.hide_unconfirmed,
                )
                if (
                    config.save_video
                    and self.save_video_dir is not None
                    and not recording_failed
                ):
                    if writer is None:
                        self.save_video_dir.mkdir(parents=True, exist_ok=True)
                        # The path was fixed in start() so the worker's
                        # log header can name the recording. If it is
                        # somehow missing, disable recording loudly
                        # instead of inventing a second filename.
                        saved_video_path = self._planned_recording_path
                        if saved_video_path is None:
                            recording_failed = True
                            warning = (
                                "Recording enabled but no recording path "
                                "was planned; recording disabled"
                            )
                            with self._lock:
                                self._warning = warning
                            print(f"WARNING: {warning}")
                            logger.warning(warning)
                        else:
                            height, width = display.shape[:2]
                            writer = cv2_module.VideoWriter(
                                saved_video_path,
                                cv2_module.VideoWriter_fourcc(*"mp4v"),
                                stream.source_fps or FALLBACK_FPS,
                                (width, height),
                            )
                            if not writer.isOpened():
                                writer.release()
                                writer = None
                                failed_path = saved_video_path
                                saved_video_path = None
                                recording_failed = True
                                warning = (
                                    "Could not open video recording "
                                    f"{failed_path!r}; recording disabled"
                                )
                                with self._lock:
                                    self._warning = warning
                                print(f"WARNING: {warning}")
                                logger.warning(warning)
                            else:
                                # True timing for the recording: one
                                # monotonic timestamp per recorded
                                # frame (raw or annotated — the timing
                                # problem is the same either way),
                                # joining the detection log on
                                # frame_index.
                                sidecar = TimestampSidecar(saved_video_path)
                    if writer is not None:
                        # Raw frames by default (record_raw): the
                        # annotated recording is useless as training
                        # data for the next model version.
                        writer.write(frame if config.record_raw else display)
                        if sidecar is not None:
                            sidecar.write(frame_index, _ts)
                ok, buf = cv2_module.imencode(
                    ".jpg", display, [cv2_module.IMWRITE_JPEG_QUALITY, 85]
                )
                if ok:
                    with self._lock:
                        self._latest_jpeg = buf.tobytes()
                        self._frame_seq += 1
                        self._frames_shown = frame_index
                        self._display_fps = display_fps
        finally:
            if writer is not None:
                writer.release()
            if sidecar is not None:
                sidecar.close()
            with self._lock:
                if saved_video_path is not None:
                    self._saved_video = saved_video_path
            # Only the CURRENT display thread finishes the run: a stale
            # thread that outlived its teardown join must not fail a
            # newer run that reused the runner.
            with self._lock:
                current = self._display_thread is me
                still_running = self._state in ("running", "paused")
            # The video ended (or inference failed, or the source died)
            # while the run was still marked running: finish it instead
            # of leaving the UI stuck on "Running".
            if current and still_running:
                if worker.error is not None:
                    self._teardown_then_fail(
                        f"Inference failed: {worker.error}"
                    )
                elif stream.ended and (stream.is_live or last_index == 0):
                    # A live source only "ends" by dying, and a file that
                    # produced zero frames never played: neither is a
                    # clean stop. Surface the end reason in the error so
                    # the UI shows why the run ended.
                    reason = (
                        "Camera disconnected"
                        if stream.is_live
                        else "Could not read video"
                    )
                    if stream.end_reason:
                        reason += f" ({stream.end_reason})"
                    self._teardown_then_fail(reason)
                else:
                    self.stop()
