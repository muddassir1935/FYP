# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""One detection run for the demo web UI.

Owns the pipeline (``FrameSource`` -> ``WeaponDetector`` -> ``ByteTrack``
-> ``TrackClassVoter``) in two threads, exactly like the CLI demo:

* the inference worker (``src.pipeline.InferenceWorker``) runs detection
  as fast as the model allows;
* a display thread reads frames at video rate, draws the latest
  detections on them, JPEG-encodes the result, and (optionally) records
  the annotated video.

The FastAPI layer only ever touches this class: it starts/stops runs,
reads the latest JPEG for the MJPEG stream, and polls ``status()``.
Inference never runs on the FastAPI event loop.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.pipeline import (
    FrameSource,
    InferenceWorker,
    RollingFps,
    TrackClassVoter,
    WeaponDetector,
    draw_annotations,
)

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
    conf_gun: float = 0.44
    conf_knife: float = 0.43
    imgsz: int = 640
    min_hits: int = 3
    tracker: str = "bytetrack.yaml"
    loop: bool = False
    hide_unconfirmed: bool = False
    save_video: bool = False


@dataclass
class RunInfo:
    """What the API returns about a run (live or finished)."""

    state: str  # idle | running | stopped | error
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
    ) -> None:
        self.videos_dir = Path(videos_dir)
        self.weights_dir = Path(weights_dir)
        self.snapshots_dir = Path(snapshots_dir)
        self.save_video_dir = Path(save_video_dir) if save_video_dir else None
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._state = "idle"
        self._config: RunConfig | None = None
        self._error: str | None = None
        self._started_at: float | None = None
        self._stopped_at: float | None = None
        self._paused_at: float | None = None
        self._paused_total_s: float = 0.0
        self._saved_video: str | None = None
        self._snapshots_at_start = 0

        # Live frame + stats, written by the display thread.
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self._frames_shown = 0
        self._display_fps = 0.0

        self._stop_event = threading.Event()
        self._display_thread: threading.Thread | None = None
        self._worker: InferenceWorker | None = None
        self._stream: FrameSource | None = None
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
            self._saved_video = None
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

        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment issue
            self._fail("opencv-python is not installed.")
            raise RuntimeError(
                "opencv-python is not installed; run: pip install opencv-python"
            ) from exc

        voter = TrackClassVoter(min_hits=config.min_hits)
        stream = FrameSource(source, loop_file=config.loop)
        try:
            stream.start()
        except Exception as exc:
            try:
                stream.release()
            except Exception:
                pass
            self._fail(f"Could not open source {config.source!r}: {exc}")
            raise RuntimeError(
                f"Could not open source {config.source!r}: {exc}"
            ) from exc

        worker = InferenceWorker(
            stream,
            detector,
            voter,
            cv2_module=cv2,
            tracker=config.tracker,
            snapshots_dir=self.snapshots_dir,
        )
        with self._lock:
            self._stream = stream
            self._worker = worker
            self._state = "running"
            self._started_at = time.time()
            self._stopped_at = None
        try:
            worker.start()
            self._display_thread = threading.Thread(
                target=self._display_loop, args=(cv2,), name="sentinel-display",
                daemon=True,
            )
            self._display_thread.start()
        except Exception as exc:
            self._teardown()
            self._fail(f"Could not start the run threads: {exc}")
            raise RuntimeError(f"Could not start the run threads: {exc}") from exc
        return self.status()

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

        ``status()`` takes ``self._lock`` itself, so it is never called
        while this method holds the lock (``threading.Lock`` is not
        re-entrant — that deadlocked the first version).
        """
        with self._lock:
            idle = self._state not in ("running", "paused")
            if not idle:
                if self._paused_at is not None:
                    self._paused_total_s += max(
                        0.0, time.time() - self._paused_at
                    )
                    self._paused_at = None
                self._state = "stopping"
        if idle:
            return self.status()
        self._teardown()
        with self._lock:
            self._state = "stopped"
            self._stopped_at = time.time()
        return self.status()

    def _teardown(self) -> None:
        """Release the worker, display thread, and stream without
        changing the run state. Safe to call from the display thread
        itself (it never joins the current thread)."""
        self._stop_event.set()
        worker, stream = self._worker, self._stream
        display_thread = self._display_thread
        if worker is not None:
            worker.stop()
            worker.join(timeout=5.0)
        if display_thread is not None and (
            display_thread is not threading.current_thread()
        ):
            display_thread.join(timeout=5.0)
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
            saved_video = self._saved_video
            frames = self._frames_shown
            display_fps = self._display_fps
            worker = self._worker
        return self._build_status(
            state, config, started, stopped, paused_at, paused_total_s,
            error, saved_video, frames, display_fps, worker,
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
                pass
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
            videos.append(
                {
                    "name": path.relative_to(self.videos_dir).as_posix(),
                    "size_mb": round(path.stat().st_size / 1_000_000, 1),
                    "duration_s": self._probe_duration(path),
                }
            )
        return videos

    def delete_uploaded_video(self, name: str) -> str:
        """Delete a video inside ``videos/uploads/``.

        Raises ``ValueError`` for invalid paths or non-uploaded files,
        ``RuntimeError`` if the video is in use by an active/paused run,
        and ``FileNotFoundError`` if the file does not exist.
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
        out = []
        for path in sorted(
            self.snapshots_dir.glob("threat_*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
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
        return removed

    # -- internals -------------------------------------------------------

    @staticmethod
    def _detect_openvino_device(weights_path: Path) -> str | None:
        """Select ``intel:gpu`` for OpenVINO folders when an Intel GPU is
        available, leaving PyTorch (.pt) and CPU-only machines untouched."""
        if not weights_path.is_dir():
            return None
        try:
            import openvino as ov

            devices = ov.Core().available_devices
            if any(d.startswith("GPU") for d in devices):
                return "intel:gpu"
        except Exception:
            pass
        return None

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
                conf_threshold=0.40,
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
                conf_threshold=0.40,
                conf_thresholds={
                    "Gun": config.conf_gun,
                    "Knife": config.conf_knife,
                },
                imgsz=config.imgsz,
                device=device,
            )
            detector.warmup(tracker=config.tracker)
        except Exception:
            if device != "intel:gpu":
                raise
            # Fallback cleanly to auto/CPU if Intel GPU compilation fails.
            device = None
            key = (weights_path.resolve(), config.imgsz, device)
            detector = WeaponDetector(
                weights_path,
                conf_threshold=0.40,
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
        if not weights_id or "/" in weights_id or "\\" in weights_id:
            raise ValueError(f"Unknown weights: {weights_id!r}")
        path = self.weights_dir / weights_id
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
                return int(config.source)
            except ValueError as exc:
                raise ValueError(
                    f"Webcam source must be a camera index, got {config.source!r}"
                ) from exc
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
        assert config is not None
        stream = self._stream
        worker = self._worker
        assert stream is not None and worker is not None
        fps_counter = RollingFps()
        last_index = 0
        writer = None
        saved_video_path: str | None = None
        try:
            while not self._stop_event.is_set():
                if self.state == "paused":
                    time.sleep(0.02)
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
                if config.save_video and self.save_video_dir is not None:
                    if writer is None:
                        self.save_video_dir.mkdir(parents=True, exist_ok=True)
                        stamp = time.strftime("%Y%m%d_%H%M%S")
                        saved_video_path = str(
                            self.save_video_dir / f"demo_{stamp}.mp4"
                        )
                        height, width = display.shape[:2]
                        writer = cv2_module.VideoWriter(
                            saved_video_path,
                            cv2_module.VideoWriter_fourcc(*"mp4v"),
                            stream.source_fps or 25.0,
                            (width, height),
                        )
                        if not writer.isOpened():
                            writer.release()
                            writer = None
                            saved_video_path = None
                    if writer is not None:
                        writer.write(display)
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
            with self._lock:
                if saved_video_path is not None:
                    self._saved_video = saved_video_path
            # The video ended (or inference failed) while the run was
            # still marked running: finish it cleanly instead of
            # leaving the UI stuck on "Running".
            with self._lock:
                still_running = self._state in ("running", "paused")
            if still_running:
                if worker.error is not None:
                    self._fail(f"Inference failed: {worker.error}")
                    self._teardown()
                else:
                    self.stop()
