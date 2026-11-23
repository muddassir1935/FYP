# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Threaded frame capture (FR-01).

Wraps OpenCV ``VideoCapture`` (webcam index, video file path, or RTSP URL)
in a background thread that always keeps only the *latest* frame. Inference
therefore never falls behind a live stream: if processing is slower than
the camera, stale frames are dropped instead of queued.

OpenCV is imported lazily in :meth:`FrameSource.start`, so importing this
module (and running ``--help`` / unit tests) works without cv2 installed.
"""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Optional, Tuple, Union

# A frame is an OpenCV/numpy image. Typed loosely so this module needs no
# numpy import at module level.
Frame = object
FramePacket = Tuple[int, float, Frame]  # (frame_index, monotonic_ts, frame)

# Fallback frame rate for recording when the source reports no FPS
# (live cameras, or files OpenCV cannot probe). Shared by the runner
# and the CLI so both record at the same rate.
FALLBACK_FPS = 25.0
# How long release() waits for the reader thread before giving up.
READER_JOIN_TIMEOUT_S = 2.0
# Idle sleep in the reader loop (paused, or between failed reads).
READER_IDLE_SLEEP_S = 0.02


class FrameSource:
    """Latest-frame video source running on its own thread.

    ``source`` is a webcam index (``0``), a video file path, or an RTSP URL;
    OpenCV accepts all three unchanged.
    """

    def __init__(
        self,
        source: Union[int, str],
        *,
        loop_file: bool = False,
        pace_file: bool = True,
        max_consecutive_failures: int = 50,
    ) -> None:
        self.source = source
        self.loop_file = loop_file
        self.pace_file = pace_file
        self.max_consecutive_failures = max_consecutive_failures
        self.source_fps: float = 0.0  # native FPS of a file source (0 = n/a)
        self._cap: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._running = False
        self._ended = False
        # Why the reader stopped: "eof" (clean end of a file), or
        # "read_failures" (live source died, or a file that cannot be
        # decoded). None while running or when release() stopped it.
        # Callers use this to tell "video finished" from "camera died".
        self._end_reason: Optional[str] = None
        self._end_error: Optional[str] = None
        self._paused = threading.Event()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> "FrameSource":
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError(
                "opencv-python is not installed; run: pip install opencv-python"
            ) from exc
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")
        if self._is_file_source():
            fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
            self.source_fps = fps if 1.0 <= fps <= 240.0 else 0.0
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="sentinel-capture", daemon=True
        )
        self._thread.start()
        return self

    def release(self) -> None:
        self._running = False
        self._paused.clear()
        if self._thread is not None:
            self._thread.join(timeout=READER_JOIN_TIMEOUT_S)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def pause(self) -> None:
        """Freeze frame capture (holds file position; keeps live camera open)."""
        self._paused.set()

    def resume(self) -> None:
        """Resume frame capture from the paused position."""
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()

    # -- reading -----------------------------------------------------------

    @property
    def ended(self) -> bool:
        """True when a file source reached EOF or the stream died."""
        return self._ended

    @property
    def end_reason(self) -> Optional[str]:
        """Why the reader stopped: ``"eof"``, ``"read_failures"``, or
        None (still running, or stopped by :meth:`release`)."""
        return self._end_reason

    @property
    def end_error(self) -> Optional[str]:
        """The exception that killed the reader thread, if any."""
        return self._end_error

    @property
    def is_live(self) -> bool:
        """True for webcams and network streams (which never EOF — if
        one ends, it died); False for video files."""
        return not self._is_file_source()

    def read(self) -> Optional[FramePacket]:
        """Return the newest ``(frame_index, timestamp, frame)`` or None.

        The frame is shared, not copied — copy it before drawing on it.
        ``frame_index`` increases by 1 per captured frame, so callers can
        skip packets they have already processed.
        """
        with self._lock:
            return self._latest

    # -- internals ----------------------------------------------------------

    def _is_file_source(self) -> bool:
        return isinstance(self.source, str) and not self.source.lower().startswith(
            ("rtsp://", "http://", "https://")
        )

    def _reader_loop(self) -> None:
        import cv2

        # The try/finally guarantees _ended is set even if read()
        # raises (e.g. after release() timed out joining this thread):
        # without it the display loop would wait forever on a dead
        # reader with _ended still False.
        try:
            assert self._cap is not None
            index = 0
            failures = 0
            while self._running:
                if self._paused.is_set():
                    if not self._is_file_source():
                        # Keep live camera/RTSP buffer drained while frozen so
                        # resuming immediately delivers fresh frames.
                        self._cap.grab()
                    time.sleep(READER_IDLE_SLEEP_S)
                    continue
                tick = time.monotonic()
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    if self.loop_file and self._rewind(cv2):
                        continue
                    failures += 1
                    if failures >= self.max_consecutive_failures:
                        # A file that stops delivering frames at (or
                        # just past) its last frame reached EOF; anything
                        # else — a live source, or a file failing
                        # mid-stream — is a read failure, not an ending.
                        self._end_reason = (
                            "eof"
                            if self._is_at_eof(cv2)
                            else "read_failures"
                        )
                        break
                    time.sleep(READER_IDLE_SLEEP_S)
                    continue
                failures = 0
                index += 1
                with self._lock:
                    self._latest = (index, time.monotonic(), frame)
                # A file would otherwise be decoded as fast as the CPU allows;
                # pace it to its native frame rate so playback looks like video.
                # Live sources (webcam/RTSP) arrive at their own rate already.
                if self.pace_file and self.source_fps > 0 and self._is_file_source():
                    remaining = (1.0 / self.source_fps) - (time.monotonic() - tick)
                    if remaining > 0:
                        time.sleep(remaining)
        except Exception as exc:
            self._end_reason = "read_failures"
            self._end_error = str(exc)
        finally:
            self._ended = True

    def _is_at_eof(self, cv2_module: object) -> bool:
        """True when a file source's read position is at its last frame.

        Live sources never EOF. Anything that cannot be probed (no
        frame count, a failed ``get``) answers False — an unprovable
        EOF is treated as a read failure, the safer direction.
        """
        if not self._is_file_source() or self._cap is None:
            return False
        try:
            total = float(self._cap.get(cv2_module.CAP_PROP_FRAME_COUNT) or 0)
            pos = float(self._cap.get(cv2_module.CAP_PROP_POS_FRAMES) or 0)
        except Exception:
            return False
        return total > 0 and pos >= total - 1

    def _rewind(self, cv2_module: object) -> bool:
        """Restart a file source at frame 0 (used when ``loop_file``)."""
        if isinstance(self.source, int):
            return False  # webcams do not rewind
        assert self._cap is not None
        return bool(self._cap.set(cv2_module.CAP_PROP_POS_FRAMES, 0))
