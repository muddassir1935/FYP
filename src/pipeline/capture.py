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
        max_consecutive_failures: int = 50,
    ) -> None:
        self.source = source
        self.loop_file = loop_file
        self.max_consecutive_failures = max_consecutive_failures
        self._cap: Optional[object] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[FramePacket] = None
        self._running = False
        self._ended = False

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
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop, name="sentinel-capture", daemon=True
        )
        self._thread.start()
        return self

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

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

    def read(self) -> Optional[FramePacket]:
        """Return the newest ``(frame_index, timestamp, frame)`` or None.

        The frame is shared, not copied — copy it before drawing on it.
        ``frame_index`` increases by 1 per captured frame, so callers can
        skip packets they have already processed.
        """
        with self._lock:
            return self._latest

    # -- internals ----------------------------------------------------------

    def _reader_loop(self) -> None:
        import cv2

        assert self._cap is not None
        index = 0
        failures = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.loop_file and self._rewind(cv2):
                    continue
                failures += 1
                if failures >= self.max_consecutive_failures:
                    break
                time.sleep(0.02)
                continue
            failures = 0
            index += 1
            with self._lock:
                self._latest = (index, time.monotonic(), frame)
        self._ended = True

    def _rewind(self, cv2_module: object) -> bool:
        """Restart a file source at frame 0 (used when ``loop_file``)."""
        if isinstance(self.source, int):
            return False  # webcams do not rewind
        assert self._cap is not None
        return bool(self._cap.set(cv2_module.CAP_PROP_POS_FRAMES, 0))
