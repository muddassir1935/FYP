"""Shared helpers for the demo tests (not a test module itself).

``make_tiny_video`` writes a short real .mp4 with OpenCV so tests can
exercise FrameSource and the upload probe against a playable file
instead of fake bytes. Skipped by callers when cv2 is not installed.
"""

from __future__ import annotations

from pathlib import Path

try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover - environment issue
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False


def make_tiny_video(
    path: Path,
    *,
    n_frames: int = 8,
    fps: float = 10.0,
    width: int = 64,
    height: int = 48,
) -> Path:
    """Write a tiny mp4 (solid-colour frames) and return its path."""
    import numpy as np

    assert cv2 is not None, "cv2 is required to generate test videos"
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not create test video {path}")
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 30) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path
