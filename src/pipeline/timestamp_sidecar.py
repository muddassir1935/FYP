# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Per-frame timestamps for a raw recording (the timestamp sidecar).

A raw recording on its own does not preserve true timing: the display
thread records the frames it happens to see, and for a webcam it writes
at the 25 FPS fallback because ``source_fps`` is 0 — so a recording
plays back at the wrong speed when the display thread falls behind.
The sidecar fixes that: one JSONL line per recorded frame,
``{"frame_index": i, "ts": <monotonic seconds>}``, written next to the
recording as ``<recording>.timestamps.jsonl``. The detection log joins
on ``frame_index`` (its ``monotonic_s`` is the same packet timestamp),
so the sidecar lets the replay benchmark tag intervals against the
exact video frames.

Shared by the UI runner's display thread and the CLI demo's recording
loop, so both recording paths write the same format. Like the
detection log, a sidecar failure (unwritable path, full disk) only
disables the sidecar with one WARNING — recording continues.
"""

from __future__ import annotations

import json
from pathlib import Path


def sidecar_path_for(recording_path: str | Path) -> Path:
    """``<recording>.timestamps.jsonl`` for a recording path."""
    return Path(f"{recording_path}.timestamps.jsonl")


# Buffered writes are flushed every N lines, mirroring
# _DetectionLog's LOG_FLUSH_EVERY_LINES in demo_worker.py: a crash
# must not truncate the sidecar to whatever fit in the buffer.
SIDECAR_FLUSH_EVERY_LINES = 50


class TimestampSidecar:
    """One JSONL line per recorded frame: frame_index + monotonic ts."""

    def __init__(self, recording_path: str | Path) -> None:
        self.path = sidecar_path_for(recording_path)
        self._handle: object | None = None
        self._broken = False
        self._lines_since_flush = 0
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = open(self.path, "w", encoding="utf-8", buffering=8192)
        except Exception as exc:
            print(f"WARNING: timestamp sidecar disabled ({exc})")
            self._broken = True
            self._handle = None

    def __enter__(self) -> "TimestampSidecar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def disabled(self) -> bool:
        """True when the sidecar could not be opened (writes no-op)."""
        return self._broken

    def write(self, frame_index: int, ts: float) -> None:
        """Append one line for a recorded frame (no-op if disabled)."""
        if self._broken or self._handle is None:
            return
        try:
            self._handle.write(
                json.dumps(
                    {"frame_index": int(frame_index), "ts": round(float(ts), 6)}
                )
                + "\n"
            )
            self._lines_since_flush += 1
            if self._lines_since_flush >= SIDECAR_FLUSH_EVERY_LINES:
                self.flush()
        except Exception as exc:
            print(
                f"WARNING: timestamp sidecar disabled ({exc}) "
                "— recording continues."
            )
            self._broken = True
            self.close()

    def flush(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.flush()
            self._lines_since_flush = 0
        except Exception:
            pass

    def close(self) -> None:
        handle, self._handle = self._handle, None
        self._lines_since_flush = 0
        if handle is None:
            return
        try:
            handle.flush()
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass
