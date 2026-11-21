# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Rolling-window frames-per-second counter for the demo HUD.

The old per-iteration ``1/dt`` EMA spiked to hundreds whenever two frames
arrived back-to-back (a buffered video frame, a fast loop iteration), so the
HUD showed fantasy numbers. A rolling window counts real frames over real
time instead: it can never exceed the actual frame rate.

The rate is the number of frames seen in the last ``window_seconds``
seconds, so it ramps up over the first second after start and decays over
one window after frames stop. Below 1 FPS it rounds up to 1.0 while any
frame is still inside the window — irrelevant for a live demo, where the
rate is an order of magnitude higher.
"""

from __future__ import annotations

import time
from collections import deque


class RollingFps:
    """Frames per second over the last ``window_seconds`` seconds.

    Call :meth:`tick` once per frame; it returns the current rate.
    """

    def __init__(self, window_seconds: float = 1.0) -> None:
        self._window = window_seconds
        self._ticks: deque[float] = deque()

    def tick(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        self._ticks.append(now)
        cutoff = now - self._window
        while self._ticks and self._ticks[0] < cutoff:
            self._ticks.popleft()
        return len(self._ticks) / self._window
