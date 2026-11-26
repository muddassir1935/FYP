# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Track-level class voting for ByteTrack-confirmed weapons.

ByteTrack (Ultralytics ``model.track(..., tracker="bytetrack.yaml")``)
assigns a stable ID to each physical object across frames, but the
detector still labels every frame independently — so one object can flip
between Gun and Knife from frame to frame. This module aggregates each
track's per-frame labels into ONE stable label: a confidence-weighted
majority vote over the track's lifetime (each sighting adds its
confidence to its class's tally; the class with the highest tally wins).

Confirmation is density over time, not a lifetime count: a track is
confirmed only while it has at least ``min_hits`` sightings within the
last ``confirm_window_s`` seconds (defaults 3 sightings in 1.0 s), and
confirmation LAPSES as soon as the density drops below that — a track
seen twice, then once more a second later, never confirms, because the
three sightings were never dense. The old rule (``hits >= min_hits`` on
a lifetime counter) confirmed exactly that sparse pattern: two
detections plus one more sighting anywhere inside the track's 30-frame
life confirmed a threat without it ever being detected consistently.

Density alone is still a frame count in a time window: at 39 FPS three
sightings span ~0.1 s, so a phone sweeping through frame for four
frames would confirm, and the rule gets weaker as the model gets
faster. The ``min_span_s`` floor (default 0.4 s) fixes that: a track
confirms only when its oldest in-window sighting is at least
``min_span_s`` old, i.e. the object has been visibly present for that
long. The rule is now a minimum observed presence in seconds, not a
frame count — a faster model just samples the same 0.4 s more densely.
Gaps inside the span are still tolerated (a sighting need only fall
inside the window), so near-threshold flicker does not drop a real
detection. The voted label itself is still the confidence-weighted
lifetime vote; only the confirmation rule changed.

Note ByteTrack itself only shows a track after detections in 2
consecutive frames (a tentative track unmatched in the very next frame
is dropped, and new tracks are hidden until matched once) — so a
genuinely present object still confirms within a few frames of becoming
visible, while 1-2 frame blips never reach the required density.

Pure Python (no torch/ultralytics import), so it is fully unit-testable —
like the temporal buffer it complements. The TemporalConsistencyBuffer
in :mod:`temporal_buffer` is kept intact for RQ3 reproducibility; the
demo now confirms via tracking instead of via the buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Tuple

from .weapon_detector import Detection


@dataclass(frozen=True)
class TrackedDetection:
    """One track visible in the current frame, with its voted label."""

    detection: Detection  # voted class; current frame's bbox/confidence
    track_id: int
    hits: int  # sightings of this track within the confirmation window
    vote_share: float  # weighted vote share of the winning class (0.0-1.0)
    raw_class_id: int  # class the model gave this frame, before voting
    confirmed: bool  # >= min_hits sightings within the window, spanning
    # >= min_span_s (see TrackClassVoter)


@dataclass
class _TrackVotes:
    votes: Dict[int, float] = field(default_factory=dict)  # class_id -> conf sum
    counts: Dict[int, int] = field(default_factory=dict)  # class_id -> sightings
    hit_times: Deque[float] = field(default_factory=deque)  # monotonic ts per sighting
    last_seen: int = 0  # frame index of the latest sighting
    last_class_id: int = -1


class TrackClassVoter:
    """Assigns one stable, voted label per ByteTrack track."""

    def __init__(
        self,
        *,
        min_hits: int = 3,
        confirm_window_s: float = 1.0,
        min_span_s: float = 0.4,
        max_age: int = 30,
    ) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if confirm_window_s <= 0:
            raise ValueError("confirm_window_s must be > 0")
        if min_span_s < 0:
            raise ValueError("min_span_s must be >= 0")
        if max_age < 1:
            raise ValueError("max_age must be >= 1")
        self.min_hits = min_hits
        self.confirm_window_s = confirm_window_s
        # 0 disables the span check (density only, the pre-span rule).
        self.min_span_s = min_span_s
        self.max_age = max_age
        self._tracks: Dict[int, _TrackVotes] = {}
        self._names: Dict[int, str] = {}  # class_id -> class_name, as seen
        self._frame = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._names.clear()
        self._frame = 0

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(
        self, tracked: List[Tuple[int, Detection]], *, now: float
    ) -> List[TrackedDetection]:
        """Feed one frame's (track_id, detection) pairs.

        ``now`` is the frame's monotonic timestamp in seconds (the
        packet timestamp from the capture thread — NOT the processing
        time, so a slow inference loop cannot stretch the window).
        Returns one :class:`TrackedDetection` per track visible this
        frame, each carrying the track's voted label. ``hits`` counts
        only sightings within the last ``confirm_window_s`` seconds
        (a sighting exactly ``confirm_window_s`` old still counts);
        ``confirmed`` is True only while that count reaches
        ``min_hits`` AND the oldest in-window sighting is at least
        ``min_span_s`` old (span = ``now - hit_times[0]`` after
        pruning), so confirmation lapses after silence — and a burst
        of sightings inside ``min_span_s`` never confirms, however
        dense it is.
        """
        self._frame += 1
        visible: List[TrackedDetection] = []
        for track_id, det in tracked:
            self._names.setdefault(det.class_id, det.class_name)
            state = self._tracks.get(track_id)
            if state is None:
                state = _TrackVotes()
                self._tracks[track_id] = state
            state.votes[det.class_id] = (
                state.votes.get(det.class_id, 0.0) + det.confidence
            )
            state.counts[det.class_id] = state.counts.get(det.class_id, 0) + 1
            state.hit_times.append(now)
            # Drop sightings older than the window (oldest first: the
            # deque is in time order because `now` is monotonic).
            cutoff = now - self.confirm_window_s
            while state.hit_times and state.hit_times[0] < cutoff:
                state.hit_times.popleft()
            hits = len(state.hit_times)
            # Minimum observed presence: the oldest surviving sighting
            # must be at least min_span_s old. A burst of sightings
            # inside the span (e.g. 3 frames at 39 FPS) has hits but
            # no span, so it never confirms.
            span_s = now - state.hit_times[0] if state.hit_times else 0.0
            confirmed = hits >= self.min_hits and span_s >= self.min_span_s
            state.last_seen = self._frame
            state.last_class_id = det.class_id

            voted_class = self._winner(state)
            total_votes = sum(state.votes.values())
            vote_share = (
                state.votes[voted_class] / total_votes if total_votes > 0 else 0.0
            )
            voted = Detection(
                class_id=voted_class,
                class_name=self._names.get(voted_class, str(voted_class)),
                confidence=det.confidence,
                bbox=det.bbox,
                track_id=track_id,
            )
            visible.append(
                TrackedDetection(
                    detection=voted,
                    track_id=track_id,
                    hits=hits,
                    vote_share=vote_share,
                    raw_class_id=det.class_id,
                    confirmed=confirmed,
                )
            )

        # Drop tracks unseen for longer than max_age frames (mirrors
        # ByteTrack's track_buffer: a reappearing object gets a fresh
        # track ID anyway, so stale votes must not linger).
        stale = [
            tid
            for tid, state in self._tracks.items()
            if self._frame - state.last_seen > self.max_age
        ]
        for tid in stale:
            del self._tracks[tid]
        return visible

    @staticmethod
    def _winner(state: _TrackVotes) -> int:
        """Winning class: highest confidence-weighted vote tally.

        Ties break toward the class seen most often, then toward the
        most recent class, then toward the lowest class id — fully
        deterministic, so the same vote history always yields the same
        label.
        """
        return max(
            state.votes,
            key=lambda c: (
                state.votes[c],
                state.counts.get(c, 0),
                1 if c == state.last_class_id else 0,
                -c,
            ),
        )
