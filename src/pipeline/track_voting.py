"""Track-level class voting for ByteTrack-confirmed weapons.

ByteTrack (Ultralytics ``model.track(..., tracker="bytetrack.yaml")``)
assigns a stable ID to each physical object across frames, but the
detector still labels every frame independently — so one object can flip
between Gun and Knife from frame to frame. This module aggregates each
track's per-frame labels into ONE stable label: a confidence-weighted
majority vote over the track's lifetime (each sighting adds its
confidence to its class's tally; the class with the highest tally wins).

Confirmation: a track is confirmed after ``min_hits`` VISIBLE sightings
(default 3). Note ByteTrack itself only shows a track after detections
in 2 consecutive frames (a tentative track unmatched in the very next
frame is dropped, and new tracks are hidden until matched once) — so
with ``min_hits=3`` a threat confirms after 4 consecutive detections,
against 6 for the 6-of-10 buffer. That recovers brief true detections
("glimpses" of 4-5 frames) the buffer drops, while 1-3 frame blips still
never confirm. Three visible votes also guarantee a majority, so no
confirmed label ever rests on a tie-break. ByteTrack's two-stage
association then keeps a confirmed track alive through weak
(low-confidence) detections and gaps of up to 30 frames.

Pure Python (no torch/ultralytics import), so it is fully unit-testable —
like the temporal buffer it complements. The TemporalConsistencyBuffer
in :mod:`temporal_buffer` is kept intact for RQ3 reproducibility; the
demo now confirms via tracking instead of via the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .weapon_detector import Detection


@dataclass(frozen=True)
class TrackedDetection:
    """One track visible in the current frame, with its voted label."""

    detection: Detection  # voted class; current frame's bbox/confidence
    track_id: int
    hits: int  # sightings of this track so far
    vote_share: float  # weighted vote share of the winning class (0.0-1.0)
    raw_class_id: int  # class the model gave this frame, before voting
    confirmed: bool  # hits >= min_hits


@dataclass
class _TrackVotes:
    votes: Dict[int, float] = field(default_factory=dict)  # class_id -> conf sum
    counts: Dict[int, int] = field(default_factory=dict)  # class_id -> sightings
    hits: int = 0
    last_seen: int = 0  # frame index of the latest sighting
    last_class_id: int = -1


class TrackClassVoter:
    """Assigns one stable, voted label per ByteTrack track."""

    def __init__(self, *, min_hits: int = 3, max_age: int = 30) -> None:
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if max_age < 1:
            raise ValueError("max_age must be >= 1")
        self.min_hits = min_hits
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
        self, tracked: List[Tuple[int, Detection]]
    ) -> List[TrackedDetection]:
        """Feed one frame's (track_id, detection) pairs.

        Returns one :class:`TrackedDetection` per track visible this
        frame, each carrying the track's voted label. ``confirmed`` is
        True once the track has ``min_hits`` sightings.
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
            state.hits += 1
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
            )
            visible.append(
                TrackedDetection(
                    detection=voted,
                    track_id=track_id,
                    hits=state.hits,
                    vote_share=vote_share,
                    raw_class_id=det.class_id,
                    confirmed=state.hits >= self.min_hits,
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
