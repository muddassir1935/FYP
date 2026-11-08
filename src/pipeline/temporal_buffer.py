"""Temporal consistency buffer (FR-05).

Pure-Python smoothing layer between per-frame detections and threat alerts.
Each detection is tracked across frames by class + bounding-box IoU. A
track is *confirmed* only when it was seen in at least ``required_hits`` of
the last ``window`` frames (default: 6 of 10 — the rule from the project
proposal). Single-frame false alarms and one-frame flicker therefore never
raise an alert, at zero extra model cost (no extra CNN — deques and IoU
only).

Deterministic and dependency-free, so it is fully unit-testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

from .weapon_detector import BBox, Detection


def bbox_iou(a: BBox, b: BBox) -> float:
    """Intersection-over-Union of two (x1, y1, x2, y2) boxes, 0.0–1.0."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = inter_x2 - inter_x1, inter_y2 - inter_y1
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class ConfirmedDetection:
    """A current-frame detection whose track passed the consistency rule."""

    detection: Detection
    track_id: int
    hits: int  # sightings inside the current window (of `window` frames)


@dataclass
class _Track:
    track_id: int
    class_id: int
    bbox: BBox
    hits: Deque[bool] = field(default_factory=deque)


class TemporalConsistencyBuffer:
    """Confirms detections that persist across a sliding frame window."""

    def __init__(
        self,
        *,
        window: int = 10,
        required_hits: int = 6,
        iou_threshold: float = 0.3,
    ) -> None:
        if not 1 <= required_hits <= window:
            raise ValueError("required_hits must be between 1 and window")
        self.window = window
        self.required_hits = required_hits
        self.iou_threshold = iou_threshold
        self._tracks: List[_Track] = []
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    @property
    def active_tracks(self) -> int:
        return len(self._tracks)

    def update(self, detections: List[Detection]) -> List[ConfirmedDetection]:
        """Feed one frame's detections; return this frame's confirmed ones."""
        matched_track_ids: set[int] = set()
        matched_det_indexes: set[int] = set()
        assignments: List[Tuple[_Track, Detection]] = []

        # Greedy best-IoU matching, same class only.
        candidates: List[Tuple[float, _Track, int, Detection]] = []
        for track in self._tracks:
            for det_index, det in enumerate(detections):
                if det.class_id != track.class_id:
                    continue
                iou = bbox_iou(track.bbox, det.bbox)
                if iou >= self.iou_threshold:
                    candidates.append((iou, track, det_index, det))
        candidates.sort(key=lambda c: c[0], reverse=True)
        for _iou, track, det_index, det in candidates:
            if track.track_id in matched_track_ids or det_index in matched_det_indexes:
                continue
            matched_track_ids.add(track.track_id)
            matched_det_indexes.add(det_index)
            track.bbox = det.bbox
            track.hits.append(True)
            assignments.append((track, det))

        # Tracks not seen this frame record a miss.
        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.hits.append(False)

        # Unmatched detections start new tracks.
        for det_index, det in enumerate(detections):
            if det_index in matched_det_indexes:
                continue
            track = _Track(
                track_id=self._next_track_id,
                class_id=det.class_id,
                bbox=det.bbox,
                hits=deque([True], maxlen=self.window),
            )
            self._next_track_id += 1
            self._tracks.append(track)
            assignments.append((track, det))

        # Every track's window is capped; drop tracks that went fully cold.
        for track in self._tracks:
            while len(track.hits) > self.window:
                track.hits.popleft()
        self._tracks = [
            t for t in self._tracks if len(t.hits) < self.window or any(t.hits)
        ]

        return [
            ConfirmedDetection(
                detection=det, track_id=track.track_id, hits=sum(track.hits)
            )
            for track, det in assignments
            if sum(track.hits) >= self.required_hits
        ]
