# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Draw detection boxes on a frame (shared by the CLI demo and the web UI)."""

from __future__ import annotations

from .track_voting import TrackedDetection
from .weapon_detector import Detection


def draw_annotations(
    cv2_module: object,
    frame: object,
    detections: list[Detection],
    confirmed: list[TrackedDetection],
    *,
    hide_unconfirmed: bool = False,
) -> object:
    """Return a copy of ``frame`` with detection/confirmation boxes drawn.

    Confirmed threats are thick red with a filled label chip showing the
    track ID, sighting count and vote share; unconfirmed tracks are thin
    muted gray (raw model output always contains some noise — the
    track-vote confirmation is the system's actual verdict). Every box
    already carries its track's voted label, so one object keeps one
    label across frames.
    """
    display = frame.copy()
    # Match confirmed threats by track_id (carried on the voted
    # Detection), not by object identity: any caller that rebuilds
    # Detection objects would otherwise draw confirmed threats as
    # unconfirmed grey boxes.
    confirmed_by_track = {cd.track_id: cd for cd in confirmed}
    font = cv2_module.FONT_HERSHEY_SIMPLEX
    for det in detections:
        cd = (
            confirmed_by_track.get(det.track_id)
            if det.track_id is not None
            else None
        )
        if cd is None and hide_unconfirmed:
            continue
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        if cd is not None:
            color, thickness = (0, 0, 255), 3
            label = (
                f"{det.class_name} {det.confidence:.2f} CONFIRMED "
                f"t{cd.track_id} {cd.hits}f vote {cd.vote_share:.2f}"
            )
        else:
            color, thickness = (170, 170, 170), 1
            label = f"{det.class_name} {det.confidence:.2f}"
        cv2_module.rectangle(display, (x1, y1), (x2, y2), color, thickness)
        (text_w, text_h), baseline = cv2_module.getTextSize(label, font, 0.55, 2)
        chip_h = text_h + baseline + 6
        chip_y1 = y1 - chip_h if y1 >= chip_h else y1
        chip_y2 = chip_y1 + chip_h
        # Clamp the chip inside the frame: near the right edge the label
        # would otherwise run off-screen and get cut.
        chip_x = min(x1, max(0, display.shape[1] - text_w - 8))
        cv2_module.rectangle(display, (chip_x, chip_y1), (chip_x + text_w + 8, chip_y2), color, -1)
        cv2_module.putText(
            display, label, (chip_x + 4, chip_y2 - baseline - 3),
            font, 0.55, (255, 255, 255), 2, cv2_module.LINE_AA,
        )
    return display
