# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Request/response models for the demo console API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.pipeline.thresholds import CONF_GUN, CONF_KNIFE, DEFAULT_CONF


class RunStartRequest(BaseModel):
    source_type: str = Field(..., pattern="^(webcam|video)$")
    source: str = Field(..., min_length=1)
    weights_id: str = Field(..., min_length=1)
    conf_gun: float = Field(CONF_GUN, gt=0.0, le=1.0)
    conf_knife: float = Field(CONF_KNIFE, gt=0.0, le=1.0)
    imgsz: int = Field(640, ge=320, le=1280)
    min_hits: int = Field(3, ge=1, le=30)
    # Confirmation window / minimum span for the track-vote voter
    # (see TrackClassVoter): the CLI only bounds them as window > 0
    # and span >= 0, so the API pins the rehearsal tuning range —
    # window 0.1–5.0 s, span 0.0–2.0 s.
    confirm_window_s: float = Field(1.0, ge=0.1, le=5.0)
    confirm_min_span_s: float = Field(0.4, ge=0.0, le=2.0)
    tracker: str = Field("bytetrack.yaml", min_length=1)
    loop: bool = False
    hide_unconfirmed: bool = False
    save_video: bool = False
    # When saving video, record the raw (un-annotated) frames by
    # default — the annotated recording is useless as training data,
    # and the gate session is the main source of v5 mining frames.
    record_raw: bool = True


class RunStatus(BaseModel):
    state: str
    source_label: str = ""
    weights_label: str = ""
    display_fps: float = 0.0
    detect_fps: float = 0.0
    frames: int = 0
    detections: int = 0
    confirmed: int = 0
    snapshots: int = 0
    elapsed_s: float = 0.0
    error: str | None = None
    # Non-fatal problems the operator should know about (e.g. the Intel
    # GPU compile failed and the run fell back to CPU, or the video
    # recording could not be opened). Shown in the console status line.
    warning: str | None = None
    saved_video: str | None = None


class VideoEntry(BaseModel):
    name: str
    size_mb: float
    duration_s: float | None = None


class WeightsEntry(BaseModel):
    id: str
    kind: str
    label: str


class SnapshotEntry(BaseModel):
    file: str
    class_: str = Field(..., alias="class")
    track: int
    timestamp: str
    url: str

    model_config = {"populate_by_name": True}


class HealthStatus(BaseModel):
    status: str = "ok"
    weights_dir_exists: bool
    videos_dir_exists: bool
    n_weights: int
    n_videos: int


class DemoDefaults(BaseModel):
    """The shipped demo operating point (from src/pipeline/thresholds).

    Served by GET /api/defaults so the frontend never hardcodes
    thresholds of its own.
    """

    conf_gun: float = CONF_GUN
    conf_knife: float = CONF_KNIFE
    conf_fallback: float = DEFAULT_CONF
