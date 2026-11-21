# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Request/response models for the demo console API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunStartRequest(BaseModel):
    source_type: str = Field(..., pattern="^(webcam|video)$")
    source: str = Field(..., min_length=1)
    weights_id: str = Field(..., min_length=1)
    conf_gun: float = Field(0.44, gt=0.0, le=1.0)
    conf_knife: float = Field(0.43, gt=0.0, le=1.0)
    imgsz: int = Field(640, ge=320, le=1280)
    min_hits: int = Field(3, ge=1, le=30)
    loop: bool = False
    hide_unconfirmed: bool = False
    save_video: bool = False


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
    started_at: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None
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
