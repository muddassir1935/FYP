# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""SENTINEL real-time pipeline: capture -> detect -> temporal confirm."""

from .annotate import draw_annotations
from .capture import FALLBACK_FPS, FrameSource
from .demo_worker import InferenceWorker
from .rolling_fps import RollingFps
from .temporal_buffer import ConfirmedDetection, TemporalConsistencyBuffer, bbox_iou
from .thresholds import CONF_GUN, CONF_KNIFE, DEFAULT_CONF
from .track_voting import TrackClassVoter, TrackedDetection
from .weapon_detector import Detection, WeaponDetector, detect_openvino_device

__all__ = [
    "FrameSource",
    "FALLBACK_FPS",
    "WeaponDetector",
    "detect_openvino_device",
    "Detection",
    "TemporalConsistencyBuffer",
    "ConfirmedDetection",
    "bbox_iou",
    "RollingFps",
    "TrackClassVoter",
    "TrackedDetection",
    "InferenceWorker",
    "draw_annotations",
    "CONF_GUN",
    "CONF_KNIFE",
    "DEFAULT_CONF",
]
