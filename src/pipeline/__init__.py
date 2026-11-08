"""SENTINEL real-time pipeline: capture -> detect -> temporal confirm."""

from .capture import FrameSource
from .temporal_buffer import ConfirmedDetection, TemporalConsistencyBuffer, bbox_iou
from .weapon_detector import Detection, WeaponDetector

__all__ = [
    "FrameSource",
    "WeaponDetector",
    "Detection",
    "TemporalConsistencyBuffer",
    "ConfirmedDetection",
    "bbox_iou",
]
