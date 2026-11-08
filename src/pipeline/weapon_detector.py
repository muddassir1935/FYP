"""YOLOv8 weapon detector (FR-02).

Thin wrapper around the fine-tuned YOLOv8s model. Class names are read from
the loaded model itself (``model.names``) — nothing about Gun/Knife is
hard-coded here, so upgrading the model is a file swap: pass a different
``weights`` path (or replace the file at the default path,
``models/weights/best.pt``) and no code changes.

Ultralytics is imported lazily in ``__init__`` so this module can be
imported (and the :class:`Detection` type used by the temporal buffer and
its tests) without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2), pixels


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class WeaponDetector:
    """Runs the weapon model on single frames.

    ``weights``: path to a ``.pt`` file (baseline v1:
    ``weapon_yolov8s_baseline_20261108_1837_best.pt`` — see
    ``docs/training_run_log.md``). ``device``: ``"cuda"``/``"cpu"`` or
    None for ultralytics' automatic choice.
    """

    def __init__(
        self,
        weights: Union[str, Path],
        *,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError(
                "ultralytics is not installed; run: pip install ultralytics"
            ) from exc
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}. Download the model "
                "from Google Drive SENTINEL_FYP/runs/ (see "
                "docs/training_run_log.md) and place it there — weights are "
                "never stored in Git."
            )
        self.weights = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self._model = YOLO(str(weights_path))
        names = self._model.names
        self.class_names: Dict[int, str] = (
            dict(names) if isinstance(names, dict) else dict(enumerate(names))
        )

    def detect(self, frame: object) -> List[Detection]:
        """Detect weapons in one frame (an OpenCV/numpy image)."""
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        boxes = results[0].boxes
        if boxes is None:
            return detections
        for box in boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=float(box.conf[0]),
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
