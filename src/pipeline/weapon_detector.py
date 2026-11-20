"""YOLOv8 weapon detector (FR-02).

Thin wrapper around the fine-tuned YOLOv8s model. Class names are read from
the loaded model itself (``model.names``) — nothing about Gun/Knife is
hard-coded here, so upgrading the model is a file swap: pass a different
``weights`` path (or replace the file at the default path,
``models/weights/best.pt``) and no code changes.

Ultralytics is imported lazily in ``__init__`` so this module can be
imported (and the :class:`Detection` type used by the temporal buffer and
its tests) without torch installed.

Per-class confidence thresholds: Gun is the false-positive-prone class
(val AP50 0.78 vs Knife 0.94), so it gets a higher threshold than Knife.
The values live in the demo (``scripts/inference/run_weapon_demo.py``),
tuned on the val split — see
``sentinel-dataset-v2/thresholds/operating_point.md``. The detector runs
the model once at the LOWEST threshold and post-filters per class, which
is exactly equivalent to filtering each class before NMS (NMS always
keeps the highest-confidence box, so a lower predict threshold can never
suppress a box that would survive at the higher per-class threshold).
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
    ``docs/training_run_log.md``) or to an exported OpenVINO model folder
    (``scripts/training/export_openvino.py``). ``device``: ``"cuda"`` /
    ``"cpu"`` / ``"intel:gpu"`` (OpenVINO models) or None for ultralytics'
    automatic choice.

    ``conf_threshold`` is the fallback threshold for any class without a
    per-class override. ``conf_thresholds`` maps a class id (``int``) or
    class name (``str``, e.g. ``"Gun"``) to that class's own threshold;
    a class's own threshold wins over the fallback.
    """

    def __init__(
        self,
        weights: Union[str, Path],
        *,
        conf_threshold: float = 0.25,
        conf_thresholds: Optional[Dict[Union[int, str], float]] = None,
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
                f"Weights not found: {weights_path}. Download the model "
                "(.pt file, or an exported OpenVINO model folder) from "
                "Google Drive SENTINEL_FYP/runs/ (see "
                "docs/training_run_log.md) and place it there — weights are "
                "never stored in Git."
            )
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError(
                f"conf_threshold must be in (0, 1], got {conf_threshold}"
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
        self.conf_thresholds = self._resolve_thresholds(conf_thresholds or {})
        # Predict at the lowest threshold so no candidate above any
        # per-class threshold is pre-filtered away; filtering happens
        # per class in detect().
        self._predict_conf = min(
            [conf_threshold] + list(self.conf_thresholds.values())
        )

    def _resolve_thresholds(
        self, thresholds: Dict[Union[int, str], float]
    ) -> Dict[int, float]:
        """Resolve per-class thresholds to a {class_id: threshold} dict."""
        by_name = {name: cid for cid, name in self.class_names.items()}
        resolved: Dict[int, float] = {}
        for key, value in thresholds.items():
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"threshold for class {key!r} must be in (0, 1], "
                    f"got {value}"
                )
            if isinstance(key, int):
                if key not in self.class_names:
                    raise ValueError(
                        f"unknown class id {key}; known ids: "
                        f"{sorted(self.class_names)}"
                    )
                resolved[key] = value
            else:
                if key not in by_name:
                    raise ValueError(
                        f"unknown class name {key!r}; known names: "
                        f"{sorted(by_name)}"
                    )
                resolved[by_name[key]] = value
        return resolved

    def threshold_for(self, class_id: int) -> float:
        """The confidence threshold that applies to ``class_id``."""
        return self.conf_thresholds.get(class_id, self.conf_threshold)

    def detect(self, frame: object) -> List[Detection]:
        """Detect weapons in one frame (an OpenCV/numpy image)."""
        results = self._model.predict(
            source=frame,
            conf=self._predict_conf,
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
            confidence = float(box.conf[0])
            if confidence < self.threshold_for(class_id):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=self.class_names.get(class_id, str(class_id)),
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                )
            )
        return detections
