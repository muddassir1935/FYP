# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""YOLOv8 weapon detector (FR-02).

Thin wrapper around the fine-tuned YOLOv8s model. Class names are read from
the loaded model itself (``model.names``) — nothing about Gun/Knife is
hard-coded here, so upgrading the model is a file swap: pass a different
``weights`` path (or replace the file at the default path,
``models/weights/best.pt``) and no code changes.

Ultralytics is imported lazily in ``__init__`` so this module can be
imported (and the :class:`Detection` type used by the temporal buffer and
its tests) without torch installed.

Per-class confidence thresholds: the shipped point runs Gun slightly
higher than Knife (0.44 vs 0.43) — that split was tuned for the v2
weights on the val split, not from live behaviour. Live, Knife is the
leaky class (it fires on any elongated object) and Gun is recall-limited.
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
    """One detected object in one frame.

    ``track_id`` is set by the track voter (``track_voting``) on the
    voted copy of a detection; raw model output leaves it ``None``.
    ``draw_annotations`` matches confirmed threats by it, not by
    object identity, so rebuilt detections still draw as confirmed.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    track_id: int | None = None


def detect_openvino_device(weights_path: Union[str, Path]) -> Optional[str]:
    """Select ``"xpu"`` for OpenVINO folders when an Intel GPU is
    available, leaving PyTorch (.pt) and CPU-only machines untouched.

    Shared by the web UI runner and the CLI demo so both pick the same
    device for the same weights. ``"xpu"`` is the PyTorch-style name
    ultralytics accepts since 8.4.210; the older ``"intel:gpu"`` alias
    is deprecated in the pinned 8.4.215 and emits a warning per run.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_dir():
        return None
    try:
        import openvino as ov

        devices = ov.Core().available_devices
        if any(d.startswith("GPU") for d in devices):
            return "xpu"
    except Exception:
        pass
    return None


class WeaponDetector:
    """Runs the weapon model on single frames.

    ``weights``: path to a ``.pt`` file (baseline v1:
    ``weapon_yolov8s_baseline_20261108_1837_best.pt`` — see
    ``docs/training_run_log.md``) or to an exported OpenVINO model folder
    (``scripts/training/export_openvino.py``). ``device``: ``"cuda"`` /
    ``"cpu"`` / ``"xpu"`` (OpenVINO models on an Intel GPU) or None for
    ultralytics' automatic choice.

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
        self._tracker_reset_pending = False
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

    def update_thresholds(
        self,
        *,
        conf_threshold: Optional[float] = None,
        conf_thresholds: Optional[Dict[Union[int, str], float]] = None,
    ) -> None:
        """Update per-class / fallback confidence thresholds in place.

        Allows callers (such as ``DemoRunner``) to reuse a warmed-up
        detector across runs when only the confidence thresholds change.
        """
        if conf_threshold is not None:
            if not 0.0 < conf_threshold <= 1.0:
                raise ValueError(
                    f"conf_threshold must be in (0, 1], got {conf_threshold}"
                )
            self.conf_threshold = conf_threshold
        if conf_thresholds is not None:
            self.conf_thresholds = self._resolve_thresholds(conf_thresholds)
        self._predict_conf = min(
            [self.conf_threshold] + list(self.conf_thresholds.values())
        )

    def warmup(self, *, tracker: Optional[str] = "bytetrack.yaml") -> None:
        """Run one dummy inference pass to trigger lazy model compilation.

        OpenVINO and CUDA backends compile kernels on the first inference
        call; warming up before starting video capture prevents opening
        frames from racing past while the model is still cold.
        """
        import numpy as np

        blank = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        if tracker:
            self.track(blank, tracker=tracker)
            self.reset_tracker()
        else:
            self.detect(blank)

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
        return self._boxes_to_detections(results[0].boxes)

    def track(
        self, frame: object, *, tracker: str = "bytetrack.yaml"
    ) -> List[Tuple[int, Detection]]:
        """Detect + ByteTrack one frame; return ``(track_id, detection)`` pairs.

        Uses ``model.track(persist=True)`` so track IDs stay stable across
        calls on consecutive frames of the same video. What keeps a track
        alive through gaps is the lost-track buffer (``track_buffer=30``
        in the stock ``bytetrack.yaml``): a track survives 30 frames
        without a sighting, then gets a fresh ID if it reappears. The
        tracker's two-stage association adds nothing here — the detector
        runs at conf 0.40, above the stock high threshold (0.25), so
        every box reaching the tracker is already a first-stage
        detection and the low-confidence second stage (0.10–0.25) has
        nothing to match. Call :meth:`reset_tracker` when the video
        source changes (new file, new camera) so stale tracks do not
        leak into the new stream.

        Works for both ``.pt`` weights and exported OpenVINO folders:
        Ultralytics runs the same tracker on either model's boxes.
        """
        results = self._model.track(
            source=frame,
            persist=not self._tracker_reset_pending,
            tracker=tracker,
            # Predict at the lowest threshold so no candidate above any
            # per-class threshold is pre-filtered away; per-class
            # filtering happens on the returned pairs below.
            conf=self._predict_conf,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        self._tracker_reset_pending = False
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []
        tracked: List[Tuple[int, Detection]] = []
        for box, track_id in zip(boxes, boxes.id.tolist()):
            detection = self._box_to_detection(box)
            if detection.confidence < self.threshold_for(detection.class_id):
                continue
            tracked.append((int(track_id), detection))
        return tracked

    def reset_tracker(self) -> None:
        """Drop all ByteTrack state; the next :meth:`track` call starts fresh IDs.

        Implemented by running the next ``track`` call with
        ``persist=False``, which makes Ultralytics create new trackers
        instead of reusing the old ones.
        """
        self._tracker_reset_pending = True

    def _boxes_to_detections(self, boxes: object) -> List[Detection]:
        if boxes is None:
            return []
        return [
            self._box_to_detection(box)
            for box in boxes
            if float(box.conf[0]) >= self.threshold_for(int(box.cls[0]))
        ]

    def _box_to_detection(self, box: object) -> Detection:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        return Detection(
            class_id=class_id,
            class_name=self.class_names.get(class_id, str(class_id)),
            confidence=float(box.conf[0]),
            bbox=(x1, y1, x2, y2),
        )
