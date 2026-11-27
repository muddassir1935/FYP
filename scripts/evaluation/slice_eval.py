#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Per-slice evaluation: where does the weapon detector fail?

Runs the model once over a YOLO-format test set (images + labels), then
breaks precision/recall/F1 down by automatically computed slices:

* lighting — dark / medium / bright, from the image's mean brightness;
* blur — sharp / blurry, from Laplacian variance;
* size — small / medium / large, from the box area as a fraction of the
  frame (a proxy for distance: small boxes are usually far objects).

Matching is greedy by confidence (IoU >= 0.5, same class, one-to-one).
A true positive or false negative counts in the size slice of its
ground-truth box; a false positive counts in the size slice of its own
predicted box. Lighting/blur slices are per image.

Thresholds were set from the cleaned frozen test set's own
distributions (Nov 2026): brightness p10 ~71 / p90 ~174, Laplacian
variance p25 ~94, box-area fraction p25 ~0.006 / p75 ~0.18 — so no
slice is starved (dark 14%, bright 15%, blurry 26%, small 31%).

Inspired by the attribute-tagged evaluation in ZeroEyes' US 11,308,335
B2 (a training database searchable by per-frame attributes), reduced to
what an academic eval needs: automatic tags, no hand labelling, and
plain P/R/F1 per slice instead of their ratio gate.

Uses ultralytics directly (like audit-v4/clean_test_eval.py) instead of
WeaponDetector, because eval needs raw detections at a chosen confidence,
including the demo's per-class thresholds (--conf-gun/--conf-knife).
Per-detection confidences are persisted in the JSON ("dets" per image,
captured at the predict floor) so slice numbers can be recomputed at any
threshold >= that floor without re-running inference.

Usage (from the repo root):
    python scripts/evaluation/slice_eval.py \\
        --weights /path/to/best.pt \\
        --images ~/workspace/sentinel-dataset-v2/audit-v4/clean-test/images \\
        --labels ~/workspace/sentinel-dataset-v2/audit-v4/clean-test/labels \\
        --out slice_eval_v4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline import bbox_iou  # noqa: E402

CLASS_NAMES = {0: "Gun", 1: "Knife"}

# Slice thresholds, from the cleaned frozen test set's distributions
# (see module docstring). Fixed values, so runs stay comparable.
DARK_MAX = 80.0  # mean grayscale below this -> dark
BRIGHT_MIN = 150.0  # mean grayscale above this -> bright
BLURRY_MAX = 100.0  # Laplacian variance below this -> blurry
SMALL_MAX = 0.01  # box area fraction below this -> small
LARGE_MIN = 0.05  # box area fraction above this -> large

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Box:
    """One box in pixel xyxy, with a class id."""

    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def area_frac(self, img_w: int, img_h: int) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1) / (img_w * img_h)


@dataclass
class SliceStats:
    """TP/FP/FN counts for one slice, overall and per class."""

    images: int = 0
    gt: int = 0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    per_class: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES.values()}
    )

    def add_tp(self, class_id: int) -> None:
        self.tp += 1
        self.per_class[CLASS_NAMES[class_id]]["tp"] += 1

    def add_fp(self, class_id: int) -> None:
        self.fp += 1
        self.per_class[CLASS_NAMES[class_id]]["fp"] += 1

    def add_fn(self, class_id: int) -> None:
        self.fn += 1
        self.per_class[CLASS_NAMES[class_id]]["fn"] += 1

    @staticmethod
    def _prf1(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
        precision = tp / (tp + fp) if tp + fp > 0 else None
        recall = tp / (tp + fn) if tp + fn > 0 else None
        if precision is None or recall is None or precision + recall == 0:
            f1 = None
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return {"P": precision, "R": recall, "F1": f1}

    def as_dict(self) -> dict:
        out = {
            "images": self.images,
            "gt": self.gt,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            **self._prf1(self.tp, self.fp, self.fn),
            "per_class": {},
        }
        for name, counts in self.per_class.items():
            out["per_class"][name] = {
                **counts,
                **self._prf1(counts["tp"], counts["fp"], counts["fn"]),
            }
        return out


def read_gt(label_path: Path, img_w: int, img_h: int) -> Tuple[List[Box], int]:
    """Parse a YOLO label file. Missing/empty file = no objects (negative)."""
    boxes: List[Box] = []
    skipped = 0
    if not label_path.exists():
        return boxes, skipped
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            skipped += 1
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            skipped += 1
            continue
        if class_id not in CLASS_NAMES or w <= 0 or h <= 0:
            skipped += 1
            continue
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append(Box(class_id, x1, y1, x2, y2))
    return boxes, skipped


def tag_image(gray) -> Tuple[str, str]:
    """(lighting, blur) buckets from a grayscale image array."""
    mean_brightness = float(gray.mean())
    if mean_brightness < DARK_MAX:
        lighting = "dark"
    elif mean_brightness > BRIGHT_MIN:
        lighting = "bright"
    else:
        lighting = "medium"
    import cv2

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = "blurry" if sharpness < BLURRY_MAX else "sharp"
    return lighting, blur


def size_bucket(area_frac: float) -> str:
    if area_frac < SMALL_MAX:
        return "small"
    if area_frac > LARGE_MIN:
        return "large"
    return "medium"


def match(
    preds: List[Tuple[Box, float]], gts: List[Box], iou_thresh: float
) -> Tuple[List[Tuple[Box, Box]], List[Box], List[Box]]:
    """Greedy match by confidence desc. Returns (tp_pairs, fps, fns).

    Each pair is (pred_box, gt_box). Unmatched preds are FPs, unmatched
    GTs are FNs. A pred only matches a GT of the same class.
    """
    matched_gt: set[int] = set()
    tp_pairs: List[Tuple[Box, Box]] = []
    fps: List[Box] = []
    for pred, _conf in sorted(preds, key=lambda p: p[1], reverse=True):
        best_iou = 0.0
        best_idx = -1
        for i, gt in enumerate(gts):
            if i in matched_gt or gt.class_id != pred.class_id:
                continue
            iou = bbox_iou(pred.xyxy, gt.xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_thresh:
            matched_gt.add(best_idx)
            tp_pairs.append((pred, gts[best_idx]))
        else:
            fps.append(pred)
    fns = [gt for i, gt in enumerate(gts) if i not in matched_gt]
    return tp_pairs, fps, fns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-slice P/R/F1: lighting x blur x size."
    )
    parser.add_argument("--weights", required=True, help="Path to best.pt.")
    parser.add_argument("--images", required=True, help="Test images dir.")
    parser.add_argument("--labels", required=True, help="YOLO labels dir.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Detection confidence (flat, both classes). "
                        "0.25 is the v3/v4 comparability setting; the "
                        "hard-frame evals use 0.40 flat and the shipped "
                        "demo uses per-class Gun 0.44 / Knife 0.43 "
                        "(--conf-gun/--conf-knife). The v5 slice gate runs "
                        "at the gate operating point, not 0.25.")
    parser.add_argument("--conf-gun", type=float, default=None,
                        help="Gun confidence (overrides --conf for Gun).")
    parser.add_argument("--conf-knife", type=float, default=None,
                        help="Knife confidence (overrides --conf for Knife).")
    parser.add_argument("--iou", type=float, default=0.5, help="Matching IoU.")
    parser.add_argument("--nms-iou", type=float, default=0.45,
                        help="NMS IoU for inference (default 0.45, matching "
                        "WeaponDetector.iou_threshold — Ultralytics' 0.7 "
                        "predict default would keep boxes the demo "
                        "suppresses).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Max images (0 = all).")
    parser.add_argument("--out", default=None, help="JSON path for the numbers.")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("opencv-python is not installed", file=sys.stderr)
        return 1
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed", file=sys.stderr)
        return 1

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        print(f"No images found in {images_dir}", file=sys.stderr)
        return 1

    # Effective per-class thresholds; inference runs at the lower of the
    # two so per-class filtering happens in scoring, and every captured
    # detection (with its confidence) is persisted for later re-scoring.
    conf_gun = args.conf_gun if args.conf_gun is not None else args.conf
    conf_knife = args.conf_knife if args.conf_knife is not None else args.conf
    conf_by_class = {0: conf_gun, 1: conf_knife}
    capture_conf = min(conf_gun, conf_knife)

    model = YOLO(str(args.weights))

    slices: Dict[str, SliceStats] = {
        "overall": SliceStats(),
        "lighting:dark": SliceStats(),
        "lighting:medium": SliceStats(),
        "lighting:bright": SliceStats(),
        "blur:sharp": SliceStats(),
        "blur:blurry": SliceStats(),
        "size:small": SliceStats(),
        "size:medium": SliceStats(),
        "size:large": SliceStats(),
    }
    image_rows: List[dict] = []
    skipped_lines = 0
    unreadable = 0
    n_pred_total = 0

    for image_path in image_paths:
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            unreadable += 1
            continue
        img_h, img_w = gray.shape
        lighting, blur = tag_image(gray)
        label_path = labels_dir / f"{image_path.stem}.txt"
        gts, skipped = read_gt(label_path, img_w, img_h)
        skipped_lines += skipped

        results = model.predict(
            source=str(image_path),
            conf=capture_conf,
            iou=args.nms_iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )
        preds: List[Tuple[Box, float]] = []
        captured_dets: List[dict] = []
        for box in results[0].boxes:
            class_id = int(box.cls.item())
            if class_id not in CLASS_NAMES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            conf = float(box.conf.item())
            captured_dets.append(
                {"cls": class_id, "conf": round(conf, 4),
                 "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]}
            )
            if conf < conf_by_class[class_id]:
                continue
            preds.append((Box(class_id, x1, y1, x2, y2), conf))
        n_pred_total += len(preds)

        tp_pairs, fps, fns = match(preds, gts, args.iou)

        # Slices this image belongs to (image-level) + overall.
        img_slices = [
            slices["overall"],
            slices[f"lighting:{lighting}"],
            slices[f"blur:{blur}"],
        ]
        for s in img_slices:
            s.images += 1
            s.gt += len(gts)
        for _pred, gt in tp_pairs:
            slices[f"size:{size_bucket(gt.area_frac(img_w, img_h))}"].add_tp(gt.class_id)
            for s in img_slices:
                s.add_tp(gt.class_id)
        for pred in fps:
            slices[f"size:{size_bucket(pred.area_frac(img_w, img_h))}"].add_fp(pred.class_id)
            for s in img_slices:
                s.add_fp(pred.class_id)
        for gt in fns:
            slices[f"size:{size_bucket(gt.area_frac(img_w, img_h))}"].add_fn(gt.class_id)
            for s in img_slices:
                s.add_fn(gt.class_id)
        # Size slices count GTs, not images.
        for gt in gts:
            slices[f"size:{size_bucket(gt.area_frac(img_w, img_h))}"].gt += 1

        image_rows.append(
            {
                "file": image_path.name,
                "lighting": lighting,
                "blur": blur,
                "n_gt": len(gts),
                "n_pred": len(preds),
                # All detections at/above the capture floor, with
                # confidences — slice numbers can be recomputed at any
                # threshold >= capture_conf without re-running inference.
                "dets": captured_dets,
            }
        )

    n_images = len(image_rows)
    n_no_gt = sum(1 for r in image_rows if r["n_gt"] == 0)
    n_gt = sum(r["n_gt"] for r in image_rows)

    def fmt(value: Optional[float]) -> str:
        return f"{value:.3f}" if value is not None else "—"

    print(f"Weights: {args.weights}")
    print(f"Images: {n_images} ({n_no_gt} negatives), GT boxes: {n_gt}, "
          f"preds: {n_pred_total} (Gun conf={conf_gun}, Knife conf={conf_knife}, "
          f"capture floor={capture_conf}, IoU={args.iou})")
    if skipped_lines:
        print(f"Warning: {skipped_lines} malformed label lines skipped")
    if unreadable:
        print(f"Warning: {unreadable} unreadable images skipped")
    print(f"\n{'slice':<16}{'imgs':>6}{'GT':>6}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'P':>8}{'R':>8}{'F1':>8}")
    for name, stats in slices.items():
        d = stats.as_dict()
        # Size slices count GT boxes, not images.
        imgs = "—" if name.startswith("size:") else str(d["images"])
        print(f"{name:<16}{imgs:>6}{d['gt']:>6}{d['tp']:>6}{d['fp']:>6}"
              f"{d['fn']:>6}{fmt(d['P']):>8}{fmt(d['R']):>8}{fmt(d['F1']):>8}")
    print("\nPer class (overall):")
    for name, counts in slices["overall"].as_dict()["per_class"].items():
        print(f"  {name:<6}TP {counts['tp']:>4}  FP {counts['fp']:>4}  "
              f"FN {counts['fn']:>4}  P {fmt(counts['P'])}  "
              f"R {fmt(counts['R'])}  F1 {fmt(counts['F1'])}")

    if args.out:
        result = {
            "weights": str(args.weights),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "conf": args.conf,
            "conf_gun": conf_gun,
            "conf_knife": conf_knife,
            "capture_conf": capture_conf,
            "iou": args.iou,
            "nms_iou": args.nms_iou,
            "imgsz": args.imgsz,
            "n_images": n_images,
            "n_images_no_gt": n_no_gt,
            "n_gt": n_gt,
            "n_pred": n_pred_total,
            "skipped_label_lines": skipped_lines,
            "unreadable_images": unreadable,
            "slices": {name: stats.as_dict() for name, stats in slices.items()},
            "images": image_rows,
        }
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"\nNumbers written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
