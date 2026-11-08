#!/usr/bin/env python3
"""SENTINEL - YOLOv8s weapon detector training (FR-02).

Fine-tunes YOLOv8s on the curated weapon dataset (Gun / Knife), evaluates the
best checkpoint on the held-out test split, and writes a ``run_summary.json``
into the run directory (config + validation/test metrics + weights SHA-256)
so every run can be recorded in the repo's run log.

Usage:
    python scripts/training/train_weapon_detector.py --data <dataset.yaml> \
        --epochs 50 --imgsz 640 --batch 16 \
        --project runs/weapon --name weapon_yolov8s_baseline

The Ultralytics import is lazy (inside ``main``) so ``--help`` and argument
parsing also work on machines without torch/ultralytics installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the SENTINEL YOLOv8s weapon detector (Gun/Knife)."
    )
    parser.add_argument("--data", required=True,
                        help="Path to dataset YAML (nc=2, names: Gun, Knife).")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="Base weights or local .pt path (default: yolov8s.pt).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="",
                        help="'' = auto (CUDA if available), or e.g. '0' / 'cpu'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/weapon")
    parser.add_argument("--name", default="weapon_yolov8s")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted run from its last.pt.")
    return parser.parse_args(argv)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_block(metrics: Any, names: dict) -> dict[str, Any]:
    """Headline + per-class detection metrics, defensive across versions."""
    out: dict[str, Any] = {}
    box = getattr(metrics, "box", None)
    if box is None:
        return out
    for attr, key in (("map50", "mAP50"), ("map", "mAP50_95"),
                      ("mp", "precision"), ("mr", "recall")):
        value = getattr(box, attr, None)
        if value is not None:
            out[key] = round(float(value), 4)
    if "precision" in out and "recall" in out:
        p, r = out["precision"], out["recall"]
        out["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
    per_class: dict[str, Any] = {}
    try:
        ap50 = box.ap50  # AP@0.5 per class, aligned with ap_class_index
        for i, cls_idx in enumerate(box.ap_class_index):
            name = names.get(int(cls_idx), str(cls_idx))
            per_class[str(name)] = {"AP50": round(float(ap50[i]), 4)}
    except Exception:
        per_class = {}
    if per_class:
        out["per_class"] = per_class
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO  # lazy: keeps --help torch-free

    data_yaml = str(Path(args.data).resolve())
    run_dir = Path(args.project) / args.name

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            print(f"Resume requested but {last} not found.")
            return 1
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            patience=args.patience,
            seed=args.seed,
            project=args.project,
            name=args.name,
            exist_ok=True,
        )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"

    train_results = {}
    for key, value in (getattr(results, "results_dict", {}) or {}).items():
        try:
            train_results[str(key)] = round(float(value), 4)
        except (TypeError, ValueError):
            continue

    summary: dict[str, Any] = {
        "run_name": args.name,
        "config": {
            "data": data_yaml,
            "model": args.model,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "seed": args.seed,
        },
        "train_val_metrics": train_results,
        "test_metrics": {},
        "weights": {},
    }

    if best.exists():
        detector = YOLO(str(best))
        test = detector.val(data=data_yaml, split="test",
                            imgsz=args.imgsz, batch=args.batch, verbose=False)
        summary["test_metrics"] = metric_block(test, detector.names)
        summary["weights"] = {
            "file": best.name,
            "path": str(best),
            "size_mb": round(best.stat().st_size / 1e6, 2),
            "sha256": sha256_of(best),
        }

    summary_path = save_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n================ SENTINEL RUN SUMMARY ================")
    print(f"Run dir      : {save_dir}")
    print(f"Test metrics : {json.dumps(summary['test_metrics'])}")
    print(f"Weights      : {json.dumps(summary['weights'])}")
    print(f"SUMMARY_JSON : {summary_path}")
    print("======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
