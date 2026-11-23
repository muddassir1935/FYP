#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""SENTINEL - YOLOv8s weapon detector training (FR-02).

Fine-tunes YOLOv8s on the curated weapon dataset (Gun / Knife), evaluates the
best checkpoint on the held-out test split, and writes a ``run_summary.json``
into the run directory (config + validation/test metrics + weights SHA-256)
so every run can be recorded in the repo's run log.

The summary is the reproducibility record: besides the hyperparameters it
logs the device/workers, seed + determinism, the ultralytics/torch/Python
versions, the repo commit (``--code-revision``), and the dataset identity
(slug + manifest SHA-256) — everything needed to re-create the run. It also
records the best-epoch validation metrics (from ``results.csv``) alongside
the final-epoch ones, and the best-mAP50 epoch vs the best-fitness epoch,
because ``best.pt`` is picked by mAP50-95 while the adoption gate uses mAP50.

Usage:
    python scripts/training/train_weapon_detector.py --data <dataset.yaml> \
        --epochs 50 --imgsz 640 --batch 16 \
        --project runs/weapon --name weapon_yolov8s_baseline

The Ultralytics import is lazy (inside ``main``) so ``--help`` and argument
parsing also work on machines without torch/ultralytics installed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
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
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Deterministic training (Ultralytics default is "
                             "True; passed explicitly so the run record shows "
                             "it). Use --no-deterministic to disable.")
    parser.add_argument("--code-revision", default="",
                        help="Git SHA of the repo the training script ran "
                             "from (the Kaggle notebook passes `git rev-parse "
                             "HEAD`); recorded in the summary.")
    parser.add_argument("--dataset-slug", default="",
                        help="Kaggle dataset slug (e.g. sentinel-weapon-v5); "
                             "recorded in the summary.")
    parser.add_argument("--dataset-version", default="",
                        help="Kaggle dataset version, when known; recorded "
                             "in the summary.")
    parser.add_argument("--exist-ok", action="store_true", default=False,
                        help="Allow overwriting an existing run directory. "
                             "Default False: Ultralytics auto-increments the "
                             "run name, so a re-run cannot silently overwrite "
                             "a previous run's results.csv / best.pt.")
    # Fine-tune / augmentation hyperparameters (YOLOv8 defaults, so a plain
    # run behaves exactly as before). The v2.1 fine-tune passes lower lr0,
    # rotation + perspective, and close_mosaic via these flags.
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (v2 used 0.01; fine-tune "
                             "with 0.001).")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate fraction (lr0 * lrf).")
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="Random rotation range in degrees (+/-).")
    parser.add_argument("--translate", type=float, default=0.1)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--shear", type=float, default=0.0)
    parser.add_argument("--perspective", type=float, default=0.0,
                        help="Random perspective warp (0.0005-0.001).")
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--copy-paste", type=float, default=0.0)
    parser.add_argument("--erasing", type=float, default=0.4,
                        help="Random erasing probability (occlusion).")
    parser.add_argument("--hsv-h", type=float, default=0.015)
    parser.add_argument("--hsv-s", type=float, default=0.7)
    parser.add_argument("--hsv-v", type=float, default=0.4)
    parser.add_argument("--close-mosaic", type=int, default=10,
                        help="Disable mosaic for the final N epochs.")
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
    except Exception as exc:
        # NICE-1: never silently drop per-class metrics — if a future
        # Ultralytics version changes box.ap50 / ap_class_index, say so.
        print(f"WARNING: per-class metrics failed ({exc}); "
              "per_class omitted from the summary.", file=sys.stderr)
        per_class = {}
    if per_class:
        out["per_class"] = per_class
    return out


def resolve_train_output(results: Any, run_dir: Path) -> tuple[Path, dict[str, float]]:
    """Resolve the run directory + train metrics from ``model.train()``.

    Ultralytics changed ``model.train()``'s return type across versions:
    older releases return the trainer (with ``.save_dir`` / ``.results_dict``),
    newer ones return a plain metrics dict. With the trainer the true save
    dir comes from ``.save_dir`` (which matters now that ``exist_ok`` defaults
    to False and Ultralytics may auto-increment the run name); with a plain
    dict the run dir is the ``project/name`` fallback.
    """
    if isinstance(results, dict):
        raw_metrics: dict = results
        save_dir = run_dir
    else:
        raw_metrics = getattr(results, "results_dict", {}) or {}
        save_dir = Path(getattr(results, "save_dir", run_dir))
    train_results: dict[str, float] = {}
    for key, value in raw_metrics.items():
        try:
            train_results[str(key)] = round(float(value), 4)
        except (TypeError, ValueError):
            # NICE-9: log dropped non-numeric metrics instead of silently
            # losing them — a future Ultralytics version may return a
            # non-numeric value for a metric.
            print(f"[debug] resolve_train_output: dropped non-numeric "
                  f"metric {key!r}={value!r}")
            continue
    return save_dir, train_results


def analyze_results_csv(results_csv: Path) -> dict[str, Any]:
    """Best-epoch analysis from ``results.csv`` (SHOULD-2 / SHOULD-3).

    Ultralytics picks ``best.pt`` by fitness, and in 8.4.215 fitness *is*
    mAP50-95 (``DetMetrics.fitness`` weights ``[0, 0, 0, 1]``) — while the
    adoption gate uses mAP50. This records both best epochs, the mAP50 gap
    between them, and the best-epoch (max-fitness) validation metrics, so a
    run where the two criteria disagree is visible in the summary instead
    of silently adopting a checkpoint that is not best on the gated metric.

    Returns ``{}`` when the file is missing or has no usable metric rows —
    a degraded record is better than a crashed run.
    """
    if not results_csv.exists():
        return {}
    map50_col = "metrics/mAP50(B)"
    map5095_col = "metrics/mAP50-95(B)"
    rows: list[dict[str, float]] = []
    try:
        with results_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return {}
            if map50_col not in reader.fieldnames or map5095_col not in reader.fieldnames:
                return {}
            for raw in reader:
                try:
                    row = {
                        "epoch": int(float(raw["epoch"])),
                        "precision": float(raw.get("metrics/precision(B)", "nan")),
                        "recall": float(raw.get("metrics/recall(B)", "nan")),
                        "mAP50": float(raw[map50_col]),
                        "mAP50_95": float(raw[map5095_col]),
                    }
                except (TypeError, ValueError):
                    continue
                if math.isnan(row["mAP50"]) or math.isnan(row["mAP50_95"]):
                    continue
                rows.append(row)
    except OSError:
        return {}
    if not rows:
        return {}
    best_fitness = max(rows, key=lambda r: r["mAP50_95"])
    best_map50 = max(rows, key=lambda r: r["mAP50"])
    best_metrics: dict[str, Any] = {"epoch": best_fitness["epoch"]}
    for key in ("precision", "recall", "mAP50", "mAP50_95"):
        value = best_fitness[key]
        if not math.isnan(value):
            best_metrics[key] = round(value, 4)
    best_metrics["fitness"] = round(best_fitness["mAP50_95"], 4)
    return {
        "best_fitness_epoch": best_fitness["epoch"],
        "best_map50_epoch": best_map50["epoch"],
        "map50_gap": round(best_map50["mAP50"] - best_fitness["mAP50"], 4),
        "fitness_definition": "mAP50-95 (ultralytics 8.4.215 fitness weights [0,0,0,1])",
        "best_val_metrics": best_metrics,
    }


def dataset_manifest_info(data_yaml: Path) -> dict[str, Any]:
    """Identity of the dataset behind ``data_yaml`` (MUST-12).

    The assembly manifest (``manifest.json``) sits next to the data YAML in
    the dataset root. Its SHA-256 pins exactly which dataset a run trained
    on — the Kaggle path in ``config["data"]`` is ephemeral and says
    nothing about the dataset's contents. Returns ``{}`` when no manifest
    is found (e.g. a local smoke run on a hand-written YAML).
    """
    manifest = data_yaml.parent / "manifest.json"
    if not manifest.exists():
        return {}
    info: dict[str, Any] = {"dataset_manifest_sha256": sha256_of(manifest)}
    try:
        parsed = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return info
    if isinstance(parsed, dict):
        if parsed.get("review_status") is not None:
            info["dataset_manifest_review_status"] = parsed["review_status"]
        audit = parsed.get("audit")
        if isinstance(audit, dict) and audit.get("verdict") is not None:
            info["dataset_manifest_audit_verdict"] = audit["verdict"]
    return info


def load_resumed_config(save_dir: Path) -> dict[str, Any] | None:
    """Config of the run being resumed, from its ``args.yaml`` (SHOULD-5).

    On ``--resume`` Ultralytics restores the original run's arguments and
    ignores every CLI flag, so the summary must not describe the resume
    invocation's flags as the run's config. Returns ``None`` when the file
    is missing, unparseable, or PyYAML is unavailable — the caller then
    omits the hyperparameters and notes the run as resumed.
    """
    args_yaml = save_dir / "args.yaml"
    if not args_yaml.exists():
        return None
    try:
        import yaml  # PyYAML ships with ultralytics; absent on a bare box
    except ImportError:
        return None
    try:
        parsed = yaml.safe_load(args_yaml.read_text())
    except Exception:  # corrupt args.yaml: fall back to "resumed, config omitted"
        return None
    return parsed if isinstance(parsed, dict) else None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from ultralytics import YOLO  # lazy: keeps --help torch-free
    import torch
    import ultralytics

    data_yaml = str(Path(args.data).resolve())
    run_dir = Path(args.project) / args.name

    if args.resume:
        last = run_dir / "weights" / "last.pt"
        if not last.exists():
            print(f"Resume requested but {last} not found.")
            return 1
        # SHOULD-5: Ultralytics restores the original run's arguments from
        # last.pt and ignores every CLI flag on the resume invocation.
        print("WARNING: --resume ignores all CLI hyperparameter flags "
              "(epochs, lr0, batch, ...); the original run's args.yaml "
              "governs. The summary config is loaded from the resumed "
              "run's args.yaml, not from these flags.", file=sys.stderr)
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
            deterministic=args.deterministic,
            lr0=args.lr0,
            lrf=args.lrf,
            warmup_epochs=args.warmup_epochs,
            degrees=args.degrees,
            translate=args.translate,
            scale=args.scale,
            shear=args.shear,
            perspective=args.perspective,
            flipud=args.flipud,
            fliplr=args.fliplr,
            mosaic=args.mosaic,
            mixup=args.mixup,
            copy_paste=args.copy_paste,
            erasing=args.erasing,
            hsv_h=args.hsv_h,
            hsv_s=args.hsv_s,
            hsv_v=args.hsv_v,
            close_mosaic=args.close_mosaic,
            project=args.project,
            name=args.name,
            exist_ok=args.exist_ok,
        )

    save_dir, train_results = resolve_train_output(results, run_dir)
    best = save_dir / "weights" / "best.pt"

    # Reproducibility record (MUST-12, SHOULD-1): library versions, code
    # revision, dataset identity, device/workers, seed + determinism —
    # everything needed to re-create the run, none of which the v3/v4
    # summaries recorded.
    config: dict[str, Any] = {
        "data": data_yaml,
        "device": args.device,
        "workers": args.workers,
        "deterministic": args.deterministic,
        "seed": args.seed,
        "code_revision": args.code_revision or None,
        "dataset_slug": args.dataset_slug or None,
        "dataset_version": args.dataset_version or None,
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "python_version": sys.version.split()[0],
    }
    config.update(dataset_manifest_info(Path(args.data)))
    hyperparams = ("model", "epochs", "imgsz", "batch", "patience",
                   "lr0", "lrf", "warmup_epochs", "degrees", "translate",
                   "scale", "shear", "perspective", "flipud", "fliplr",
                   "mosaic", "mixup", "copy_paste", "erasing", "hsv_h",
                   "hsv_s", "hsv_v", "close_mosaic")
    if args.resume:
        # SHOULD-5: the resumed run's real config comes from its args.yaml,
        # not from the (ignored) CLI flags of this invocation.
        resumed = load_resumed_config(save_dir)
        config["resumed"] = True
        if resumed is not None:
            config["resumed_config_source"] = "args.yaml"
            # Seed/determinism/device/workers are also restored from the
            # original run, not from this invocation's (ignored) flags.
            for key in hyperparams + ("device", "workers", "deterministic",
                                      "seed"):
                if key in resumed:
                    config[key] = resumed[key]
        else:
            config["resumed_config_source"] = (
                "unavailable — args.yaml missing; hyperparameters omitted, "
                "CLI flags were ignored on resume")
    else:
        for key in hyperparams:
            config[key] = getattr(args, key)

    # SHOULD-2 / SHOULD-3: best-epoch metrics from results.csv.
    epoch_analysis = analyze_results_csv(save_dir / "results.csv")
    summary: dict[str, Any] = {
        "run_name": args.name,
        "config": config,
        # Final-epoch metrics, relabelled: these are the LAST epoch's
        # validation numbers, not the adopted checkpoint's — see
        # best_val_metrics for the checkpoint best.pt came from.
        "train_val_metrics_final_epoch": train_results,
        "best_val_metrics": epoch_analysis.get("best_val_metrics", {}),
        "best_epoch_metrics": {k: v for k, v in epoch_analysis.items()
                               if k != "best_val_metrics"},
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
    else:
        # SHOULD-4: a run with no best.pt is a failed run, not a
        # successful run with empty metrics — exit non-zero and write no
        # summary that could be mistaken for a valid record.
        print(f"ERROR: best.pt missing at {best} — training did not "
              "produce a checkpoint; no summary written.", file=sys.stderr)
        return 1

    summary_path = save_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n================ SENTINEL RUN SUMMARY ================")
    print(f"Run dir      : {save_dir}")
    print(f"Test metrics : {json.dumps(summary['test_metrics'])}")
    if summary["best_epoch_metrics"]:
        print(f"Best epochs  : {json.dumps(summary['best_epoch_metrics'])}")
    print(f"Weights      : {json.dumps(summary['weights'])}")
    print(f"SUMMARY_JSON : {summary_path}")
    print("======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
