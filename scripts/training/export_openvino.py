#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Export the trained weapon model to OpenVINO format (Intel hardware).

OpenVINO is Intel's inference toolkit — NOT a different model. It converts
the same YOLOv8 weights into an Intel-optimised format (a folder holding
``.xml`` + ``.bin``) that runs much faster on Intel CPUs and integrated
GPUs (e.g. Iris Xe) than plain PyTorch on CPU. The ``.pt`` file stays the
master model; the OpenVINO folder is a derived copy for Intel machines.
NVIDIA/RTX demo machines keep using the ``.pt`` + PyTorch/CUDA path.

Two precisions are supported:

* **FP32** (default): the faithful export — same accuracy as the ``.pt``,
  kept as the fallback whenever an INT8 model is also exported.
* **INT8** (``--int8``): NNCF static quantization, calibrated on a sample
  of the *validation* split (never the frozen test split). Typically
  ~1.5–2x faster on Intel iGPUs for a small (<1 point) mAP cost — verify
  on the frozen test set at a milestone before trusting it.

Usage (from the repo root, after ``pip install openvino``;
INT8 additionally needs ``pip install nncf``):

    python scripts/training/export_openvino.py
    # -> creates models/weights/best_openvino_model/

    python scripts/training/export_openvino.py --int8 --data path/to/dataset.yaml
    # -> creates models/weights/best_int8_openvino_model/ (300 val images)

    python scripts/training/export_openvino.py --int8 --data dataset.yaml \\
        --calibration-size 200 --out-dir /path/to/artifacts
    # -> /path/to/artifacts/best_int8_openvino_model/

    python scripts/inference/run_weapon_demo.py \\
        --weights models/weights/best_openvino_model --source demo.mp4

The script finishes with a self-check: it loads the exported folder and
times inference — on a real validation image when ``--data`` is given and
one can be found, otherwise on a blank frame (labelled explicitly as a
blank-frame upper bound) — printing an approximate FPS so the speed-up
over the ``.pt`` model is visible before running the demo. A self-check
that cannot run does not fail the export: the exit code is 0 with a
warning, and 1 is reserved for an actual export failure.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the weapon model to OpenVINO format")
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to the trained .pt weights (default: models/weights/best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size baked into the export (default 640; "
        "use the same value when running the demo).",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Quantize to INT8 with NNCF (needs --data). Calibrates on a "
        "sample of the validation split; never on the frozen test split.",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Dataset YAML (ultralytics format) for INT8 calibration. "
        "The val split is used; required with --int8.",
    )
    parser.add_argument(
        "--calibration-size",
        type=int,
        default=300,
        help="Number of val images used for NNCF calibration (default 300).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Parent directory the exported model folder is moved into "
        "(keeps its ultralytics name, e.g. best_openvino_model — "
        "ultralytics detects the OpenVINO format from that name). "
        "Without it the folder stays next to the .pt weights.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip the post-export load + timing self-check.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        print(f"Weights not found: {weights}", file=sys.stderr)
        print(
            "Place the trained model there first (Drive SENTINEL_FYP/runs/, "
            "see docs/training_run_log.md).",
            file=sys.stderr,
        )
        return 1
    if args.int8 and not args.data:
        print("--int8 needs --data (dataset YAML for calibration).", file=sys.stderr)
        return 1
    if args.data and not Path(args.data).exists():
        print(f"Dataset YAML not found: {args.data}", file=sys.stderr)
        return 1
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed; run: pip install ultralytics", file=sys.stderr)
        return 1

    precision = "INT8" if args.int8 else "FP32"
    print(f"Exporting {weights} to OpenVINO {precision} (imgsz={args.imgsz}) ...")
    if args.int8:
        print(
            f"Calibrating NNCF on {args.calibration_size} val images "
            f"from {args.data} ..."
        )
    try:
        export_kwargs = {"format": "openvino", "imgsz": args.imgsz}
        if args.int8:
            export_kwargs["quantize"] = 8
            export_kwargs["data"] = args.data
            # ultralytics treats an int fraction as an exact image count
            export_kwargs["fraction"] = args.calibration_size
        exported = Path(YOLO(str(weights)).export(**export_kwargs))
    except ImportError as exc:
        missing = "nncf" if args.int8 else "openvino"
        print(f"{missing} is not installed ({exc}); run: pip install {missing}", file=sys.stderr)
        return 1
    print(f"Exported model folder: {exported}")

    if args.out_dir:
        dest = Path(args.out_dir) / exported.name
        if dest.exists():
            print(f"Destination already exists: {dest}", file=sys.stderr)
            print(f"Exported folder left in place: {exported}", file=sys.stderr)
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(dest))
        exported = dest
        print(f"Moved model folder to: {exported}")

    if not args.skip_check:
        # NICE-4: the export already succeeded — a self-check that cannot
        # run must not turn it into a failed export (exit 1 is reserved
        # for actual export failure).
        checked = self_check(exported, args.imgsz, data_yaml=args.data)
        if checked is None:
            print("WARNING: export succeeded but the self-check could not "
                  "run (see above); the exported model folder is usable.",
                  file=sys.stderr)
        else:
            fps, frame_kind = checked
            if frame_kind == "real":
                print(f"Self-check OK — inference on a real validation "
                      f"image ≈ {fps:.1f} FPS on this machine.")
            else:
                print(f"Self-check OK — blank-frame upper bound "
                      f"≈ {fps:.1f} FPS on this machine (no validation "
                      f"image found; real frames will be slower).")

    print()
    print("Run the demo with the exported model:")
    print(
        "  python scripts/inference/run_weapon_demo.py "
        f"--weights {exported} --imgsz {args.imgsz} --source demo.mp4"
    )
    print("(Use --device intel:gpu to select the Intel iGPU explicitly.)")
    return 0


def find_val_image(data_yaml: str | None) -> Path | None:
    """First image of the dataset's val split, or ``None`` if not found.

    The YAML is ultralytics format (``path`` + ``val`` relative to it).
    Any failure — no YAML given, unparsable, no images — returns ``None``
    and the caller falls back to a blank frame.
    """
    if not data_yaml:
        return None
    try:
        import yaml  # PyYAML ships with ultralytics
    except ImportError:
        return None
    yaml_path = Path(data_yaml)
    if not yaml_path.exists():
        return None
    try:
        parsed = yaml.safe_load(yaml_path.read_text())
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    root = Path(str(parsed.get("path", yaml_path.parent)))
    val = parsed.get("val")
    if not val:
        return None
    val_dir = root / str(val) if not Path(str(val)).is_absolute() else Path(str(val))
    if not val_dir.is_dir():
        return None
    for ext in (".jpg", ".jpeg", ".png"):
        found = sorted(val_dir.glob(f"*{ext}"))
        if found:
            return found[0]
    return None


def self_check(exported: Path, imgsz: int,
               data_yaml: str | None = None) -> tuple[float, str] | None:
    """Load the exported model and time inference.

    Times on a real validation image when one can be found via
    ``data_yaml`` (NICE-5); otherwise falls back to a blank frame, which
    the caller must label a "blank-frame upper bound" — blank frames
    produce no detections, so NMS/post-processing cost is minimal and
    real frames will be slower. Returns ``(fps, kind)`` with kind
    ``"real"`` or ``"blank"``, or ``None`` when the check could not run.
    """
    try:
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Self-check could not run ({exc}); the export itself is done.", file=sys.stderr)
        return None
    val_image = find_val_image(data_yaml)
    try:
        model = YOLO(str(exported))
        if val_image is not None:
            from PIL import Image  # Pillow ships with ultralytics
            frame = np.array(Image.open(val_image).convert("RGB"))
            kind = "real"
            print(f"Self-check timing on validation image: {val_image.name}")
        else:
            frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            kind = "blank"
        for _ in range(3):  # warm-up
            model.predict(frame, verbose=False)
        runs = 20
        start = time.perf_counter()
        for _ in range(runs):
            model.predict(frame, verbose=False)
        elapsed = time.perf_counter() - start
    except Exception as exc:  # report, don't crash the export result
        print(f"Self-check FAILED: {exc}", file=sys.stderr)
        return None
    if elapsed <= 0:
        return None
    return runs / elapsed, kind


if __name__ == "__main__":
    raise SystemExit(main())
