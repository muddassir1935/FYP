#!/usr/bin/env python3
"""Export the trained weapon model to OpenVINO format (Intel hardware).

OpenVINO is Intel's inference toolkit — NOT a different model. It converts
the same YOLOv8 weights into an Intel-optimised format (a folder holding
``.xml`` + ``.bin``) that runs much faster on Intel CPUs and integrated
GPUs (e.g. Iris Xe) than plain PyTorch on CPU. The ``.pt`` file stays the
master model; the OpenVINO folder is a derived copy for Intel machines.
NVIDIA/RTX demo machines keep using the ``.pt`` + PyTorch/CUDA path.

Usage (from the repo root, after ``pip install openvino``):

    python scripts/training/export_openvino.py
    # -> creates models/weights/best_openvino_model/

    python scripts/inference/run_weapon_demo.py \
        --weights models/weights/best_openvino_model --source demo.mp4

The script finishes with a self-check: it loads the exported folder and
times inference on a blank frame, printing an approximate FPS so the
speed-up over the ``.pt`` model is visible before running the demo.
"""

from __future__ import annotations

import argparse
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
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed; run: pip install ultralytics", file=sys.stderr)
        return 1

    print(f"Exporting {weights} to OpenVINO (imgsz={args.imgsz}) ...")
    try:
        exported = Path(YOLO(str(weights)).export(format="openvino", imgsz=args.imgsz))
    except ImportError:
        print(
            "OpenVINO is not installed; run: pip install openvino — then re-run this script.",
            file=sys.stderr,
        )
        return 1
    print(f"Exported model folder: {exported}")

    if not args.skip_check:
        fps = self_check(exported, args.imgsz)
        if fps is None:
            return 1
        print(f"Self-check OK — blank-frame inference ≈ {fps:.1f} FPS on this machine.")

    print()
    print("Run the demo with the exported model:")
    print(
        "  python scripts/inference/run_weapon_demo.py "
        f"--weights {exported} --imgsz {args.imgsz} --source demo.mp4"
    )
    print("(On recent ultralytics versions, --device intel:gpu selects the Intel iGPU explicitly.)")
    return 0


def self_check(exported: Path, imgsz: int) -> float | None:
    """Load the exported model and time blank-frame inference."""
    try:
        import numpy as np
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"Self-check could not run ({exc}); the export itself is done.", file=sys.stderr)
        return None
    try:
        model = YOLO(str(exported))
        frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
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
    return runs / elapsed if elapsed > 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
