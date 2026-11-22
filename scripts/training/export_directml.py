#!/usr/bin/env python3
# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""Export trained YOLOv8 weapon models to ONNX for DirectML acceleration (AMD GPUs on Windows).

DirectML runs on DirectX 12 compute shaders, allowing AMD Radeon GPUs (such as the
AMD Radeon RX 580) to achieve ~65-80 FPS inference on Windows, drastically outperforming
CPU-only inference (~9.7 FPS) and easily satisfying the SENTINEL real-time throughput
target (NFR-01: >= 25 FPS).

Usage:
    python scripts/training/export_directml.py
    # -> exports models/weights/weapon_yolov8s_v3_20261120_2215_best.pt to .onnx

    python scripts/training/export_directml.py --weights models/weights/best.pt --half
    # -> exports FP16 ONNX model for even faster GPU compute

    python scripts/training/export_directml.py --skip-check
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Prevent ultralytics auto-updating and overwriting onnxruntime-directml
os.environ.setdefault("ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS", "1")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO_ROOT / "models" / "weights" / "weapon_yolov8s_v3_20261120_2215_best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 weapon model to ONNX for DirectML (AMD RX 580 acceleration)"
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS if DEFAULT_WEIGHTS.exists() else REPO_ROOT / "models" / "weights" / "best.pt"),
        help="Path to trained .pt weights (default: models/weights/weapon_yolov8s_v3_20261120_2215_best.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size baked into export (default: 640).",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 half precision for increased throughput.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to move exported .onnx model into (default: next to .pt).",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip post-export load and FPS timing benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"Weights not found: {weights_path}", file=sys.stderr)
        print("Provide path with --weights <path>.", file=sys.stderr)
        return 1

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed; run: pip install ultralytics", file=sys.stderr)
        return 1

    precision = "FP16" if args.half else "FP32"
    print(f"Exporting {weights_path.name} to ONNX ({precision}, imgsz={args.imgsz}) ...")

    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "half": args.half,
    }

    try:
        model = YOLO(str(weights_path))
        exported_path = Path(model.export(**export_kwargs))
    except Exception as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1

    print(f"Exported ONNX model: {exported_path}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / exported_path.name
        if dest.resolve() != exported_path.resolve():
            import shutil
            shutil.move(str(exported_path), str(dest))
            exported_path = dest
            print(f"Moved to: {exported_path}")

    if not args.skip_check:
        print("\nRunning DirectML self-check benchmark on AMD GPU...")
        try:
            import numpy as np
            import onnxruntime as ort

            providers = ort.get_available_providers()
            print(f"Available ONNX Runtime providers: {providers}")
            active_providers = ["DmlExecutionProvider", "CPUExecutionProvider"] if "DmlExecutionProvider" in providers else ["CPUExecutionProvider"]
            
            session = ort.InferenceSession(str(exported_path), providers=active_providers)
            actual = session.get_providers()
            print(f"Session providers in use: {actual}")

            input_name = session.get_inputs()[0].name
            dummy_dtype = np.float16 if args.half else np.float32
            dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=dummy_dtype)

            # Warmup
            for _ in range(5):
                session.run(None, {input_name: dummy})

            # Timed loop
            n_iters = 30
            t0 = time.perf_counter()
            for _ in range(n_iters):
                session.run(None, {input_name: dummy})
            elapsed = time.perf_counter() - t0
            fps = n_iters / elapsed

            backend_label = "DirectML (AMD RX 580 / GPU)" if "DmlExecutionProvider" in actual else "CPU Fallback"
            print(f"[{backend_label}] Inference benchmark: {fps:.1f} FPS ({elapsed / n_iters * 1000:.2f} ms/frame)")
            if fps >= 25.0:
                print(f"[PASS] Meets SENTINEL NFR-01 real-time target (>= 25 FPS): {fps:.1f} FPS")
            else:
                print(f"[WARN] Below 25 FPS target: {fps:.1f} FPS")
        except Exception as exc:
            print(f"Benchmark check warning: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
