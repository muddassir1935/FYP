#!/usr/bin/env python3
"""
SENTINEL — Bi-Lateral Skeletal Joint Mirroring & Jitter Augmentation Pipeline
=============================================================================
Scales action recognition temporal sequences from base extractions to 2,500+
balanced sequences (~600 to 750 per class) across:
  0: Normal Activity
  1: Fighting
  2: Robbery
  3: Shooting

Algorithms:
1. COCO 17-Keypoint Bi-Lateral Symmetry Logic:
   - Invert horizontal axis: x_flipped = 1.0 - x_orig (for detected joints)
   - Symmetrical Joint Swapping (Left Eye <-> Right Eye, Left Shoulder <-> Right Shoulder, etc.)
   - Anatomically doubles sequence count (right-handed attack -> left-handed attack)
2. Gaussian Coordinate Jitter:
   - Inject micro-noise N(0, 0.005) to non-zero coordinates to build invariance
     against camera shake and perspective variation
3. Output Serialization:
   - Saves augmented arrays with '_flipped.npy', '_jitter.npy', and '_flipped_jitter.npy' suffixes

Author: SENTINEL Project Team (Qaim Mehdi / Muddassir Rizwan / Faiz Hussain)
FAST-NUCES Karachi Campus
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ==============================================================================
# Configuration & Paths (Path Independence)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
CURATED_LSTM_DIR = BASE_DIR / "DataSets" / "curated" / "behavior_lstm"

CLASS_NAMES = ["normal", "fighting", "robbery", "shooting"]

# COCO 17 Keypoint Left/Right Swap Pairs
# Index 0 (Nose) remains index 0
SWAP_PAIRS: List[Tuple[int, int]] = [
    (1, 2),    # Left Eye <-> Right Eye
    (3, 4),    # Left Ear <-> Right Ear
    (5, 6),    # Left Shoulder <-> Right Shoulder
    (7, 8),    # Left Elbow <-> Right Elbow
    (9, 10),   # Left Wrist <-> Right Wrist
    (11, 12),  # Left Hip <-> Right Hip
    (13, 14),  # Left Knee <-> Right Knee
    (15, 16),  # Left Ankle <-> Right Ankle
]

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PoseAugmentor")


# ==============================================================================
# Bi-Lateral Mirroring & Jitter Logic
# ==============================================================================
def mirror_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Apply bi-lateral joint symmetry mirroring to (30, 51) skeletal sequence.
    Horizontal coordinates are inverted (x = 1.0 - x) and left/right joints are swapped.
    Non-detected coordinates (zeros) are strictly preserved.
    """
    assert seq.shape == (30, 51), f"Expected (30, 51), got {seq.shape}"
    flipped = np.zeros_like(seq)

    for t in range(seq.shape[0]):
        # 1. Nose (index 0) — Keep joint index, invert x
        x0, y0, c0 = seq[t, 0], seq[t, 1], seq[t, 2]
        if c0 > 0.0 or x0 > 0.0:
            flipped[t, 0] = float(np.clip(1.0 - x0, 0.0, 1.0))
            flipped[t, 1] = y0
            flipped[t, 2] = c0

        # 2. Symmetrical anatomical pairs
        for left_idx, right_idx in SWAP_PAIRS:
            # Left receives inverted Right
            rx, ry, rc = seq[t, right_idx * 3], seq[t, right_idx * 3 + 1], seq[t, right_idx * 3 + 2]
            if rc > 0.0 or rx > 0.0:
                flipped[t, left_idx * 3] = float(np.clip(1.0 - rx, 0.0, 1.0))
                flipped[t, left_idx * 3 + 1] = ry
                flipped[t, left_idx * 3 + 2] = rc

            # Right receives inverted Left
            lx, ly, lc = seq[t, left_idx * 3], seq[t, left_idx * 3 + 1], seq[t, left_idx * 3 + 2]
            if lc > 0.0 or lx > 0.0:
                flipped[t, right_idx * 3] = float(np.clip(1.0 - lx, 0.0, 1.0))
                flipped[t, right_idx * 3 + 1] = ly
                flipped[t, right_idx * 3 + 2] = lc

    return flipped


def jitter_sequence(seq: np.ndarray, std: float = 0.005) -> np.ndarray:
    """
    Inject Gaussian coordinate micro-noise N(0, std) into detected (conf > 0.05) keypoints.
    Clamps coordinates to [0.0, 1.0] and preserves zeroes for absent keypoints.
    """
    assert seq.shape == (30, 51), f"Expected (30, 51), got {seq.shape}"
    jittered = seq.copy()

    for t in range(seq.shape[0]):
        for j in range(17):
            conf = jittered[t, j * 3 + 2]
            if conf > 0.05:
                noise_x = float(np.random.normal(0.0, std))
                noise_y = float(np.random.normal(0.0, std))
                jittered[t, j * 3] = float(np.clip(jittered[t, j * 3] + noise_x, 0.0, 1.0))
                jittered[t, j * 3 + 1] = float(np.clip(jittered[t, j * 3 + 1] + noise_y, 0.0, 1.0))

    return jittered


# ==============================================================================
# Augmentation Execution & Serialization
# ==============================================================================
def augment_directory(
    class_dir: Path,
    target_count: int = 700,
    seed: int = 42,
) -> Dict[str, int]:
    """
    Augment all base sequences in a specific split/class directory to meet target count.
    Generates:
      - _flipped.npy
      - _jitter.npy
      - _flipped_jitter.npy (if needed to reach target)
    """
    np.random.seed(seed)

    # Collect only base sequences (ignore previously generated augmented files)
    base_files = [
        f for f in class_dir.glob("*.npy")
        if not f.stem.endswith("_flipped") and not f.stem.endswith("_jitter")
    ]

    stats = {
        "base": len(base_files),
        "flipped": 0,
        "jitter": 0,
        "flipped_jitter": 0,
        "total": len(base_files),
    }

    if not base_files:
        logger.warning("No base sequences found in %s", class_dir)
        return stats

    # 1. Generate Bi-Lateral Mirroring for 100% of base sequences
    for f in base_files:
        seq = np.load(str(f))
        flipped = mirror_sequence(seq)
        out_path = class_dir / f"{f.stem}_flipped.npy"
        np.save(str(out_path), flipped)
        stats["flipped"] += 1
        stats["total"] += 1

    # 2. Generate Gaussian Coordinate Jitter for 100% of base sequences
    for f in base_files:
        seq = np.load(str(f))
        jittered = jitter_sequence(seq, std=0.005)
        out_path = class_dir / f"{f.stem}_jitter.npy"
        np.save(str(out_path), jittered)
        stats["jitter"] += 1
        stats["total"] += 1

    # 3. If still below target count, generate flipped-jitter or second jitter
    deficit = max(0, target_count - stats["total"])
    if deficit > 0:
        step = max(1, len(base_files) // deficit) if deficit < len(base_files) else 1
        selected = base_files[::step][:deficit]
        for f in selected:
            seq = np.load(str(f))
            flipped = mirror_sequence(seq)
            flipped_jittered = jitter_sequence(flipped, std=0.005)
            out_path = class_dir / f"{f.stem}_flipped_jitter.npy"
            np.save(str(out_path), flipped_jittered)
            stats["flipped_jitter"] += 1
            stats["total"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Bi-Lateral Skeletal Joint Augmentation")
    parser.add_argument(
        "--target_per_class",
        type=int,
        default=700,
        help="Target total sequences per class across Train/Val/Test (default: 700, yields ~2800 total)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SENTINEL — Bi-Lateral Skeletal Joint Mirroring & Jitter Pipeline")
    logger.info("=" * 70)
    logger.info("Curated LSTM Root  : %s", CURATED_LSTM_DIR)
    logger.info("Target / Class     : %d sequences", args.target_per_class)

    # Proportional split targets: Train 70%, Val 20%, Test 10%
    split_ratios = {"train": 0.70, "val": 0.20, "test": 0.10}

    overall_stats: Dict[str, Dict[str, int]] = {
        s: {c: 0 for c in CLASS_NAMES} for s in ("train", "val", "test")
    }

    nan_check_count = 0
    total_augmented = 0

    for split_name in ("train", "val", "test"):
        split_target = int(round(args.target_per_class * split_ratios[split_name]))
        for class_name in CLASS_NAMES:
            class_dir = CURATED_LSTM_DIR / split_name / class_name
            if not class_dir.exists():
                logger.warning("Directory does not exist: %s", class_dir)
                continue

            dir_stats = augment_directory(class_dir, target_count=split_target)
            overall_stats[split_name][class_name] = dir_stats["total"]
            total_augmented += dir_stats["total"]

            # Sanity audit all files in class_dir
            for npy_file in class_dir.glob("*.npy"):
                arr = np.load(str(npy_file))
                if arr.shape != (30, 51) or not np.isfinite(arr).all():
                    nan_check_count += 1
                    logger.error("Corrupted array detected: %s", npy_file)

    logger.info("=" * 70)
    logger.info("BI-LATERAL AUGMENTATION COMPLETE — AUDIT SUMMARY:")
    logger.info("=" * 70)
    for s_name in ("train", "val", "test"):
        counts = overall_stats[s_name]
        s_total = sum(counts.values())
        logger.info(
            "[%s] Normal: %d | Fighting: %d | Robbery: %d | Shooting: %d | Total: %d",
            s_name.upper(),
            counts["normal"],
            counts["fighting"],
            counts["robbery"],
            counts["shooting"],
            s_total,
        )
    logger.info("-" * 70)
    logger.info("Grand Total Sequences Extracted & Augmented : %d", total_augmented)
    logger.info("Integrity Verification (NaN/Inf Errors)    : %d", nan_check_count)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
