#!/usr/bin/env python3
"""
SENTINEL — Weapon Detection Preprocessing Pipeline (YOLOv8s)
============================================================
Harmonizes heterogeneous Kaggle weapon datasets into an Ultralytics YOLO
2-class format (Gun: 0, Knife: 1) with ~15% negative background sampling,
bounding box normalization/validation, and anti-leakage splitting.

Author: SENTINEL Project Team (Qaim Mehdi / Muddassir Rizwan / Faiz Hussain)
FAST-NUCES Karachi Campus
"""

import argparse
import logging
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ==============================================================================
# Configuration & Paths (Path Independence)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "DataSets" / "raw" / "weapons"
RAW_ALT_DIR = BASE_DIR / "raw_datasets"
CURATED_DIR = BASE_DIR / "DataSets" / "curated" / "weapon_yolo"
QUARANTINE_LOG = BASE_DIR / "DataSets" / "raw" / "quarantine_log.txt"

# Canonical 2-Class Schema
CLASS_MAPPING: Dict[str, int] = {
    # Class 0: Gun
    "gun": 0,
    "pistol": 0,
    "handgun": 0,
    "rifle": 0,
    "automatic rifle": 0,
    "shotgun": 0,
    "firearm": 0,
    "revolver": 0,
    "smg": 0,
    "sniper": 0,
    "weapon": 0,
    # Class 1: Knife
    "knife": 1,
    "knives": 1,
    "dagger": 1,
    "blade": 1,
    "machete": 1,
    "sword": 1,
}

OUT_OF_SCOPE = {
    "bazooka",
    "grenade",
    "grenade launcher",
    "baton",
    "bomb",
    "military",
    "tank",
    "missile",
}

# Metadata.csv target mapping for weapon_source_A
SOURCE_A_TARGETS: Dict[int, Optional[int]] = {
    0: 0,  # Automatic Rifle -> Gun (0)
    1: None,  # Bazooka -> Out of scope (discard)
    2: None,  # Grenade Launcher -> Out of scope (discard)
    3: 0,  # Handgun -> Gun (0)
    4: 1,  # Knife -> Knife (1)
    5: 0,  # Shotgun -> Gun (0)
    6: 0,  # SMG -> Gun (0)
    7: 0,  # Sniper -> Gun (0)
    8: 1,  # Sword -> Knife (1)
}

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WeaponPreprocessor")


def log_quarantine(file_path: Path, reason: str) -> None:
    """Log corrupted or discarded media into the quarantine log."""
    QUARANTINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(QUARANTINE_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [WEAPON_CORRUPT] {file_path} — Reason: {reason}\n")
    logger.warning("Quarantined: %s (Reason: %s)", file_path.name, reason)


def validate_image(image_path: Path) -> Optional[Tuple[int, int]]:
    """Validate image file integrity and return (width, height)."""
    if not image_path.exists() or image_path.stat().st_size == 0:
        log_quarantine(image_path, "0-byte or non-existent image file")
        return None
    try:
        img = cv2.imread(str(image_path))
        if img is None or img.size == 0:
            log_quarantine(image_path, "OpenCV failed to decode image")
            return None
        h, w = img.shape[:2]
        if w <= 0 or h <= 0:
            log_quarantine(image_path, f"Invalid image dimensions: {w}x{h}")
            return None
        return (w, h)
    except Exception as exc:
        log_quarantine(image_path, f"Exception during image read: {exc}")
        return None


def validate_and_harmonize_bbox(
    line: str,
    target_class_id: int,
    image_dims: Tuple[int, int],
) -> Optional[str]:
    """
    Validate and clamp bounding box coordinates to [0.0, 1.0].
    Returns harmonized line: '<target_class_id> <x> <y> <w> <h>' or None.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        # Extract coordinates
        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    except ValueError:
        return None

    # Discard degenerate boxes
    if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
        return None

    # Check bounds with minor tolerance
    if x < -0.05 or x > 1.05 or y < -0.05 or y > 1.05:
        return None

    # Clamp strictly to [0.0, 1.0] interval
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    w = float(np.clip(w, 0.001, 1.0))
    h = float(np.clip(h, 0.001, 1.0))

    # Ensure box doesn't exceed image boundary
    x_min = max(0.0, x - w / 2.0)
    x_max = min(1.0, x + w / 2.0)
    y_min = max(0.0, y - h / 2.0)
    y_max = min(1.0, y + h / 2.0)

    # Recalculate clamped center and dimensions
    w = x_max - x_min
    h = y_max - y_min
    x = x_min + w / 2.0
    y = y_min + h / 2.0

    if w <= 0.0 or h <= 0.0:
        return None

    return f"{target_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def collect_source_a_samples() -> List[Dict]:
    """Collect samples from weapon_source_A using metadata.csv."""
    source_dir = RAW_DIR / "source_A"
    if not source_dir.exists():
        source_dir = RAW_ALT_DIR / "weapon_source_A"
    if not source_dir.exists():
        logger.warning("Source A not found at %s", source_dir)
        return []

    metadata_path = source_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.warning("metadata.csv not found at %s", metadata_path)
        return []

    import csv

    samples: List[Dict] = []

    # Build index of all files in source_dir for fast lookup
    file_map: Dict[str, Path] = {}
    for p in source_dir.rglob("*"):
        if p.is_file():
            file_map[p.name] = p

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_name = str(row["imagefile"])
            lbl_name = str(row["labelfile"])
            target = int(row["target"])

            # Check target class
            target_class = SOURCE_A_TARGETS.get(target)
            if target_class is None:
                continue  # Discard out-of-scope classes (Bazooka, Grenade Launcher)

            img_path = file_map.get(img_name)
            lbl_path = file_map.get(lbl_name)

            if not img_path or not lbl_path:
                continue

            samples.append(
                {
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "target_class": target_class,
                    "is_negative": False,
                    "source": "source_A",
                }
            )

    logger.info("Source A: Collected %d valid weapon samples", len(samples))
    return samples


def collect_source_b_samples(
    max_positives: Optional[int] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Collect positive and negative samples from weapon_source_B.
    Returns: (positive_samples, negative_samples)
    """
    source_dir = RAW_DIR / "source_B"
    if not source_dir.exists():
        source_dir = RAW_ALT_DIR / "weapon_source_B"
    if not source_dir.exists():
        logger.warning("Source B not found at %s", source_dir)
        return [], []

    # Find all image files
    img_files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(source_dir.rglob(ext))

    positives: List[Dict] = []
    negatives: List[Dict] = []

    for img_path in img_files:
        lbl_path = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        is_neg = img_path.name.startswith("neg_") or lbl_path.stat().st_size == 0

        if is_neg:
            negatives.append(
                {
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "target_class": -1,
                    "is_negative": True,
                    "source": "source_B",
                }
            )
        else:
            # Default class 0: Gun
            positives.append(
                {
                    "image_path": img_path,
                    "label_path": lbl_path,
                    "target_class": 0,
                    "is_negative": False,
                    "source": "source_B",
                }
            )

    logger.info(
        "Source B: Found %d positives and %d negatives",
        len(positives),
        len(negatives),
    )
    if max_positives and len(positives) > max_positives:
        random.seed(42)
        positives = random.sample(positives, max_positives)
        logger.info("Subsampled Source B positives to %d", len(positives))

    return positives, negatives


def process_and_copy_sample(
    sample: Dict,
    dest_img_path: Path,
    dest_lbl_path: Path,
) -> bool:
    """Validate, normalize bboxes, copy image and write standardized label file."""
    img_path = sample["image_path"]
    lbl_path = sample["label_path"]
    target_class = sample["target_class"]
    is_neg = sample["is_negative"]

    # Validate image
    dims = validate_image(img_path)
    if dims is None:
        return False

    if is_neg:
        # Standard empty YOLO annotation for negative/background images
        shutil.copy2(img_path, dest_img_path)
        dest_lbl_path.write_text("", encoding="utf-8")
        return True

    # Validate bounding boxes
    if not lbl_path.exists():
        log_quarantine(img_path, "Missing label file for positive sample")
        return False

    try:
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as exc:
        log_quarantine(lbl_path, f"Cannot read label file: {exc}")
        return False

    valid_lines: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        valid_bbox = validate_and_harmonize_bbox(line, target_class, dims)
        if valid_bbox:
            valid_lines.append(valid_bbox)

    if not valid_lines:
        log_quarantine(img_path, "No valid bounding boxes after coordinate normalization")
        return False

    # Copy image and write validated label
    shutil.copy2(img_path, dest_img_path)
    dest_lbl_path.write_text("\n".join(valid_lines) + "\n", encoding="utf-8")
    return True


def split_and_write_dataset(
    all_samples: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    """Partition dataset into Train/Val/Test splits with stratified class balance."""
    random.seed(seed)
    np.random.seed(seed)

    # Separate by class for stratified splitting
    guns = [s for s in all_samples if not s["is_negative"] and s["target_class"] == 0]
    knives = [s for s in all_samples if not s["is_negative"] and s["target_class"] == 1]
    negatives = [s for s in all_samples if s["is_negative"]]

    random.shuffle(guns)
    random.shuffle(knives)
    random.shuffle(negatives)

    splits = {"train": [], "val": [], "test": []}

    for group in (guns, knives, negatives):
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])

    # Create destination directories
    for split_name in ("train", "val", "test"):
        (CURATED_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (CURATED_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    stats = {
        "train": {"Gun": 0, "Knife": 0, "Background": 0},
        "val": {"Gun": 0, "Knife": 0, "Background": 0},
        "test": {"Gun": 0, "Knife": 0, "Background": 0},
    }

    for split_name, sample_list in splits.items():
        logger.info("Processing '%s' split (%d samples)...", split_name, len(sample_list))
        for idx, sample in enumerate(sample_list):
            prefix = sample["source"]
            stem = f"{prefix}_{sample['image_path'].stem}_{idx:05d}"
            ext = sample["image_path"].suffix.lower()
            dest_img = CURATED_DIR / "images" / split_name / f"{stem}{ext}"
            dest_lbl = CURATED_DIR / "labels" / split_name / f"{stem}.txt"

            success = process_and_copy_sample(sample, dest_img, dest_lbl)
            if success:
                if sample["is_negative"]:
                    stats[split_name]["Background"] += 1
                elif sample["target_class"] == 0:
                    stats[split_name]["Gun"] += 1
                elif sample["target_class"] == 1:
                    stats[split_name]["Knife"] += 1

    return stats


def generate_dataset_yaml() -> Path:
    """Generate Ultralytics YOLOv8 dataset configuration YAML."""
    yaml_path = CURATED_DIR / "dataset.yaml"
    config = {
        "path": str(CURATED_DIR.as_posix()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 2,
        "names": {
            0: "Gun",
            1: "Knife",
        },
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info("Generated YOLO configuration at: %s", yaml_path)
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Weapon Preprocessing Pipeline")
    parser.add_argument(
        "--max_positives",
        type=int,
        default=4000,
        help="Maximum positive weapon samples to incorporate from Source B (ensures balance and fast training)",
    )
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=0.15,
        help="Target ratio of negative/background samples (~15% false-alarm mitigation)",
    )
    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("SENTINEL — Weapon Detection Preprocessing Pipeline (YOLOv8s)")
    logger.info("=" * 65)
    logger.info("Workspace Root : %s", BASE_DIR)
    logger.info("Raw Input Dir  : %s", RAW_DIR)
    logger.info("Curated Output : %s", CURATED_DIR)

    # 1. Collect from Source A (Guns + Knives)
    source_a_samples = collect_source_a_samples()

    # 2. Collect from Source B (Guns + Negatives)
    source_b_positives, source_b_negatives = collect_source_b_samples(
        max_positives=args.max_positives
    )

    all_positives = source_a_samples + source_b_positives
    logger.info("Total positive samples: %d", len(all_positives))

    # Calculate required negative background samples (~15% of total dataset)
    # N / (P + N) = 0.15 => N = P * 0.15 / (1 - 0.15)
    target_neg_count = int(len(all_positives) * args.neg_ratio / (1.0 - args.neg_ratio))
    target_neg_count = min(target_neg_count, len(source_b_negatives))

    random.seed(42)
    selected_negatives = random.sample(source_b_negatives, target_neg_count)
    logger.info("Selected %d negative background samples (~%.1f%% of dataset)",
                len(selected_negatives),
                100.0 * len(selected_negatives) / (len(all_positives) + len(selected_negatives)))

    all_samples = all_positives + selected_negatives
    logger.info("Total combined samples to process: %d", len(all_samples))

    # 3. Clean existing curated output dir if present
    if CURATED_DIR.exists():
        logger.info("Cleaning existing curated directory: %s", CURATED_DIR)
        shutil.rmtree(CURATED_DIR)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    # 4. Stratified Split and Write
    stats = split_and_write_dataset(all_samples, seed=42)

    # 5. Generate dataset.yaml
    yaml_path = generate_dataset_yaml()

    # 6. Report Summary
    logger.info("=" * 65)
    logger.info("PREPROCESSING PIPELINE COMPLETE — SUMMARY REPORT:")
    logger.info("=" * 65)
    total_samples = 0
    for split_name, counts in stats.items():
        split_total = sum(counts.values())
        total_samples += split_total
        logger.info(
            "[%s] Gun: %d | Knife: %d | Background: %d | Total: %d",
            split_name.upper(),
            counts["Gun"],
            counts["Knife"],
            counts["Background"],
            split_total,
        )
    logger.info("-" * 65)
    logger.info("Total Curated Dataset Size : %d samples", total_samples)
    logger.info("Dataset YAML Configuration  : %s", yaml_path)
    logger.info("Quarantine Log File         : %s", QUARANTINE_LOG)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
