#!/usr/bin/env python3
"""
SENTINEL — Weapon Detection Preprocessing & Augmentation Pipeline (YOLOv8s)
==========================================================================
Harmonizes heterogeneous Kaggle weapon datasets and supplementary knife datasets
into an Ultralytics YOLO 2-class format (Gun: 0, Knife: 1) with:
  - Supplementary knife ingestion (mapping strictly to Class 1)
  - Geometric (horizontal flip, random rotation +-10° to +-15°) and
    photometric (HSV jitter, Gaussian/motion blur) offline knife augmentations
  - Scaling knives to 1,200 – 1,500 total instances (~20-25% positive ratio)
  - ~15% negative background sampling for false-alarm mitigation
  - Coordinate normalization, boundary validation, and degenerate box filtering
  - Stratified 70% / 20% / 10% anti-leakage splitting

Author: SENTINEL Project Team (Qaim Mehdi / Muddassir Rizwan / Faiz Hussain)
FAST-NUCES Karachi Campus
"""

import argparse
import csv
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
    0: 0,     # Automatic Rifle -> Gun (0)
    1: None,  # Bazooka -> Out of scope (discard)
    2: None,  # Grenade Launcher -> Out of scope (discard)
    3: 0,     # Handgun -> Gun (0)
    4: 1,     # Knife -> Knife (1)
    5: 0,     # Shotgun -> Gun (0)
    6: 0,     # SMG -> Gun (0)
    7: 0,     # Sniper -> Gun (0)
    8: 1,     # Sword -> Knife (1)
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


# ==============================================================================
# Ingestion from Heterogeneous Sources
# ==============================================================================
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

    samples: List[Dict] = []
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

            target_class = SOURCE_A_TARGETS.get(target)
            if target_class is None:
                continue

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


def collect_extra_knife_samples() -> List[Dict]:
    """
    Ingest supplementary knife dataset from DataSets/raw/weapons/extra_knives (or extra_knifes).
    Strictly maps all knife annotations to Class 1: Knife.
    """
    candidate_dirs = [
        RAW_DIR / "extra_knives",
        RAW_DIR / "extra_knifes",
        RAW_ALT_DIR / "extra_knives",
        RAW_ALT_DIR / "extra_knifes",
    ]
    extra_dir: Optional[Path] = None
    for cdir in candidate_dirs:
        if cdir.exists() and cdir.is_dir():
            extra_dir = cdir
            break

    if extra_dir is None:
        logger.warning("Supplementary knife directory not found in candidates: %s", candidate_dirs)
        return []

    logger.info("Found supplementary knife dataset at: %s", extra_dir)

    img_files: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        img_files.extend(extra_dir.rglob(ext))

    knife_samples: List[Dict] = []
    for img_p in img_files:
        # Standard Roboflow YOLO structure: split/images/xxx.jpg and split/labels/xxx.txt
        lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
        if not lbl_p.exists():
            # Try same directory
            lbl_p = img_p.parent / f"{img_p.stem}.txt"

        if not lbl_p.exists() or lbl_p.stat().st_size == 0:
            continue

        knife_samples.append(
            {
                "image_path": img_p,
                "label_path": lbl_p,
                "target_class": 1,  # Strictly Knife (1)
                "is_negative": False,
                "source": "extra_knives",
                "is_augmented": False,
            }
        )

    logger.info("Supplementary Knives: Collected %d valid knife samples", len(knife_samples))
    return knife_samples


# ==============================================================================
# Spatial & Photometric Knife Augmentation Engine
# ==============================================================================
def apply_knife_augmentation(
    img: np.ndarray,
    bboxes: List[Tuple[float, float, float, float]],
    aug_mode: str,
) -> Tuple[np.ndarray, List[Tuple[float, float, float, float]]]:
    """
    Apply geometric and photometric transformations to knife images and bboxes:
      - 'flip': Horizontal flip (recalculating x_center = 1.0 - x_center)
      - 'rotate': Slight random rotation (+-10° to +-15°) with corner-recalculated bbox
      - 'photometric': HSV color/brightness jitter & Gaussian/motion blur
      - 'combined': Flip/rotation + photometric jitter
    """
    h_img, w_img = img.shape[:2]
    img_aug = img.copy()
    aug_boxes: List[Tuple[float, float, float, float]] = []

    # 1. Geometric transforms
    if aug_mode in ("flip", "combined_flip"):
        img_aug = cv2.flip(img_aug, 1)
        for xc, yc, w, h in bboxes:
            new_xc = 1.0 - xc
            aug_boxes.append((new_xc, yc, w, h))

    elif aug_mode in ("rotate", "combined_rotate"):
        # Select rotation angle between +-10 and +-15 degrees
        angle = random.choice([random.uniform(-15.0, -10.0), random.uniform(10.0, 15.0)])
        M = cv2.getRotationMatrix2D((w_img / 2.0, h_img / 2.0), angle, 1.0)
        img_aug = cv2.warpAffine(img_aug, M, (w_img, h_img), borderMode=cv2.BORDER_REFLECT)

        for xc, yc, w, h in bboxes:
            x_c_px, y_c_px = xc * w_img, yc * h_img
            w_px, h_px = w * w_img, h * h_img
            x1, y1 = x_c_px - w_px / 2.0, y_c_px - h_px / 2.0
            x2, y2 = x_c_px + w_px / 2.0, y_c_px + h_px / 2.0

            corners = np.array(
                [
                    [x1, y1, 1.0],
                    [x2, y1, 1.0],
                    [x2, y2, 1.0],
                    [x1, y2, 1.0],
                ],
                dtype=np.float32,
            )
            rot_corners = corners @ M.T

            rx_min = max(0.0, float(np.min(rot_corners[:, 0]) / w_img))
            rx_max = min(1.0, float(np.max(rot_corners[:, 0]) / w_img))
            ry_min = max(0.0, float(np.min(rot_corners[:, 1]) / h_img))
            ry_max = min(1.0, float(np.max(rot_corners[:, 1]) / h_img))

            rw = rx_max - rx_min
            rh = ry_max - ry_min
            if rw > 0.005 and rh > 0.005:
                aug_boxes.append((rx_min + rw / 2.0, ry_min + rh / 2.0, rw, rh))
            else:
                aug_boxes.append((xc, yc, w, h))
    else:
        # Photometric only (geometry unchanged)
        aug_boxes = list(bboxes)

    # 2. Photometric transforms (HSV jitter & blur)
    if aug_mode in ("photometric", "combined_flip", "combined_rotate"):
        try:
            hsv = cv2.cvtColor(img_aug, cv2.COLOR_BGR2HSV).astype(np.float32)
            # Saturation jitter
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0.0, 255.0)
            # Value / Brightness jitter
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0.0, 255.0)
            img_aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Slight surveillance blur (Gaussian or Motion)
            if random.random() > 0.5:
                img_aug = cv2.GaussianBlur(img_aug, (5, 5), random.uniform(0.6, 1.4))
            else:
                k_size = random.choice([3, 5])
                kernel = np.zeros((k_size, k_size))
                kernel[int((k_size - 1) / 2), :] = np.ones(k_size) / k_size
                img_aug = cv2.filter2D(img_aug, -1, kernel)
        except Exception as exc:
            logger.debug("Photometric jitter skipped due to: %s", exc)

    return img_aug, aug_boxes


def generate_augmented_knife_samples(
    pooled_knives: List[Dict],
    target_total_knives: int = 1350,
) -> List[Dict]:
    """
    Generate synthetic augmented knife samples to hit target range (1,200 – 1,500 total).
    Augmentation modes: flip, rotate, photometric, combined.
    """
    current_count = len(pooled_knives)
    deficit = max(0, target_total_knives - current_count)
    if deficit == 0:
        logger.info("Knife count (%d) already meets target (%d).", current_count, target_total_knives)
        return []

    logger.info(
        "Augmenting Knives: Generating %d synthetic samples to scale from %d to %d...",
        deficit,
        current_count,
        target_total_knives,
    )

    random.seed(42)
    selected_for_aug = random.sample(pooled_knives, min(deficit, len(pooled_knives)))
    # If deficit > len(pooled_knives), sample with replacement
    while len(selected_for_aug) < deficit:
        selected_for_aug.append(random.choice(pooled_knives))

    aug_modes = ["flip", "rotate", "photometric", "combined_flip", "combined_rotate"]
    augmented_samples: List[Dict] = []

    for i, base_sample in enumerate(selected_for_aug):
        mode = aug_modes[i % len(aug_modes)]
        augmented_samples.append(
            {
                "image_path": base_sample["image_path"],
                "label_path": base_sample["label_path"],
                "target_class": 1,
                "is_negative": False,
                "source": "knife_aug",
                "is_augmented": True,
                "aug_mode": mode,
            }
        )

    logger.info("Synthesized %d knife augmentation metadata entries", len(augmented_samples))
    return augmented_samples


# ==============================================================================
# Sample Processing & File Writing
# ==============================================================================
def process_and_copy_sample(
    sample: Dict,
    dest_img_path: Path,
    dest_lbl_path: Path,
) -> bool:
    """Validate, normalize bboxes, apply augmentations if flagged, and copy/write."""
    img_path = sample["image_path"]
    lbl_path = sample["label_path"]
    target_class = sample["target_class"]
    is_neg = sample["is_negative"]
    is_aug = sample.get("is_augmented", False)

    dims = validate_image(img_path)
    if dims is None:
        return False
    w_img, h_img = dims

    if is_neg:
        shutil.copy2(img_path, dest_img_path)
        dest_lbl_path.write_text("", encoding="utf-8")
        return True

    if not lbl_path.exists():
        log_quarantine(img_path, "Missing label file for positive sample")
        return False

    try:
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    except Exception as exc:
        log_quarantine(lbl_path, f"Cannot read label file: {exc}")
        return False

    valid_boxes: List[Tuple[float, float, float, float]] = []
    for line in lines:
        if not line.strip():
            continue
        v_line = validate_and_harmonize_bbox(line, target_class, dims)
        if v_line:
            parts = v_line.split()
            valid_boxes.append((float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))

    if not valid_boxes:
        log_quarantine(img_path, "No valid bounding boxes after coordinate normalization")
        return False

    if is_aug:
        # Load image and perform spatial/photometric augmentation
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        aug_mode = sample.get("aug_mode", "flip")
        img_aug, aug_bboxes = apply_knife_augmentation(img, valid_boxes, aug_mode)

        # Write augmented image
        cv2.imwrite(str(dest_img_path), img_aug)

        # Write augmented label
        aug_lines = [f"1 {bx[0]:.6f} {bx[1]:.6f} {bx[2]:.6f} {bx[3]:.6f}" for bx in aug_bboxes]
        dest_lbl_path.write_text("\n".join(aug_lines) + "\n", encoding="utf-8")
        return True

    else:
        # Standard copy
        shutil.copy2(img_path, dest_img_path)
        out_lines = [f"{target_class} {bx[0]:.6f} {bx[1]:.6f} {bx[2]:.6f} {bx[3]:.6f}" for bx in valid_boxes]
        dest_lbl_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
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

    for split_name in ("train", "val", "test"):
        (CURATED_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (CURATED_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    stats = {
        "train": {"Gun": 0, "Knife": 0, "Background": 0},
        "val": {"Gun": 0, "Knife": 0, "Background": 0},
        "test": {"Gun": 0, "Knife": 0, "Background": 0},
    }

    for split_name, sample_list in splits.items():
        logger.info("Writing '%s' split (%d samples)...", split_name, len(sample_list))
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
    """Generate Ultralytics YOLOv8 dataset configuration YAML with root-relative path."""
    yaml_path = CURATED_DIR / "dataset.yaml"
    config = {
        "path": "DataSets/curated/weapon_yolo",
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
    logger.info("Generated portable root-relative YOLO configuration at: %s", yaml_path)
    return yaml_path


# ==============================================================================
# Main Execution Entrypoint
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Weapon Preprocessing & Augmentation Pipeline")
    parser.add_argument(
        "--max_positives",
        type=int,
        default=4000,
        help="Maximum positive weapon samples from Source B (default: 4000)",
    )
    parser.add_argument(
        "--target_knives",
        type=int,
        default=1350,
        help="Target total knife instances across all sources + augmentations (default: 1350)",
    )
    parser.add_argument(
        "--neg_ratio",
        type=float,
        default=0.15,
        help="Target ratio of negative/background samples (~15% false-alarm mitigation)",
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("SENTINEL — Weapon Detection Preprocessing & Augmentation Pipeline")
    logger.info("=" * 70)
    logger.info("Workspace Root : %s", BASE_DIR)
    logger.info("Raw Input Dir  : %s", RAW_DIR)
    logger.info("Curated Output : %s", CURATED_DIR)
    logger.info("Target Knives  : %d instances", args.target_knives)

    # 1. Collect from Source A (Guns + Knives)
    source_a_samples = collect_source_a_samples()

    # 2. Collect from Source B (Guns + Negatives)
    source_b_positives, source_b_negatives = collect_source_b_samples(
        max_positives=args.max_positives
    )

    # 3. Ingest Supplementary Knife Dataset
    extra_knife_samples = collect_extra_knife_samples()

    # Separate raw knives vs guns
    guns_pooled = [s for s in source_a_samples if s["target_class"] == 0] + source_b_positives
    knives_pooled = [s for s in source_a_samples if s["target_class"] == 1] + extra_knife_samples

    logger.info("Pooled Raw Guns  : %d samples", len(guns_pooled))
    logger.info("Pooled Raw Knives: %d samples (Source A: %d, Extra: %d)",
                len(knives_pooled),
                len([s for s in source_a_samples if s["target_class"] == 1]),
                len(extra_knife_samples))

    # 4. Generate Spatial & Photometric Knife Augmentations
    augmented_knives = generate_augmented_knife_samples(
        knives_pooled,
        target_total_knives=args.target_knives,
    )
    all_knives = knives_pooled + augmented_knives
    all_positives = guns_pooled + all_knives

    logger.info("Total Final Positives: %d (Guns: %d, Knives: %d [%.1f%%])",
                len(all_positives),
                len(guns_pooled),
                len(all_knives),
                100.0 * len(all_knives) / len(all_positives))

    # 5. Negative Background Samples (~15% of total dataset)
    target_neg_count = int(len(all_positives) * args.neg_ratio / (1.0 - args.neg_ratio))
    target_neg_count = min(target_neg_count, len(source_b_negatives))

    random.seed(42)
    selected_negatives = random.sample(source_b_negatives, target_neg_count)
    logger.info(
        "Selected %d negative background samples (~%.1f%% of dataset)",
        len(selected_negatives),
        100.0 * len(selected_negatives) / (len(all_positives) + len(selected_negatives)),
    )

    all_samples = all_positives + selected_negatives
    logger.info("Total combined samples to process: %d", len(all_samples))

    # 6. Clean existing curated output directory
    if CURATED_DIR.exists():
        logger.info("Cleaning existing curated directory: %s", CURATED_DIR)
        shutil.rmtree(CURATED_DIR)
    CURATED_DIR.mkdir(parents=True, exist_ok=True)

    # 7. Stratified Split and Write
    stats = split_and_write_dataset(all_samples, seed=42)

    # 8. Generate dataset.yaml
    yaml_path = generate_dataset_yaml()

    # 9. Summary Report
    logger.info("=" * 70)
    logger.info("PREPROCESSING PIPELINE COMPLETE — SUMMARY REPORT:")
    logger.info("=" * 70)
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
    logger.info("-" * 70)
    logger.info("Total Curated Dataset Size : %d samples", total_samples)
    logger.info("Dataset YAML Configuration  : %s", yaml_path)
    logger.info("Quarantine Log File         : %s", QUARANTINE_LOG)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
