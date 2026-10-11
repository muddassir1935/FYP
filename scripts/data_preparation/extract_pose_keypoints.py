#!/usr/bin/env python3
"""
SENTINEL — Action Recognition Offline Pose Extraction Pipeline
==============================================================
Extracts 17 skeletal keypoints (51-dim: X, Y, Conf) per person using
YOLOv8m-Pose, filters strictly to 4 behavioral classes:
  0: Normal Activity
  1: Fighting
  2: Robbery
  3: Shooting
Enforces anti-data-leakage video-level splitting and segments continuous
keypoint streams into fixed sliding windows of 30 consecutive frames (30, 51).

Author: SENTINEL Project Team (Qaim Mehdi / Muddassir Rizwan / Faiz Hussain)
FAST-NUCES Karachi Campus
"""

import argparse
import logging
import os
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# ==============================================================================
# Configuration & Paths (Path Independence)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_UCF_DIR = BASE_DIR / "DataSets" / "raw" / "ucf_crime"
RAW_RWF_DIR = BASE_DIR / "DataSets" / "raw" / "rwf_2000"
ALT_RAW_DIR = BASE_DIR / "raw_datasets"
CURATED_LSTM_DIR = BASE_DIR / "DataSets" / "curated" / "behavior_lstm"
WEIGHTS_DIR = BASE_DIR / "models" / "weights"
QUARANTINE_LOG = BASE_DIR / "DataSets" / "raw" / "quarantine_log.txt"

# Canonical 4-Class Architecture
CLASS_NAMES: Dict[int, str] = {
    0: "normal",
    1: "fighting",
    2: "robbery",
    3: "shooting",
}

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PoseExtractor")


def get_device() -> str:
    """Dynamically auto-detect computation device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def log_quarantine(file_path: Path, reason: str) -> None:
    """Log corrupted video streams or frames into quarantine log."""
    QUARANTINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(QUARANTINE_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] [VIDEO_CORRUPT] {file_path} — Reason: {reason}\n")
    logger.warning("Quarantined: %s (Reason: %s)", file_path.name, reason)


# ==============================================================================
# Video Dataset Discovery & Anti-Data-Leakage Splitting
# ==============================================================================
def discover_rwf2000_videos() -> Dict[str, List[Path]]:
    """
    Discover RWF-2000 video files.
    Maps:
      'Fight' -> 'fighting' (Class 1)
      'NonFight' -> 'normal' (Class 0)
    """
    rwf_dir = RAW_RWF_DIR if RAW_RWF_DIR.exists() else ALT_RAW_DIR / "rwf2000"
    categories = {"normal": [], "fighting": []}
    if not rwf_dir.exists():
        logger.warning("RWF-2000 directory not found at: %s", rwf_dir)
        return categories

    for p in rwf_dir.rglob("*.avi"):
        parent_name = p.parent.name.lower()
        if "nonfight" in parent_name:
            categories["normal"].append(p)
        elif "fight" in parent_name:
            categories["fighting"].append(p)

    logger.info("RWF-2000 discovered: %d Fight, %d NonFight videos",
                len(categories["fighting"]), len(categories["normal"]))
    return categories


def discover_ucf_crime_video_sequences() -> Dict[str, Dict[str, List[Path]]]:
    """
    Discover UCF-Crime pre-extracted frame sequences.
    Strictly filters down to: 'Fighting', 'Robbery', 'Shooting'.
    Groups frames by unique video identifier:
      {
         'fighting': { video_id: [frame_path_0, frame_path_10, ...] },
         'robbery':  { video_id: [...] },
         'shooting': { video_id: [...] }
      }
    """
    ucf_dir = RAW_UCF_DIR if RAW_UCF_DIR.exists() else ALT_RAW_DIR / "ucf_crime"
    categories = {"fighting": {}, "robbery": {}, "shooting": {}}
    if not ucf_dir.exists():
        logger.warning("UCF-Crime directory not found at: %s", ucf_dir)
        return categories

    target_map = {
        "fighting": "fighting",
        "robbery": "robbery",
        "shooting": "shooting",
    }

    # Directly check target category folders across splits (Train and Test)
    for split in ("Train", "Test", "train", "test"):
        split_dir = ucf_dir / split
        if not split_dir.is_dir():
            continue
        for cat_name, matched_class in (("Fighting", "fighting"), ("Robbery", "robbery"), ("Shooting", "shooting")):
            cat_folder = split_dir / cat_name
            if not cat_folder.is_dir():
                continue
            for frame_file in cat_folder.iterdir():
                if not frame_file.name.endswith(".png"):
                    continue
                # Extract video ID by removing the trailing frame number
                # e.g. "Fighting002_x264_10.png" -> "Fighting002_x264"
                parts = frame_file.stem.split("_")
                if len(parts) >= 2 and parts[-1].isdigit():
                    video_id = "_".join(parts[:-1])
                else:
                    video_id = frame_file.stem

                if video_id not in categories[matched_class]:
                    categories[matched_class][video_id] = []
                categories[matched_class][video_id].append(frame_file)

    # Sort each video sequence by frame index
    def parse_frame_idx(p: Path) -> int:
        match = re.search(r"(\d+)$", p.stem)
        return int(match.group(1)) if match else 0

    for c in categories:
        for vid, frames in categories[c].items():
            frames.sort(key=parse_frame_idx)

    for c, vids in categories.items():
        logger.info("UCF-Crime discovered: %d unique videos for '%s'", len(vids), c)

    return categories


def create_anti_leakage_splits(
    rwf_data: Dict[str, List[Path]],
    ucf_data: Dict[str, Dict[str, List[Path]]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Dict[str, List]]:
    """
    Partition video clips (NOT frames) across Train, Val, and Test splits.
    Ensures frames/windows from the same video never appear in multiple splits.
    """
    random.seed(seed)
    np.random.seed(seed)

    splits: Dict[str, Dict[str, List]] = {
        "train": {"normal": [], "fighting": [], "robbery": [], "shooting": []},
        "val": {"normal": [], "fighting": [], "robbery": [], "shooting": []},
        "test": {"normal": [], "fighting": [], "robbery": [], "shooting": []},
    }

    # Class 0: Normal Activity (from RWF-2000 NonFight)
    normal_vids = [("rwf_file", p) for p in rwf_data["normal"]]
    random.shuffle(normal_vids)
    n = len(normal_vids)
    splits["train"]["normal"] = normal_vids[: int(n * train_ratio)]
    splits["val"]["normal"] = normal_vids[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    splits["test"]["normal"] = normal_vids[int(n * (train_ratio + val_ratio)) :]

    # Class 1: Fighting (RWF-2000 Fight + UCF-Crime Fighting)
    fighting_vids = [("rwf_file", p) for p in rwf_data["fighting"]] + [
        ("ucf_seq", (vid, frames)) for vid, frames in ucf_data["fighting"].items()
    ]
    random.shuffle(fighting_vids)
    n = len(fighting_vids)
    splits["train"]["fighting"] = fighting_vids[: int(n * train_ratio)]
    splits["val"]["fighting"] = fighting_vids[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    splits["test"]["fighting"] = fighting_vids[int(n * (train_ratio + val_ratio)) :]

    # Class 2: Robbery (UCF-Crime Robbery)
    robbery_vids = [("ucf_seq", (vid, frames)) for vid, frames in ucf_data["robbery"].items()]
    random.shuffle(robbery_vids)
    n = len(robbery_vids)
    splits["train"]["robbery"] = robbery_vids[: int(n * train_ratio)]
    splits["val"]["robbery"] = robbery_vids[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    splits["test"]["robbery"] = robbery_vids[int(n * (train_ratio + val_ratio)) :]

    # Class 3: Shooting (UCF-Crime Shooting)
    shooting_vids = [("ucf_seq", (vid, frames)) for vid, frames in ucf_data["shooting"].items()]
    random.shuffle(shooting_vids)
    n = len(shooting_vids)
    splits["train"]["shooting"] = shooting_vids[: int(n * train_ratio)]
    splits["val"]["shooting"] = shooting_vids[int(n * train_ratio) : int(n * (train_ratio + val_ratio))]
    splits["test"]["shooting"] = shooting_vids[int(n * (train_ratio + val_ratio)) :]

    for s_name in ("train", "val", "test"):
        logger.info(
            "Split '%s' video counts -> Normal: %d, Fighting: %d, Robbery: %d, Shooting: %d",
            s_name,
            len(splits[s_name]["normal"]),
            len(splits[s_name]["fighting"]),
            len(splits[s_name]["robbery"]),
            len(splits[s_name]["shooting"]),
        )

    return splits


# ==============================================================================
# YOLOv8m-Pose Feature Extraction Engine
# ==============================================================================
def extract_keypoints_from_frame(
    model,
    frame: np.ndarray,
    device: str,
) -> np.ndarray:
    """
    Extract 17 anatomical keypoints (51 floats: [x_norm, y_norm, conf] * 17)
    from a single frame using YOLOv8m-Pose.
    Multi-person strategy: selects primary person by largest bbox area * confidence.
    """
    h, w = frame.shape[:2]
    # Default 51 zeros if no person is detected
    empty_keypoints = np.zeros(51, dtype=np.float32)

    try:
        results = model.predict(
            source=frame,
            device=device,
            verbose=False,
            conf=0.25,
            imgsz=384,
            classes=[0],  # Person class only
        )
    except Exception as exc:
        logger.debug("Model prediction error: %s", exc)
        return empty_keypoints

    if not results or len(results) == 0:
        return empty_keypoints

    res = results[0]
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return empty_keypoints

    # Locate primary interacting person (highest bbox area * score)
    boxes = res.boxes.xywh.cpu().numpy()  # [N, 4]: x_c, y_c, bw, bh
    confs = res.boxes.conf.cpu().numpy()
    best_idx = -1
    best_score = -1.0

    for i in range(len(boxes)):
        area = boxes[i][2] * boxes[i][3]
        score = area * confs[i]
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx < 0:
        return empty_keypoints

    # Extract 17 keypoints for the primary person
    # shape: [17, 3] or [17, 2] with conf
    kp_data = res.keypoints.data[best_idx].cpu().numpy()  # [17, 3]: x, y, conf

    flattened = np.zeros(51, dtype=np.float32)
    for j in range(min(17, len(kp_data))):
        kx, ky = kp_data[j][0], kp_data[j][1]
        kc = kp_data[j][2] if kp_data.shape[-1] >= 3 else 1.0

        # Coordinate Normalization relative to frame dimensions
        kx_norm = float(np.clip(kx / max(w, 1), 0.0, 1.0))
        ky_norm = float(np.clip(ky / max(h, 1), 0.0, 1.0))
        kc = float(np.clip(kc, 0.0, 1.0))

        flattened[j * 3] = kx_norm
        flattened[j * 3 + 1] = ky_norm
        flattened[j * 3 + 2] = kc

    return flattened


def process_video_frames(
    model,
    video_item: Tuple,
    device: str,
    target_fps: int = 15,
    max_frames: int = 120,
) -> List[np.ndarray]:
    """Decode video frames and run pose extraction."""
    item_type, data = video_item
    keypoint_sequence: List[np.ndarray] = []

    if item_type == "rwf_file":
        video_path: Path = data
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            log_quarantine(video_path, "Cannot open video stream")
            return []

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        orig_fps = orig_fps if orig_fps > 0 else 30.0
        step = max(1, int(round(orig_fps / target_fps)))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or len(keypoint_sequence) >= max_frames:
                break
            if frame_idx % step == 0:
                kps = extract_keypoints_from_frame(model, frame, device)
                keypoint_sequence.append(kps)
            frame_idx += 1
        cap.release()

    elif item_type == "ucf_seq":
        vid_id, frame_paths = data
        for fp in frame_paths[:max_frames]:
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            kps = extract_keypoints_from_frame(model, frame, device)
            keypoint_sequence.append(kps)

    return keypoint_sequence


def create_sliding_windows(
    keypoints: List[np.ndarray],
    window_size: int = 30,
    stride: int = 15,
) -> List[np.ndarray]:
    """
    Segment continuous keypoint stream into fixed sliding windows of 30 frames.
    Output array shape: (30, 51).
    """
    windows: List[np.ndarray] = []
    total_frames = len(keypoints)

    if total_frames < window_size:
        # If sequence is slightly shorter than 30 frames, pad by repeating last frame
        if total_frames >= 15:
            arr = np.array(keypoints, dtype=np.float32)
            pad_len = window_size - total_frames
            padding = np.tile(arr[-1:], (pad_len, 1))
            padded = np.vstack([arr, padding])
            windows.append(padded)
        return windows

    for start in range(0, total_frames - window_size + 1, stride):
        window = np.array(keypoints[start : start + window_size], dtype=np.float32)
        windows.append(window)

    return windows


# ==============================================================================
# Pipeline Execution & Serialization
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="SENTINEL Offline Pose Extraction Pipeline")
    parser.add_argument(
        "--window_size",
        type=int,
        default=30,
        help="Sliding window size (frames) for 2-layer LSTM sequence (default: 30)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Sliding window stride (frames) (default: 15)",
    )
    parser.add_argument(
        "--max_videos_per_class",
        type=int,
        default=None,
        help="Maximum video clips per class to process (optional cap for fast execution)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on: 'cuda' or 'cpu' (default: auto-detected)",
    )
    args = parser.parse_args()

    device = args.device or get_device()
    logger.info("=" * 65)
    logger.info("SENTINEL — Offline Pose Extraction Pipeline (YOLOv8m-Pose)")
    logger.info("=" * 65)
    logger.info("Selected Device   : %s", device.upper())
    logger.info("Window Size       : %d frames", args.window_size)
    logger.info("Window Stride     : %d frames", args.stride)
    logger.info("Output Directory  : %s", CURATED_LSTM_DIR)

    # 1. Discover Video Datasets
    rwf_data = discover_rwf2000_videos()
    ucf_data = discover_ucf_crime_video_sequences()

    # 2. Partition with Anti-Data-Leakage Splitting (70% / 20% / 10%)
    splits = create_anti_leakage_splits(rwf_data, ucf_data, seed=42)

    # 3. Load YOLOv8m-Pose Model
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = WEIGHTS_DIR / "yolov8m-pose.pt"
    logger.info("Loading YOLOv8m-Pose model (%s)...", weights_path)
    model = YOLO("yolov8m-pose.pt")

    # 4. Process Splits and Extract Keypoint Sequences
    CURATED_LSTM_DIR.mkdir(parents=True, exist_ok=True)
    summary_stats = {
        "train": {c: 0 for c in CLASS_NAMES.values()},
        "val": {c: 0 for c in CLASS_NAMES.values()},
        "test": {c: 0 for c in CLASS_NAMES.values()},
    }

    for split_name in ("train", "val", "test"):
        for class_id, class_name in CLASS_NAMES.items():
            dest_dir = CURATED_LSTM_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            video_items = splits[split_name][class_name]
            if args.max_videos_per_class:
                ratio = 0.7 if split_name == "train" else (0.2 if split_name == "val" else 0.1)
                limit = max(1, int(round(args.max_videos_per_class * ratio)))
                video_items = video_items[:limit]

            logger.info(
                "Extracting '%s/%s' (%d videos)...",
                split_name.upper(),
                class_name,
                len(video_items),
            )

            seq_count = 0
            for idx, item in enumerate(tqdm(video_items, desc=f"{split_name}/{class_name}")):
                kps_stream = process_video_frames(model, item, device=device)
                if not kps_stream:
                    continue

                windows = create_sliding_windows(
                    kps_stream,
                    window_size=args.window_size,
                    stride=args.stride,
                )

                for w_idx, win in enumerate(windows):
                    # Sanity check: must be exactly (30, 51) and finite
                    if win.shape != (args.window_size, 51) or not np.isfinite(win).all():
                        continue

                    out_name = f"seq_{class_name}_{idx:04d}_{w_idx:03d}.npy"
                    out_path = dest_dir / out_name
                    np.save(str(out_path), win)
                    seq_count += 1

            summary_stats[split_name][class_name] = seq_count
            logger.info("Saved %d sequences for '%s/%s'", seq_count, split_name, class_name)

    # 5. Summary Report
    logger.info("=" * 65)
    logger.info("POSE EXTRACTION COMPLETE — DATASET AUDIT:")
    logger.info("=" * 65)
    total_sequences = 0
    for s_name, counts in summary_stats.items():
        s_total = sum(counts.values())
        total_sequences += s_total
        logger.info(
            "[%s] Normal: %d | Fighting: %d | Robbery: %d | Shooting: %d | Total: %d",
            s_name.upper(),
            counts["normal"],
            counts["fighting"],
            counts["robbery"],
            counts["shooting"],
            s_total,
        )
    logger.info("-" * 65)
    logger.info("Total Extracted (30, 51) Sequences: %d", total_sequences)
    logger.info("Curated Behavior LSTM Root        : %s", CURATED_LSTM_DIR)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
