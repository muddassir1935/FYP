#!/usr/bin/env python3
"""Assemble the SENTINEL weapon_yolo_v2.1 fine-tune dataset.

Mixes the v2.1 hard-example pool (mined frames + Roboflow knife images)
with a replay of the v2 train split, so the fine-tune run learns the new
hard cases without catastrophically forgetting the v2 distribution.

Inputs (all passed as CLI args -- no machine paths are hardcoded):
  --v2-dir DIR            v2 dataset (images/{train,val,test} + labels/*,
                          nc=2, 0=Gun, 1=Knife). Val + test are copied
                          UNCHANGED; the 680-image test split stays FROZEN.
  --v21 LABELS:IMG,...    one v2.1 label dir (YOLO .txt, empty file = hard
                          negative) + the image dir(s) its stems live in.
                          Repeatable, one per mining source.
  --roboflow-kept JSON    kept-list JSON (entries "<split>/<stem>") of
                          Roboflow knife images to fold in as knife coverage.
  --roboflow-dir DIR      the Roboflow dataset root holding those images.

Rules:
  - Replay guard: at least 50% of the v2 train split is mixed back in
    (--replay-fraction, default 0.5, values below 0.5 are rejected).
  - pHash audit FIRST: every new-pool frame is hashed and compared against
    the FROZEN test set (Hamming distance <= 5 = leakage, grouped by source
    video). On leakage the run FAILS (exit 2) and no dataset is written.
    Near-dupes of the v2 train split and within the new pool are dropped;
    within the pool, positives are audited first so a near-dupe cluster
    keeps a positive representative when one exists.
  - Labels are NEVER invented: only the YOLO boxes already present in the
    label files are copied, verbatim (after format validation). Empty label
    files stay empty (hard negatives). Frames listed in excluded.txt must
    not appear in the label dirs at all -- if one does, the run aborts.
  - Offline weather/blur augmentation (one variant per accepted v2.1
    POSITIVE frame, same boxes): gaussian blur, motion blur, low light,
    fog, sensor noise, downscale-upscale. Variants inherit the audit
    verdict of their source frame (pixel-level transform of an audited
    original). Disable with --no-augment-v21.

Outputs (--out DIR):
  images/{train,val,test} + labels/{train,val,test} + dataset.yaml
  + manifest.json (counts per source and per class, audit result,
  review status, replay fraction, seed). The manifest travels with the
  dataset to Kaggle; the training notebook refuses to train on a dataset
  whose manifest says review_status == "unreviewed".

Usage:
  python scripts/data_preparation/assemble_v21_dataset.py \\
      --v2-dir <v2 dataset> \\
      --v21 <ucf labels>:<ucf frames> \\
      --v21 <test-clips labels>:<test1 frames>,<test2 frames>,... \\
      --v21 <webcam2 labels>:<webcam2 frames> \\
      --roboflow-kept <mahad_knife_kept.json> \\
      --roboflow-dir <gun-and-knife-detection-1-yolov8> \\
      --out <weapon_yolo_v21> --review-status reviewed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    import imagehash
except ImportError:  # pragma: no cover - import-time guard for --help
    imagehash = None  # type: ignore[assignment]

HAMMING_MAX = 5
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = {0: "Gun", 1: "Knife"}
# A source-video group is the frame stem minus its trailing _NNNN index,
# e.g. Shooting004_0022 -> Shooting004, test5_0018 -> test5.
GROUP_RE = re.compile(r"^(.*)_\d+$")


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class Item:
    """One candidate image + its (already human-written) label lines."""

    img_path: Path
    lines: list[str]
    stem: str
    source: str
    group: str  # source video / clip, for leakage grouping
    phash: int | None = None


@dataclass
class SourceStats:
    received: int = 0
    accepted: int = 0
    missing_image: int = 0
    bad_label: int = 0
    corrupt: int = 0
    leakage: int = 0
    dup_v2_train: int = 0
    dup_pool: int = 0
    gun_boxes: int = 0
    knife_boxes: int = 0
    positive_images: int = 0
    negative_images: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "received": self.received,
            "accepted": self.accepted,
            "missing_image": self.missing_image,
            "bad_label": self.bad_label,
            "corrupt": self.corrupt,
            "leakage": self.leakage,
            "dup_v2_train": self.dup_v2_train,
            "dup_pool": self.dup_pool,
            "gun_boxes": self.gun_boxes,
            "knife_boxes": self.knife_boxes,
            "positive_images": self.positive_images,
            "negative_images": self.negative_images,
        }


def parse_yolo_label(path: Path) -> list[str] | None:
    """Parse + validate a YOLO label file. None if malformed.

    Empty file (or whitespace only) is valid: a hard-negative image.
    Boxes are returned verbatim, re-formatted to 6 decimals.
    """
    try:
        raw = path.read_text()
    except OSError:
        return None
    lines: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return None
        try:
            cls = int(parts[0])
            x, y, w, h = (float(p) for p in parts[1:])
        except ValueError:
            return None
        if cls not in (0, 1):
            return None
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            return None
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    return lines


def phash_of(path: Path) -> int | None:
    """64-bit perceptual hash as int, or None if the image is unreadable."""
    assert imagehash is not None, "imagehash is required (pip install ImageHash)"
    try:
        with Image.open(path) as img:
            img.load()
            if img.size[0] < 16 or img.size[1] < 16:
                return None
            return int(str(imagehash.phash(img)), 16)
    except Exception:
        return None


def walk_images(d: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def find_image(stem: str, image_dirs: list[Path]) -> Path | None:
    for d in image_dirs:
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            cand = d / f"{stem}{ext}"
            if cand.is_file():
                return cand
    return None


def source_group(stem: str) -> str:
    m = GROUP_RE.match(stem)
    return m.group(1) if m else stem


def read_excluded(labels_dir: Path) -> set[str]:
    """Stems listed in excluded.txt (labels dir or its parent)."""
    excluded: set[str] = set()
    for cand in (labels_dir / "excluded.txt", labels_dir.parent / "excluded.txt"):
        if cand.is_file():
            for line in cand.read_text().splitlines():
                name = line.strip()
                if name:
                    excluded.add(Path(name).stem)
            break
    return excluded


def collect_v21(labels_dir: Path, image_dirs: list[Path],
                prefix: str) -> tuple[list[Item], SourceStats]:
    """Collect one v2.1 mining source. Label files are authoritative."""
    stats = SourceStats()
    items: list[Item] = []
    excluded = read_excluded(labels_dir)
    label_files = sorted(labels_dir.glob("*.txt"))
    log(f"  {prefix}: {len(label_files)} label files, "
        f"{len(excluded)} excluded frames on record")
    for lbl in label_files:
        stem = lbl.stem
        if stem in excluded:
            raise SystemExit(
                f"ERROR: {stem} is in excluded.txt but has a label file in "
                f"{labels_dir} -- excluded frames must not be labelled. "
                "Fix the label dir and re-run.")
        stats.received += 1
        lines = parse_yolo_label(lbl)
        if lines is None:
            stats.bad_label += 1
            continue
        img = find_image(stem, image_dirs)
        if img is None:
            stats.missing_image += 1
            continue
        # Prefix the stem per source (skip it when the stem already starts
        # with it, e.g. webcam2_0001, to avoid webcam2_webcam2_0001).
        out_stem = stem if stem.startswith(prefix) else f"{prefix}{stem}"
        items.append(Item(img_path=img, lines=lines, stem=out_stem,
                          source=prefix.rstrip("_"),
                          group=source_group(stem)))
    return items, stats


def collect_roboflow(kept_json: Path, rf_dir: Path) -> tuple[list[Item], SourceStats]:
    """Collect the kept Roboflow knife images (entries '<split>/<stem>')."""
    stats = SourceStats()
    items: list[Item] = []
    kept: list[str] = json.loads(kept_json.read_text())
    log(f"  roboflow_knife: {len(kept)} kept entries")
    for entry in kept:
        stats.received += 1
        parts = entry.split("/")
        if len(parts) != 2:
            stats.bad_label += 1
            continue
        split, stem = parts
        img = find_image(stem, [rf_dir / split / "images"])
        lbl = rf_dir / split / "labels" / f"{stem}.txt"
        lines = parse_yolo_label(lbl) if lbl.is_file() else None
        if img is None:
            stats.missing_image += 1
            continue
        if lines is None:
            stats.bad_label += 1
            continue
        items.append(Item(img_path=img, lines=lines, stem=f"rfk_{stem}",
                          source="roboflow_knife", group=f"roboflow/{split}"))
    return items, stats


def min_hamming(h: int, arr: np.ndarray) -> tuple[int, int]:
    """Min Hamming distance from hash h to each hash in uint64 array."""
    if len(arr) == 0:
        return 999, -1
    dists = np.bitwise_count(arr ^ np.uint64(h))
    i = int(np.argmin(dists))
    return int(dists[i]), i


# ------------------------------------------------- weather augmentation
# Offline blur/weather variants for v2.1 hard positives (PIL + numpy only,
# no albumentations dependency). One variant per positive frame, same boxes.
def weather_variant(img: Image.Image, rng: random.Random) -> tuple[Image.Image, str]:
    choice = rng.choice(["gaussian_blur", "motion_blur", "low_light",
                         "fog", "noise", "downscale"])
    if choice == "gaussian_blur":
        return img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.0, 2.5))), choice
    if choice == "motion_blur":
        kernel = [0] * 25
        if rng.random() < 0.5:  # horizontal streak
            for x in range(5):
                kernel[10 + x] = 1
        else:  # vertical streak
            for y in range(5):
                kernel[y * 5 + 2] = 1
        return img.filter(ImageFilter.Kernel((5, 5), kernel, scale=5)), choice
    if choice == "low_light":
        out = ImageEnhance.Brightness(img).enhance(rng.uniform(0.5, 0.65))
        return ImageEnhance.Contrast(out).enhance(0.9), choice
    if choice == "fog":
        grey = Image.new("RGB", img.size, (200, 208, 212))
        out = Image.blend(img.convert("RGB"), grey, rng.uniform(0.30, 0.45))
        return ImageEnhance.Contrast(out).enhance(0.85), choice
    if choice == "noise":
        sigma = rng.uniform(8.0, 15.0)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, sigma, np.asarray(img.convert("RGB")).shape)
        arr = np.asarray(img.convert("RGB")).astype(np.float32) + noise
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)), choice
    # downscale-upscale: distant / low-resolution CCTV look
    w, h = img.size
    factor = rng.uniform(0.4, 0.5)
    small = img.resize((max(8, int(w * factor)), max(8, int(h * factor))),
                       Image.BILINEAR)
    return small.resize((w, h), Image.BICUBIC), choice


# ------------------------------------------------------------------- main
def parse_v21_spec(spec: str) -> tuple[Path, list[Path], str]:
    """'LABELS_DIR:IMG_DIR[,IMG_DIR2]' -> (labels, [images], prefix).

    The prefix comes from the labels dir name: '<name>-labels' -> '<name>_'
    (when the inner 'labels/' subdir is passed, the parent dir name is used).
    Keeps output stems unique per source.
    """
    if ":" not in spec:
        raise SystemExit(
            f"ERROR: --v21 expects LABELS_DIR:IMG_DIR[,...], got {spec!r}")
    labels_s, images_s = spec.split(":", 1)
    labels = Path(labels_s)
    image_dirs = [Path(p) for p in images_s.split(",") if p]
    if not labels.is_dir():
        raise SystemExit(f"ERROR: v2.1 labels dir not found: {labels}")
    for d in image_dirs:
        if not d.is_dir():
            raise SystemExit(f"ERROR: v2.1 image dir not found: {d}")
    # The name usually comes from the labels dir itself ('ucf-shooting-labels'),
    # but when the caller passes the inner 'labels/' subdir, use the parent.
    name = labels.parent.name if labels.name == "labels" else labels.name
    if name.endswith("-labels"):
        name = name[: -len("-labels")]
    return labels, image_dirs, name.replace("-", "_") + "_"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Assemble the SENTINEL weapon_yolo_v2.1 fine-tune "
                    "dataset (v2.1 hard examples + v2 replay, pHash-audited "
                    "against the frozen test set).")
    p.add_argument("--v2-dir", required=True,
                   help="v2 dataset dir (images/{train,val,test} + labels/*).")
    p.add_argument("--v21", action="append", required=True, metavar="SPEC",
                   help="v2.1 source as LABELS_DIR:IMG_DIR[,...]; repeatable.")
    p.add_argument("--roboflow-kept",
                   help="kept-list JSON of Roboflow knife images.")
    p.add_argument("--roboflow-dir",
                   help="Roboflow dataset root for --roboflow-kept.")
    p.add_argument("--out", required=True, help="Output dataset dir.")
    p.add_argument("--replay-fraction", type=float, default=0.5,
                   help="Fraction of v2 train mixed back in (>= 0.5).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--review-status", choices=["unreviewed", "reviewed"],
                   default="unreviewed",
                   help="'reviewed' only once the user has approved the "
                        "v2.1 labels; the Kaggle notebook refuses to train "
                        "on an 'unreviewed' dataset.")
    p.add_argument("--leakage-mode", choices=["fail", "flag"], default="fail",
                   help="'fail' (default) aborts without writing a dataset "
                        "when a new frame is a near-dupe of the frozen test "
                        "set; 'flag' only records it in the manifest.")
    p.add_argument("--no-augment-v21", action="store_true",
                   help="Skip the offline weather/blur variants.")
    p.add_argument("--force", action="store_true",
                   help="Replace a non-empty --out dir.")
    return p


def write_label(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if imagehash is None:
        print("ERROR: the 'imagehash' package is required "
              "(pip install ImageHash).", file=sys.stderr)
        return 1
    if args.replay_fraction < 0.5:
        print(f"ERROR: --replay-fraction {args.replay_fraction} < 0.5 -- "
              "the fine-tune mix must keep at least 50% of v2 train "
              "(catastrophic-forgetting guard).", file=sys.stderr)
        return 1
    if bool(args.roboflow_kept) != bool(args.roboflow_dir):
        print("ERROR: --roboflow-kept and --roboflow-dir go together.",
              file=sys.stderr)
        return 1

    v2_dir = Path(args.v2_dir)
    out_dir = Path(args.out)
    for split in ("train", "val", "test"):
        if not (v2_dir / "images" / split).is_dir():
            print(f"ERROR: {v2_dir}/images/{split} not found -- is "
                  "--v2-dir a YOLO dataset dir?", file=sys.stderr)
            return 1
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        print(f"ERROR: --out {out_dir} exists and is not empty "
              "(pass --force to replace it).", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    aug_rng = random.Random(args.seed + 1)
    manifest: dict[str, Any] = {
        "dataset": "weapon_yolo_v2.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "review_status": args.review_status,
        "seed": args.seed,
        "replay_fraction": args.replay_fraction,
        "leakage_mode": args.leakage_mode,
        "offline_weather_augmentation": not args.no_augment_v21,
        "sources": {},
        "audit": {},
        "final": {},
    }
    if args.review_status == "unreviewed":
        manifest["warning"] = (
            "UNREVIEWED LABELS -- pipeline proof only, NOT training data. "
            "Re-run with --review-status reviewed after the user approves "
            "the v2.1 labels on Drive.")
        log("!!! review-status is UNREVIEWED: this output is a pipeline "
            "proof, NOT training data !!!")

    # ---- collect new pool -------------------------------------------
    log("Collecting v2.1 sources ...")
    pool: list[Item] = []
    stats_by_source: dict[str, SourceStats] = {}
    for spec in args.v21:
        labels_dir, image_dirs, prefix = parse_v21_spec(spec)
        items, stats = collect_v21(labels_dir, image_dirs, prefix)
        pool.extend(items)
        stats_by_source[prefix.rstrip("_")] = stats
        log(f"  {prefix.rstrip('_')}: {len(items)} usable "
            f"({stats.bad_label} bad labels, {stats.missing_image} missing images)")
    if args.roboflow_kept:
        items, stats = collect_roboflow(Path(args.roboflow_kept),
                                        Path(args.roboflow_dir))
        pool.extend(items)
        stats_by_source["roboflow_knife"] = stats
        log(f"  roboflow_knife: {len(items)} usable "
            f"({stats.bad_label} bad labels, {stats.missing_image} missing images)")
    log(f"New pool: {len(pool)} images")

    # ---- hash frozen test + v2 train ----------------------------------
    log("Hashing frozen test set ...")
    test_imgs = walk_images(v2_dir / "images" / "test")
    test_hashes, test_stems = [], []
    test_bad = 0
    for i, p in enumerate(test_imgs, 1):
        h = phash_of(p)
        if h is None:
            test_bad += 1
            continue
        test_hashes.append(h)
        test_stems.append(p.stem)
        if i % 250 == 0:
            log(f"  ... {i}/{len(test_imgs)} test images hashed")
    test_arr = np.array(test_hashes, dtype=np.uint64)
    log(f"  test: {len(test_hashes)} hashed, {test_bad} unreadable")

    log("Hashing v2 train (dedupe reference) ...")
    train_imgs = walk_images(v2_dir / "images" / "train")
    train_hashes: list[int] = []
    train_bad = 0
    for i, p in enumerate(train_imgs, 1):
        h = phash_of(p)
        if h is None:
            train_bad += 1
            continue
        train_hashes.append(h)
        if i % 5000 == 0:
            log(f"  ... {i}/{len(train_imgs)} v2 train images hashed")
    train_arr = np.array(train_hashes, dtype=np.uint64)
    log(f"  v2 train: {len(train_hashes)} hashed, {train_bad} unreadable")

    # ---- pHash audit + dedupe ----------------------------------------
    # Audit positives FIRST (stable sort, so source/frame order is kept
    # within each group): a near-dupe cluster then keeps a positive
    # representative when one exists. Positives are the scarce training
    # signal; without this, a positive frame could lose its cluster to an
    # earlier near-identical negative (weapon entering/leaving the frame).
    pool.sort(key=lambda it: 0 if it.lines else 1)
    log("Auditing new pool vs frozen test (Hamming <= 5 = leakage) ...")
    accepted: list[Item] = []
    accepted_hashes: list[int] = []
    leakage_items: list[dict[str, Any]] = []
    leakage_by_group: dict[str, int] = {}
    for i, item in enumerate(pool, 1):
        if i % 500 == 0:
            log(f"  ... {i}/{len(pool)} pool images audited, "
                f"{len(accepted)} accepted")
        stats = stats_by_source[item.source]
        h = phash_of(item.img_path)
        if h is None:
            stats.corrupt += 1
            continue
        item.phash = h
        d_test, j = min_hamming(h, test_arr)
        if d_test <= HAMMING_MAX:
            stats.leakage += 1
            leakage_by_group[item.group] = leakage_by_group.get(item.group, 0) + 1
            leakage_items.append({
                "frame": item.img_path.name, "source": item.source,
                "group": item.group, "distance": d_test,
                "test_match": test_stems[j] if j >= 0 else None,
            })
            continue
        d_train, _ = min_hamming(h, train_arr)
        if d_train <= HAMMING_MAX:
            stats.dup_v2_train += 1
            continue
        d_pool = 999
        if accepted_hashes:
            d_pool, _ = min_hamming(h, np.array(accepted_hashes, dtype=np.uint64))
        if d_pool <= HAMMING_MAX:
            stats.dup_pool += 1
            continue
        accepted.append(item)
        accepted_hashes.append(h)
        stats.accepted += 1
        if not item.lines:
            stats.negative_images += 1
        else:
            stats.positive_images += 1
        for ln in item.lines:
            if ln.startswith("0 "):
                stats.gun_boxes += 1
            else:
                stats.knife_boxes += 1

    n_leak = len(leakage_items)
    audit: dict[str, Any] = {
        "hamming_max": HAMMING_MAX,
        "test_images_hashed": len(test_hashes),
        "test_images_unreadable": test_bad,
        "v2_train_hashed": len(train_hashes),
        "v2_train_unreadable": train_bad,
        "leakage_count": n_leak,
        "leakage_by_source_video": leakage_by_group,
        "leakage_items": leakage_items[:50],
        "verdict": "LEAK" if n_leak else "CLEAN",
    }
    manifest["audit"] = audit
    manifest["sources"] = {k: v.as_dict() for k, v in stats_by_source.items()}
    log(f"  audit verdict: {audit['verdict']} "
        f"({n_leak} leakage, "
        f"{sum(s.dup_v2_train for s in stats_by_source.values())} dup vs v2 train, "
        f"{sum(s.dup_pool for s in stats_by_source.values())} dup in pool)")
    for group, n in sorted(leakage_by_group.items()):
        log(f"    leakage in {group}: {n} frames")
    if n_leak and args.leakage_mode == "fail":
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest["final"] = {"status": "FAILED_LEAKAGE",
                             "reason": "pHash audit found near-dupes of the "
                                       "frozen test set; no dataset written."}
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        log(f"FAIL: {n_leak} new-pool frames leak into the frozen test set "
            f"(see {out_dir / 'manifest.json'}). No dataset written.")
        return 2

    # ---- replay sample -------------------------------------------------
    n_replay = math.ceil(args.replay_fraction * len(train_imgs))
    replay = rng.sample(train_imgs, n_replay)
    log(f"Replay: {n_replay} / {len(train_imgs)} v2 train images "
        f"({args.replay_fraction:.0%}, seed {args.seed})")

    # ---- write output dataset ------------------------------------------
    log(f"Writing dataset to {out_dir} ...")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True)
        (out_dir / "labels" / split).mkdir(parents=True)

    def copy_pair(img: Path, lbl_dir: Path, dest_stem: str,
                  dest_split: str = "train") -> None:
        shutil.copy2(img, out_dir / "images" / dest_split
                     / f"{dest_stem}{img.suffix.lower()}")
        src_lbl = lbl_dir / f"{img.stem}.txt"
        dst_lbl = out_dir / "labels" / dest_split / f"{dest_stem}.txt"
        if src_lbl.is_file():
            shutil.copy2(src_lbl, dst_lbl)
        else:
            dst_lbl.touch()  # no label file = background image

    copied = 0
    for img in replay:
        copy_pair(img, v2_dir / "labels" / "train", img.stem)
        copied += 1
        if copied % 5000 == 0:
            log(f"  ... {copied} replay images copied")
    log(f"  replay copied: {copied}")

    # val + test copied UNCHANGED (test stays frozen)
    for split in ("val", "test"):
        n = 0
        for img in walk_images(v2_dir / "images" / split):
            copy_pair(img, v2_dir / "labels" / split, img.stem, split)
            n += 1
        log(f"  {split} copied unchanged: {n}")

    # new pool -> train
    n_v21 = n_rf = 0
    variant_counts: dict[str, int] = {}
    for i, item in enumerate(accepted, 1):
        if i % 500 == 0:
            log(f"  ... {i}/{len(accepted)} new-pool images written")
        ext = item.img_path.suffix.lower()
        shutil.copy2(item.img_path, out_dir / "images" / "train" / f"{item.stem}{ext}")
        write_label(out_dir / "labels" / "train" / f"{item.stem}.txt", item.lines)
        if item.source == "roboflow_knife":
            n_rf += 1
        else:
            n_v21 += 1
        # one weather/blur variant per accepted v2.1 POSITIVE frame
        if (not args.no_augment_v21 and item.source != "roboflow_knife"
                and item.lines):
            try:
                with Image.open(item.img_path) as im:
                    variant, kind = weather_variant(im.convert("RGB"), aug_rng)
                    variant.save(out_dir / "images" / "train"
                                 / f"{item.stem}_wx.jpg", quality=92)
                write_label(out_dir / "labels" / "train"
                            / f"{item.stem}_wx.txt", item.lines)
                variant_counts[kind] = variant_counts.get(kind, 0) + 1
            except Exception as exc:  # never let one bad image stop the run
                log(f"  WARNING: weather variant failed for {item.stem}: {exc}")
    n_variants = sum(variant_counts.values())
    log(f"  new pool written: {n_v21} v2.1 + {n_rf} roboflow_knife, "
        f"+ {n_variants} weather variants {variant_counts}")

    (out_dir / "dataset.yaml").write_text(
        f"path: {out_dir.name}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnc: 2\nnames:\n  0: Gun\n  1: Knife\n")

    final_train = n_replay + len(accepted) + n_variants
    gun_boxes = sum(s.gun_boxes for s in stats_by_source.values())
    knife_boxes = sum(s.knife_boxes for s in stats_by_source.values())
    neg_images = sum(s.negative_images for s in stats_by_source.values())
    manifest["final"] = {
        "status": "OK",
        "train": final_train,
        "val": len(walk_images(v2_dir / "images" / "val")),
        "test": len(test_imgs),
        "train_breakdown": {
            "v2_replay": n_replay,
            "v21_new": n_v21,
            "roboflow_knife": n_rf,
            "weather_variants": n_variants,
            "weather_variants_by_kind": variant_counts,
        },
        "new_pool_boxes": {"Gun": gun_boxes, "Knife": knife_boxes,
                           "negative_images": neg_images},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    log("DONE")
    log(f"  final: train {final_train} (replay {n_replay} + v2.1 {n_v21} "
        f"+ roboflow {n_rf} + weather {n_variants}), "
        f"val {manifest['final']['val']}, test {manifest['final']['test']} (frozen)")
    log(f"  new boxes: Gun {gun_boxes}, Knife {knife_boxes}, "
        f"negatives {neg_images}")
    log(f"  manifest: {out_dir / 'manifest.json'}")
    if args.review_status == "unreviewed":
        log("  REMINDER: review_status is UNREVIEWED -- this is a pipeline "
            "proof, NOT training data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
