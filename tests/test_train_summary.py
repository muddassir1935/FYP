"""Unit tests for the training run-summary helpers (pure Python).

Covers the best-epoch analysis of ``results.csv`` (SHOULD-2 / SHOULD-3),
the dataset-manifest identity + resumed-config helpers (MUST-12 /
SHOULD-5), the new argument-parser flags, and the OpenVINO self-check's
validation-image lookup (NICE-5). No torch/ultralytics needed: the
training script imports those lazily, and the helpers under test are
stdlib-only (PyYAML tests skip when it is unavailable).

Run from the repo root:  python -m unittest tests.test_train_summary -v
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.training.export_openvino import find_val_image
from scripts.training.train_weapon_detector import (
    analyze_results_csv,
    dataset_manifest_info,
    load_resumed_config,
    parse_args,
)

try:
    import yaml  # noqa: F401  (only needed for the args.yaml / data-yaml tests)

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

HEADER = (
    "epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,"
    "metrics/precision(B),metrics/recall(B),metrics/mAP50(B),"
    "metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,"
    "lr/pg0,lr/pg1,lr/pg2\n"
)


def _row(epoch, precision, recall, map50, map5095):
    return (
        f"{epoch},{epoch * 10},1.0,1.0,1.0,{precision},{recall},{map50},"
        f"{map5095},1.0,1.0,1.0,0.01,0.01,0.01\n"
    )


class AnalyzeResultsCsvTest(unittest.TestCase):
    def _write_csv(self, directory: Path, text: str) -> Path:
        path = directory / "results.csv"
        path.write_text(text)
        return path

    def test_best_epochs_differ_gap_recorded(self):
        # v3-like: best-fitness (mAP50-95) epoch 100, best-mAP50 epoch 98.
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(
                Path(tmp),
                HEADER
                + _row(97, 0.80, 0.75, 0.8600, 0.5300)
                + _row(98, 0.81, 0.76, 0.8688, 0.5350)
                + _row(99, 0.80, 0.75, 0.8650, 0.5380)
                + _row(100, 0.82, 0.77, 0.8684, 0.5404),
            )
            out = analyze_results_csv(csv_path)
        self.assertEqual(out["best_fitness_epoch"], 100)
        self.assertEqual(out["best_map50_epoch"], 98)
        self.assertAlmostEqual(out["map50_gap"], 0.0004)
        self.assertIn("mAP50-95", out["fitness_definition"])
        # best_val_metrics come from the max-fitness row (epoch 100).
        best = out["best_val_metrics"]
        self.assertEqual(best["epoch"], 100)
        self.assertAlmostEqual(best["mAP50"], 0.8684)
        self.assertAlmostEqual(best["mAP50_95"], 0.5404)
        self.assertAlmostEqual(best["fitness"], 0.5404)
        self.assertAlmostEqual(best["precision"], 0.82)
        self.assertAlmostEqual(best["recall"], 0.77)

    def test_best_epochs_coincide_gap_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(
                Path(tmp),
                HEADER
                + _row(1, 0.50, 0.40, 0.5000, 0.3000)
                + _row(2, 0.70, 0.60, 0.8000, 0.5000),
            )
            out = analyze_results_csv(csv_path)
        self.assertEqual(out["best_fitness_epoch"], 2)
        self.assertEqual(out["best_map50_epoch"], 2)
        self.assertAlmostEqual(out["map50_gap"], 0.0)

    def test_missing_file_returns_empty(self):
        self.assertEqual(analyze_results_csv(Path("/nonexistent/results.csv")), {})

    def test_missing_metric_columns_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(Path(tmp), "epoch,time\n1,10\n")
            self.assertEqual(analyze_results_csv(csv_path), {})

    def test_bad_rows_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = self._write_csv(
                Path(tmp),
                HEADER
                + "not-an-epoch,10,1,1,1,0.5,0.5,0.5,0.5,1,1,1,0.01,0.01,0.01\n"
                + _row(1, 0.50, 0.40, 0.5000, 0.3000),
            )
            out = analyze_results_csv(csv_path)
        self.assertEqual(out["best_fitness_epoch"], 1)
        self.assertEqual(out["best_map50_epoch"], 1)


class DatasetManifestInfoTest(unittest.TestCase):
    def test_manifest_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "review_status": "reviewed",
                "audit": {"verdict": "CLEAN", "leakage_count": 0},
            }
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "sentinel_data.yaml").write_text("path: .\n")
            info = dataset_manifest_info(root / "sentinel_data.yaml")
        expected = hashlib.sha256(
            json.dumps(manifest).encode()
        ).hexdigest()
        self.assertEqual(info["dataset_manifest_sha256"], expected)
        self.assertEqual(info["dataset_manifest_review_status"], "reviewed")
        self.assertEqual(info["dataset_manifest_audit_verdict"], "CLEAN")

    def test_manifest_absent_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            yaml_path = Path(tmp) / "data.yaml"
            yaml_path.write_text("path: .\n")
            self.assertEqual(dataset_manifest_info(yaml_path), {})


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class LoadResumedConfigTest(unittest.TestCase):
    def test_missing_args_yaml_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_resumed_config(Path(tmp)))

    def test_valid_args_yaml_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "args.yaml").write_text(
                "epochs: 100\nlr0: 0.01\nseed: 42\n"
            )
            config = load_resumed_config(Path(tmp))
        self.assertEqual(config["epochs"], 100)
        self.assertAlmostEqual(config["lr0"], 0.01)
        self.assertEqual(config["seed"], 42)


class ParseArgsTest(unittest.TestCase):
    def test_new_flag_defaults(self):
        args = parse_args(["--data", "x.yaml"])
        self.assertTrue(args.deterministic)
        self.assertEqual(args.code_revision, "")
        self.assertEqual(args.dataset_slug, "")
        self.assertEqual(args.dataset_version, "")
        self.assertFalse(args.exist_ok)
        self.assertEqual(args.seed, 42)

    def test_new_flags_accepted(self):
        args = parse_args([
            "--data", "x.yaml",
            "--no-deterministic",
            "--code-revision", "abc123",
            "--dataset-slug", "sentinel-weapon-v5",
            "--dataset-version", "3",
            "--exist-ok",
            "--seed", "7",
        ])
        self.assertFalse(args.deterministic)
        self.assertEqual(args.code_revision, "abc123")
        self.assertEqual(args.dataset_slug, "sentinel-weapon-v5")
        self.assertEqual(args.dataset_version, "3")
        self.assertTrue(args.exist_ok)
        self.assertEqual(args.seed, 7)


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class FindValImageTest(unittest.TestCase):
    def test_no_yaml_returns_none(self):
        self.assertIsNone(find_val_image(None))
        self.assertIsNone(find_val_image("/nonexistent/data.yaml"))

    def test_val_image_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_dir = root / "images" / "val"
            val_dir.mkdir(parents=True)
            (val_dir / "b.jpg").write_bytes(b"fake")
            (val_dir / "a.jpg").write_bytes(b"fake")
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                f"path: {root}\ntrain: images/train\nval: images/val\n"
            )
            self.assertEqual(find_val_image(str(yaml_path)), val_dir / "a.jpg")

    def test_missing_val_dir_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yaml_path = root / "data.yaml"
            yaml_path.write_text(
                f"path: {root}\nval: images/val\n"
            )
            self.assertIsNone(find_val_image(str(yaml_path)))


if __name__ == "__main__":
    unittest.main()
