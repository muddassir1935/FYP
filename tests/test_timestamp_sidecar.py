"""Unit tests for the recording timestamp sidecar (pure Python).

One JSONL line per recorded frame: {"frame_index": i, "ts": <monotonic
seconds>}, written next to the recording as
<recording>.timestamps.jsonl. The detection log joins on frame_index.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.timestamp_sidecar import TimestampSidecar, sidecar_path_for


class TestSidecarPath(unittest.TestCase):
    def test_name_appends_timestamps_jsonl(self) -> None:
        self.assertEqual(
            sidecar_path_for("/tmp/rec/demo_20261126_160000_raw.mp4"),
            Path("/tmp/rec/demo_20261126_160000_raw.mp4.timestamps.jsonl"),
        )


class TestTimestampSidecar(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.rec = Path(self._tmp.name) / "rec" / "demo.mp4"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _read_lines(self, path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_lines_match_recorded_frames(self) -> None:
        # A fake recording loop: 5 frames, packet timestamps.
        sidecar = TimestampSidecar(self.rec)
        frames = [(1, 100.0), (2, 100.1), (3, 100.25), (4, 100.3), (5, 100.5)]
        for index, ts in frames:
            sidecar.write(index, ts)
        sidecar.close()
        lines = self._read_lines(sidecar.path)
        self.assertEqual(len(lines), 5)
        self.assertEqual(
            [(line["frame_index"], line["ts"]) for line in lines], frames
        )

    def test_close_is_idempotent_and_write_after_close_noops(self) -> None:
        sidecar = TimestampSidecar(self.rec)
        sidecar.write(1, 100.0)
        sidecar.close()
        sidecar.close()
        sidecar.write(2, 100.1)  # no-op, must not raise or append
        lines = self._read_lines(sidecar.path)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["frame_index"], 1)

    def test_context_manager_closes(self) -> None:
        with TimestampSidecar(self.rec) as sidecar:
            sidecar.write(1, 100.0)
        lines = self._read_lines(sidecar.path)
        self.assertEqual(len(lines), 1)

    def test_unwritable_path_disables_sidecar_without_raising(self) -> None:
        # Parent is a file, so mkdir fails: writes become no-ops.
        blocker = Path(self._tmp.name) / "not-a-dir"
        blocker.write_text("x")
        sidecar = TimestampSidecar(blocker / "demo.mp4")
        self.assertTrue(sidecar.disabled)
        sidecar.write(1, 100.0)  # must not raise
        sidecar.close()
        self.assertFalse(blocker.joinpath("demo.mp4.timestamps.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
