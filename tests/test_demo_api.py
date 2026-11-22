"""API tests for the demo web UI (FastAPI TestClient).

Skipped when FastAPI is not installed (e.g. a minimal checkout without
the web dependencies) — run `pip install -r requirements.txt` first.

Run from the repo root:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from fastapi.testclient import TestClient

    from src.backend import create_app

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "fastapi not installed")
class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.videos = root / "videos"
        self.weights = root / "weights"
        self.snaps = root / "snaps"
        self.videos.mkdir()
        self.weights.mkdir()
        (self.weights / "best.pt").write_bytes(b"not a model")
        (self.videos / "clip.mp4").write_bytes(b"not a video")
        app = create_app(
            videos_dir=self.videos,
            weights_dir=self.weights,
            snapshots_dir=self.snaps,
            save_video_dir=root / "rec",
        )
        self.app = app
        self.client = TestClient(app)
        self.runner = app.state.runner

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_index_serves_the_console(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("SENTINEL", res.text)
        self.assertIn("Start run", res.text)

    def test_static_files_served(self) -> None:
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/styles.css").status_code, 200)

    def test_health(self) -> None:
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["n_weights"], 1)
        self.assertEqual(body["n_videos"], 1)

    def test_sources_and_weights(self) -> None:
        sources = self.client.get("/api/sources").json()
        self.assertEqual([s["name"] for s in sources], ["clip.mp4"])
        weights = self.client.get("/api/weights").json()
        self.assertEqual(weights[0]["id"], "best.pt")
        self.assertEqual(weights[0]["kind"], "pytorch")

    def test_upload_accepts_video_rejects_other(self) -> None:
        res = self.client.post(
            "/api/upload",
            files={"file": ("new.mp4", b"fake video bytes", "video/mp4")},
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["name"], "uploads/new.mp4")
        self.assertTrue((self.videos / "uploads" / "new.mp4").is_file())

        res = self.client.post(
            "/api/upload",
            files={"file": ("evil.txt", b"nope", "text/plain")},
        )
        self.assertEqual(res.status_code, 400)

    def test_upload_strips_directory_parts(self) -> None:
        res = self.client.post(
            "/api/upload",
            files={"file": ("../escape.mp4", b"x", "video/mp4")},
        )
        self.assertEqual(res.status_code, 201)
        self.assertEqual(res.json()["name"], "uploads/escape.mp4")
        self.assertFalse((self.videos.parent / "escape.mp4").exists())

    def test_run_start_validation_errors(self) -> None:
        # Unknown weights -> 400.
        res = self.client.post(
            "/api/run/start",
            json={"source_type": "video", "source": "clip.mp4",
                  "weights_id": "ghost.pt"},
        )
        self.assertEqual(res.status_code, 400)
        # Bad confidence -> 422 (Pydantic).
        res = self.client.post(
            "/api/run/start",
            json={"source_type": "video", "source": "clip.mp4",
                  "weights_id": "best.pt", "conf_gun": 1.5},
        )
        self.assertEqual(res.status_code, 422)
        # Bad source type -> 422 (Pydantic pattern).
        res = self.client.post(
            "/api/run/start",
            json={"source_type": "rtsp", "source": "x",
                  "weights_id": "best.pt"},
        )
        self.assertEqual(res.status_code, 422)

    def test_run_start_conflict_when_running(self) -> None:
        self.runner._state = "running"
        res = self.client.post(
            "/api/run/start",
            json={"source_type": "video", "source": "clip.mp4",
                  "weights_id": "best.pt"},
        )
        self.assertEqual(res.status_code, 409)

    def test_run_start_conflict_when_starting(self) -> None:
        self.runner._state = "starting"
        res = self.client.post(
            "/api/run/start",
            json={"source_type": "video", "source": "clip.mp4",
                  "weights_id": "best.pt"},
        )
        self.assertEqual(res.status_code, 409)
        self.assertEqual(self.client.get("/api/run/status").json()["state"], "starting")

    def test_run_stop_when_idle(self) -> None:
        res = self.client.post("/api/run/stop")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["state"], "idle")

    def test_stream_without_run_is_503(self) -> None:
        self.assertEqual(self.client.get("/api/stream").status_code, 503)

    def test_shutdown_without_server_is_409(self) -> None:
        # create_app leaves uvicorn_server None (the plain
        # `uvicorn ...:main_app` launch): nothing to signal.
        res = self.client.post("/api/server/shutdown")
        self.assertEqual(res.status_code, 409)

    def test_shutdown_signals_attached_server(self) -> None:
        class FakeServer:
            should_exit = False

        server = FakeServer()
        self.app.state.uvicorn_server = server
        self.runner._state = "running"
        res = self.client.post("/api/server/shutdown")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json(), {"status": "shutdown_requested"})
        self.assertTrue(server.should_exit)
        # The active run is stopped first.
        self.assertEqual(self.runner.state, "stopped")

    def test_snapshots_flow(self) -> None:
        self.assertEqual(self.client.get("/api/snapshots").json(), [])
        self.snaps.mkdir(exist_ok=True)
        (self.snaps / "threat_Gun_20261121_120000_t1.jpg").write_bytes(b"x")
        snaps = self.client.get("/api/snapshots").json()
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["class"], "Gun")
        res = self.client.get("/snapshots/threat_Gun_20261121_120000_t1.jpg")
        self.assertEqual(res.status_code, 200)
        res = self.client.post("/api/snapshots/clear")
        self.assertEqual(res.json(), {"removed": 1})
        self.assertEqual(self.client.get("/api/snapshots").json(), [])

    def test_snapshot_traversal_is_404(self) -> None:
        self.assertEqual(
            self.client.get("/snapshots/..%2Fsecret.jpg").status_code, 404
        )
        self.assertEqual(
            self.client.get("/snapshots/nope.jpg").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
