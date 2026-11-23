# Copyright (c) 2026 Muddassir Rizwan, Faiz Hussain, Qaim Mehdi. Licensed under the MIT License.
"""FastAPI app for the SENTINEL demo console.

Serves the vanilla frontend in ``src/frontend/`` plus the ``/api/*``
routes in ``docs/demo_ui_plan.md`` §5. Created through
:func:`create_app` so tests can inject temporary directories.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .runner import SNAPSHOT_RE, VIDEO_EXTENSIONS, DemoRunner, RunConfig
from .schemas import (
    DemoDefaults,
    HealthStatus,
    RunStartRequest,
    RunStatus,
    SnapshotEntry,
    VideoEntry,
    WeightsEntry,
)
from src.pipeline.thresholds import as_dict as operating_point_dict

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "src" / "frontend"
MAX_UPLOAD_MB = 500


def create_app(
    *,
    videos_dir: Path | None = None,
    weights_dir: Path | None = None,
    snapshots_dir: Path | None = None,
    save_video_dir: Path | None = None,
) -> FastAPI:
    videos_dir = Path(videos_dir) if videos_dir else REPO_ROOT / "videos"
    weights_dir = Path(weights_dir) if weights_dir else REPO_ROOT / "models" / "weights"
    snapshots_dir = (
        Path(snapshots_dir) if snapshots_dir else REPO_ROOT / "snapshots"
    )
    save_video_dir = (
        Path(save_video_dir)
        if save_video_dir
        else REPO_ROOT / "videos" / "recordings"
    )
    videos_dir.mkdir(parents=True, exist_ok=True)

    runner = DemoRunner(
        videos_dir=videos_dir,
        weights_dir=weights_dir,
        snapshots_dir=snapshots_dir,
        save_video_dir=save_video_dir,
    )

    app = FastAPI(title="SENTINEL Demo Console", version="0.1.0")
    app.state.runner = runner
    # The uvicorn Server object, when one owns this app. Set by
    # scripts/run_demo_ui.py; stays None under a plain
    # `uvicorn src.backend.app:main_app` launch, where /api/server/
    # shutdown has nothing to signal and answers 409.
    app.state.uvicorn_server = None

    # -- pages & static -----------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        page = FRONTEND_DIR / "index.html"
        if not page.is_file():
            return JSONResponse(
                {"detail": "Frontend not built: src/frontend/index.html is missing."},
                status_code=500,
            )
        return FileResponse(page, media_type="text/html")

    if FRONTEND_DIR.is_dir():
        app.mount(
            "/static", StaticFiles(directory=FRONTEND_DIR), name="static"
        )

    # -- health & listings --------------------------------------------
    @app.get("/api/health", response_model=HealthStatus)
    def health():
        weights = runner.list_weights()
        videos = runner.list_videos()
        return HealthStatus(
            weights_dir_exists=weights_dir.is_dir(),
            videos_dir_exists=videos_dir.is_dir(),
            n_weights=len(weights),
            n_videos=len(videos),
        )

    @app.get("/api/defaults", response_model=DemoDefaults)
    def defaults():
        # The shipped operating point, from the single source of truth
        # (src/pipeline/thresholds.py). The frontend fills its inputs
        # from here instead of hardcoding thresholds of its own.
        return DemoDefaults(**operating_point_dict())

    @app.get("/api/sources", response_model=list[VideoEntry])
    def sources():
        return [VideoEntry(**v) for v in runner.list_videos()]

    @app.delete("/api/sources/{name:path}")
    def delete_source(name: str):
        try:
            deleted = runner.delete_uploaded_video(name)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except PermissionError as exc:
            # Unlinking an open file (Windows): the video is in use.
            raise HTTPException(
                409, f"Cannot delete {name!r}: the file is in use."
            ) from exc
        except OSError as exc:
            raise HTTPException(500, f"Could not delete {name!r}: {exc}") from exc
        return {"deleted": deleted}

    @app.get("/api/weights", response_model=list[WeightsEntry])
    def weights():
        return [WeightsEntry(**w) for w in runner.list_weights()]

    @app.post("/api/upload", response_model=VideoEntry, status_code=201)
    def upload(file: UploadFile = File(...)):
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in VIDEO_EXTENSIONS:
            raise HTTPException(
                400,
                f"Not a video file: {file.filename!r} "
                f"(allowed: {', '.join(sorted(VIDEO_EXTENSIONS))}).",
            )
        uploads_dir = videos_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        # Keep only the bare filename: no directory parts from the client.
        name = Path(file.filename or "upload").name
        target = uploads_dir / name
        if target.exists():
            stem, ext = target.stem, target.suffix
            target = uploads_dir / f"{stem}_{int(time.time())}{ext}"
        limit = MAX_UPLOAD_MB * 1_000_000
        written = 0
        try:
            with open(target, "wb") as fh:
                while True:
                    chunk = file.file.read(1 << 20)
                    if not chunk:
                        break
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(
                            413, f"Upload exceeds the {MAX_UPLOAD_MB} MB cap."
                        )
                    fh.write(chunk)
        except HTTPException:
            target.unlink(missing_ok=True)
            raise
        except OSError as exc:
            target.unlink(missing_ok=True)
            raise HTTPException(500, f"Could not save the upload: {exc}") from exc
        # Probe the content, not just the extension: a .txt renamed to
        # .mp4 used to upload with 201 and fail only at run start.
        duration = DemoRunner._probe_duration(target)
        if duration is None:
            target.unlink(missing_ok=True)
            raise HTTPException(
                400,
                f"OpenCV could not open {file.filename!r} as a video — "
                "the file is not a playable video.",
            )
        return VideoEntry(
            name=target.relative_to(videos_dir).as_posix(),
            size_mb=round(written / 1_000_000, 1),
            duration_s=duration,
        )

    # -- run control ---------------------------------------------------
    @app.post("/api/run/start", response_model=RunStatus)
    def run_start(body: RunStartRequest):
        if runner.state in ("starting", "running", "paused"):
            raise HTTPException(409, "A run is already active — stop it first.")
        config = RunConfig(
            source_type=body.source_type,
            source=body.source,
            weights_id=body.weights_id,
            conf_gun=body.conf_gun,
            conf_knife=body.conf_knife,
            imgsz=body.imgsz,
            min_hits=body.min_hits,
            tracker=body.tracker,
            loop=body.loop,
            hide_unconfirmed=body.hide_unconfirmed,
            save_video=body.save_video,
            record_raw=body.record_raw,
        )
        try:
            info = runner.start(config)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(500, str(exc)) from exc
        return RunStatus(**vars(info))

    @app.post("/api/run/stop", response_model=RunStatus)
    def run_stop():
        return RunStatus(**vars(runner.stop()))

    @app.post("/api/run/pause", response_model=RunStatus)
    def run_pause():
        try:
            info = runner.pause()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return RunStatus(**vars(info))

    @app.post("/api/run/resume", response_model=RunStatus)
    def run_resume():
        try:
            info = runner.resume()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        return RunStatus(**vars(info))

    @app.get("/api/run/status", response_model=RunStatus)
    def run_status():
        return RunStatus(**vars(runner.status()))

    # -- server ---------------------------------------------------------
    @app.post("/api/server/shutdown")
    def server_shutdown():
        """Stop the active run (if any) and shut the server down.

        Closing the browser tab does not stop the process, so the
        console needs its own off-switch. uvicorn finishes this
        response, then exits gracefully on ``should_exit``.
        """
        runner.stop()
        server = getattr(app.state, "uvicorn_server", None)
        if server is None:
            raise HTTPException(
                409,
                "This server was not started by run_demo_ui.py — "
                "stop it from the terminal (Ctrl+C).",
            )
        server.should_exit = True
        return {"status": "shutdown_requested"}

    # -- stream ---------------------------------------------------------
    @app.get("/api/stream", include_in_schema=False)
    def stream():
        if runner.state not in ("running", "paused"):
            return JSONResponse(
                {"detail": "No run is active — start one first."}, status_code=503
            )

        def generate():
            last_seq = -1
            while runner.state in ("running", "paused"):
                seq, jpeg = runner.latest_frame()
                if jpeg is not None and seq != last_seq:
                    last_seq = seq
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                else:
                    time.sleep(0.01)

        return StreamingResponse(
            generate(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    # -- snapshots ------------------------------------------------------
    @app.get("/api/snapshots", response_model=list[SnapshotEntry])
    def snapshots():
        return [
            SnapshotEntry(
                file=s["file"],
                **{"class": s["class"]},
                track=s["track"],
                timestamp=s["timestamp"],
                url=s["url"],
            )
            for s in runner.list_snapshots()
        ]

    @app.get("/snapshots/{name}", include_in_schema=False)
    def snapshot_file(name: str):
        # Only files the worker actually writes (threat_<class>_<stamp>_t<id>.jpg):
        # same regex as the listing, so a stray .jpg in the dir is not served.
        if not SNAPSHOT_RE.match(name):
            raise HTTPException(404, "Snapshot not found.")
        path = snapshots_dir / name
        if not path.is_file():
            raise HTTPException(404, "Snapshot not found.")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/api/snapshots/clear")
    def snapshots_clear():
        return {"removed": runner.clear_snapshots()}

    return app


def main_app() -> FastAPI:
    """Default app for ``uvicorn src.backend.app:main_app`` (factory form)."""
    return create_app()
