#!/usr/bin/env python3
"""Pre-demo smoke test for the SENTINEL demo web UI.

Starts the server on a throwaway port, checks every endpoint, runs a
short clip end to end (stream frame + stats + pause/resume), checks
upload/delete and the missing-weights error, and prints PASS/FAIL per
check. Run it the morning of the review:

    python scripts/smoke_demo_ui.py

Exit code 0 means every check passed. A failed check prints what to
fix (e.g. "no weights in models/weights/") plus the tail of the server
log, which is captured to a temp file for exactly this purpose.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKS: list[tuple[str, bool, str]] = []
LOG_TAIL_LINES = 30


def check(name: str, ok: bool, extra: str = "") -> None:
    CHECKS.append((name, ok, extra))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")


def print_log_tail(log_path: Path, lines: int = LOG_TAIL_LINES) -> None:
    """Print the last lines of the captured server log (diagnostics
    for a failed smoke run — the log used to be discarded entirely)."""
    try:
        text = log_path.read_text(errors="replace").splitlines()
    except OSError:
        print(f"(server log {log_path} unavailable)")
        return
    print(f"--- server log tail ({log_path}) ---")
    for line in text[-lines:]:
        print(line)
    print("--- end of server log ---")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the demo web UI")
    parser.add_argument("--port", type=int, default=8129)
    parser.add_argument("--video", default=None,
                        help="Clip to run (default: the shortest in videos/).")
    parser.add_argument("--weights", default=None,
                        help="Weights id to run (default: the first listed).")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Seconds to wait for the run to produce frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        import httpx
    except ImportError:
        print("httpx is not installed; run: pip install -r requirements.txt",
              file=sys.stderr)
        return 1

    base = f"http://127.0.0.1:{args.port}"
    # Capture the server's output (unbuffered, -u) to a temp log file
    # instead of discarding it: on failure the tail below is the only
    # diagnostic the morning-of-review operator gets.
    log_file = tempfile.NamedTemporaryFile(
        prefix="smoke_demo_ui_", suffix=".log", delete=False, mode="wb"
    )
    log_path = Path(log_file.name)
    print(f"Server log: {log_path}")
    # Smoke artifacts (snapshots, recorded videos, JSONL logs) go in
    # a temp dir, NOT the repo's real snapshots/, videos/recordings/
    # and logs/ — otherwise a morning-of-review smoke run leaves its
    # recordings sitting in the jury demo UI.
    smoke_tmp = tempfile.TemporaryDirectory(prefix="smoke_demo_ui_")
    server = subprocess.Popen(
        [sys.executable, "-u", str(REPO_ROOT / "scripts" / "run_demo_ui.py"),
         "--port", str(args.port), "--no-browser",
         "--snapshots-dir", str(Path(smoke_tmp.name) / "snapshots"),
         "--save-video-dir", str(Path(smoke_tmp.name) / "recordings"),
         "--log-dir", str(Path(smoke_tmp.name) / "logs")],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        # Localhost traffic must never go through a proxy — and
        # httpx's env-proxy parsing chokes on bracketed IPv6 entries
        # in NO_PROXY, so ignore the environment entirely.
        client = httpx.Client(base_url=base, timeout=30.0, trust_env=False)
        # -- wait for the server --
        healthy = False
        for _ in range(30):
            try:
                if client.get("/api/health").status_code == 200:
                    healthy = True
                    break
            except httpx.RequestError:
                pass
            time.sleep(1)
        check("server starts", healthy)
        if not healthy:
            print("The server did not come up — see the server log tail below.")
            failed = [name for name, ok, _ in CHECKS if not ok]
            print(f"FAILED: {len(failed)} check(s): {failed}")
            return 1

        # -- listings --
        weights = client.get("/api/weights").json()
        check("weights listed", len(weights) > 0,
              f"({len(weights)} found)" if weights
              else "— place best.pt in models/weights/")
        videos = client.get("/api/sources").json()
        check("videos listed", len(videos) > 0,
              f"({len(videos)} found)" if videos
              else "— put a clip in videos/ or upload one from the UI")
        page = client.get("/")
        check("console page loads",
              page.status_code == 200 and "SENTINEL" in page.text)
        check("snapshots endpoint",
              client.get("/api/snapshots").status_code == 200)
        defaults = client.get("/api/defaults")
        check("defaults endpoint", defaults.status_code == 200
              and "conf_gun" in defaults.json())

        if not weights or not videos:
            print("Skipping the live run: no weights or no videos.")
        else:
            # -- a bad start is a client error, not a crash --
            bad = client.post("/api/run/start", json={
                "source_type": "video", "source": videos[0]["name"],
                "weights_id": "no-such-weights.pt",
            })
            check("start with missing weights is 400", bad.status_code == 400,
                  bad.text[:120] if bad.status_code != 400 else "")

            # -- live run on the shortest clip (with recording on, so the
            # saved-video check below actually runs) --
            video = args.video or min(
                videos, key=lambda v: v["duration_s"] or 1e9
            )["name"]
            weights_id = args.weights or weights[0]["id"]
            # Model load + warmup can take a while on first start
            # (OpenVINO compile, slow disk), so the start call gets a
            # longer timeout than the 30 s client default — and a
            # failure here is a failed check, not a crash.
            res = None
            try:
                res = client.post("/api/run/start", json={
                    "source_type": "video", "source": video,
                    "weights_id": weights_id, "save_video": True,
                }, timeout=180.0)
            except httpx.RequestError as exc:
                check("run starts", False, f"(request failed: {exc})")
            else:
                check("run starts", res.status_code == 200,
                      res.text[:120] if res.status_code != 200 else f"({video})")
            if res is not None and res.status_code == 200:
                # -- one MJPEG frame --
                got_frame = False
                try:
                    with client.stream("GET", "/api/stream", timeout=30.0) as stream:
                        buf = b""
                        for chunk in stream.iter_bytes():
                            buf += chunk
                            if b"\xff\xd8" in buf and b"\xff\xd9" in buf:
                                got_frame = True
                                break
                            if len(buf) > 5_000_000:
                                break
                except httpx.RequestError as exc:
                    print(f"stream error: {exc}")
                check("MJPEG stream delivers a frame", got_frame)

                # -- stats advance --
                frames_seen = 0
                status = {}
                t0 = time.time()
                while time.time() - t0 < args.timeout:
                    status = client.get("/api/run/status").json()
                    frames_seen = max(frames_seen, status["frames"])
                    if status["state"] != "running" or frames_seen > 10:
                        break
                    time.sleep(1)
                check("frames advance", frames_seen > 10, f"(frames={frames_seen})")

                # -- pause -> resume -> stop mid-run --
                if status.get("state") == "running":
                    paused = client.post("/api/run/pause")
                    check("run pauses",
                          paused.status_code == 200
                          and paused.json()["state"] == "paused",
                          paused.text[:120] if paused.status_code != 200 else "")
                    resumed = client.post("/api/run/resume")
                    check("run resumes",
                          resumed.status_code == 200
                          and resumed.json()["state"] == "running",
                          resumed.text[:120] if resumed.status_code != 200 else "")
                else:
                    print("Run already finished — skipping pause/resume checks.")

                # -- stop --
                stopped = client.post("/api/run/stop").json()
                check("run stops", stopped["state"] in ("stopped", "idle"),
                      f"(state={stopped['state']})")
                check("saved video recorded", bool(stopped.get("saved_video")),
                      stopped.get("saved_video")
                      or "— save_video was on but no file was recorded")

                # -- upload + delete a clip --
                try:
                    import cv2
                    import numpy as np
                except ImportError:
                    cv2 = None
                if cv2 is None:
                    print("Skipping upload/delete checks: cv2 not installed.")
                else:
                    with tempfile.TemporaryDirectory() as tmp:
                        clip = Path(tmp) / "smoke_upload.mp4"
                        writer = cv2.VideoWriter(
                            str(clip), cv2.VideoWriter_fourcc(*"mp4v"),
                            10.0, (64, 48),
                        )
                        for _ in range(8):
                            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
                        writer.release()
                        with open(clip, "rb") as fh:
                            up = client.post(
                                "/api/upload",
                                files={"file": ("smoke_upload.mp4", fh, "video/mp4")},
                            )
                        check("upload a clip", up.status_code == 201,
                              up.text[:120] if up.status_code != 201 else "")
                        if up.status_code == 201:
                            name = up.json()["name"]
                            deleted = client.delete(f"/api/sources/{name}")
                            check("delete the uploaded clip",
                                  deleted.status_code == 200,
                                  deleted.text[:120]
                                  if deleted.status_code != 200 else "")

        # -- server shutdown (the console's off-switch) --
        shut = client.post("/api/server/shutdown")
        check("server shutdown endpoint", shut.status_code == 200,
              shut.text[:120] if shut.status_code != 200 else "")
        exited = False
        try:
            server.wait(timeout=10)
            exited = True
        except subprocess.TimeoutExpired:
            pass
        check("server exits after shutdown", exited)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        log_file.close()
        smoke_tmp.cleanup()
        # The tail prints here (not at the bottom) so even the
        # early-return failure paths get diagnostics.
        if any(not ok for _, ok, _ in CHECKS):
            print_log_tail(log_path)

    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failed:
        print(f"FAILED: {len(failed)} check(s): {failed}")
        return 1
    print("ALL SMOKE CHECKS PASSED — the demo console is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
