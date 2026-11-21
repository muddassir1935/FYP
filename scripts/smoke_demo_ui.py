#!/usr/bin/env python3
"""Pre-demo smoke test for the SENTINEL demo web UI.

Starts the server on a throwaway port, checks every endpoint, runs a
short clip end to end (stream frame + stats), and prints PASS/FAIL per
check. Run it the morning of the review:

    python scripts/smoke_demo_ui.py

Exit code 0 means every check passed. A failed check prints what to
fix (e.g. "no weights in models/weights/").
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, extra: str = "") -> None:
    CHECKS.append((name, ok, extra))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {extra}")


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
    server = subprocess.Popen(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_demo_ui.py"),
         "--port", str(args.port), "--no-browser"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
            print("The server did not come up — see the log above.")
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

        if not weights or not videos:
            print("Skipping the live run: no weights or no videos.")
            return 0 if all(ok for _, ok, _ in CHECKS) else 1

        # -- live run on the shortest clip --
        video = args.video or min(
            videos, key=lambda v: v["duration_s"] or 1e9
        )["name"]
        weights_id = args.weights or weights[0]["id"]
        res = client.post("/api/run/start", json={
            "source_type": "video", "source": video,
            "weights_id": weights_id,
        })
        check("run starts", res.status_code == 200,
              res.text[:120] if res.status_code != 200 else f"({video})")
        if res.status_code != 200:
            return 1

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
        t0 = time.time()
        while time.time() - t0 < args.timeout:
            status = client.get("/api/run/status").json()
            frames_seen = max(frames_seen, status["frames"])
            if status["state"] != "running" or frames_seen > 10:
                break
            time.sleep(1)
        check("frames advance", frames_seen > 10, f"(frames={frames_seen})")

        # -- stop --
        stopped = client.post("/api/run/stop").json()
        check("run stops", stopped["state"] in ("stopped", "idle"),
              f"(state={stopped['state']})")
        if stopped.get("saved_video"):
            check("saved video recorded", True, stopped["saved_video"])
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()

    failed = [name for name, ok, _ in CHECKS if not ok]
    print()
    if failed:
        print(f"FAILED: {len(failed)} check(s): {failed}")
        return 1
    print("ALL SMOKE CHECKS PASSED — the demo console is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
