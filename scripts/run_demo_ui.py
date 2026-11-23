#!/usr/bin/env python3
"""Start the SENTINEL demo web UI.

One command:  python scripts/run_demo_ui.py
Then open http://127.0.0.1:8000 (the browser opens automatically).

The server binds to localhost only — this is a demo console, not a
deployed service. Put video clips in videos/ and weights in
models/weights/ before starting (or upload a clip from the UI).
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _port_listening(host: str, port: int) -> bool:
    """True when something accepts TCP on (host, port)."""
    target = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    family = socket.AF_INET6 if ":" in target else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            return sock.connect_ex((target, port)) == 0
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SENTINEL demo web UI")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically.")
    parser.add_argument("--videos-dir", default=None,
                        help="Folder with video clips (default: videos/).")
    parser.add_argument("--weights-dir", default=None,
                        help="Folder with weights (default: models/weights/).")
    parser.add_argument("--snapshots-dir", default=None,
                        help="Folder for threat snapshots (default: snapshots/).")
    parser.add_argument("--save-video-dir", default=None,
                        help="Folder for recorded videos "
                        "(default: videos/recordings/).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # The runner logs run start/stop/error + per-run summaries through
    # the logging module; make sure they reach the terminal.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.host not in LOOPBACK_HOSTS:
        print(
            f"WARNING: binding to {args.host} — the demo console has no "
            "authentication, including on POST /api/server/shutdown. "
            "Only do this on a trusted network.",
            file=sys.stderr,
        )
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is not installed; run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1
    from src.backend import create_app

    port = args.port
    if not _port_free(port):
        for candidate in range(args.port + 1, args.port + 11):
            if _port_free(candidate):
                print(f"Port {port} is busy; using {candidate} instead.")
                port = candidate
                break
        else:
            print(f"No free port near {args.port}; pick another with --port.",
                  file=sys.stderr)
            return 1

    app = create_app(
        videos_dir=Path(args.videos_dir) if args.videos_dir else None,
        weights_dir=Path(args.weights_dir) if args.weights_dir else None,
        snapshots_dir=Path(args.snapshots_dir) if args.snapshots_dir else None,
        save_video_dir=Path(args.save_video_dir) if args.save_video_dir else None,
    )
    # Own the Server object (instead of uvicorn.run) so the console's
    # Stop server button can shut the process down via should_exit.
    config = uvicorn.Config(app, host=args.host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    app.state.uvicorn_server = server
    url = f"http://{args.host}:{port}"
    print("SENTINEL demo console")
    print(f"  Open {url}")
    print("  Stop: Ctrl+C here, or the Stop server button in the console.")
    # Run the server on a thread so the browser opens only AFTER the
    # server has bound the port (it used to open first, so the first
    # load could hit connection-refused on a slow machine). Waiting for
    # the bind also settles the _port_free check-then-bind race: if the
    # port was taken in between, the server thread dies and we report
    # it instead of opening a dead URL.
    thread = threading.Thread(
        target=server.run, name="sentinel-uvicorn", daemon=True
    )
    thread.start()
    bound = False
    for _ in range(100):  # up to ~10 s for a slow first bind
        if not thread.is_alive():
            break
        if _port_listening(args.host, port):
            bound = True
            break
        time.sleep(0.1)
    if not bound:
        print(
            f"The server did not start listening on {url} — is the port "
            "taken by another process?",
            file=sys.stderr,
        )
        return 1
    if not args.no_browser:
        webbrowser.open(url)
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        server.should_exit = True
        thread.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
