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
import socket
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) != 0


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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
    )
    url = f"http://{args.host}:{port}"
    print("SENTINEL demo console")
    print(f"  Open {url}")
    print("  Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
