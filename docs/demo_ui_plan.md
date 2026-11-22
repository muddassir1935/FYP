# Demo Web UI — Plan & Design (FYP-I)

**Status:** Plan approved by user (2026-11-21). Build in progress on
`feature/demo-web-ui`.
**Scope change (user decision, 2026-11-21):** FYP-I originally had no web
UI (pipeline + standalone OpenCV demo only). The user asked for a complete
demo UI now, because the jury demo cannot be run from terminal commands —
and this work has to be built for FYP-II anyway. This UI is the foundation
of the FYP-II dashboard (FR-06/FR-07), built one step at a time.

---

## 1. Problem

The current demo (`scripts/inference/run_weapon_demo.py`) is driven entirely
from the terminal: every source change, weights change, or threshold tweak
means retyping a command with flags. For a jury demo that is slow, fragile,
and looks unfinished. The OpenCV window is also sized by the video
resolution, and going fullscreen leaves bars around the frame.

The demo UI replaces the terminal with a single browser page: pick a video
or webcam, pick weights, set thresholds, press Start, watch the annotated
stream with live stats, and review confirmed-threat snapshots — no commands.

---

## 2. Scope

### In scope (FYP-I demo UI v1)

- Start/stop a detection run from the browser: source (webcam or video
  file), weights (.pt or OpenVINO folder), per-class confidence thresholds,
  imgsz, min-hits, loop, hide-unconfirmed.
- Live annotated video stream in the browser, with a proper fullscreen
  mode (no bars).
- Live stats: display FPS, detect FPS, frames processed, detections now,
  confirmed threats, run time.
- Confirmed-threat alert banner + audible siren (Web Audio, off by default
  until the user enables sound — browsers block autoplay audio).
- Snapshot gallery of confirmed threats (from the snapshots dir).
- Video upload (short clips) + a server-side `videos/` folder listing.
- Run logging + health endpoint + a pre-demo smoke script.

### Out of scope (FYP-II, documented as hooks — see §9)

- WebSocket streaming (v1 uses MJPEG; WebSocket is the FYP-II upgrade).
- React/Vite frontend (v1 is vanilla HTML/CSS/JS, no build step).
- SQLite incident log, search/filter history (FR-08/FR-10).
- Email alerts (FR-09).
- Pose/LSTM behavior stream (FR-03/FR-04).
- RTSP input, multi-camera (out of scope per proposal).
- Authentication (local demo only, binds to localhost).

---

## 3. Decisions

| # | Decision | Why | Rejected alternative |
|---|----------|-----|----------------------|
| D1 | **FastAPI** backend in `src/backend/` | The approved proposal already specifies a FastAPI backend (FR-06/07); this is Faiz's area and the FYP-II direction. | Flask (no async story), stdlib http.server (too primitive for uploads + streaming). |
| D2 | **Vanilla HTML/CSS/JS** frontend in `src/frontend/`, no build step | Zero build step = one command to run the demo; repo rule is Vanilla CSS only. React is the FYP-II step and reuses this API. | React+Vite now (needs node + build on the demo laptop), Streamlit/Gradio (generic look, heavy deps, leads nowhere). |
| D3 | **MJPEG** stream (`/api/stream`) | Simple, robust, works in every browser with an `<img>` tag; boxes burned in server-side like the CLI demo. | WebSocket + Canvas overlays (the FYP-II design; more moving parts than v1 needs). |
| D4 | **Shared worker refactor**: move `InferenceWorker` + `draw_annotations` from the demo script into `src/pipeline/` (`demo_worker.py`, `annotate.py`) | One inference engine for the CLI demo and the web UI — no duplicated glue. The CLI demo imports it and behaves identically. | Duplicating the worker in the web backend (two engines to maintain and drift). |
| D5 | **Reuse `RollingFps`** (FPS fix) | The UI shows FPS; the 1/dt EMA spikes to hundreds. The FPS fix commit is cherry-picked onto this branch and rebased away after it merges. | Re-implementing FPS in the runner (duplication). |
| D6 | **Videos from `videos/` + upload** | A web page cannot browse the server's disk, and the browser cannot hand the server a local path. A `videos/` folder (gitignored media) + upload covers both. | File-path text input (fragile, leaks server paths, bad UX). |
| D7 | **Weights picker scans `models/weights/`** | The weights already live there (.pt + OpenVINO dirs). The picker lists what exists; missing weights is a clear UI error, not a crash. | Hardcoded weights list (goes stale with every retrain). |
| D8 | **Status via polling** (`GET /api/run/status` every 500 ms) | Simple and robust for v1 stats/alerts. | SSE/WebSocket (FYP-II upgrade; polling is fine at 2 Hz). |
| D9 | **Snapshots from the filesystem** (no DB in v1) | The pipeline already writes snapshots with timestamps; the UI lists them. SQLite incident log is FR-08 (FYP-II). | Building the DB now (scope creep; the FYP-II design owns it). |
| D10 | **One-command start**: `python scripts/run_demo_ui.py` | Opens `http://127.0.0.1:8000` in the browser. Localhost only — this is a demo console, not a deployed service. | Requiring manual uvicorn + browser steps (the exact friction this UI removes). |

**New dependencies** (user to be told): `fastapi`, `uvicorn`,
`python-multipart` (uploads), `httpx` (test client). Recorded in
`requirements.txt`. Everything else (ultralytics, opencv, numpy) is
already required by the pipeline.

---

## 4. Architecture

```
Browser (src/frontend: index.html + styles.css + app.js)
   │  HTTP: page, MJPEG stream, REST polling, snapshot images
   ▼
FastAPI app (src/backend/app.py)
   │  /api/* routes, static files, snapshot files
   ▼
DemoRunner (src/backend/runner.py) — one run at a time
   │  owns the worker thread, latest JPEG, stats, snapshots
   ▼
InferenceWorker (src/pipeline/demo_worker.py) — shared with the CLI demo
   │  FrameSource → WeaponDetector → TrackClassVoter → RollingFps
   ▼
Pipeline (src/pipeline/): capture, weapon_detector, track_voting,
temporal_buffer (kept for RQ3), rolling_fps, annotate
```

One run at a time: starting a new run stops the current one. The runner
holds the latest annotated JPEG (encoded once, served to all stream
clients) and a stats dict. The worker thread does inference; FastAPI
never blocks the event loop on inference (PROJECT_CONTEXT §7.2).

---

## 5. API design

| Method & path | Purpose | Notes |
|---|---|---|
| `GET /` | Serve the UI | `src/frontend/index.html` |
| `GET /static/*` | CSS/JS | Mounted static dir |
| `GET /api/health` | Liveness | `{status: "ok", weights_dir_exists, videos_dir_exists}` |
| `GET /api/sources` | List video files in `videos/` + webcam option | `{videos: [{name, size_mb, duration_s?}], webcams: [0]}` — duration only if cheap to read (cv2 probe, cached). |
| `DELETE /api/sources/{name}` | Delete an uploaded video | Only files inside `videos/uploads/` (400 otherwise / traversal); 404 if missing; 409 if used by an active/paused run. |
| `GET /api/weights` | List available weights | `{weights: [{id, kind: "pytorch"\|"openvino", label, path}]}` — `.pt` files + `*_openvino_model` dirs. |
| `POST /api/upload` | Upload a video clip | Multipart; saved to `videos/uploads/`; size cap 500 MB; only video extensions. |
| `POST /api/run/start` | Start a run | Body: `{source_type, source, weights_id, conf_gun, conf_knife, imgsz, min_hits, loop, hide_unconfirmed, save_video}`. Enters `starting` while loading/warming up the detector. 409 if a run is already starting/running/paused. |
| `POST /api/run/pause` | Pause the active run | Holds video file position (or freezes webcam display), pauses inference and elapsed timer, keeps `/api/stream` alive. 409 if not running. |
| `POST /api/run/resume` | Resume a paused run | Resumes from the paused position and timer. 409 if not paused. |
| `POST /api/run/stop` | Stop the run | Idempotent (works from running or paused); returns the final stats. |
| `GET /api/run/status` | Run state + live stats | `{state: idle\|starting\|running\|paused\|stopped\|error, source, weights, display_fps, detect_fps, frames, detections, confirmed, started_at, elapsed_s, error}`. |
| `GET /api/stream` | MJPEG annotated stream | `multipart/x-mixed-replace`; active during `running` and `paused` (holds last frame when paused); 503 with a JSON body otherwise. |
| `GET /api/snapshots` | List snapshots (newest first) | `{snapshots: [{file, class, track, timestamp, url}]}` — parsed from filenames. |
| `GET /snapshots/{file}` | Serve one snapshot image | Path-traversal guarded (name must be inside the snapshots dir). |
| `POST /api/snapshots/clear` | Delete all snapshots | For a clean demo start. Asks for confirmation in the UI. |

Validation: unknown weights id → 400; missing video file → 400; bad
thresholds (outside 0–1) → 422 (FastAPI/Pydantic). Errors return
`{detail: "..."}` and the UI shows the message in place — never a blank
page or a traceback.

---

## 6. UI design

### Design direction (per the frontend_design skill: two passes)

**Pass 1 — plan.** The subject is a security-operations console, not a
SaaS landing page. The video is the hero: it takes the whole left ~70%
of the screen, and everything else is a quiet instrument panel. The
visual language comes from CCTV/ops rooms: dark, dense, monospace data
readouts, one signal colour reserved for confirmed threats. Boldness is
spent in one place — the alert state (a threat banner that turns the
whole top edge red). Everything else stays restrained.

**Tokens:**
- Base: `#0A0E14` page, `#11161F` panels, `#1A212C` raised surfaces,
  `#232C3A` borders.
- Text: `#E6EAF0` primary, `#8B94A3` secondary.
- Signal: `#E5484D` confirmed threat (used ONLY for threats),
  `#46A758` running/healthy, `#F5A524` warning/idle attention.
- Type: system sans stack for UI (`-apple-system, "Segoe UI", Roboto,
  sans-serif`); `ui-monospace, "Cascadia Mono", Consolas` for FPS,
  timestamps, and data readouts. No webfonts — the demo must work
  offline on the jury laptop.
- Radius: 10px panels, 6px controls. No gradients, no shadows-as-
  decoration; borders carry the structure.

**Pass 2 — review against the brief.** A near-black page with a single
red accent risks reading as the generic "dark dashboard" default (skill
trait #2). What makes this specific to SENTINEL: the layout is driven
by the surveillance workflow (video first, controls second, incident
tape third), the red is semantically reserved (it means "confirmed
threat", never decoration), and the readouts use the pipeline's real
vocabulary (detect FPS, track votes, min-hits). Kept: dark ops console.
Cut: any hero headline, marketing copy, or card-grid chrome — this is a
tool, not a page.

### Layout (desktop, ≥1100px)

```
┌──────────────────────────────────────────────────────────────────┐
│ SENTINEL · Demo Console              [● RUNNING  00:42]  [Stop]  │
├──────────────────────────────────────────────────┬───────────────┤
│                                                  │ SOURCE        │
│                                                  │ ○ Webcam  ○ Video│
│   VIDEO (annotated stream, fills panel,          │ [Test4.mp4 ▾] │
│   aspect kept, black bars only if the            │ [Upload clip] │
│   video itself is narrower)                      │               │
│   [⛶ fullscreen]  [REC ● when saving]           │ WEIGHTS       │
│                                                  │ [v3 best.pt ▾]│
│                                                  │ THRESHOLDS    │
│   ┌────────────────────────────────────────┐     │ Gun   [0.44]  │
│   │ ⚠ CONFIRMED: Knife — track 3 (only     │     │ Knife [0.43]  │
│   │    when a threat is confirmed)         │     │               │
│   └────────────────────────────────────────┘     │ OPTIONS       │
│                                                  │ imgsz [640]   │
│   FPS 30.0 · detect 28.4 · frame 1,204 · 2 now  │ min-hits [3]  │
│   confirmed 5                                    │ ☑ loop ☐ hide │
├──────────────────────────────────────────────────┤   unconfirmed │
│ SNAPSHOTS (newest first, click to enlarge)       │ ☑ save video  │
│ [img] [img] [img] [img] [img] ...   [Clear]      │               │
│                                                  │ [▶ Start run] │
└──────────────────────────────────────────────────┴───────────────┘
```

- **Fullscreen fix:** the video panel is a CSS container with
  `aspect-ratio: 16 / 9` and `object-fit: contain` on the stream image;
  the fullscreen button uses the browser Fullscreen API on the panel,
  so the frame scales to the screen with no stray bars. (The old
  OpenCV window sized itself to the video; the browser panel sizes
  itself to the screen.)
- **Narrow screens (<1100px):** the control panel moves below the
  video; snapshots stay a horizontal scroll strip. Mobile is not a
  target (jury demo is on a laptop) but nothing may break.
- **States:** idle (video panel shows "No run — pick a source and
  press Start"), running, stopped (last frame + final stats kept),
  error (message in the panel + in the status line, with what to do).
  Empty snapshots: "No confirmed threats yet — snapshots appear here."
- **Siren:** a sound toggle, off by default; when on, a confirmed
  threat plays a short Web Audio siren. Browsers require a user
  gesture before audio, so the toggle itself is the gesture.
- **Accessibility:** visible keyboard focus, labelled controls, status
  text (not colour alone) for run state, `prefers-reduced-motion`
  respected (no flashing banner animation).

### Copy rules (from the skill)

Plain verbs, sentence case, no filler. The Start button says "Start
run"; the status then says "Running". Errors say what happened and what
to do ("Weights file not found — place best.pt in models/weights/").

---

## 7. Build plan

Branch: `feature/demo-web-ui` (from `origin/main` + cherry-picked FPS
fix commit `fe5d924` as `199a0ab`; rebase onto main after the FPS fix
merges so the duplicate drops out).

**Step 1 — shared worker refactor (no behaviour change).**
- New `src/pipeline/annotate.py`: `draw_annotations` moved unchanged.
- New `src/pipeline/demo_worker.py`: `InferenceWorker` moved unchanged
  (imports `RollingFps`, `draw_annotations`).
- `scripts/inference/run_weapon_demo.py`: import from the pipeline;
  `draw_hud` stays (CLI-only). Verify: full test suite green + module
  import + headless worker smoke test on a real video file.
- `src/pipeline/__init__.py`: export `RollingFps` (done in FPS fix),
  `DemoWorker`, `draw_annotations`.

**Step 2 — backend (`src/backend/`).**
- `runner.py`: `DemoRunner` — start/stop, config validation, latest
  JPEG + stats, snapshot listing, save-video writer. One run at a time.
- `app.py`: FastAPI app factory `create_app(videos_dir, weights_dir,
  snapshots_dir)` — routes per §5, static mount, snapshot mount.
  Injectable dirs make it testable.
- `schemas.py`: Pydantic request/response models.
- `scripts/run_demo_ui.py`: entry point — finds the repo root, starts
  uvicorn on 127.0.0.1:8000, opens the browser. `--port`, `--no-browser`,
  `--videos-dir` overrides.

**Step 3 — frontend (`src/frontend/`).**
- `index.html`, `styles.css`, `app.js` per §6. No build step, no
  webfonts, no framework. Vanilla JS with `fetch` + polling.

**Step 4 — repo wiring.**
- `requirements.txt`: fastapi, uvicorn, python-multipart, httpx (tests),
  ultralytics, opencv-python, numpy — with the versions tested on.
- `videos/.gitkeep` + note in README that clips go there (media is
  gitignored already).
- README: "Demo UI" section (one-command start, what it does).
- `docs/training_run_log.md`: not touched (training log, not UI work).

Commits (conventional, small, one per step):
`refactor(pipeline): extract shared demo worker and annotations`,
`feat(backend): add FastAPI demo runner and API`,
`feat(ui): add vanilla demo console frontend`,
`chore(docs): document the demo UI and its requirements`.

---

## 8. Test plan

| Layer | What | How | Pass bar |
|---|---|---|---|
| Unit | `DemoRunner` start/stop/config, stats shape, snapshot listing/parsing, upload validation | `tests/test_demo_runner.py` (unittest, fake weights rejected before inference starts) | All pass |
| API | Every endpoint: happy path + errors (400/409/422/503) | `tests/test_demo_api.py` (FastAPI TestClient, temp dirs) | All pass |
| Refactor | CLI demo unchanged | Full suite + `python -c "import"` of the demo script + headless worker smoke on Test7.mp4 (real inference, CPU) | Suite green; worker produces annotated frames + stats |
| Live walkthrough | The real UI in a real browser: load, pick video + weights, start, stream renders, stats update, stop, snapshots appear, fullscreen works, console clean, mobile width sane | Playwright (`webapp_testing` skill + `live_webapp_walkthrough` skill), server on the VM with Test7.mp4 | Every check passes; screenshots kept |
| Manual (user's laptop) | Webcam source, OpenVINO weights, full-length clip, siren toggle | User runs `scripts/run_demo_ui.py` before Dec 5 | Checklist in §10 |

No claim of "ready" until: unit + API tests pass, the live
walkthrough is clean (console + network), and the CLI demo still works.

### Verification results (2026-11-21, branch `feature/demo-web-ui`)

- **Unit + API tests:** 62/62 green in ~1 s (35 pre-existing + 15
  runner + 12 API).
- **Headless worker smoke:** real v3 weights + Test7.mp4 — 145 frames,
  display FPS ~19, detect FPS ramping to ~5 (CPU-bound VM), clean
  auto-stop at end of video, JPEG frames produced.
- **Playwright live walkthrough** (`walkthrough_demo_ui.py`, kept in
  the dataset workspace, not the repo): **20/20 checks pass** —
  idle/running/stopped states, stream frames, advancing stats, alert
  banner + snapshot + gallery + dialog, fullscreen in/out, 390 px
  width without overflow, zero console errors, zero failed requests.
  The walkthrough runs on `walkthrough_dense_30s.mp4` (Test2 45–75 s,
  the densest confirmed-threat segment per the RQ3 v3 data) with
  min-hits 1 — see the note below.
- **Pre-demo smoke script:** `scripts/smoke_demo_ui.py` — 9/9 checks
  pass (server, listings, page, snapshots, run, MJPEG frame, frames
  advance, stop).

**Bugs found by testing and fixed on this branch:**
1. `stop()` called `status()` while holding the runner lock —
   `threading.Lock` is not re-entrant, so the first `stop()` hung
   forever. Fixed by snapshotting under the lock and building the
   status outside it.
2. CSS `display` rules overrode the `hidden` attribute, so the alert
   banner and REC badge showed on the idle page. Fixed with
   `[hidden] { display: none !important; }`.
3. The alert banner only showed while the latest frame had a
   confirmation — a brief hit flashed and vanished. It now latches
   for 5 s after the last confirmed observation.
4. The topbar overflowed horizontally at 390 px. It now wraps and
   hides the "Demo Console" subtitle below 640 px.
5. httpx's env-proxy parsing chokes on bracketed IPv6 entries in
   `NO_PROXY` (this VM's environment). The smoke script creates its
   client with `trust_env=False` — localhost traffic never needs a
   proxy.

### Fixes from the first live test (2026-11-21, branch `fix/demo-ui-live-bugs`)

6. **Stale frame after switching sources.** The stream `<img>` was
   attached once per page (`hasStreamed`) while the backend MJPEG
   generator ends with each run, so stopping a webcam run and starting
   a video left the webcam's last frame on screen even though the
   backend was running and taking snapshots. Fixed: the attachment is
   now per-run — `applyStatus` re-attaches on every transition into
   running and detaches (back to the idle placeholder) on
   stop/error/idle.
7. **No way to stop the server.** Closing the tab leaves the uvicorn
   process running. Fixed: `POST /api/server/shutdown` stops the
   active run and sets `uvicorn.Server.should_exit` (D11). The
   launcher now owns the Server object instead of calling
   `uvicorn.run()`; under a plain `uvicorn ...:main_app` launch the
   endpoint answers 409 (nothing to signal). The console has a quiet
   "Stop server" button at the bottom of the control panel with a
   confirm dialog; after shutdown it stops polling and says how to
   restart. The smoke script checks the endpoint and the exit (11
   checks).

**D11 — server shutdown:** closing the browser tab does not stop the
uvicorn process, so the console has its own off-switch:
`POST /api/server/shutdown` stops the active run, then sets
`uvicorn.Server.should_exit`. The launcher owns the Server object for
this; the factory form (`main_app`) has no server to signal.

### Run controls, warmup, and upload cleanup (2026-11-23, branch `feature/demo-ui-run-controls`)

- **D12 — explicit `starting` state + detector warmup & cache:**
  `DemoRunner.start()` sets `state = "starting"` before loading and
  warming up `WeaponDetector` (`warmup()` runs one dummy frame + resets
  ByteTrack) and before opening `FrameSource`, so the UI never flickers
  back to "Run stopped." during polling and frame 1 of the video runs at
  full detect FPS. Warmed detectors are cached in `DemoRunner` by
  `(weights_path, imgsz, device)` so consecutive runs start immediately;
  for OpenVINO directories, `intel:gpu` is auto-selected when available
  (with automatic CPU fallback) and `OPENVINO_CACHE_DIR` points to
  `models/weights/.ov_cache`.
- **D13 — explicit Pause / Resume endpoints:** `POST /api/run/pause` and
  `POST /api/run/resume` (rather than a single toggle, so retries are
  explicit and safe). Pausing holds file capture position in
  `FrameSource`, pauses `InferenceWorker` and the `elapsed_s` timer, and
  keeps `/api/stream` alive showing the frozen annotated frame; stopping
  while paused works cleanly.
- **D14 — delete uploaded videos:** `DELETE /api/sources/{name:path}`
  deletes only files inside `videos/uploads/` (rejecting traversal and
  non-upload files with 400, missing files with 404, and active-run
  conflicts with 409). The UI shows a `×` button beside the Video
  selector whenever an `uploads/...` clip is selected.

**Why the walkthrough uses min-hits 1:** this VM runs inference at
~2 FPS, so analyzed frames are ~25 video-frames apart and ByteTrack
cannot hold one track ID long enough for 3 votes (Test4 confirmed
only 8/633 frames even at 10 FPS sampling in RQ3). On the demo laptop
(OpenVINO ~30 FPS) the worker analyzes near-consecutive frames and
the default min-hits 3 works — the field sessions proved it
(36 + 52 + 48 snapshots from the same engine). The walkthrough
verifies the alert/snapshot UI path, not tracking at 2 FPS.

---

## 9. Future work (FYP-II hooks — named now so v1 doesn't block them)

- **WebSocket streaming + Canvas overlays** (FR-06): replace MJPEG with
  a WS frame channel + a detections channel; the runner already owns
  the latest frame and detections, so the endpoint is additive.
- **React dashboard** (FR-06/07): the API in §5 is the contract the
  React app consumes; the vanilla UI stays as the fallback.
- **SQLite incident log** (FR-08): snapshots gain a DB row; the
  snapshot list endpoint becomes a query over the DB.
- **Email alerts** (FR-09): hook into the confirmed-threat event the
  runner already emits.
- **Pose/LSTM stream** (FR-03/04): a second worker beside the weapon
  worker; the runner aggregates both.
- **RTSP input**: a third `source_type` in the start body.

---

## 10. Monitor & demo-day operations

- **Logging:** the runner logs run start/stop, source, weights,
  errors, and per-run summary (frames, confirmed) to stdout; the
  smoke script checks the log for errors.
- **Health:** `GET /api/health` + the UI shows a "backend unreachable"
  banner if polling fails twice in a row.
- **Pre-demo smoke script** (`scripts/smoke_demo_ui.py`): starts the
  server on a test port, hits every endpoint, runs a 5-second clip
  end to end, asserts a stream frame + stats + (if a threat confirms)
  a snapshot, prints PASS/FAIL per check. Run it the morning of the
  review.
- **Failure modes the UI must survive:** weights file missing (clear
  error, no crash), video file deleted mid-list (400 on start),
  webcam busy (error + "close other apps using the camera"), port
  8000 taken (entry script tries 8001 and prints the URL), browser
  closed mid-run (run keeps going server-side; reopening shows the
  live state), server killed mid-run (next start is clean — no stale
  lock files; the runner is in-memory only).
- **Demo-day checklist (for the user):** weights in `models/weights/`,
  clips in `videos/`, Redis not needed (this UI does not use it),
  run the smoke script, keep the terminal visible behind the browser
  (logs are the fallback diagnosis).

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| CPU inference on the VM is ~4 FPS, so live tests are slow | Test with short clips (Test7, 6 s); the jury laptop runs OpenVINO at ~30 FPS. |
| Webcam only works when the server runs on the demo laptop | Documented; the jury demo runs locally, so this is the normal case. |
| OpenVINO weights need the `openvino` package on the laptop | The weights picker labels the kind; a missing package is a clear error naming the pip install. |
| Cherry-picked FPS fix diverges if the FPS branch changes | The FPS fix is finished and frozen; rebase drops the duplicate on merge. |
| Scope creep into FYP-II features | §2 is the fence; anything in §9 needs a new decision, not a quiet addition. |
